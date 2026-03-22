"""Causal transformer decoder for marine mammal communication.

GPT-style autoregressive transformer with:
- RoPE (Rotary Positional Embeddings)
- Flash Attention via PyTorch SDPA
- Configurable size presets
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.model.config import TransformerConfig


class RoPE(nn.Module):
    """Rotary Positional Embeddings."""

    def __init__(self, d_head: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for rotary embeddings.

        Args:
            x: Tensor of shape (batch, seq_len, ...)

        Returns:
            (cos, sin) each of shape (1, seq_len, 1, d_head)
        """
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, d_head/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, d_head)
        cos = emb.cos().unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, d_head)
        sin = emb.sin().unsqueeze(0).unsqueeze(2)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor."""
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and Flash Attention."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        cos = cos[:, :T, :, :]  # (1, T, 1, d_head)
        sin = sin[:, :T, :, :]
        q = apply_rope(q, cos.transpose(1, 2), sin.transpose(1, 2))
        k = apply_rope(k, cos.transpose(1, 2), sin.transpose(1, 2))

        # Flash Attention via PyTorch SDPA (handles causal masking efficiently)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ff_norm = nn.RMSNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, attention_mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class CausalTransformer(nn.Module):
    """Autoregressive causal transformer decoder.

    Used for both symbolic (Track 1) and audio token (Track 2) modeling.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RoPE(config.d_head, config.max_seq_len)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass.

        Args:
            input_ids: (B, T) token IDs
            attention_mask: (B, T) mask (1=real, 0=padding)
            targets: (B, T) target token IDs for loss computation

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        B, T = input_ids.shape

        x = self.token_emb(input_ids)
        x = self.drop(x)

        cos, sin = self.rope(x)

        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, cos, sin, attention_mask, use_reentrant=False)
            else:
                x = block(x, cos, sin, attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}

        if targets is not None:
            # Flatten for cross entropy, ignore padding (target == 0)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,  # PAD token
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """Autoregressive generation.

        Args:
            input_ids: (B, T) starting token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            eos_token_id: Stop generation when this token is produced

        Returns:
            (B, T + generated) full sequence including generated tokens
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]

            result = self.forward(idx_cond)
            logits = result["logits"][:, -1, :]  # (B, vocab_size)

            if temperature == 0:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = -float("inf")

                probs = logits.softmax(dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if all sequences have generated EOS
            if (next_token == eos_token_id).all():
                break

        return input_ids

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
