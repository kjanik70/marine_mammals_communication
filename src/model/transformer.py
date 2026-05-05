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

    def forward(self, x: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for rotary embeddings.

        Args:
            x: Tensor of shape (batch, seq_len, ...)
            offset: Position offset for KV-cached generation

        Returns:
            (cos, sin) each of shape (1, seq_len, 1, d_head)
        """
        seq_len = x.shape[1]
        t = torch.arange(offset, offset + seq_len, device=x.device, dtype=self.inv_freq.dtype)
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
        self.split_qkv = getattr(config, 'use_split_qkv', False)
        if self.split_qkv:
            self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        else:
            self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        if self.split_qkv:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE (cos/sin already offset-aware)
        cos = cos[:, :T, :, :]  # (1, T, 1, d_head)
        sin = sin[:, :T, :, :]
        q = apply_rope(q, cos.transpose(1, 2), sin.transpose(1, 2))
        k = apply_rope(k, cos.transpose(1, 2), sin.transpose(1, 2))

        # KV cache: append new K/V to cached
        new_cache = None
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
            new_cache = (k, v)
            # Single new token attending to full history — no causal mask needed
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            new_cache = (k, v)
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out), new_cache


class SlidingWindowAttention(nn.Module):
    """Multi-head causal self-attention with sliding window and RoPE.

    Same interface as CausalSelfAttention but restricts each token to
    attend only within a local window of swa_window_size tokens back.
    The mask buffer is shared across all SWA layers (set by CausalTransformer).
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.window_size = config.swa_window_size
        self.split_qkv = getattr(config, 'use_split_qkv', False)

        if self.split_qkv:
            self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        else:
            self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        if self.split_qkv:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.split(self.d_model, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        cos = cos[:, :T, :, :]
        sin = sin[:, :T, :, :]
        q = apply_rope(q, cos.transpose(1, 2), sin.transpose(1, 2))
        k = apply_rope(k, cos.transpose(1, 2), sin.transpose(1, 2))

        # KV cache: append and trim to window size
        new_cache = None
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
            # Trim to window size
            if k.shape[2] > self.window_size:
                k = k[:, :, -self.window_size:, :]
                v = v[:, :, -self.window_size:, :]
            new_cache = (k, v)
            # Single new token attending to cached window — no mask needed
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            new_cache = (k, v)
            W = self.window_size
            if self.training:
                # Chunked SWA: split into window-sized chunks with overlap,
                # using is_causal=True (flash kernel, no mask tensor needed).
                if T <= W:
                    attn_out = F.scaled_dot_product_attention(
                        q, k, v, is_causal=True,
                        dropout_p=self.dropout.p if self.training else 0.0)
                else:
                    n_chunks = (T + W - 1) // W
                    outputs = []
                    for c in range(n_chunks):
                        q_start = c * W
                        q_end = min((c + 1) * W, T)
                        k_start = max(0, (c - 1) * W)
                        k_end = q_end
                        out_chunk = F.scaled_dot_product_attention(
                            q[:, :, q_start:q_end, :],
                            k[:, :, k_start:k_end, :],
                            v[:, :, k_start:k_end, :],
                            is_causal=True,
                            dropout_p=self.dropout.p if self.training else 0.0)
                        outputs.append(out_chunk)
                    attn_out = torch.cat(outputs, dim=2)
            else:
                # Eval (loss computation): same chunked approach as training.
                # Generation always uses the kv_cache branch above, so this
                # path is never hit for autoregressive decoding.
                if T <= W:
                    attn_out = F.scaled_dot_product_attention(
                        q, k, v, is_causal=True, dropout_p=0.0)
                else:
                    n_chunks = (T + W - 1) // W
                    outputs = []
                    for c in range(n_chunks):
                        q_start = c * W
                        q_end = min((c + 1) * W, T)
                        k_start = max(0, (c - 1) * W)
                        k_end = q_end
                        outputs.append(F.scaled_dot_product_attention(
                            q[:, :, q_start:q_end, :],
                            k[:, :, k_start:k_end, :],
                            v[:, :, k_start:k_end, :],
                            is_causal=True, dropout_p=0.0))
                    attn_out = torch.cat(outputs, dim=2)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out), new_cache


class CompressedGlobalAttention(nn.Module):
    """Causal global attention over stride-compressed K/V (NSA-style).

    Every global attention layer, K and V are downsampled by taking every
    `stride`-th token, reducing attended context from T to T//stride tokens.
    Queries remain full-resolution.

    Causal rule: query at position i may attend to compressed key j
    only when j * stride <= i.  The mask is built per query-chunk so peak
    memory is O(chunk_size × T_c) regardless of total sequence length.

    For stride=72 and seq_len=128K:
      T_c ≈ 1820 anchors spanning ~169s of audio (~0.093s per anchor).
      Peak SDPA memory per chunk: O(2048 × 1820) ≈ 90 MB for all heads —
      vs O(128K²) ≈ 200 GB for naive full attention.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.stride = config.compressed_attn_stride
        self.chunk = config.compressed_attn_chunk
        self.split_qkv = getattr(config, 'use_split_qkv', False)
        if self.split_qkv:
            self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        else:
            self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        H, D = self.n_heads, self.d_head
        S = self.stride

        if self.split_qkv:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.split(self.d_model, dim=-1)

        q = q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Apply RoPE to Q and K at their original sequence positions
        cos_t = cos[:, :T, :, :].transpose(1, 2)  # (1, 1, T, D)
        sin_t = sin[:, :T, :, :].transpose(1, 2)
        q = apply_rope(q, cos_t, sin_t)
        k = apply_rope(k, cos_t, sin_t)

        if kv_cache is not None:
            # Generation: full-resolution K/V cache; compress on the fly for attention
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
            new_cache = (k, v)
            k_c = k[:, :, ::S, :]  # (B, H, T_c, D)
            v_c = v[:, :, ::S, :]
            # Single new query token — sees all past compressed keys, no mask needed
            attn_out = F.scaled_dot_product_attention(q, k_c, v_c, dropout_p=0.0)
        else:
            new_cache = (k, v)
            # Compress K/V: take every S-th position (aligned to original positions)
            k_c = k[:, :, ::S, :]  # (B, H, T_c, D)
            v_c = v[:, :, ::S, :]
            T_c = k_c.shape[2]

            # Process Q in chunks to bound peak attention memory
            CQ = self.chunk
            outputs = []
            for q_start in range(0, T, CQ):
                q_end = min(q_start + CQ, T)
                q_ch = q[:, :, q_start:q_end, :]  # (B, H, cq, D)
                cq = q_end - q_start

                # Compressed keys accessible to this chunk:
                # key j (at position j*S) is accessible if j*S <= q_end - 1
                n_keys = min((q_end - 1) // S + 1, T_c)
                k_ch = k_c[:, :, :n_keys, :]
                v_ch = v_c[:, :, :n_keys, :]

                # Causal mask: query at global position i sees key j iff j*S <= i
                query_pos = torch.arange(q_start, q_end, device=x.device)  # (cq,)
                key_pos = torch.arange(n_keys, device=x.device) * S         # (n_keys,)
                # True = mask out (future)
                causal = key_pos.unsqueeze(0) > query_pos.unsqueeze(1)  # (cq, n_keys)
                bias = x.new_zeros(1, 1, cq, n_keys)
                bias = bias.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

                out_ch = F.scaled_dot_product_attention(
                    q_ch, k_ch, v_ch,
                    attn_mask=bias,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                outputs.append(out_ch)

            attn_out = torch.cat(outputs, dim=2)  # (B, H, T, D)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out), new_cache


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


class MoEFeedForward(nn.Module):
    """Mixture of Experts feed-forward with top-K routing.

    Replaces dense FeedForward with N expert FFNs and a learned router.
    Includes load-balancing auxiliary loss (Switch Transformer style).
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.moe_top_k
        self.use_bias_routing = getattr(config, 'use_bias_routing', False)

        self.gate = nn.Linear(config.d_model, config.n_experts, bias=False)

        # Bias routing (DeepSeek V4-style): per-expert learned bias instead of aux_loss
        if self.use_bias_routing:
            self.expert_bias = nn.Parameter(torch.zeros(config.n_experts))

        # Build experts with controllable size
        expert_ff = config.expert_d_ff if config.expert_d_ff > 0 else config.d_ff
        expert_config = TransformerConfig(
            d_model=config.d_model, d_ff=expert_ff, dropout=config.dropout)
        self.experts = nn.ModuleList(
            [FeedForward(expert_config) for _ in range(config.n_experts)])
        self._aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        N = x_flat.shape[0]

        # Router: optionally add per-expert bias for load balancing
        logits = self.gate(x_flat)  # (N, n_experts)
        if self.use_bias_routing:
            logits = logits + self.expert_bias
        top_k_logits, top_k_idx = logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (N, top_k)

        if self.use_bias_routing:
            # Bias routing: no auxiliary loss — expert_bias handles load balancing
            self._aux_loss = x_flat.new_zeros(())
        else:
            # Switch Transformer auxiliary loss for load balancing
            router_probs = F.softmax(logits, dim=-1)
            tokens_per_expert = torch.zeros(self.n_experts, device=x.device)
            for k in range(self.top_k):
                tokens_per_expert.scatter_add_(
                    0, top_k_idx[:, k],
                    torch.ones(N, device=x.device))
            f = tokens_per_expert / tokens_per_expert.sum()
            P = router_probs.mean(dim=0)
            self._aux_loss = self.n_experts * (f * P).sum()

        # Dispatch to experts
        output = torch.zeros_like(x_flat)
        flat_idx = top_k_idx.view(-1)
        flat_w = top_k_weights.view(-1)
        tok_idx = torch.arange(N, device=x.device).repeat_interleave(self.top_k)

        for i, expert in enumerate(self.experts):
            mask = flat_idx == i
            if not mask.any():
                continue
            tokens = x_flat[tok_idx[mask]]
            out = expert(tokens)
            idx = tok_idx[mask].unsqueeze(-1).expand_as(out)
            output.scatter_add_(0, idx, out * flat_w[mask].unsqueeze(-1))

        return output.view(B, T, D)


class TransformerBlock(nn.Module):
    """Single transformer decoder block.

    Supports configurable attention (full or sliding window) and
    FFN (dense or MoE) based on config and layer position.
    """

    def __init__(self, config: TransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.d_model)

        # Attention type per layer:
        #  - Local layers: SlidingWindowAttention (when swa_window_size > 0)
        #  - Global layers (every full_attention_every_n):
        #      * CompressedGlobalAttention when compressed_attn_stride > 0 (NSA-style)
        #      * CausalSelfAttention otherwise (full O(T²) attention)
        is_global = (config.full_attention_every_n > 0 and
                     (layer_idx + 1) % config.full_attention_every_n == 0)
        use_compressed = (is_global and
                          getattr(config, 'compressed_attn_stride', 0) > 0)
        use_swa = (config.swa_window_size > 0 and
                   config.full_attention_every_n > 0 and not is_global)

        if use_compressed:
            self.attn = CompressedGlobalAttention(config)
        elif use_swa:
            self.attn = SlidingWindowAttention(config)
        else:
            self.attn = CausalSelfAttention(config)

        self.ff_norm = nn.RMSNorm(config.d_model)

        # MoE on ALL FFN layers when n_experts > 1
        self.ff = MoEFeedForward(config) if config.n_experts > 1 else FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, attention_mask, kv_cache)
        x = x + attn_out
        x = x + self.ff(self.ff_norm(x))
        return x, new_cache


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
            TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)
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
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> dict:
        """Forward pass.

        Args:
            input_ids: (B, T) token IDs
            attention_mask: (B, T) mask (1=real, 0=padding)
            targets: (B, T) target token IDs for loss computation
            past_kv: List of (K, V) caches per layer for generation

        Returns:
            dict with 'logits' and optionally 'loss', 'past_kv'
        """
        B, T = input_ids.shape

        x = self.token_emb(input_ids)
        x = self.drop(x)

        # Position offset for KV-cached generation
        # Use max cache size across layers — SWA layers trim their cache,
        # but full attention layers keep the full history.
        offset = max(kv[0].shape[2] for kv in past_kv) if past_kv is not None else 0
        cos, sin = self.rope(x, offset=offset)

        new_kv = []
        for i, block in enumerate(self.blocks):
            layer_cache = past_kv[i] if past_kv is not None else None
            if self.config.use_gradient_checkpointing and self.training:
                # Gradient checkpointing doesn't use KV cache (training only)
                x, _ = checkpoint(block, x, cos, sin, attention_mask, None, use_reentrant=False)
            else:
                x, cache = block(x, cos, sin, attention_mask, layer_cache)
                if targets is None:  # only accumulate KV cache for generation, not eval/training
                    new_kv.append(cache)

        x = self.norm(x)

        result = {}

        if new_kv:
            result["past_kv"] = new_kv

        # Collect MoE auxiliary losses
        aux_losses = [block.ff._aux_loss for block in self.blocks
                      if hasattr(block.ff, '_aux_loss')]
        if aux_losses:
            result["aux_loss"] = sum(aux_losses)

        if targets is not None:
            # Gradient-checkpointed chunked cross-entropy: each chunk's logit
            # tensor is never retained in the autograd graph — it's freed after
            # the chunk forward and recomputed during backward.  This keeps peak
            # VRAM at O(chunk × vocab) rather than O(seq_len × vocab), which is
            # critical at 128K context (2.4 GB in bf16 if retained naively).
            flat_x = x.view(-1, x.size(-1))
            flat_targets = targets.view(-1)
            chunk = 2048
            total_loss = x.new_zeros(())
            n_valid = 0
            weight = self.lm_head.weight

            def _ce_chunk(x_c, t_c, w):
                return F.cross_entropy(
                    F.linear(x_c, w), t_c,
                    ignore_index=0, reduction="sum",
                )

            for s in range(0, flat_x.size(0), chunk):
                e = min(s + chunk, flat_x.size(0))
                tgt_chunk = flat_targets[s:e]
                valid = (tgt_chunk != 0).sum().item()
                if valid == 0:
                    continue
                chunk_loss = checkpoint(
                    _ce_chunk, flat_x[s:e], tgt_chunk, weight,
                    use_reentrant=False,
                )
                total_loss = total_loss + chunk_loss
                n_valid += valid
            result["loss"] = total_loss / max(n_valid, 1)
            result["logits"] = None
        else:
            logits = self.lm_head(x)
            result["logits"] = logits

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
        """Autoregressive generation with KV cache.

        Prefills the cache with the prompt in one pass, then generates
        one token at a time using cached K/V for O(1) per step.

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

        # Prefill: process full prompt and populate KV cache
        prompt = input_ids[:, :self.config.max_seq_len]
        result = self.forward(prompt)
        past_kv = result.get("past_kv", None)
        logits = result["logits"][:, -1, :]

        generated = []
        for _ in range(max_new_tokens):
            # Sample next token
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = logits / temperature

                if top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[:, [-1]]] = -float("inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    scaled[indices_to_remove] = -float("inf")

                probs = scaled.softmax(dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token)

            if (next_token == eos_token_id).all():
                break

            # Forward single token with KV cache
            result = self.forward(next_token, past_kv=past_kv)
            past_kv = result.get("past_kv", None)
            logits = result["logits"][:, -1, :]

        if generated:
            input_ids = torch.cat([input_ids] + generated, dim=1)
        return input_ids

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
