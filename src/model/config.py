"""Model configuration for the causal transformer."""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for the causal transformer decoder."""

    vocab_size: int = 74          # Set to match tokenizer vocab
    max_seq_len: int = 256        # Maximum sequence length
    n_layers: int = 6             # Number of transformer blocks
    n_heads: int = 4              # Number of attention heads
    d_model: int = 256            # Model dimension
    d_ff: int = 1024              # Feed-forward inner dimension
    dropout: float = 0.1          # Dropout rate
    use_gradient_checkpointing: bool = False

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def param_count_estimate(self) -> int:
        """Rough parameter count estimate."""
        # Embedding
        emb = self.vocab_size * self.d_model
        # Per layer: attn (4 * d_model^2) + ffn (2 * d_model * d_ff) + norms
        per_layer = 4 * self.d_model ** 2 + 2 * self.d_model * self.d_ff + 4 * self.d_model
        # Output head
        head = self.d_model * self.vocab_size
        return emb + self.n_layers * per_layer + head


# Presets
TINY = TransformerConfig(
    n_layers=6, n_heads=4, d_model=256, d_ff=1024,
)  # ~8M params

SMALL = TransformerConfig(
    n_layers=8, n_heads=8, d_model=512, d_ff=2048,
)  # ~35M params

MEDIUM = TransformerConfig(
    n_layers=12, n_heads=12, d_model=768, d_ff=3072,
)  # ~85M params

LARGE = TransformerConfig(
    n_layers=16, n_heads=16, d_model=1024, d_ff=4096,
    use_gradient_checkpointing=True,
)  # ~200M params

XLARGE = TransformerConfig(
    n_layers=24, n_heads=16, d_model=1280, d_ff=5120,
    use_gradient_checkpointing=True,
)  # ~350M params

PRESETS = {
    "tiny": TINY,
    "small": SMALL,
    "medium": MEDIUM,
    "large": LARGE,
    "xlarge": XLARGE,
}


def get_config(preset: str, **overrides) -> TransformerConfig:
    """Get a config preset with optional overrides."""
    cfg = PRESETS[preset]
    # Create a new instance with overrides
    fields = {k: v for k, v in cfg.__dict__.items()}
    fields.update(overrides)
    return TransformerConfig(**fields)
