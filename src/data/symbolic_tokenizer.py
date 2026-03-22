"""Symbolic tokenizer for CETI sperm whale coda annotations.

Converts coda annotations (type, ICI pattern, duration) into discrete token sequences.
Each coda becomes a single token (its type ID). Special tokens encode whale identity,
pauses between codas, and sequence boundaries.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Pause duration bins (seconds between codas)
PAUSE_BINS = [0.5, 2.0, 5.0, 15.0]  # boundaries -> 5 bins


@dataclass
class SymbolicVocab:
    """Vocabulary for symbolic coda tokenization."""

    # Special tokens
    PAD: int = 0
    BOS: int = 1
    EOS: int = 2

    # Pause tokens (between codas)
    PAUSE_VERY_SHORT: int = 3   # < 0.5s
    PAUSE_SHORT: int = 4        # 0.5-2s
    PAUSE_MEDIUM: int = 5       # 2-5s
    PAUSE_LONG: int = 6         # 5-15s
    PAUSE_VERY_LONG: int = 7    # > 15s

    # Whale ID tokens start at offset 8
    WHALE_OFFSET: int = 8
    MAX_WHALES: int = 20  # supports up to 20 whales per dialogue

    # Coda type tokens start after whale tokens
    CODA_OFFSET: int = 28  # 8 + 20

    # Mapping from coda type string to token ID
    coda_type_to_id: dict = field(default_factory=dict)
    id_to_coda_type: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.coda_type_to_id:
            self._build_default_vocab()

    def _build_default_vocab(self):
        """Build vocabulary from known CETI coda types."""
        coda_types = [
            # Named types from DominicaCodas.csv
            "1+1+3", "1+31", "1+32",
            "2+3",
            "3D", "3R",
            "4D", "4R1", "4R2",
            "5R1", "5R2", "5R3",
            "6R", "6i",
            "7D1", "7D2", "7R", "7i",
            "8D", "8R", "8i",
            "9R", "9i",
            "10R", "10i",
            # Noise types (kept separate - could filter these)
            "1-NOISE", "2-NOISE", "3-NOISE", "4-NOISE", "5-NOISE",
            "6-NOISE", "7-NOISE", "8-NOISE", "9-NOISE", "10-NOISE",
            # Click-count types for dialogues (which lack CodaType labels)
            "clicks:1", "clicks:2", "clicks:3", "clicks:4", "clicks:5",
            "clicks:6", "clicks:7", "clicks:8", "clicks:9", "clicks:10",
            "clicks:11+",
        ]
        for i, ct in enumerate(coda_types):
            self.coda_type_to_id[ct] = self.CODA_OFFSET + i
            self.id_to_coda_type[self.CODA_OFFSET + i] = ct

    @property
    def vocab_size(self) -> int:
        return self.CODA_OFFSET + len(self.coda_type_to_id)

    def whale_token(self, whale_id: int) -> int:
        """Get token ID for a whale identifier (1-indexed)."""
        assert 1 <= whale_id <= self.MAX_WHALES
        return self.WHALE_OFFSET + whale_id - 1

    def pause_token(self, duration_seconds: float) -> int:
        """Get pause token for a given inter-coda duration."""
        if duration_seconds < PAUSE_BINS[0]:
            return self.PAUSE_VERY_SHORT
        elif duration_seconds < PAUSE_BINS[1]:
            return self.PAUSE_SHORT
        elif duration_seconds < PAUSE_BINS[2]:
            return self.PAUSE_MEDIUM
        elif duration_seconds < PAUSE_BINS[3]:
            return self.PAUSE_LONG
        else:
            return self.PAUSE_VERY_LONG

    def coda_token(self, coda_type: str) -> Optional[int]:
        """Get token ID for a coda type string. Returns None if unknown."""
        return self.coda_type_to_id.get(coda_type)

    def decode_token(self, token_id: int) -> str:
        """Convert a token ID back to a human-readable string."""
        if token_id == self.PAD:
            return "<pad>"
        elif token_id == self.BOS:
            return "<bos>"
        elif token_id == self.EOS:
            return "<eos>"
        elif token_id == self.PAUSE_VERY_SHORT:
            return "<pause:0-0.5s>"
        elif token_id == self.PAUSE_SHORT:
            return "<pause:0.5-2s>"
        elif token_id == self.PAUSE_MEDIUM:
            return "<pause:2-5s>"
        elif token_id == self.PAUSE_LONG:
            return "<pause:5-15s>"
        elif token_id == self.PAUSE_VERY_LONG:
            return "<pause:>15s>"
        elif self.WHALE_OFFSET <= token_id < self.CODA_OFFSET:
            whale_id = token_id - self.WHALE_OFFSET + 1
            return f"<whale:{whale_id}>"
        elif token_id in self.id_to_coda_type:
            return self.id_to_coda_type[token_id]
        else:
            return f"<unk:{token_id}>"

    def is_coda_token(self, token_id: int) -> bool:
        return token_id >= self.CODA_OFFSET

    def is_pause_token(self, token_id: int) -> bool:
        return self.PAUSE_VERY_SHORT <= token_id <= self.PAUSE_VERY_LONG

    def is_whale_token(self, token_id: int) -> bool:
        return self.WHALE_OFFSET <= token_id < self.CODA_OFFSET


def tokenize_coda_sequence(
    codas_df: pd.DataFrame,
    vocab: SymbolicVocab,
    include_whale_ids: bool = False,
    include_pauses: bool = True,
    filter_noise: bool = True,
) -> list[int]:
    """Convert a DataFrame of sequential codas into a token sequence.

    Args:
        codas_df: DataFrame with columns matching CETI format.
                  Must be sorted by timestamp if timestamps exist.
        vocab: The symbolic vocabulary.
        include_whale_ids: If True, insert whale ID tokens before each coda.
        include_pauses: If True, insert pause tokens between codas.
        filter_noise: If True, skip codas with '-NOISE' type.

    Returns:
        List of token IDs.
    """
    tokens = [vocab.BOS]
    prev_timestamp = None

    for _, row in codas_df.iterrows():
        coda_type = row.get("CodaType", None)

        # Filter noise codas
        if filter_noise and coda_type and "NOISE" in str(coda_type):
            continue

        # Insert pause token based on time gap
        if include_pauses and "TsTo" in row and prev_timestamp is not None:
            gap = row["TsTo"] - prev_timestamp - row.get("Duration", 0)
            if gap > 0:
                tokens.append(vocab.pause_token(gap))
        prev_timestamp = row.get("TsTo", None)

        # Insert whale ID
        if include_whale_ids and "Whale" in row:
            whale_id = int(row["Whale"])
            if 1 <= whale_id <= vocab.MAX_WHALES:
                tokens.append(vocab.whale_token(whale_id))

        # Insert coda type token
        if coda_type and str(coda_type) != "nan":
            token = vocab.coda_token(str(coda_type))
            if token is not None:
                tokens.append(token)
        elif "nClicks" in row:
            # Fall back to click-count token (for dialogue data without CodaType)
            n = int(row["nClicks"])
            key = f"clicks:{min(n, 11)}+" if n > 10 else f"clicks:{n}"
            token = vocab.coda_token(key)
            if token is not None:
                tokens.append(token)

    tokens.append(vocab.EOS)
    return tokens


def decode_token_sequence(tokens: list[int], vocab: SymbolicVocab) -> list[str]:
    """Convert a token sequence back to human-readable strings."""
    return [vocab.decode_token(t) for t in tokens]
