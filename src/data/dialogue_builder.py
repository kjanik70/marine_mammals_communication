"""Build dialogue sequences from CETI sperm whale data.

Reconstructs multi-whale conversations from the dialogue CSV,
producing ordered sequences of (whale_id, coda) events suitable
for tokenization and training.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.symbolic_tokenizer import SymbolicVocab, tokenize_coda_sequence


def load_dialogues(csv_path: str | Path) -> pd.DataFrame:
    """Load the CETI dialogue CSV."""
    df = pd.read_csv(csv_path)
    # Ensure TsTo is float
    df["TsTo"] = pd.to_numeric(df["TsTo"], errors="coerce")
    df["Whale"] = pd.to_numeric(df["Whale"], errors="coerce").astype(int)
    return df


def build_dialogue_sequences(
    df: pd.DataFrame,
    min_codas: int = 3,
    filter_noise: bool = True,
) -> list[dict]:
    """Build a list of dialogue sequences from the raw DataFrame.

    Each dialogue is a dict with:
        - "recording": recording ID string
        - "codas": DataFrame of codas sorted by timestamp
        - "n_codas": number of codas
        - "n_whales": number of distinct whales
        - "duration_s": dialogue span in seconds
        - "whale_ids": list of unique whale IDs

    Args:
        df: The dialogues DataFrame.
        min_codas: Minimum codas to include a dialogue.
        filter_noise: Whether to filter noise codas from count.
    """
    dialogues = []

    for rec_id, group in df.groupby("REC"):
        group = group.sort_values("TsTo").reset_index(drop=True)

        if len(group) < min_codas:
            continue

        whale_ids = sorted(group["Whale"].unique().tolist())
        duration = group["TsTo"].max() - group["TsTo"].min()

        dialogues.append({
            "recording": rec_id,
            "codas": group,
            "n_codas": len(group),
            "n_whales": len(whale_ids),
            "duration_s": duration,
            "whale_ids": whale_ids,
        })

    # Sort by number of codas (largest first)
    dialogues.sort(key=lambda d: d["n_codas"], reverse=True)
    return dialogues


def tokenize_dialogue(
    dialogue: dict,
    vocab: SymbolicVocab,
    include_pauses: bool = True,
) -> list[int]:
    """Tokenize a single dialogue into a token sequence.

    Format: <bos> [<whale_X> <coda_type> <pause>]* <eos>
    """
    return tokenize_coda_sequence(
        dialogue["codas"],
        vocab,
        include_whale_ids=True,
        include_pauses=include_pauses,
        filter_noise=True,
    )


def split_dialogues(
    dialogues: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split dialogues into train/val/test sets.

    Splits by recording to prevent data leakage.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(dialogues))

    n_train = int(len(dialogues) * train_ratio)
    n_val = int(len(dialogues) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = [dialogues[i] for i in train_idx]
    val = [dialogues[i] for i in val_idx]
    test = [dialogues[i] for i in test_idx]

    return train, val, test


def load_coda_sequences(csv_path: str | Path) -> list[dict]:
    """Load the main coda CSV and group into per-whale sequences.

    Groups codas by (Date, Unit, UnitNum) to create sequences
    from the same whale in the same session.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    sequences = []
    for (date, unit, whale_num), group in df.groupby(["Date", "Unit", "UnitNum"]):
        group = group.sort_values("codaNUM2018").reset_index(drop=True)
        if len(group) < 3:
            continue
        sequences.append({
            "date": date,
            "unit": unit,
            "whale_num": whale_num,
            "codas": group,
            "n_codas": len(group),
        })

    sequences.sort(key=lambda s: s["n_codas"], reverse=True)
    return sequences
