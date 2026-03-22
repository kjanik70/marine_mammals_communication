"""PyTorch Dataset classes for marine mammal communication training."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.symbolic_tokenizer import SymbolicVocab, tokenize_coda_sequence
from src.data.dialogue_builder import (
    build_dialogue_sequences,
    load_dialogues,
    load_coda_sequences,
    split_dialogues,
    tokenize_dialogue,
)


class CodaSequenceDataset(Dataset):
    """Dataset of individual whale coda sequences (Track 1, step 1).

    Each item is a sequence of coda type tokens from a single whale
    in a single recording session. Used for next-token prediction.
    """

    def __init__(
        self,
        sequences: list[dict],
        vocab: SymbolicVocab,
        max_seq_len: int = 128,
        filter_noise: bool = True,
    ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.filter_noise = filter_noise

        # Tokenize all sequences
        self.token_sequences = []
        for seq in sequences:
            tokens = tokenize_coda_sequence(
                seq["codas"],
                vocab,
                include_whale_ids=False,
                include_pauses=False,
                filter_noise=filter_noise,
            )
            if len(tokens) > 3:  # BOS + at least 1 coda + EOS
                self.token_sequences.append(tokens)

    def __len__(self) -> int:
        return len(self.token_sequences)

    def __getitem__(self, idx: int) -> dict:
        tokens = self.token_sequences[idx]

        # Truncate to max_seq_len + 1 (need input + target)
        if len(tokens) > self.max_seq_len + 1:
            tokens = tokens[: self.max_seq_len + 1]

        # Pad if shorter
        pad_len = (self.max_seq_len + 1) - len(tokens)
        if pad_len > 0:
            tokens = tokens + [self.vocab.PAD] * pad_len

        tokens = torch.tensor(tokens, dtype=torch.long)
        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = (input_ids != self.vocab.PAD).long()

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
        }


class DialogueDataset(Dataset):
    """Dataset of multi-whale dialogue sequences (Track 1, step 2).

    Each item is a full dialogue: interleaved whale IDs, coda types,
    and pause tokens. Format: <bos> <whale_X> <coda> <pause> <whale_Y> <coda> ... <eos>
    """

    def __init__(
        self,
        dialogues: list[dict],
        vocab: SymbolicVocab,
        max_seq_len: int = 256,
        include_pauses: bool = True,
    ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len

        # Tokenize all dialogues
        self.token_sequences = []
        for dlg in dialogues:
            tokens = tokenize_dialogue(dlg, vocab, include_pauses=include_pauses)
            if len(tokens) > 3:
                self.token_sequences.append(tokens)

        # For longer dialogues, create sliding windows
        self.windows = []
        for tokens in self.token_sequences:
            if len(tokens) <= max_seq_len + 1:
                self.windows.append(tokens)
            else:
                # Sliding window with 50% overlap
                stride = max_seq_len // 2
                for start in range(0, len(tokens) - max_seq_len, stride):
                    window = tokens[start: start + max_seq_len + 1]
                    self.windows.append(window)
                # Include the last window
                self.windows.append(tokens[-(max_seq_len + 1):])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        tokens = self.windows[idx]

        # Truncate
        if len(tokens) > self.max_seq_len + 1:
            tokens = tokens[: self.max_seq_len + 1]

        # Pad
        pad_len = (self.max_seq_len + 1) - len(tokens)
        if pad_len > 0:
            tokens = tokens + [self.vocab.PAD] * pad_len

        tokens = torch.tensor(tokens, dtype=torch.long)
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        attention_mask = (input_ids != self.vocab.PAD).long()

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
        }


class AudioTokenDataset(Dataset):
    """Dataset of pre-tokenized audio sequences (Track 2).

    Loads token arrays from disk (produced by the audio tokenizer).
    Each item is a fixed-length sequence of audio codec tokens.

    Supports data augmentation to reduce overfitting:
    - token_noise_prob: probability of perturbing each token by ±1-3
    - token_mask_prob: probability of replacing each token with a random value
    - time_stretch_prob: probability of stretching/compressing the sequence
    """

    def __init__(
        self,
        token_dir: str | Path,
        max_seq_len: int = 512,
        pad_token: int = 0,
        vocab_size: int = 1026,
        augment: bool = False,
        token_noise_prob: float = 0.05,
        token_mask_prob: float = 0.02,
        time_stretch_prob: float = 0.3,
    ):
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token
        self.vocab_size = vocab_size
        self.augment = augment
        self.token_noise_prob = token_noise_prob
        self.token_mask_prob = token_mask_prob
        self.time_stretch_prob = time_stretch_prob

        # Support multiple token directories
        if isinstance(token_dir, (list, tuple)):
            token_dirs = [Path(d) for d in token_dir]
        else:
            token_dirs = [Path(token_dir)]

        # Load all into memory (small enough for this dataset)
        self.token_sequences = []
        for td in token_dirs:
            for f in sorted(td.glob("*.npy")):
                tokens = np.load(f)
                if len(tokens) > 2:
                    self.token_sequences.append(tokens)

        # Create sliding windows for sequences longer than max_seq_len
        self.windows = []
        for tokens in self.token_sequences:
            if len(tokens) <= max_seq_len + 1:
                self.windows.append(tokens)
            else:
                stride = max_seq_len // 2
                for start in range(0, len(tokens) - max_seq_len, stride):
                    self.windows.append(tokens[start: start + max_seq_len + 1])
                self.windows.append(tokens[-(max_seq_len + 1):])

    def _augment_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """Apply token-level data augmentation."""
        tokens = tokens.copy()
        real_mask = tokens > self.pad_token  # only augment non-pad tokens

        # Token noise: perturb values by ±1-3 (like slight pitch/timing jitter)
        if self.token_noise_prob > 0:
            noise_mask = np.random.random(len(tokens)) < self.token_noise_prob
            noise_mask &= real_mask
            perturbation = np.random.choice([-2, -1, 1, 2], size=len(tokens))
            tokens[noise_mask] = np.clip(
                tokens[noise_mask] + perturbation[noise_mask],
                1, self.vocab_size - 1  # keep in valid range (above PAD)
            )

        # Token masking: replace with random token (like dropout at token level)
        if self.token_mask_prob > 0:
            mask = np.random.random(len(tokens)) < self.token_mask_prob
            mask &= real_mask
            tokens[mask] = np.random.randint(1, self.vocab_size, size=mask.sum())

        # Time stretch: duplicate or drop tokens (like tempo perturbation)
        if self.time_stretch_prob > 0 and np.random.random() < self.time_stretch_prob:
            factor = np.random.uniform(0.9, 1.1)
            new_len = max(3, int(len(tokens) * factor))
            indices = np.linspace(0, len(tokens) - 1, new_len).astype(int)
            tokens = tokens[indices]

        return tokens

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        tokens = self.windows[idx]

        # Apply augmentation during training
        if self.augment:
            tokens = self._augment_tokens(tokens)

        if len(tokens) > self.max_seq_len + 1:
            tokens = tokens[: self.max_seq_len + 1]

        pad_len = (self.max_seq_len + 1) - len(tokens)
        if pad_len > 0:
            tokens = np.concatenate([tokens, np.full(pad_len, self.pad_token)])

        tokens = torch.tensor(tokens, dtype=torch.long)
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        attention_mask = (input_ids != self.pad_token).long()

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
        }


def create_symbolic_datasets(
    codas_csv: str | Path,
    dialogues_csv: str | Path,
    max_seq_len: int = 128,
    dialogue_max_seq_len: int = 256,
    seed: int = 42,
) -> dict:
    """Create all symbolic datasets from CETI data.

    Returns dict with keys: vocab, coda_train, coda_val, coda_test,
    dialogue_train, dialogue_val, dialogue_test.
    """
    vocab = SymbolicVocab()

    # Individual coda sequences
    coda_seqs = load_coda_sequences(codas_csv)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(coda_seqs))
    n_train = int(len(coda_seqs) * 0.8)
    n_val = int(len(coda_seqs) * 0.1)

    coda_train = CodaSequenceDataset(
        [coda_seqs[i] for i in perm[:n_train]], vocab, max_seq_len
    )
    coda_val = CodaSequenceDataset(
        [coda_seqs[i] for i in perm[n_train:n_train + n_val]], vocab, max_seq_len
    )
    coda_test = CodaSequenceDataset(
        [coda_seqs[i] for i in perm[n_train + n_val:]], vocab, max_seq_len
    )

    # Dialogue sequences
    dlg_df = load_dialogues(dialogues_csv)
    dialogues = build_dialogue_sequences(dlg_df, min_codas=3)
    dlg_train, dlg_val, dlg_test = split_dialogues(dialogues, seed=seed)

    dialogue_train = DialogueDataset(dlg_train, vocab, dialogue_max_seq_len)
    dialogue_val = DialogueDataset(dlg_val, vocab, dialogue_max_seq_len)
    dialogue_test = DialogueDataset(dlg_test, vocab, dialogue_max_seq_len)

    return {
        "vocab": vocab,
        "coda_train": coda_train,
        "coda_val": coda_val,
        "coda_test": coda_test,
        "dialogue_train": dialogue_train,
        "dialogue_val": dialogue_val,
        "dialogue_test": dialogue_test,
    }
