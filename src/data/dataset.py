"""PyTorch Dataset classes for marine mammal communication training."""

import bisect
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
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


def _load_npy(path):
    """Load a single .npy file. Returns (stem, tokens) or (stem, None) if too short."""
    tokens = np.load(path)
    if tokens.ndim == 2:
        # 2D array (n_codebooks, T) — use time dimension for length check
        if tokens.shape[1] > 2:
            return (path.stem, tokens)
    elif len(tokens) > 2:
        return (path.stem, tokens)
    return (path.stem, None)


def _scan_npy(path):
    """Scan a .npy file header only. Returns (path, stem, n_tokens) or None if too short.

    Handles both 1D (interleaved) and 2D (n_codebooks, T) arrays.
    For 2D arrays, n_tokens is the time dimension (shape[1]).
    """
    arr = np.load(path, mmap_mode='r')
    if arr.ndim == 2:
        n = arr.shape[1]  # time dimension
    else:
        n = arr.shape[0]
    if n > 2:
        return (path, path.stem, n)
    return None


@dataclass
class _ConcatGroup:
    """Index for a group of concatenated files."""
    file_entries: list = field(default_factory=list)  # [(Path, n_tokens), ...]
    cum_offsets: list = field(default_factory=list)    # start position of each file
    total_length: int = 0


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


def _load_accepted_files(score_file, min_score):
    """Return set of .npy filenames with detector_score >= min_score."""
    import csv
    accepted = set()
    with open(score_file) as f:
        for row in csv.DictReader(f):
            ds = row.get('detector_score', '')
            if ds == '' or float(ds) >= min_score:
                accepted.add(row['npy_file'])
    return accepted


class AudioTokenDataset(Dataset):
    """Dataset of pre-tokenized audio sequences (Track 2).

    Lazy-loading: builds a lightweight index at init time (~125MB for 500K files),
    loads individual windows from disk on demand in __getitem__. Scales to billions
    of tokens without OOM.

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
        concat: bool = False,
        sep_token: int | None = None,
        codebook_index: int | None = None,
        score_file: str | Path | None = None,
        min_detector_score: float | None = None,
    ):
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token
        self.vocab_size = vocab_size
        self.augment = augment
        self.token_noise_prob = token_noise_prob
        self.token_mask_prob = token_mask_prob
        self.time_stretch_prob = time_stretch_prob
        self.concat = concat
        self.sep_token = sep_token
        self.codebook_index = codebook_index

        # Support multiple token directories
        if isinstance(token_dir, (list, tuple)):
            token_dirs = [Path(d) for d in token_dir]
        else:
            token_dirs = [Path(token_dir)]

        # Collect all .npy file paths
        all_paths = []
        for td in token_dirs:
            all_paths.extend(sorted(td.glob("*.npy")))

        # Filter by detector score if score_file provided
        if score_file and min_detector_score is not None:
            accepted = _load_accepted_files(score_file, min_detector_score)
            n_before = len(all_paths)
            all_paths = [p for p in all_paths if p.name in accepted]
            print(f"Score filter: {len(all_paths)}/{n_before} files with "
                  f"detector_score >= {min_detector_score}")

        # Phase 1: Scan file headers (parallel, no data loaded)
        n_workers = min(len(all_paths), os.cpu_count() or 4)
        if n_workers > 1 and len(all_paths) > 100:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_scan_npy, all_paths, chunksize=256))
        else:
            results = [_scan_npy(f) for f in all_paths]

        scanned = [r for r in results if r is not None]

        # Phase 2: Build index
        self._groups = []      # concat mode: list of _ConcatGroup
        self._file_list = []   # non-concat mode: list of (Path, n_tokens)
        self._windows = []     # list of (source_idx, start, end) tuples

        if concat and sep_token is not None:
            self._build_concat_index(scanned, max_seq_len)
        else:
            self._build_simple_index(scanned, max_seq_len)

    def _build_concat_index(self, scanned, max_seq_len):
        """Build window index for concat mode (group by prefix, SEP between files)."""
        # Group files by source prefix (everything before _NNNNNN)
        groups = {}
        for path, stem, n_tokens in scanned:
            parts = stem.rsplit("_", 1)
            prefix = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
            groups.setdefault(prefix, []).append((path, n_tokens))

        for prefix in sorted(groups):
            files = groups[prefix]
            group = _ConcatGroup()
            pos = 0
            for i, (path, n_tokens) in enumerate(files):
                group.file_entries.append((path, n_tokens))
                group.cum_offsets.append(pos)
                pos += n_tokens
                if i < len(files) - 1:
                    pos += 1  # SEP token between files
            group.total_length = pos

            group_idx = len(self._groups)
            self._groups.append(group)

            # Generate sliding windows
            total = group.total_length
            if total <= max_seq_len + 1:
                self._windows.append((group_idx, 0, total))
            else:
                stride = max_seq_len // 2
                for start in range(0, total - max_seq_len, stride):
                    self._windows.append((group_idx, start, start + max_seq_len + 1))
                self._windows.append((group_idx, total - max_seq_len - 1, total))

    def _build_simple_index(self, scanned, max_seq_len):
        """Build window index for non-concat mode (each file independent)."""
        for path, stem, n_tokens in scanned:
            file_idx = len(self._file_list)
            self._file_list.append((path, n_tokens))

            if n_tokens <= max_seq_len + 1:
                self._windows.append((file_idx, 0, n_tokens))
            else:
                stride = max_seq_len // 2
                for start in range(0, n_tokens - max_seq_len, stride):
                    self._windows.append((file_idx, start, start + max_seq_len + 1))
                self._windows.append((file_idx, n_tokens - max_seq_len - 1, n_tokens))

    def _extract_1d(self, arr):
        """Extract 1D token sequence from array, handling 2D codebook files."""
        if arr.ndim == 2 and self.codebook_index is not None:
            return arr[self.codebook_index, :]
        elif arr.ndim == 2:
            # No codebook_index specified — flatten first row as fallback
            return arr[0, :]
        return arr

    def _load_concat_window(self, group_idx, start, end):
        """Load tokens for a window that spans concatenated files."""
        group = self._groups[group_idx]
        offsets = group.cum_offsets

        # Binary search to find first and last file in range
        first = bisect.bisect_right(offsets, start) - 1
        last = bisect.bisect_right(offsets, end - 1) - 1

        pieces = []
        for fi in range(first, last + 1):
            path, flen = group.file_entries[fi]
            file_start = offsets[fi]
            file_end = file_start + flen

            # SEP token between previous file and this one
            if fi > first:
                sep_pos = offsets[fi] - 1  # SEP sits just before this file
                if start <= sep_pos < end:
                    pieces.append(np.array([self.sep_token], dtype=np.int16))

            # Slice of this file that falls within [start, end)
            lo = max(0, start - file_start)
            hi = min(flen, end - file_start)
            if lo < hi:
                arr = self._extract_1d(np.load(path))
                pieces.append(arr[lo:hi])

        return np.concatenate(pieces) if pieces else np.array([], dtype=np.int16)

    def _load_simple_window(self, file_idx, start, end):
        """Load tokens for a window from a single file."""
        path, _ = self._file_list[file_idx]
        arr = self._extract_1d(np.load(path))
        return arr[start:end]

    def _augment_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """Apply token-level data augmentation."""
        tokens = tokens.copy()
        real_mask = tokens > self.pad_token  # only augment non-pad tokens
        if self.sep_token is not None:
            real_mask &= tokens != self.sep_token  # don't augment SEP tokens

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
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        source_idx, start, end = self._windows[idx]

        # Load tokens from disk on demand
        if self.concat and self.sep_token is not None:
            tokens = self._load_concat_window(source_idx, start, end)
        else:
            tokens = self._load_simple_window(source_idx, start, end)

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
