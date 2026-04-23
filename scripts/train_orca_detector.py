#!/usr/bin/env python3
"""Train a binary orca-call detector on log-mel spectrogram windows.

Phase 1 – cache build
    Reads raw audio from curated orca sources (positives) and ambient /
    non-orca recordings (negatives).  Slices each file into 3 s windows,
    computes log-mel spectrograms, and saves float16 .npy files to
    data/orca_detector_cache/.

Phase 2 – training
    Loads the cached spectrograms, trains OrcaDetectorCNN with BCE loss,
    saves the best checkpoint to models/orca_detector.pt.

Usage
-----
    PYTHONPATH=. python3 scripts/train_orca_detector.py          # full run
    PYTHONPATH=. python3 scripts/train_orca_detector.py --skip-cache  # reuse cache
    PYTHONPATH=. python3 scripts/train_orca_detector.py --cache-only  # build cache only

Positive sources
    data/raw/dori_orcasound_full/data/   (*.flac, 60 s, 44.1 kHz stereo)
    data/raw/esp_orcas/audio/            (*.wav,  ~4 s, 44.1 kHz mono)
    data/raw/kw_pei/kw_pei.wav           (single long file)
    data/raw/orcasound/2017-09-05-SRKW*/ (*.wav, ~10 min, 44.1 kHz)

Negative sources
    data/raw/mbari/                      (*.wav, 10 min, 16 kHz ambient)
    data/raw/watkins/audio/*/            (all species except Killer_Whale)
"""

import argparse
import csv
import gc
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detector.orca_detector import (
    OrcaDetectorCNN,
    SR,
    WIN_SAMPLES,
    audio_to_spec,
    make_mel_transform,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = Path("data/orca_detector_cache")
MODEL_OUT = Path("models/orca_detector.pt")
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff"}

# Window sampling caps (per file, to avoid over-representing long files)
MAX_WIN_DORI = 15       # DORI FLACs are 60 s → ~38 possible windows; take 15
MAX_WIN_LONG = 40       # KW PEI / Orcasound long files
MAX_WIN_MBARI = 50      # MBARI 10-min files
MAX_WIN_WATKINS = 10    # Watkins clips are short; take up to 10 per file

HOP_S = 1.5             # window hop in seconds
MIN_WIN_SAMPLES = int(1.5 * SR)   # discard windows shorter than 1.5 s

EPOCHS = 30
BATCH = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 6            # early stopping patience (epochs without val improvement)
VAL_FRAC = 0.15
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mono_44k(path: Path) -> np.ndarray | None:
    """Load audio file → mono float32 at 44 100 Hz. Returns None on error."""
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e:
        print(f"    [skip] {path.name}: {e}")
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio.astype(np.float32)


def windows_from_audio(audio: np.ndarray, max_wins: int | None = None) -> list[np.ndarray]:
    """Slice audio into WIN_SAMPLES windows with HOP_S-second hop."""
    hop = int(HOP_S * SR)
    wins = []
    for start in range(0, len(audio) - MIN_WIN_SAMPLES + 1, hop):
        chunk = audio[start:start + WIN_SAMPLES]
        if len(chunk) < WIN_SAMPLES:
            padded = np.zeros(WIN_SAMPLES, dtype=np.float32)
            padded[:len(chunk)] = chunk
            chunk = padded
        wins.append(chunk)
        if max_wins and len(wins) >= max_wins:
            break
    return wins


def collect_source_files(
    directory: Path,
    glob: str = "**/*",
    exclude_dirs: set[str] | None = None,
) -> list[Path]:
    files = []
    for p in sorted(directory.glob(glob)):
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        if exclude_dirs and p.parent.name in exclude_dirs:
            continue
        files.append(p)
    return files


# ---------------------------------------------------------------------------
# Phase 1: cache build
# ---------------------------------------------------------------------------

def build_cache(mel_transform: nn.Sequential) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE_DIR / "manifest.csv"

    rows: list[dict] = []
    idx = 0

    def save_windows(wins: list[np.ndarray], label: int, source: str) -> int:
        nonlocal idx
        saved = 0
        for w in wins:
            spec = audio_to_spec(w, mel_transform)   # (N_MELS, T) float16
            out = CACHE_DIR / f"{label}_{idx:07d}.npy"
            np.save(out, spec)
            rows.append({"path": str(out), "label": label, "source": source})
            idx += 1
            saved += 1
        return saved

    # ---- positives --------------------------------------------------------

    # DORI
    dori_files = collect_source_files(Path("data/raw/dori_orcasound_full/data"), "*.flac")
    print(f"DORI: {len(dori_files)} files")
    pos_dori = 0
    for f in tqdm(dori_files, desc="DORI"):
        audio = load_mono_44k(f)
        if audio is None:
            continue
        wins = windows_from_audio(audio, max_wins=MAX_WIN_DORI)
        random.shuffle(wins)
        pos_dori += save_windows(wins, label=1, source="dori")
        del audio
        gc.collect()
    print(f"  → {pos_dori} windows")

    # ESP orcas
    esp_files = collect_source_files(Path("data/raw/esp_orcas/audio"), "*.wav")
    print(f"ESP: {len(esp_files)} files")
    pos_esp = 0
    for f in tqdm(esp_files, desc="ESP"):
        audio = load_mono_44k(f)
        if audio is None:
            continue
        wins = windows_from_audio(audio)
        pos_esp += save_windows(wins, label=1, source="esp")
        del audio
    print(f"  → {pos_esp} windows")

    # KW PEI
    kw_path = Path("data/raw/kw_pei/kw_pei.wav")
    if kw_path.exists():
        print("KW PEI: 1 file")
        audio = load_mono_44k(kw_path)
        if audio is not None:
            wins = windows_from_audio(audio, max_wins=MAX_WIN_LONG)
            n = save_windows(wins, label=1, source="kw_pei")
            print(f"  → {n} windows")
            del audio

    # Orcasound SRKW (files live directly in the orcasound dir, flat naming)
    srkw_dir = Path("data/raw/orcasound")
    srkw_files = [
        f for f in srkw_dir.glob("2017-09-05-SRKW*.wav")
        if f.suffix.lower() in AUDIO_EXTS
    ]
    print(f"Orcasound SRKW: {len(srkw_files)} files")
    pos_srkw = 0
    for f in tqdm(srkw_files, desc="SRKW"):
        audio = load_mono_44k(f)
        if audio is None:
            continue
        wins = windows_from_audio(audio, max_wins=MAX_WIN_LONG)
        pos_srkw += save_windows(wins, label=1, source="orcasound_srkw")
        del audio
    print(f"  → {pos_srkw} windows")

    # ---- negatives --------------------------------------------------------

    # MBARI ambient hydrophone
    mbari_files = collect_source_files(Path("data/raw/mbari"), "*.wav")
    print(f"MBARI: {len(mbari_files)} files")
    neg_mbari = 0
    for f in tqdm(mbari_files, desc="MBARI"):
        audio = load_mono_44k(f)
        if audio is None:
            continue
        wins = windows_from_audio(audio, max_wins=MAX_WIN_MBARI)
        random.shuffle(wins)
        neg_mbari += save_windows(wins, label=0, source="mbari")
        del audio
        gc.collect()
    print(f"  → {neg_mbari} windows")

    # Watkins non-orca species (hard negatives: other marine mammals)
    watkins_root = Path("data/raw/watkins/audio")
    watkins_files = collect_source_files(
        watkins_root, "**/*", exclude_dirs={"Killer_Whale"}
    )
    print(f"Watkins non-orca: {len(watkins_files)} files across "
          f"{len(set(f.parent.name for f in watkins_files))} species")
    neg_watkins = 0
    for f in tqdm(watkins_files, desc="Watkins"):
        audio = load_mono_44k(f)
        if audio is None:
            continue
        wins = windows_from_audio(audio, max_wins=MAX_WIN_WATKINS)
        neg_watkins += save_windows(wins, label=0, source="watkins_non_orca")
        del audio
    print(f"  → {neg_watkins} windows")

    # ---- write manifest ---------------------------------------------------
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = sum(1 for r in rows if r["label"] == 0)
    print(f"\nCache: {len(rows)} windows total  ({n_pos} pos / {n_neg} neg)")
    print(f"Saved to {CACHE_DIR}/")
    return manifest_path


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpecDataset(Dataset):
    def __init__(self, records: list[dict], augment: bool = False):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        spec = np.load(r["path"]).astype(np.float32)   # (N_MELS, T)
        x = torch.tensor(spec).unsqueeze(0)            # (1, N_MELS, T)

        if self.augment:
            # Time shift (roll along time axis)
            shift = random.randint(0, x.shape[-1] // 4)
            x = torch.roll(x, shift, dims=-1)
            # Frequency masking: zero out up to 16 consecutive mel bins
            if random.random() < 0.4:
                f0 = random.randint(0, max(0, x.shape[-2] - 16))
                fw = random.randint(1, 16)
                x[:, f0:f0 + fw, :] = 0.0
            # Time masking: zero out up to 20 consecutive frames
            if random.random() < 0.4:
                t0 = random.randint(0, max(0, x.shape[-1] - 20))
                tw = random.randint(1, 20)
                x[:, :, t0:t0 + tw] = 0.0

        return x, torch.tensor(float(r["label"]))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(manifest_path: Path, device: str) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load manifest
    with open(manifest_path) as f:
        records = list(csv.DictReader(f))
    random.shuffle(records)
    print(f"Loaded {len(records)} samples from {manifest_path}")

    # Train/val split by index (fast, good enough since we shuffled)
    n_val = max(1, int(len(records) * VAL_FRAC))
    val_records = records[:n_val]
    train_records = records[n_val:]
    print(f"Train: {len(train_records)}  Val: {len(val_records)}")

    # Weighted sampler to balance classes in training batches
    labels = [int(r["label"]) for r in train_records]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative samples")
    w_pos = 1.0 / n_pos
    w_neg = 1.0 / n_neg
    weights = [w_pos if l == 1 else w_neg for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_ds = SpecDataset(train_records, augment=True)
    val_ds = SpecDataset(val_records, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = OrcaDetectorCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_correct += ((logits > 0) == y.bool()).sum().item()
        scheduler.step()

        # --- val ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(y)
                val_correct += ((logits > 0) == y.bool()).sum().item()

        t_loss = train_loss / len(train_ds)
        v_loss = val_loss / len(val_ds)
        t_acc = train_correct / len(train_ds)
        v_acc = val_correct / len(val_ds)
        print(f"Epoch {epoch:3d} | "
              f"train loss {t_loss:.4f} acc {t_acc:.3f} | "
              f"val loss {v_loss:.4f} acc {v_acc:.3f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_count = 0
            torch.save({"model": model.state_dict(), "val_loss": v_loss, "epoch": epoch},
                       MODEL_OUT)
            print(f"  ✓ saved {MODEL_OUT} (val_loss={v_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val_loss: {best_val_loss:.4f}  →  {MODEL_OUT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cache", action="store_true",
                        help="Skip cache build, reuse existing manifest.csv")
    parser.add_argument("--cache-only", action="store_true",
                        help="Build cache only, skip training")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest_path = CACHE_DIR / "manifest.csv"

    if not args.skip_cache:
        print("=== Phase 1: building spectrogram cache ===")
        mel = make_mel_transform(SR)
        manifest_path = build_cache(mel)
    else:
        print(f"Skipping cache build, using {manifest_path}")

    if args.cache_only:
        return

    print(f"\n=== Phase 2: training (device={args.device}) ===")
    train(manifest_path, args.device)


if __name__ == "__main__":
    main()
