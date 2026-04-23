#!/usr/bin/env python3
"""Fine-tune the orca detector on SanctSound passive acoustic data.

Why this works: positives (annotation windows) and negatives (quiet regions of
the same FLAC) share the same hydrophone noise floor, so the model learns to
distinguish orca calls from *that specific ambient*, not just clean vs noisy.

Phases
------
1. Download   – pull targeted FLACs from GCS for high-annotation deployments
2. Cache      – extract positive/negative 3-s spectrogram windows per FLAC
3. Fine-tune  – start from models/orca_detector.pt, low LR, mixed data

Positive windows  : inside annotation ± ANN_BUFFER_S
Negative windows  : from same FLAC, at least NEG_GUARD_S from any annotation

Usage
-----
    PYTHONPATH=. python3 scripts/finetune_orca_detector_sanctsound.py
    PYTHONPATH=. python3 scripts/finetune_orca_detector_sanctsound.py --skip-download
    PYTHONPATH=. python3 scripts/finetune_orca_detector_sanctsound.py --skip-download --skip-cache
"""

import argparse
import csv
import gc
import random
from datetime import datetime, timezone
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

# Deployments to use, ordered by annotation density (station, deployment_num)
FINETUNE_DEPLOYMENTS = [
    ("oc02", 2),   # 180 annotations – highest density
    ("oc01", 3),   # 139 annotations
    ("oc02", 5),   # 108 annotations
    ("oc02", 4),   #  78 annotations
]

MAX_FLACS_PER_DEPLOY = 8      # cap downloads per deployment
ANN_BUFFER_S = 30.0           # expand each annotation window by ±30 s
NEG_GUARD_S  = 180.0          # negatives must be ≥ 180 s from any annotation
MAX_POS_PER_FLAC = 20         # max positive windows per FLAC
MAX_NEG_PER_FLAC = 30         # max negative windows per FLAC
HOP_S = 1.5                   # window stride in seconds

FLAC_AUDIO_DIR  = Path("data/sanctsound/audio")
DETECTION_DIR   = Path("data/sanctsound/detections")
CACHE_DIR       = Path("data/orca_detector_cache")
CACHE_SS_DIR    = Path("data/orca_detector_cache_sanctsound")
ORIG_MANIFEST   = CACHE_DIR / "manifest.csv"
SS_MANIFEST     = CACHE_SS_DIR / "manifest.csv"
MODEL_IN        = Path("models/orca_detector.pt")
MODEL_OUT       = Path("models/orca_detector_ft.pt")

EPOCHS   = 20
BATCH    = 64
LR       = 5e-5          # low LR for fine-tuning
WD       = 1e-4
PATIENCE = 5
VAL_FRAC = 0.15
SEED     = 42

# Ratio of original-cache samples to SanctSound samples in fine-tune batches.
# 1.0 = equal; 2.0 = twice as many original.  Keeps clean-data performance.
ORIG_OVERSAMPLE = 2.0


# ---------------------------------------------------------------------------
# Annotation helpers (mirrors process_sanctsound_orca.py)
# ---------------------------------------------------------------------------

def _parse_iso(s: str) -> datetime:
    s = s.strip().rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {s!r}")


def _parse_flac_dt(fname: str) -> datetime | None:
    """Extract UTC datetime from FLAC filename.

    Two SanctSound formats:
    - …_20191106T205844Z.flac  (YYYYMMDDTHHMMSSz)
    - …_191106205844.flac      (YYMMDDHHMMSS, 12-digit numeric)
    """
    stem = Path(fname).stem
    for part in stem.split("_"):
        if len(part) >= 15 and "T" in part:
            try:
                return datetime.strptime(part.rstrip("Z"), "%Y%m%dT%H%M%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue
        if len(part) == 12 and part.isdigit():
            try:
                return datetime.strptime(part, "%y%m%d%H%M%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue
    return None


def load_killerwhale_csv(station: str, deploy_num: int) -> list[tuple[datetime, datetime]]:
    """Return list of (start, end) UTC datetimes from killerwhale detection CSV."""
    deploy_str = f"{station.upper()}_{deploy_num:02d}"
    # Files are named e.g. SanctSound_OC01_03_killerwhale.csv (case-sensitive match)
    matches = list((DETECTION_DIR / station).glob(f"*{deploy_str}_killerwhale*.csv"))
    if not matches:
        # Fallback: case-insensitive scan
        matches = [f for f in (DETECTION_DIR / station).glob("*.csv")
                   if "killerwhale" in f.name.lower() and deploy_str.lower() in f.name.lower()]
    if not matches:
        return []
    intervals = []
    with open(matches[0]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = _parse_iso(
                    row.get("ISOStartTime") or row.get("start_time") or
                    row.get("detection_time") or ""
                )
                end_raw = (row.get("ISOEndTime") or row.get("end_time") or
                           row.get("end_detection_time") or "")
                import datetime as _dt
                end = _parse_iso(end_raw) if end_raw.strip() else start + _dt.timedelta(seconds=60)
                intervals.append((start, end))
            except (ValueError, KeyError):
                continue
    return intervals


def merge_intervals(
    intervals: list[tuple[float, float]], buffer: float = 0.0
) -> list[tuple[float, float]]:
    """Merge overlapping float (start, end) intervals, optionally expanded by buffer."""
    expanded = [(max(0.0, s - buffer), e + buffer) for s, e in intervals]
    expanded.sort()
    merged: list[tuple[float, float]] = []
    for s, e in expanded:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def find_quiet_regions(
    ann_intervals: list[tuple[float, float]],
    file_dur: float,
    guard: float,
    min_region: float = 10.0,
) -> list[tuple[float, float]]:
    """Return time regions that are ≥ guard seconds from every annotation."""
    blocked = merge_intervals(ann_intervals, buffer=guard)
    quiet: list[tuple[float, float]] = []
    cursor = 0.0
    for bs, be in blocked:
        if bs - cursor >= min_region:
            quiet.append((cursor, bs))
        cursor = be
    if file_dur - cursor >= min_region:
        quiet.append((cursor, file_dur))
    return quiet


def sample_region_starts(
    regions: list[tuple[float, float]],
    n: int,
    win_s: float = 3.0,
    rng: random.Random | None = None,
) -> list[float]:
    """Uniformly sample n window start times from non-overlapping quiet regions."""
    if rng is None:
        rng = random.Random()
    total = sum(max(0.0, e - s - win_s) for s, e in regions)
    if total <= 0:
        return []
    starts: list[float] = []
    for _ in range(n * 4):          # over-sample then deduplicate
        t = rng.uniform(0, total)
        acc = 0.0
        for rs, re in regions:
            span = max(0.0, re - rs - win_s)
            if t <= acc + span:
                starts.append(rs + (t - acc))
                break
            acc += span
        if len(starts) >= n:
            break
    return starts[:n]


def resample_chunk(audio: np.ndarray, orig_sr: int, target_sr: int = SR) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


# ---------------------------------------------------------------------------
# Phase 1: Download
# ---------------------------------------------------------------------------

def download_flacs(dry_run: bool = False) -> None:
    from google.cloud import storage

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket("noaa-passive-bioacoustic")

    for station, dep_num in FINETUNE_DEPLOYMENTS:
        deploy_str = f"sanctsound_{station}_{dep_num:02d}"
        prefix = f"sanctsound/audio/{station}/{deploy_str}/audio/"
        out_dir = FLAC_AUDIO_DIR / station
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nListing {prefix}...")
        blobs = sorted(
            [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".flac")],
            key=lambda b: b.name,
        )
        print(f"  {len(blobs)} FLACs available")

        # Load annotations to prioritise FLACs that overlap detections
        annotations = load_killerwhale_csv(station, dep_num)
        ann_times = set()
        for s, e in annotations:
            ann_times.add(s.date())

        def has_annotation(blob_name: str) -> bool:
            dt = _parse_flac_dt(blob_name.split("/")[-1])
            return dt is not None and dt.date() in ann_times

        priority = [b for b in blobs if has_annotation(b.name)]
        rest = [b for b in blobs if not has_annotation(b.name)]
        selected = (priority + rest)[:MAX_FLACS_PER_DEPLOY]
        print(f"  Selecting {len(selected)} FLACs "
              f"({len([b for b in selected if has_annotation(b.name)])} overlap annotations)")

        if dry_run:
            for b in selected:
                print(f"    would download: {b.name.split('/')[-1]} ({b.size/1e6:.0f} MB)")
            continue

        for blob in tqdm(selected, desc=f"{station}/{dep_num:02d}"):
            fname = blob.name.split("/")[-1]
            local = out_dir / fname
            if local.exists() and local.stat().st_size > 0:
                continue
            blob.download_to_filename(str(local))


# ---------------------------------------------------------------------------
# Phase 2: Cache build
# ---------------------------------------------------------------------------

def build_sanctsound_cache(mel: nn.Sequential) -> Path:
    CACHE_SS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    rows: list[dict] = []
    idx = 0

    def save_spec(audio_chunk: np.ndarray, label: int, source: str) -> None:
        nonlocal idx
        spec = audio_to_spec(audio_chunk.astype(np.float32), mel)
        out = CACHE_SS_DIR / f"{label}_{idx:07d}.npy"
        np.save(out, spec)
        rows.append({"path": str(out), "label": label, "source": source})
        idx += 1

    for station, dep_num in FINETUNE_DEPLOYMENTS:
        annotations = load_killerwhale_csv(station, dep_num)
        if not annotations:
            print(f"  No annotations for {station}/{dep_num:02d}, skipping")
            continue

        station_dir = FLAC_AUDIO_DIR / station
        deploy_str = f"sanctsound_{station}_{dep_num:02d}"
        flac_files = sorted(station_dir.glob(f"*{deploy_str.upper()}*.flac"))
        if not flac_files:
            # Try case-insensitive glob via listdir
            flac_files = sorted(
                f for f in station_dir.iterdir()
                if f.suffix.lower() == ".flac" and deploy_str.split("_")[-1] in f.name.lower()
            )
        print(f"\n{station}/{dep_num:02d}: {len(flac_files)} FLACs, "
              f"{len(annotations)} annotations")

        for flac_path in tqdm(flac_files, desc=f"  {station}/{dep_num:02d}"):
            flac_dt = _parse_flac_dt(flac_path.name)
            if flac_dt is None:
                continue

            with sf.SoundFile(str(flac_path)) as f:
                orig_sr = f.samplerate
                n_frames = f.frames
            file_dur = n_frames / orig_sr

            flac_end_dt = flac_dt.replace(tzinfo=timezone.utc) + \
                __import__("datetime").timedelta(seconds=file_dur)
            flac_start_dt = flac_dt.replace(tzinfo=timezone.utc)

            # Annotation intervals in file-relative seconds
            rel_intervals: list[tuple[float, float]] = []
            for ann_s, ann_e in annotations:
                if ann_e < flac_start_dt or ann_s > flac_end_dt:
                    continue
                t0 = max(0.0, (ann_s - flac_start_dt).total_seconds())
                t1 = min(file_dur, (ann_e - flac_start_dt).total_seconds())
                if t1 > t0:
                    rel_intervals.append((t0, t1))

            if not rel_intervals:
                continue

            merged_pos = merge_intervals(rel_intervals, buffer=ANN_BUFFER_S)
            win_s = WIN_SAMPLES / SR

            # --- Positive windows ---
            pos_count = 0
            for t0, t1 in merged_pos:
                if pos_count >= MAX_POS_PER_FLAC:
                    break
                region_dur = t1 - t0
                if region_dur < win_s:
                    continue
                # Slide windows through the annotation region
                hop = HOP_S
                win_starts = list(np.arange(t0, t1 - win_s + 0.01, hop))
                rng.shuffle(win_starts)
                for ws in win_starts:
                    if pos_count >= MAX_POS_PER_FLAC:
                        break
                    start_frame = int(ws * orig_sr)
                    end_frame   = min(start_frame + int(win_s * orig_sr), n_frames)
                    try:
                        chunk, _ = sf.read(
                            str(flac_path), start=start_frame, stop=end_frame,
                            dtype="float32", always_2d=False,
                        )
                    except Exception:
                        continue
                    if chunk.ndim > 1:
                        chunk = chunk.mean(axis=1)
                    chunk = resample_chunk(chunk, orig_sr)
                    if len(chunk) < WIN_SAMPLES:
                        padded = np.zeros(WIN_SAMPLES, dtype=np.float32)
                        padded[:len(chunk)] = chunk
                        chunk = padded
                    else:
                        chunk = chunk[:WIN_SAMPLES]
                    save_spec(chunk, label=1, source=f"ss_{station}_{dep_num:02d}")
                    pos_count += 1

            # --- Negative windows ---
            quiet = find_quiet_regions(merged_pos, file_dur, guard=NEG_GUARD_S)
            neg_starts = sample_region_starts(quiet, MAX_NEG_PER_FLAC, win_s=win_s, rng=rng)
            neg_count = 0
            for ws in neg_starts:
                start_frame = int(ws * orig_sr)
                end_frame   = min(start_frame + int(win_s * orig_sr), n_frames)
                try:
                    chunk, _ = sf.read(
                        str(flac_path), start=start_frame, stop=end_frame,
                        dtype="float32", always_2d=False,
                    )
                except Exception:
                    continue
                if chunk.ndim > 1:
                    chunk = chunk.mean(axis=1)
                chunk = resample_chunk(chunk, orig_sr)
                if len(chunk) < WIN_SAMPLES:
                    padded = np.zeros(WIN_SAMPLES, dtype=np.float32)
                    padded[:len(chunk)] = chunk
                    chunk = padded
                else:
                    chunk = chunk[:WIN_SAMPLES]
                save_spec(chunk, label=0, source=f"ss_{station}_{dep_num:02d}_neg")
                neg_count += 1

            gc.collect()

        print(f"  Deployment total so far: {idx} cached windows")

    with open(SS_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = sum(1 for r in rows if r["label"] == 0)
    print(f"\nSanctSound cache: {len(rows)} windows  ({n_pos} pos / {n_neg} neg)")
    return SS_MANIFEST


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
        spec = np.load(r["path"]).astype(np.float32)
        x = torch.tensor(spec).unsqueeze(0)    # (1, N_MELS, T)

        if self.augment:
            if random.random() < 0.5:
                x = torch.roll(x, random.randint(0, x.shape[-1] // 4), dims=-1)
            if random.random() < 0.4:
                f0 = random.randint(0, max(0, x.shape[-2] - 16))
                x[:, f0:f0 + random.randint(1, 16), :] = 0.0
            if random.random() < 0.4:
                t0 = random.randint(0, max(0, x.shape[-1] - 20))
                x[:, :, t0:t0 + random.randint(1, 20)] = 0.0

        return x, torch.tensor(float(r["label"]))


# ---------------------------------------------------------------------------
# Phase 3: Fine-tune
# ---------------------------------------------------------------------------

def finetune(device: str) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load SanctSound records
    with open(SS_MANIFEST) as f:
        ss_records = list(csv.DictReader(f))
    n_ss_pos = sum(1 for r in ss_records if r["label"] == "1")
    n_ss_neg = sum(1 for r in ss_records if r["label"] == "0")
    print(f"SanctSound records: {len(ss_records)}  ({n_ss_pos} pos / {n_ss_neg} neg)")

    # Load original records (subsample for balance)
    orig_records: list[dict] = []
    if ORIG_MANIFEST.exists():
        with open(ORIG_MANIFEST) as f:
            all_orig = list(csv.DictReader(f))
        # Subsample original to ORIG_OVERSAMPLE × SanctSound size
        target_orig = int(len(ss_records) * ORIG_OVERSAMPLE)
        random.shuffle(all_orig)
        orig_records = all_orig[:target_orig]
        print(f"Original records sampled: {len(orig_records)} "
              f"(of {len(all_orig)} total, {ORIG_OVERSAMPLE}× oversample)")

    all_records = ss_records + orig_records
    random.shuffle(all_records)

    n_val = max(1, int(len(all_records) * VAL_FRAC))
    val_records   = all_records[:n_val]
    train_records = all_records[n_val:]
    print(f"Train: {len(train_records)}  Val: {len(val_records)}")

    # Weighted sampler
    labels = [int(r["label"]) for r in train_records]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative samples in training set")
    w_pos, w_neg = 1.0 / n_pos, 1.0 / n_neg
    weights = [w_pos if l == 1 else w_neg for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_ds = SpecDataset(train_records, augment=True)
    val_ds   = SpecDataset(val_records, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Load model from checkpoint
    ckpt = torch.load(MODEL_IN, map_location=device, weights_only=False)
    model = OrcaDetectorCNN().to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded weights from {MODEL_IN} (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?'):.4f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = t_correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(y)
            t_correct += ((logits > 0) == y.bool()).sum().item()
        scheduler.step()

        model.eval()
        v_loss = v_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                v_loss += criterion(logits, y).item() * len(y)
                v_correct += ((logits > 0) == y.bool()).sum().item()

        tl = t_loss / len(train_ds)
        vl = v_loss / len(val_ds)
        ta = t_correct / len(train_ds)
        va = v_correct / len(val_ds)
        print(f"Epoch {epoch:3d} | train {tl:.4f} acc {ta:.3f} | "
              f"val {vl:.4f} acc {va:.3f} | lr {scheduler.get_last_lr()[0]:.2e}")

        if vl < best_val_loss:
            best_val_loss = vl
            patience_count = 0
            torch.save({"model": model.state_dict(), "val_loss": vl, "epoch": epoch},
                       MODEL_OUT)
            print(f"  ✓ saved {MODEL_OUT}")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val_loss: {best_val_loss:.4f}  →  {MODEL_OUT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-cache",    action="store_true")
    parser.add_argument("--cache-only",    action="store_true")
    parser.add_argument("--dry-run-download", action="store_true",
                        help="List what would be downloaded without fetching")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.skip_download:
        print("=== Phase 1: downloading SanctSound FLACs ===")
        download_flacs(dry_run=args.dry_run_download)
        if args.dry_run_download:
            return

    if not args.skip_cache:
        print("\n=== Phase 2: building SanctSound spectrogram cache ===")
        mel = make_mel_transform(SR)
        build_sanctsound_cache(mel)

    if args.cache_only:
        return

    print(f"\n=== Phase 3: fine-tuning (device={args.device}) ===")
    finetune(args.device)


if __name__ == "__main__":
    main()
