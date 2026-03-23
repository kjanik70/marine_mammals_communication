#!/usr/bin/env python3
"""Batch denoise all raw audio datasets.

Applies medium denoising (bandpass + two-pass spectral gating + loudness normalization)
to all raw audio files and saves cleaned WAVs to data/denoised/<source>/.
"""

import argparse
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfilt
from tqdm import tqdm


def denoise_audio(audio, sr, target_lufs=-20.0):
    """Medium denoise: bandpass + two-pass spectral gating + loudness normalization.

    Adapts bandpass range to the actual sample rate (handles low-SR sources
    like right_whale at 2kHz).
    """
    nyq = sr / 2

    # Bandpass: 400 Hz – 20 kHz, clamped to Nyquist
    low_hz = 200 if nyq < 5000 else 400
    low = min(low_hz / nyq, 0.95)
    high = min(20000 / nyq, 0.95)
    if low < high:
        sos = butter(5, [low, high], btype='band', output='sos')
        audio = sosfilt(sos, audio).astype(np.float32)

    # First pass: stationary noise reduction (90%)
    audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=True,
        prop_decrease=0.90,
        n_fft=min(2048, len(audio)),
        freq_mask_smooth_hz=250,
        time_mask_smooth_ms=60,
    )

    # Second pass: non-stationary (75%)
    audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,
        prop_decrease=0.75,
        n_fft=min(2048, len(audio)),
        freq_mask_smooth_hz=200,
        time_mask_smooth_ms=50,
    )

    # Loudness normalization (ITU-R BS.1770)
    meter = pyln.Meter(sr)
    current_loudness = meter.integrated_loudness(audio)
    if np.isfinite(current_loudness):
        audio = pyln.normalize.loudness(audio, current_loudness, target_lufs)
    else:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.5 / peak)

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


DATASETS = [
    ("dswp", "data/raw/dswp"),
    ("watkins", "data/raw/watkins/audio"),
    ("esp_orcas", "data/raw/esp_orcas/audio"),
    ("orcasound", "data/raw/orcasound"),
    ("mbari", "data/raw/mbari"),
    ("dori_orca", "data/raw/dori_orcasound"),
    ("dori_orca_full", "data/raw/dori_orcasound_full"),
    ("humpback_tsujii", "data/raw/humpback_zenodo"),
    ("kw_pei", "data/raw/kw_pei"),
    ("right_whale", "data/raw/right_whale/v1"),
]


def process_source(name, audio_dir, output_dir, target_sr=44100, target_lufs=-20.0):
    """Denoise all audio files from a source."""
    audio_dir = Path(audio_dir)
    out_dir = Path(output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for pat in ("*.wav", "*.flac"):
        files.extend(audio_dir.rglob(pat))
    files = sorted(f for f in files if f.stat().st_size > 0)

    if not files:
        print(f"  No audio files found in {audio_dir}")
        return {"n_files": 0, "n_failed": 0, "total_duration": 0.0}

    stats = {"n_files": 0, "n_failed": 0, "total_duration": 0.0}

    for audio_path in tqdm(files, desc=f"  {name}"):
        try:
            audio, sr = sf.read(str(audio_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if sr < 2000:
                continue

            duration = len(audio) / sr
            stats["total_duration"] += duration

            # Resample to target SR
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            # Denoise
            audio_clean = denoise_audio(audio, target_sr, target_lufs=target_lufs)

            # Save — preserve relative path structure for watkins subdirs
            rel = audio_path.relative_to(audio_dir)
            out_path = out_dir / rel.with_suffix(".wav")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), audio_clean, target_sr)
            stats["n_files"] += 1

        except Exception as e:
            stats["n_failed"] += 1
            tqdm.write(f"    FAILED {audio_path.name}: {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Batch denoise all audio datasets")
    parser.add_argument("--output-dir", default="data/denoised",
                        help="Output directory for denoised WAVs")
    parser.add_argument("--target-sr", type=int, default=44100)
    parser.add_argument("--target-lufs", type=float, default=-20.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_failed = 0
    total_duration = 0.0

    for name, audio_dir in DATASETS:
        if not Path(audio_dir).exists():
            print(f"\nSkipping {name}: {audio_dir} not found")
            continue

        print(f"\n=== {name} ===")
        stats = process_source(name, audio_dir, args.output_dir,
                               target_sr=args.target_sr, target_lufs=args.target_lufs)
        total_files += stats["n_files"]
        total_failed += stats["n_failed"]
        total_duration += stats["total_duration"]

        print(f"  Files: {stats['n_files']}, Duration: {stats['total_duration']:.0f}s")
        if stats["n_failed"]:
            print(f"  Failed: {stats['n_failed']}")

    print(f"\n{'='*50}")
    print(f"TOTAL: {total_files} files denoised, {total_duration/60:.1f} min")
    if total_failed:
        print(f"  Failed: {total_failed}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
