#!/usr/bin/env python3
"""Medium denoising: two-pass spectral gating + tighter bandpass, no Wiener/spectral subtraction.

Sweet spot between standard and aggressive — more noise removal than single-pass
but avoids the musical noise artifacts from Wiener and spectral subtraction.
"""

import argparse
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfilt


def medium_denoise(audio, sr, target_lufs=-20.0):
    """Two-pass spectral gating with tighter bandpass."""

    # 1. Bandpass: 400 Hz – 20 kHz (tighter than standard, less harsh than aggressive)
    nyq = sr / 2
    low = min(400 / nyq, 0.99)
    high = min(20000 / nyq, 0.99)
    if low < high:
        sos = butter(5, [low, high], btype='band', output='sos')
        audio = sosfilt(sos, audio).astype(np.float32)

    # 2. First pass: stationary noise reduction (90% — between 85% and 95%)
    audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=True,
        prop_decrease=0.90,
        n_fft=2048,
        freq_mask_smooth_hz=250,
        time_mask_smooth_ms=60,
    )

    # 3. Second pass: non-stationary (gentler — 75%)
    audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,
        prop_decrease=0.75,
        n_fft=2048,
        freq_mask_smooth_hz=200,
        time_mask_smooth_ms=50,
    )

    # 4. Loudness normalization
    meter = pyln.Meter(sr)
    current_loudness = meter.integrated_loudness(audio)
    if np.isfinite(current_loudness):
        audio = pyln.normalize.loudness(audio, current_loudness, target_lufs)
    else:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.5 / peak)

    audio = np.clip(audio, -1.0, 1.0)
    return audio


def main():
    parser = argparse.ArgumentParser(description="Medium denoise + normalize")
    parser.add_argument("input_files", nargs="+")
    parser.add_argument("--output-dir", default="data/denoised_medium")
    parser.add_argument("--target-lufs", type=float, default=-20.0)
    parser.add_argument("--target-sr", type=int, default=44100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fpath in args.input_files:
        fpath = Path(fpath)
        print(f"\n=== {fpath.name} ===")

        audio, sr = sf.read(str(fpath), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        print(f"  Loaded: {len(audio)/sr:.1f}s @ {sr} Hz")

        if sr != args.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=args.target_sr)
            sr = args.target_sr
            print(f"  Resampled to {sr} Hz")

        rms_before = float(np.sqrt(np.mean(audio ** 2)))
        print(f"  Before: RMS={rms_before:.4f}")

        audio_clean = medium_denoise(audio, sr, target_lufs=args.target_lufs)

        rms_after = float(np.sqrt(np.mean(audio_clean ** 2)))
        print(f"  After:  RMS={rms_after:.4f}")

        out_path = output_dir / f"{fpath.stem}_medium.wav"
        sf.write(str(out_path), audio_clean, sr)
        print(f"  Saved: {out_path}")

    print(f"\nAll files in: {output_dir}/")


if __name__ == "__main__":
    main()
