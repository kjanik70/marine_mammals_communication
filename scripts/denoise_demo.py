#!/usr/bin/env python3
"""Demo: denoise and normalize dolphin audio clips.

Applies spectral gating noise reduction + RMS normalization so all clips
have consistent loudness and reduced background noise.
"""

import argparse
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def denoise_and_normalize(audio, sr, target_lufs=-20.0):
    """Denoise audio via spectral gating, then loudness-normalize.

    Steps:
      1. Bandpass filter: 200 Hz – 22 kHz (cut ocean rumble + equipment noise)
      2. Spectral gating noise reduction (noisereduce)
      3. ITU-R BS.1770 loudness normalization to target LUFS
    """
    # 1. Bandpass filter
    audio_bp = librosa.effects.preemphasis(audio, coef=0.0)  # identity, placeholder
    # Use a butterworth bandpass via scipy
    from scipy.signal import butter, sosfilt
    nyq = sr / 2
    low = min(200 / nyq, 0.99)
    high = min(22000 / nyq, 0.99)
    if low < high:
        sos = butter(4, [low, high], btype='band', output='sos')
        audio_bp = sosfilt(sos, audio).astype(np.float32)
    else:
        audio_bp = audio

    # 2. Spectral gating noise reduction
    # Use stationary mode — estimates noise from overall statistics
    audio_clean = nr.reduce_noise(
        y=audio_bp,
        sr=sr,
        stationary=True,
        prop_decrease=0.85,  # how much to reduce noise (0-1)
        n_fft=2048,
        freq_mask_smooth_hz=200,
        time_mask_smooth_ms=50,
    )

    # 3. Loudness normalization (ITU-R BS.1770)
    meter = pyln.Meter(sr)
    current_loudness = meter.integrated_loudness(audio_clean)
    if np.isfinite(current_loudness):
        audio_norm = pyln.normalize.loudness(audio_clean, current_loudness, target_lufs)
    else:
        # Fallback: peak normalize
        peak = np.max(np.abs(audio_clean))
        if peak > 0:
            audio_norm = audio_clean * (0.5 / peak)
        else:
            audio_norm = audio_clean

    # Clip to prevent any overs
    audio_norm = np.clip(audio_norm, -1.0, 1.0)
    return audio_norm


def main():
    parser = argparse.ArgumentParser(description="Denoise and normalize audio clips")
    parser.add_argument("input_files", nargs="+", help="Input WAV/FLAC files")
    parser.add_argument("--output-dir", default="data/denoised_demo",
                        help="Output directory for cleaned files")
    parser.add_argument("--target-lufs", type=float, default=-20.0,
                        help="Target loudness in LUFS (default: -20)")
    parser.add_argument("--target-sr", type=int, default=44100,
                        help="Output sample rate (default: 44100)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fpath in args.input_files:
        fpath = Path(fpath)
        print(f"\n=== {fpath.name} ===")

        # Load
        audio, sr = sf.read(str(fpath), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        print(f"  Loaded: {len(audio)/sr:.1f}s @ {sr} Hz")

        # Resample to target SR
        if sr != args.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=args.target_sr)
            sr = args.target_sr
            print(f"  Resampled to {sr} Hz")

        # Stats before
        rms_before = float(np.sqrt(np.mean(audio ** 2)))
        peak_before = float(np.max(np.abs(audio)))
        print(f"  Before: RMS={rms_before:.4f}, Peak={peak_before:.4f}")

        # Denoise + normalize
        audio_clean = denoise_and_normalize(audio, sr, target_lufs=args.target_lufs)

        # Stats after
        rms_after = float(np.sqrt(np.mean(audio_clean ** 2)))
        peak_after = float(np.max(np.abs(audio_clean)))
        print(f"  After:  RMS={rms_after:.4f}, Peak={peak_after:.4f}")

        # Save
        out_path = output_dir / f"{fpath.stem}_clean.wav"
        sf.write(str(out_path), audio_clean, sr)
        print(f"  Saved: {out_path}")

    print(f"\nAll cleaned files in: {output_dir}/")


if __name__ == "__main__":
    main()
