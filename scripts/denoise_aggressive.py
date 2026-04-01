#!/usr/bin/env python3
"""Aggressive denoising: spectral gating + Wiener filter + tight bandpass.

Compared to the standard version:
  - Tighter bandpass (500 Hz – 20 kHz)
  - Non-stationary noise reduction (adapts over time)
  - Two-pass spectral gating (95% reduction)
  - Wiener filter for residual noise
  - Spectral subtraction as final cleanup
"""

import argparse
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfilt, wiener


def aggressive_denoise(audio, sr, target_lufs=-20.0):
    """Multi-stage aggressive denoising pipeline."""

    # 1. Tight bandpass: 500 Hz – 20 kHz
    #    Dolphins vocalize mostly 2–20 kHz; cuts more ocean rumble
    nyq = sr / 2
    low = min(500 / nyq, 0.99)
    high = min(20000 / nyq, 0.99)
    if low < high:
        sos = butter(6, [low, high], btype='band', output='sos')
        audio = sosfilt(sos, audio).astype(np.float32)

    # 2. First pass: stationary noise reduction (aggressive)
    audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=True,
        prop_decrease=0.95,
        n_fft=2048,
        freq_mask_smooth_hz=300,
        time_mask_smooth_ms=80,
    )

    # 3. Second pass: non-stationary noise reduction
    #    Adapts to time-varying noise (wave splashes, boat engines, etc.)
    audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,
        prop_decrease=0.90,
        n_fft=2048,
        freq_mask_smooth_hz=200,
        time_mask_smooth_ms=50,
    )

    # 4. Wiener filter on the STFT magnitude
    #    Smooths residual noise in the spectral domain
    n_fft = 2048
    hop = 512
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    mag = np.abs(S)
    phase = np.angle(S)

    # Apply Wiener filter to each frequency band
    mag_filtered = np.zeros_like(mag)
    for i in range(mag.shape[0]):
        mag_filtered[i] = wiener(mag[i], mysize=7)
    mag_filtered = np.maximum(mag_filtered, 0)

    S_filtered = mag_filtered * np.exp(1j * phase)
    audio = librosa.istft(S_filtered, hop_length=hop, length=len(audio)).astype(np.float32)

    # 5. Spectral subtraction: estimate noise from quietest 10% of frames
    S2 = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    mag2 = np.abs(S2)
    phase2 = np.angle(S2)
    frame_energy = np.sum(mag2 ** 2, axis=0)
    n_noise_frames = max(1, int(0.10 * mag2.shape[1]))
    noise_idx = np.argsort(frame_energy)[:n_noise_frames]
    noise_profile = np.mean(mag2[:, noise_idx], axis=1, keepdims=True)

    # Subtract with flooring
    mag_sub = mag2 - 1.5 * noise_profile
    mag_sub = np.maximum(mag_sub, 0.02 * mag2)  # floor at 2% of original

    S_sub = mag_sub * np.exp(1j * phase2)
    audio = librosa.istft(S_sub, hop_length=hop, length=len(audio)).astype(np.float32)

    # 6. Loudness normalization (ITU-R BS.1770)
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
    parser = argparse.ArgumentParser(description="Aggressive denoise + normalize")
    parser.add_argument("input_files", nargs="+")
    parser.add_argument("--output-dir", default="data/denoised_aggressive")
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

        audio_clean = aggressive_denoise(audio, sr, target_lufs=args.target_lufs)

        rms_after = float(np.sqrt(np.mean(audio_clean ** 2)))
        print(f"  After:  RMS={rms_after:.4f}")

        out_path = output_dir / f"{fpath.stem}_aggressive.wav"
        sf.write(str(out_path), audio_clean, sr)
        print(f"  Saved: {out_path}")

    print(f"\nAll files in: {output_dir}/")


if __name__ == "__main__":
    main()
