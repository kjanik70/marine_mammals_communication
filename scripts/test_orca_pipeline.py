#!/usr/bin/env python3
"""Test the orca audio processing pipeline on local samples.

Runs N audio files through the full orca pipeline (bandpass → segment →
normalize → DAC 9CB tokenize → decode) and saves side-by-side WAV files
so you can listen and compare. Also prints reconstruction metrics.

Usage:
    # 12 random esp_orcas samples
    PYTHONPATH=. python3 scripts/test_orca_pipeline.py

    # specific source directory
    PYTHONPATH=. python3 scripts/test_orca_pipeline.py --input data/raw/esp_orcas/audio --n 15

    # specific files
    PYTHONPATH=. python3 scripts/test_orca_pipeline.py --input path/to/file1.wav path/to/file2.wav

    # save output to a different directory
    PYTHONPATH=. python3 scripts/test_orca_pipeline.py --output-dir data/pipeline_test
"""

import argparse
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfilt

try:
    import noisereduce as nr
    _HAS_NOISEREDUCE = True
except ImportError:
    _HAS_NOISEREDUCE = False


# --- Metrics ---

def spectral_convergence(a, b, n_fft=2048):
    Sa = np.abs(librosa.stft(a, n_fft=n_fft))
    Sb = np.abs(librosa.stft(b, n_fft=n_fft))
    t = min(Sa.shape[1], Sb.shape[1])
    return float(np.linalg.norm(Sa[:, :t] - Sb[:, :t]) / (np.linalg.norm(Sa[:, :t]) + 1e-10))


def snr_db(original, reconstructed):
    n = min(len(original), len(reconstructed))
    orig, recon = original[:n], reconstructed[:n]
    noise_power = np.mean((orig - recon) ** 2)
    if noise_power < 1e-12:
        return float('inf')
    return float(10 * np.log10(np.mean(orig ** 2) / noise_power))


def mel_cepstral_distortion(a, b, sr=44100):
    mfcc_a = librosa.feature.mfcc(y=a, sr=sr, n_mfcc=13, n_mels=80)
    mfcc_b = librosa.feature.mfcc(y=b, sr=sr, n_mfcc=13, n_mels=80)
    t = min(mfcc_a.shape[1], mfcc_b.shape[1])
    return float(np.mean(np.sqrt(np.sum((mfcc_a[:, :t] - mfcc_b[:, :t]) ** 2, axis=0))))


# --- Processing steps (mirrors process_sanctsound_orca.py) ---

def bandpass_audio(audio, sr, low_hz=80, high_hz=20000):
    nyq = sr / 2
    low = min(low_hz / nyq, 0.95)
    high = min(high_hz / nyq, 0.95)
    if low < high:
        sos = butter(5, [low, high], btype='band', output='sos')
        audio = sosfilt(sos, audio).astype(np.float32)
    return audio


def loudness_normalize(audio, sr, target_lufs=-20.0):
    import pyloudnorm as pyln
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    if np.isfinite(loudness):
        audio = pyln.normalize.loudness(audio, loudness, target_lufs)
    else:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.5 / peak)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def spectral_gate(audio, sr, stationary_prop=0.90, nonstationary_prop=0.75):
    """Two-pass spectral gating (mirrors medium denoising pipeline).

    Pass 1: stationary noise reduction (targets constant background hiss/hum)
    Pass 2: non-stationary reduction (gentler, targets variable noise)

    Note: can remove faint calls on low-SNR hydrophone recordings. Works well
    on clean labeled clips (esp_orcas) but use with care on SanctSound data.
    """
    audio = nr.reduce_noise(
        y=audio, sr=sr, stationary=True,
        prop_decrease=stationary_prop,
        n_fft=2048, freq_mask_smooth_hz=250, time_mask_smooth_ms=60,
    ).astype(np.float32)
    audio = nr.reduce_noise(
        y=audio, sr=sr, stationary=False,
        prop_decrease=nonstationary_prop,
        n_fft=2048, freq_mask_smooth_hz=200, time_mask_smooth_ms=50,
    ).astype(np.float32)
    return audio


def process_chunk(audio, sr, target_sr=44100, denoise=False):
    """Apply the full orca processing chain to a chunk of audio."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    audio = bandpass_audio(audio, sr)
    if denoise:
        audio = spectral_gate(audio, sr)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (0.9 / peak)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    audio = loudness_normalize(audio, sr)
    return audio, sr


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Test orca pipeline: process → tokenize → decode → compare")
    parser.add_argument("--input", nargs="+", default=None,
                        help="Audio file(s) or a single directory to sample from")
    parser.add_argument("--n", type=int, default=12,
                        help="Number of files to test when --input is a directory (default: 12)")
    parser.add_argument("--output-dir", type=str, default="data/pipeline_test_orca",
                        help="Directory to write processed + decoded WAV files")
    parser.add_argument("--n-codebooks", type=int, default=9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--denoise", action="store_true",
                        help="Apply two-pass spectral gating before tokenization")
    args = parser.parse_args()

    if args.denoise and not _HAS_NOISEREDUCE:
        print("ERROR: noisereduce not installed. Run: pip install noisereduce")
        return

    random.seed(args.seed)

    # Resolve input files
    if args.input is None:
        src_dir = Path("data/raw/esp_orcas/audio")
        all_files = sorted(src_dir.glob("*.wav"))
        input_files = random.sample(all_files, min(args.n, len(all_files)))
    elif len(args.input) == 1 and Path(args.input[0]).is_dir():
        src_dir = Path(args.input[0])
        all_files = sorted(src_dir.glob("*.wav")) + sorted(src_dir.glob("*.flac"))
        input_files = random.sample(all_files, min(args.n, len(all_files)))
    else:
        input_files = [Path(p) for p in args.input]

    print(f"Testing {len(input_files)} files → {args.output_dir}"
          + (" (with spectral gating)" if args.denoise else ""))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    from src.tokenizer.dac_tokenizer import DACTokenizer
    print(f"Loading DACTokenizer ({args.n_codebooks} codebooks)...")
    tokenizer = DACTokenizer(device=args.device, n_codebooks=args.n_codebooks)
    sr_out = tokenizer.sample_rate  # 44100

    # Header
    print(f"\n{'File':<30} {'SC':>6} {'SNR(dB)':>8} {'MCD':>6}  Status")
    print("-" * 60)

    results = []
    for path in input_files:
        try:
            audio, sr = sf.read(str(path), dtype='float32')
            processed, sr_proc = process_chunk(audio, sr, denoise=args.denoise)

            if len(processed) < sr_proc * 0.1:
                print(f"{path.name:<30}  (too short, skipped)")
                continue

            # Tokenize
            chunk_tensor = torch.tensor(processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            codes, z = tokenizer.encode(chunk_tensor)
            del chunk_tensor, z

            codes_np = codes[0].cpu().numpy().astype(np.int32)
            del codes

            # Decode
            decoded = tokenizer.decode_2d_to_audio(codes_np, n_codebooks=args.n_codebooks)

            # Align lengths for metrics (decoded may be slightly longer due to hop rounding)
            n = min(len(processed), len(decoded))
            proc_aligned = processed[:n]
            dec_aligned = decoded[:n]

            sc = spectral_convergence(proc_aligned, dec_aligned)
            snr = snr_db(proc_aligned, dec_aligned)
            mcd = mel_cepstral_distortion(proc_aligned, dec_aligned, sr=sr_out)

            stem = path.stem
            proc_path = output_dir / f"{stem}_processed.wav"
            dec_path = output_dir / f"{stem}_decoded.wav"
            sf.write(str(proc_path), processed, sr_out)
            sf.write(str(dec_path), decoded, sr_out)

            print(f"{path.name:<30} {sc:>6.3f} {snr:>8.1f} {mcd:>6.2f}  OK")
            results.append({"file": path.name, "sc": sc, "snr": snr, "mcd": mcd})

        except Exception as e:
            print(f"{path.name:<30}  ERROR: {e}")

        torch.cuda.empty_cache()

    if results:
        scs = [r["sc"] for r in results]
        snrs = [r["snr"] for r in results]
        mcds = [r["mcd"] for r in results]
        print("-" * 60)
        print(f"{'Mean':<30} {np.mean(scs):>6.3f} {np.mean(snrs):>8.1f} {np.mean(mcds):>6.2f}")
        print(f"\nSC: spectral convergence (lower=better, <0.3 is good)")
        print(f"SNR: signal-to-noise ratio in dB (higher=better, >15dB is good)")
        print(f"MCD: mel cepstral distortion (lower=better, <5 is good)")
        print(f"\nWAV pairs saved to: {output_dir}/")
        print(f"  *_processed.wav  ← after pipeline, before tokenization")
        print(f"  *_decoded.wav    ← tokenized then decoded back")


if __name__ == "__main__":
    main()
