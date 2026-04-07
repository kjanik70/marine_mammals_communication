#!/usr/bin/env python3
"""Evaluate audio tokenization quality across different codebook counts.

Measures roundtrip reconstruction quality: original → encode → decode → compare.
Tests 1, 2, 4, 8, and 14 codebooks to quantify information loss.

Metrics:
- Spectral Convergence (SC): lower = better, measures spectral shape match
- Signal-to-Noise Ratio (SNR): higher = better
- Mel Cepstral Distortion (MCD): lower = better, perceptual similarity
- Cross-correlation: higher = better, waveform alignment
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


def spectral_convergence(original: np.ndarray, reconstructed: np.ndarray,
                         sr: int = 44100, n_fft: int = 2048) -> float:
    """Spectral convergence: ||S_orig - S_recon|| / ||S_orig||. Lower = better."""
    S_orig = np.abs(librosa.stft(original, n_fft=n_fft))
    S_recon = np.abs(librosa.stft(reconstructed, n_fft=n_fft))
    # Align lengths
    min_t = min(S_orig.shape[1], S_recon.shape[1])
    S_orig = S_orig[:, :min_t]
    S_recon = S_recon[:, :min_t]
    return np.linalg.norm(S_orig - S_recon) / (np.linalg.norm(S_orig) + 1e-10)


def signal_to_noise(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """SNR in dB. Higher = better."""
    min_len = min(len(original), len(reconstructed))
    orig = original[:min_len]
    recon = reconstructed[:min_len]
    noise = orig - recon
    signal_power = np.mean(orig ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return 100.0
    return 10 * np.log10(signal_power / noise_power)


def mel_cepstral_distortion(original: np.ndarray, reconstructed: np.ndarray,
                            sr: int = 44100, n_mels: int = 80) -> float:
    """Mel Cepstral Distortion. Lower = better."""
    mfcc_orig = librosa.feature.mfcc(y=original, sr=sr, n_mfcc=13, n_mels=n_mels)
    mfcc_recon = librosa.feature.mfcc(y=reconstructed, sr=sr, n_mfcc=13, n_mels=n_mels)
    min_t = min(mfcc_orig.shape[1], mfcc_recon.shape[1])
    mfcc_orig = mfcc_orig[:, :min_t]
    mfcc_recon = mfcc_recon[:, :min_t]
    return np.mean(np.sqrt(np.sum((mfcc_orig - mfcc_recon) ** 2, axis=0)))


def cross_correlation(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Normalized cross-correlation. Higher = better (1.0 = perfect)."""
    min_len = min(len(original), len(reconstructed))
    orig = original[:min_len]
    recon = reconstructed[:min_len]
    orig = orig - np.mean(orig)
    recon = recon - np.mean(recon)
    denom = np.sqrt(np.sum(orig ** 2) * np.sum(recon ** 2))
    if denom < 1e-10:
        return 0.0
    return np.sum(orig * recon) / denom


def roundtrip(tokenizer, audio_tensor: torch.Tensor, n_cb: int) -> np.ndarray:
    """Encode audio with full codebooks, decode with only n_cb codebooks."""
    # Encode to get all 14 codebook codes
    result = tokenizer.codec.encode(audio_tensor.to(tokenizer.device), tokenizer.sample_rate)
    codes = result["codes"]  # (1, 14, T)

    # Zero out codebooks beyond n_cb
    T = codes.shape[-1]
    full_codes = torch.zeros(1, 14, T, dtype=torch.long, device=tokenizer.device)
    full_codes[:, :n_cb, :] = codes[:, :n_cb, :]

    # Decode
    z = tokenizer.codec.quantizer.from_codes(full_codes)[0]
    audio = tokenizer.codec.decode(z)["audio"]
    return audio.squeeze().cpu().numpy()


def find_test_files(token_dir: str, audio_dirs: list[str], n: int = 10) -> list[Path]:
    """Find audio files that have corresponding tokenized versions.

    Falls back to just finding audio files if no token_dir.
    """
    # Try finding whale audio in common locations
    candidates = []
    for d in audio_dirs:
        p = Path(d)
        if p.exists():
            candidates.extend(sorted(p.glob("*.wav"))[:50])
            candidates.extend(sorted(p.glob("*.flac"))[:50])

    if not candidates:
        print("No audio files found!")
        return []

    # Pick a diverse sample
    rng = np.random.default_rng(42)
    if len(candidates) > n:
        indices = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[i] for i in sorted(indices)]

    return candidates[:n]


def main():
    parser = argparse.ArgumentParser(description="Evaluate codec reconstruction quality")
    parser.add_argument("--codec-path", default="models/codec.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-files", type=int, default=10,
                        help="Number of audio files to test")
    parser.add_argument("--max-duration", type=float, default=10.0,
                        help="Max duration in seconds per file (truncate longer)")
    parser.add_argument("--output-dir", default="runs/codec_quality",
                        help="Directory for comparison audio files")
    parser.add_argument("audio_dirs", nargs="*",
                        default=["data/raw/dswp", "data/denoised/dswp",
                                 "data/sanctsound/audio/hi01"],
                        help="Directories to search for audio files")
    args = parser.parse_args()

    from src.tokenizer.audio_tokenizer import AudioTokenizer

    print("Loading codec...")
    tokenizer = AudioTokenizer(codec_path=args.codec_path, device=args.device, n_codebooks=14)
    sr = tokenizer.sample_rate

    # Find test files
    files = find_test_files("", args.audio_dirs, n=args.n_files)
    if not files:
        return

    print(f"Testing {len(files)} files across codebook counts [1, 2, 4, 8, 14]\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cb_counts = [1, 2, 4, 8, 14]
    # Aggregate metrics
    all_metrics = {n: {"sc": [], "snr": [], "mcd": [], "xcorr": []} for n in cb_counts}

    for fi, fpath in enumerate(files):
        print(f"[{fi}] {fpath.name}")

        # Load and preprocess
        audio, file_sr = sf.read(str(fpath), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

        # Truncate to max duration
        max_samples = int(args.max_duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        dur = len(audio) / sr
        print(f"    Duration: {dur:.1f}s, samples: {len(audio)}")

        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Save original
        orig_path = out_dir / f"file{fi}_original.wav"
        sf.write(str(orig_path), audio, sr)

        for n_cb in cb_counts:
            with torch.no_grad():
                recon = roundtrip(tokenizer, audio_tensor, n_cb)

            # Compute metrics
            sc = spectral_convergence(audio, recon, sr)
            snr = signal_to_noise(audio, recon)
            mcd = mel_cepstral_distortion(audio, recon, sr)
            xcorr = cross_correlation(audio, recon)

            all_metrics[n_cb]["sc"].append(sc)
            all_metrics[n_cb]["snr"].append(snr)
            all_metrics[n_cb]["mcd"].append(mcd)
            all_metrics[n_cb]["xcorr"].append(xcorr)

            print(f"    {n_cb:2d} CB: SC={sc:.4f}  SNR={snr:.1f}dB  MCD={mcd:.1f}  xcorr={xcorr:.4f}")

            # Save reconstruction
            recon_path = out_dir / f"file{fi}_{n_cb}cb.wav"
            sf.write(str(recon_path), recon, sr)

        print()

    # Summary table
    print("=" * 70)
    print(f"{'CBs':>4} | {'SC (↓)':>8} | {'SNR dB (↑)':>10} | {'MCD (↓)':>8} | {'xcorr (↑)':>10}")
    print("-" * 70)
    for n_cb in cb_counts:
        m = all_metrics[n_cb]
        print(f"{n_cb:4d} | {np.mean(m['sc']):8.4f} | {np.mean(m['snr']):10.1f} | "
              f"{np.mean(m['mcd']):8.1f} | {np.mean(m['xcorr']):10.4f}")
    print("=" * 70)

    # Save metrics
    import json
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            str(n_cb): {k: [round(v, 4) for v in vals] for k, vals in m.items()}
            for n_cb, m in all_metrics.items()
        }, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Audio comparisons saved to {out_dir}/")


if __name__ == "__main__":
    main()
