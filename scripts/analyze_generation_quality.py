#!/usr/bin/env python3
"""Analyze and compare generated audio samples across temperature settings.

Computes spectral properties, energy distribution, and diversity metrics.

Usage:
    PYTHONPATH=. python3 scripts/analyze_generation_quality.py \
      --comparison-dir runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/comparison/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from scipy import signal


def compute_spectral_centroid(audio: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[float, float]:
    """Compute spectral centroid and its std dev."""
    _, _, S = signal.stft(audio, sr, nperseg=2048, noverlap=1536, window='hann')
    S = np.abs(S)
    freqs = np.fft.rfftfreq(2048, 1 / sr)
    mag = np.mean(S, axis=1)
    centroid = np.sum(freqs * mag) / np.sum(mag)
    centroid_std = np.std([np.sum(S[:, i] * freqs) / np.sum(S[:, i]) for i in range(S.shape[1])])
    return centroid, centroid_std


def compute_energy_distribution(audio: np.ndarray, sr: int, n_bands: int = 5) -> np.ndarray:
    """Compute energy in frequency bands."""
    freqs, _, Sxx = signal.spectrogram(audio, sr)
    energy_per_frame = np.sum(Sxx, axis=0)
    return np.percentile(energy_per_frame, np.linspace(0, 100, n_bands + 1))


def compute_novelty(audio: np.ndarray, sr: int) -> float:
    """Measure spectral novelty (how much the spectrum changes over time)."""
    _, _, S = signal.stft(audio, sr, nperseg=2048, noverlap=1536, window='hann')
    S = np.abs(S)
    diffs = np.diff(S, axis=1)
    novelty = np.mean(np.linalg.norm(diffs, axis=0))
    return novelty


def compute_peak_frequency(audio: np.ndarray, sr: int) -> float:
    """Compute dominant frequency."""
    freqs, _, Sxx = signal.spectrogram(audio, sr)
    mean_spectrum = np.mean(Sxx, axis=1)
    peak_idx = np.argmax(mean_spectrum)
    return freqs[peak_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-dir", required=True)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    comparison_dir = Path(args.comparison_dir)
    summary_path = Path(args.summary_json) if args.summary_json else comparison_dir / "generation_summary.json"

    # Load generation metadata
    with open(summary_path) as f:
        metadata = json.load(f)

    # Group by prompt
    by_prompt: Dict[int, List[Dict]] = {}
    for m in metadata:
        pid = m["prompt_idx"]
        if pid not in by_prompt:
            by_prompt[pid] = []
        by_prompt[pid].append(m)

    print("\n" + "=" * 100)
    print("GENERATION QUALITY ANALYSIS")
    print("=" * 100)

    all_metrics = []

    for prompt_idx in sorted(by_prompt.keys()):
        entries = by_prompt[prompt_idx]
        prompt_file = entries[0]["prompt_file"]
        detector_score = entries[0]["detector_score"]

        print(f"\n[Prompt {prompt_idx}] {prompt_file} (det={detector_score:.4f})")
        print("-" * 100)
        print(f"{'Config':<15} {'Temp':>6} {'Top-K':>6} {'Duration':>10} {'Novelty':>10} {'Peak Freq':>10} {'Spectral Centroid':>18}")
        print("-" * 100)

        for entry in entries:
            config_name = entry["config_name"]
            temp = entry["temperature"]
            top_k = entry["top_k"]
            total_dur = entry["total_duration"]

            # Load audio
            filename = f"full_{prompt_idx:02d}_{prompt_file}_T{temp:.2f}_{top_k}.wav"
            audio_path = comparison_dir / filename
            if not audio_path.exists():
                print(f"  {config_name:<13} {temp:6.2f} {top_k:6d} MISSING")
                continue

            audio, sr = sf.read(str(audio_path))
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Compute metrics
            novelty = compute_novelty(audio, sr)
            peak_freq = compute_peak_frequency(audio, sr)
            centroid, centroid_std = compute_spectral_centroid(audio, sr)

            print(
                f"  {config_name:<13} {temp:6.2f} {top_k:6d} {total_dur:9.2f}s {novelty:10.4f} "
                f"{peak_freq:9.1f} Hz {centroid:10.1f}±{centroid_std:5.1f} Hz"
            )

            all_metrics.append({
                "prompt_idx": prompt_idx,
                "prompt_file": prompt_file,
                "config_name": config_name,
                "temperature": temp,
                "top_k": top_k,
                "novelty": novelty,
                "peak_frequency": peak_freq,
                "spectral_centroid": centroid,
                "centroid_std": centroid_std,
            })

    # Summary across all configs
    print("\n" + "=" * 100)
    print("SUMMARY: Novelty by Temperature")
    print("=" * 100)

    for temp in [0.70, 0.85, 1.00]:
        temps = [m for m in all_metrics if m["temperature"] == temp]
        if temps:
            novelties = [m["novelty"] for m in temps]
            print(f"T={temp:.2f}: novelty {np.mean(novelties):.4f} (±{np.std(novelties):.4f}) "
                  f"[range {np.min(novelties):.4f}–{np.max(novelties):.4f}]")

    # Save detailed metrics
    metrics_path = comparison_dir / "quality_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nDetailed metrics saved to {metrics_path.name}")

    # Interpretation
    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("""
Novelty Score:
  - Measures how much the spectrum changes over time
  - Higher = more variation, lower = more repetitive
  - Low T (0.70): expect lower novelty (conservative sampling)
  - High T (1.00): expect higher novelty (diverse sampling)

Peak Frequency:
  - Dominant frequency in the audio
  - Should be in whale call range (~5-40 kHz depending on species)

Spectral Centroid:
  - Center of mass of the spectrum
  - Higher = more high-frequency energy
  - Useful for comparing timbral characteristics
""")


if __name__ == "__main__":
    main()
