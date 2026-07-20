#!/usr/bin/env python3
"""Find prompts with continuous audio (not sparse detections)."""

import numpy as np
from pathlib import Path
import csv

from src.tokenizer.dac_tokenizer import DACTokenizer

N_CB = 9
TOKENS_PER_SEC = 86.133
INTERLEAVED_PER_SEC = TOKENS_PER_SEC * N_CB

def interleave_2d(codes_2d: np.ndarray) -> np.ndarray:
    n_cb, T = codes_2d.shape
    offsets = np.arange(n_cb).reshape(n_cb, 1) * 1024
    return (codes_2d + offsets).T.reshape(-1).astype(np.int32)

def compute_audio_density(audio: np.ndarray, sr: int, window_size: int = 2048) -> float:
    """Compute fraction of audio that has significant energy."""
    # Compute RMS in sliding windows
    energy = np.array([
        np.sqrt(np.mean(audio[i:i+window_size]**2))
        for i in range(0, len(audio), window_size)
    ])

    # Consider window "active" if energy > 1% of max
    threshold = np.max(energy) * 0.01
    active_fraction = np.mean(energy > threshold)
    return active_fraction

token_dir = Path("data/tokenized/sanctsound_humpback_dac")
scores_csv = token_dir / "chunk_scores.csv"

print("Analyzing audio density of prompts...")
print("=" * 80)

candidates = []

with open(scores_csv) as f:
    for row in csv.DictReader(f):
        npy_file = row["npy_file"]
        path = token_dir / npy_file

        if not path.exists():
            continue

        codes_2d = np.load(str(path))
        tokens_1d = interleave_2d(codes_2d)

        # Get 4-second prompt worth
        prompt_tokens = int(round(4.0 * INTERLEAVED_PER_SEC / N_CB)) * N_CB
        if len(tokens_1d) < prompt_tokens + N_CB:
            continue

        prompt_tokens_subset = tokens_1d[:prompt_tokens]

        # Decode to audio to check density
        tokenizer = DACTokenizer(device="cpu", n_codebooks=N_CB)
        try:
            audio = tokenizer.decode_tokens_to_audio(
                prompt_tokens_subset, n_codebooks=N_CB, sep_token=9218
            )
        except:
            continue

        density = compute_audio_density(audio, tokenizer.sample_rate)
        det_score = float(row["detector_score"]) if row["detector_score"] else 0.0

        candidates.append({
            "file": npy_file,
            "detector_score": det_score,
            "density": density,
        })

# Sort by density (continuous audio first)
candidates.sort(key=lambda x: x["density"], reverse=True)

print(f"{'File':<50} {'Detector Score':>15} {'Density':>10}")
print("-" * 80)

for i, cand in enumerate(candidates[:20]):
    print(f"{cand['file']:<50} {cand['detector_score']:>15.3f} {cand['density']:>10.1%}")
    if i == 2:
        print("  ↑ Top 3 candidates for continuous prompts")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("Use prompts with density > 50% for continuous audio throughout the 4s window")
print(f"\nTop candidate: {candidates[0]['file']} (density: {candidates[0]['density']:.1%})")
