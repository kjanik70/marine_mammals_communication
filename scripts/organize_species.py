#!/usr/bin/env python3
"""Tokenize audio files into species-group directories.

Creates separate tokenized directories for:
- sperm_whale: DSWP + Watkins Sperm_Whale
- toothed: All toothed cetaceans (Odontoceti) — includes sperm_whale + orcas + dolphins
- baleen: All baleen whales (Mysticeti)
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
from tqdm import tqdm

from scripts.tokenize_all_audio import segment_audio

# Species taxonomy
TOOTHED_SPECIES = [
    "Sperm_Whale", "Killer_Whale",
    "Atlantic_Spotted_Dolphin", "Bottlenose_Dolphin", "Clymene_Dolphin",
    "Common_Dolphin", "Frasers_Dolphin", "Grampus_Rissos_Dolphin",
    "Pantropical_Spotted_Dolphin", "Rough-Toothed_Dolphin",
    "Spinner_Dolphin", "Striped_Dolphin", "White-beaked_Dolphin",
    "White-sided_Dolphin", "Long-Finned_Pilot_Whale",
    "Short-Finned_Pacific_Pilot_Whale", "False_Killer_Whale",
    "Melon_Headed_Whale", "Narwhal", "Beluga_White_Whale",
]

BALEEN_SPECIES = [
    "Humpback_Whale", "Fin_Finback_Whale", "Bowhead_Whale",
    "Minke_Whale", "Northern_Right_Whale", "Southern_Right_Whale",
]


def tokenize_audio_files(audio_paths, tokenizer, output_dir, prefix, max_duration=5.0):
    """Tokenize a list of audio files into output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_sr = tokenizer.sample_rate
    seg_idx = 0
    n_tokens = 0

    for audio_path in tqdm(audio_paths, desc=f"  {prefix}"):
        try:
            audio, sr = sf.read(str(audio_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr < 2000:
                continue
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            segments = segment_audio(audio, target_sr, max_duration=max_duration)
            for seg in segments:
                seg_tensor = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                codes, z = tokenizer.encode(seg_tensor)
                tokens = tokenizer.codes_to_sequence(codes)
                if len(tokens) > 2:
                    out_path = output_dir / f"{prefix}_{seg_idx:06d}.npy"
                    np.save(out_path, tokens)
                    n_tokens += len(tokens)
                    seg_idx += 1
        except Exception as e:
            tqdm.write(f"    FAILED {audio_path.name}: {e}")

    return seg_idx, n_tokens


def collect_audio_files(directory, patterns=("*.wav", "*.flac")):
    """Find all audio files in directory recursively."""
    d = Path(directory)
    if not d.exists():
        return []
    files = []
    for pat in patterns:
        files.extend(d.rglob(pat))
    return sorted(f for f in files if f.stat().st_size > 0)


def main():
    parser = argparse.ArgumentParser(description="Organize and tokenize by species group")
    parser.add_argument("--codec-path", default="models/codec.pth")
    parser.add_argument("--output-base", default="data/tokenized")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-duration", type=float, default=5.0)
    args = parser.parse_args()

    from src.tokenizer.audio_tokenizer import AudioTokenizer

    print(f"Loading AudioTokenizer from {args.codec_path}...")
    tokenizer = AudioTokenizer(
        codec_path=args.codec_path, device=args.device, n_codebooks=1
    )

    base = Path(args.output_base)
    watkins_base = Path("data/raw/watkins/audio")

    # ===== SPERM WHALE =====
    print("\n=== SPERM WHALE (DSWP + Watkins Sperm_Whale) ===")
    sperm_files = collect_audio_files("data/raw/dswp")
    sperm_files += collect_audio_files(watkins_base / "Sperm_Whale")
    n_segs, n_toks = tokenize_audio_files(
        sperm_files, tokenizer, base / "sperm_whale", "sperm",
        max_duration=args.max_duration,
    )
    print(f"  -> {n_segs} segments, {n_toks:,} tokens")

    # ===== TOOTHED CETACEANS =====
    print("\n=== TOOTHED CETACEANS (Odontoceti) ===")
    toothed_out = base / "toothed"
    # Start with DSWP (sperm whales)
    toothed_files = collect_audio_files("data/raw/dswp")
    # Add ESP Orcas
    toothed_files += collect_audio_files("data/raw/esp_orcas/audio")
    # Add Orcasound sperm whale + orca files
    for f in collect_audio_files("data/raw/orcasound"):
        fname = f.name.lower()
        if "sperm" in fname or "yukusam" in fname or "srkw" in fname or "orca" in fname:
            toothed_files.append(f)
    # Add all toothed Watkins species
    for species in TOOTHED_SPECIES:
        toothed_files += collect_audio_files(watkins_base / species)
    # Add DORI-Orcasound (orca hydrophone recordings)
    toothed_files += collect_audio_files("data/raw/dori_orcasound")
    # Add Killer Whale Prince Edward Islands
    toothed_files += collect_audio_files("data/raw/kw_pei")
    n_segs, n_toks = tokenize_audio_files(
        toothed_files, tokenizer, toothed_out, "toothed",
        max_duration=args.max_duration,
    )
    print(f"  -> {n_segs} segments, {n_toks:,} tokens")

    # ===== BALEEN WHALES =====
    print("\n=== BALEEN WHALES (Mysticeti) ===")
    baleen_out = base / "baleen"
    baleen_files = []
    # Watkins baleen species
    for species in BALEEN_SPECIES:
        baleen_files += collect_audio_files(watkins_base / species)
    # Orcasound humpback files
    for f in collect_audio_files("data/raw/orcasound"):
        if "humpback" in f.name.lower():
            baleen_files.append(f)
    # Humpback Tsujii dataset
    baleen_files += collect_audio_files("data/raw/humpback_zenodo")
    # Right whale upcalls
    baleen_files += collect_audio_files("data/raw/right_whale/v1")
    n_segs, n_toks = tokenize_audio_files(
        baleen_files, tokenizer, baleen_out, "baleen",
        max_duration=args.max_duration,
    )
    print(f"  -> {n_segs} segments, {n_toks:,} tokens")

    print("\nDone! Species-specific token dirs:")
    for d in ["sperm_whale", "toothed", "baleen"]:
        p = base / d
        n = len(list(p.glob("*.npy"))) if p.exists() else 0
        print(f"  {p}: {n} files")


if __name__ == "__main__":
    main()
