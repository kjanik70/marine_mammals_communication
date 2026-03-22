#!/usr/bin/env python3
"""Tokenize all audio datasets with LAC codec for training.

Handles varying sample rates, mono/stereo, and different file formats.
Segments long recordings into coda-length chunks before tokenizing.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm


def segment_audio(audio, sr, max_duration=5.0, min_duration=0.3, silence_threshold=0.005):
    """Split long audio into segments based on energy.

    For short clips (<= max_duration), returns the whole clip.
    For long clips, splits on silence gaps to extract vocalization segments.
    """
    duration = len(audio) / sr

    if duration <= max_duration:
        if duration >= min_duration:
            return [audio]
        return []

    # For long recordings: use energy-based segmentation
    # Compute short-time energy
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    energy = np.array([
        np.sqrt(np.mean(audio[i:i+frame_length]**2))
        for i in range(0, len(audio) - frame_length, hop_length)
    ])

    # Find segments above threshold
    is_active = energy > silence_threshold
    segments = []
    in_segment = False
    seg_start = 0

    for i, active in enumerate(is_active):
        if active and not in_segment:
            seg_start = i * hop_length
            in_segment = True
        elif not active and in_segment:
            seg_end = i * hop_length
            seg_dur = (seg_end - seg_start) / sr
            if min_duration <= seg_dur <= max_duration:
                segments.append(audio[seg_start:seg_end])
            elif seg_dur > max_duration:
                # Split oversized segments into max_duration chunks
                chunk_samples = int(max_duration * sr)
                for j in range(0, seg_end - seg_start, chunk_samples):
                    chunk = audio[seg_start + j:seg_start + j + chunk_samples]
                    if len(chunk) / sr >= min_duration:
                        segments.append(chunk)
            in_segment = False

    # Handle last segment
    if in_segment:
        seg_end = len(audio)
        seg_dur = (seg_end - seg_start) / sr
        if min_duration <= seg_dur <= max_duration:
            segments.append(audio[seg_start:seg_end])

    return segments


def process_dataset(name, audio_dir, tokenizer, output_dir, max_duration=5.0,
                    file_patterns=("*.wav", "*.flac"), min_sr=2000):
    """Process a dataset directory: load, segment, tokenize, save."""
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files (including subdirectories)
    files = []
    for pattern in file_patterns:
        files.extend(audio_dir.rglob(pattern))
    files = sorted(f for f in files if f.stat().st_size > 0)

    if not files:
        print(f"  No audio files found in {audio_dir}")
        return {"n_files": 0, "n_segments": 0, "n_tokens": 0}

    stats = {"n_files": 0, "n_segments": 0, "n_tokens": 0, "n_failed": 0,
             "total_duration": 0.0, "token_lengths": []}
    target_sr = tokenizer.sample_rate
    seg_idx = 0

    for audio_path in tqdm(files, desc=f"  {name}"):
        try:
            audio, sr = sf.read(str(audio_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Skip very low sample rate files (can't resample meaningfully)
            if sr < min_sr:
                continue

            stats["total_duration"] += len(audio) / sr
            stats["n_files"] += 1

            # Resample to codec rate
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            # Segment long recordings
            segments = segment_audio(audio, target_sr, max_duration=max_duration)

            for seg in segments:
                import torch
                seg_tensor = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                codes, z = tokenizer.encode(seg_tensor)
                tokens = tokenizer.codes_to_sequence(codes)

                if len(tokens) > 2:
                    out_path = output_dir / f"{name}_{seg_idx:06d}.npy"
                    np.save(out_path, tokens)
                    stats["n_segments"] += 1
                    stats["n_tokens"] += len(tokens)
                    stats["token_lengths"].append(len(tokens))
                    seg_idx += 1

        except Exception as e:
            stats["n_failed"] += 1
            tqdm.write(f"    FAILED {audio_path.name}: {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Tokenize all audio datasets")
    parser.add_argument("--codec-path", type=str, default="models/codec.pth")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: data/tokenized/all or all_4cb)")
    parser.add_argument("--n-codebooks", type=int, default=4)
    parser.add_argument("--max-duration", type=float, default=5.0,
                        help="Max segment duration in seconds")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.output_dir is None:
        suffix = f"_{args.n_codebooks}cb" if args.n_codebooks > 1 else ""
        args.output_dir = f"data/tokenized/all{suffix}"

    from src.tokenizer.audio_tokenizer import AudioTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading AudioTokenizer from {args.codec_path}...")
    tokenizer = AudioTokenizer(
        codec_path=args.codec_path,
        device=args.device,
        n_codebooks=args.n_codebooks,
    )
    print(f"  Sample rate: {tokenizer.sample_rate}, Tokens/sec: {tokenizer.tokens_per_second:.1f}")

    # Define datasets to process
    datasets = [
        ("dswp", "data/raw/dswp"),
        ("watkins", "data/raw/watkins/audio"),
        ("esp_orcas", "data/raw/esp_orcas/audio"),
        ("orcasound", "data/raw/orcasound"),
        ("mbari", "data/raw/mbari"),
        ("dori_orca", "data/raw/dori_orcasound"),
        ("humpback_tsujii", "data/raw/humpback_zenodo"),
        ("kw_pei", "data/raw/kw_pei"),
        ("right_whale", "data/raw/right_whale/v1"),
    ]

    all_stats = {}
    total_tokens = 0
    total_segments = 0

    for name, audio_dir in datasets:
        if not Path(audio_dir).exists():
            print(f"\nSkipping {name}: {audio_dir} not found")
            continue

        print(f"\n=== {name} ===")
        stats = process_dataset(
            name, audio_dir, tokenizer, output_dir,
            max_duration=args.max_duration,
        )
        all_stats[name] = stats
        total_tokens += stats["n_tokens"]
        total_segments += stats["n_segments"]

        print(f"  Files: {stats['n_files']}, Segments: {stats['n_segments']}, "
              f"Tokens: {stats['n_tokens']:,}, Duration: {stats['total_duration']:.0f}s")
        if stats["n_failed"]:
            print(f"  Failed: {stats['n_failed']}")

    print(f"\n{'='*50}")
    print(f"TOTAL: {total_segments} segments, {total_tokens:,} tokens")
    print(f"Token files in: {output_dir}")

    # Save metadata
    meta = {
        "codec_path": args.codec_path,
        "n_codebooks": args.n_codebooks,
        "sample_rate": tokenizer.sample_rate,
        "tokens_per_second": tokenizer.tokens_per_second,
        "vocab_size": tokenizer.vocab_size,
        "max_segment_duration": args.max_duration,
        "total_segments": total_segments,
        "total_tokens": total_tokens,
        "datasets": {k: {kk: vv for kk, vv in v.items() if kk != "token_lengths"}
                     for k, v in all_stats.items()},
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
