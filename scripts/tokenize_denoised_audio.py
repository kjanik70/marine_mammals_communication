#!/usr/bin/env python3
"""Tokenize denoised audio with long chunks (30s) preserving natural pauses.

Unlike the original tokenizer that splits into ≤5s segments:
- Removes silence >4s (replaces with 0.5s gap)
- Keeps silence ≤4s (natural inter-vocalization pauses)
- Splits into ≤30s chunks at silence boundaries
- Preserves temporal continuity within chunks
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


def compute_energy_envelope(audio, sr, frame_ms=25, hop_ms=10):
    """Compute short-time RMS energy envelope."""
    frame_length = int(frame_ms / 1000 * sr)
    hop_length = int(hop_ms / 1000 * sr)
    n_frames = max(1, (len(audio) - frame_length) // hop_length)
    energy = np.array([
        np.sqrt(np.mean(audio[i * hop_length:i * hop_length + frame_length] ** 2))
        for i in range(n_frames)
    ])
    return energy, hop_length, frame_length


def find_silence_regions(energy, hop_length, sr, threshold=0.005):
    """Find contiguous silence regions (start_sample, end_sample, duration_s)."""
    is_silent = energy < threshold
    regions = []
    in_silence = False
    start = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            start = i * hop_length
            in_silence = True
        elif not silent and in_silence:
            end = i * hop_length
            dur = (end - start) / sr
            regions.append((start, end, dur))
            in_silence = False

    if in_silence:
        end = len(energy) * hop_length
        dur = (end - start) / sr
        regions.append((start, end, dur))

    return regions


def segment_audio_long(audio, sr, max_duration=30.0, min_duration=2.0,
                       max_silence=4.0, replacement_silence=0.5,
                       silence_threshold=0.005):
    """Segment audio preserving natural pauses, removing long silence.

    1. Find all silence regions
    2. Remove silence >max_silence, replace with replacement_silence gap
    3. Split into ≤max_duration chunks, preferring to split at silence gaps
    """
    duration = len(audio) / sr
    if duration < min_duration:
        return []

    # Short clips: return as-is
    if duration <= max_duration:
        # Still remove long silence even in short clips
        energy, hop_length, _ = compute_energy_envelope(audio, sr)
        silence_regions = find_silence_regions(energy, hop_length, sr, silence_threshold)
        long_silences = [r for r in silence_regions if r[2] > max_silence]

        if not long_silences:
            return [audio]

        # Rebuild audio with long silences shortened
        audio = _remove_long_silence(audio, sr, long_silences, replacement_silence)
        if len(audio) / sr >= min_duration:
            return [audio]
        return []

    # Long clips: remove long silence, then chunk
    energy, hop_length, _ = compute_energy_envelope(audio, sr)
    silence_regions = find_silence_regions(energy, hop_length, sr, silence_threshold)

    # Rebuild audio with long silences shortened
    long_silences = [r for r in silence_regions if r[2] > max_silence]
    if long_silences:
        audio = _remove_long_silence(audio, sr, long_silences, replacement_silence)
        # Recompute energy after silence removal
        energy, hop_length, _ = compute_energy_envelope(audio, sr)
        silence_regions = find_silence_regions(energy, hop_length, sr, silence_threshold)

    # If now short enough, return as single chunk
    if len(audio) / sr <= max_duration:
        if len(audio) / sr >= min_duration:
            return [audio]
        return []

    # Split at silence boundaries into ≤max_duration chunks
    return _split_at_silence(audio, sr, silence_regions, max_duration, min_duration)


def _remove_long_silence(audio, sr, long_silences, replacement_duration):
    """Replace long silence regions with short gaps."""
    replacement_samples = int(replacement_duration * sr)
    pieces = []
    prev_end = 0

    for start, end, dur in sorted(long_silences):
        # Keep audio before this silence
        if start > prev_end:
            pieces.append(audio[prev_end:start])
        # Insert short silence
        pieces.append(np.zeros(replacement_samples, dtype=audio.dtype))
        prev_end = end

    # Keep remaining audio
    if prev_end < len(audio):
        pieces.append(audio[prev_end:])

    if pieces:
        return np.concatenate(pieces)
    return audio


def _split_at_silence(audio, sr, silence_regions, max_duration, min_duration):
    """Split audio into chunks ≤max_duration, preferring silence boundaries."""
    max_samples = int(max_duration * sr)
    min_samples = int(min_duration * sr)
    total = len(audio)

    # Get silence midpoints as potential split points
    split_candidates = []
    for start, end, dur in silence_regions:
        if dur >= 0.1:  # Only split at noticeable pauses
            mid = (start + end) // 2
            split_candidates.append(mid)

    chunks = []
    chunk_start = 0

    while chunk_start < total:
        remaining = total - chunk_start

        if remaining <= max_samples:
            # Last piece fits in one chunk
            if remaining >= min_samples:
                chunks.append(audio[chunk_start:])
            break

        # Find the best split point within [chunk_start, chunk_start + max_samples]
        chunk_end_max = chunk_start + max_samples
        best_split = None

        # Prefer the latest silence boundary before max_duration
        for sp in reversed(split_candidates):
            if chunk_start + min_samples <= sp <= chunk_end_max:
                best_split = sp
                break

        if best_split is None:
            # No silence boundary found — hard split at max_duration
            best_split = chunk_end_max

        chunk = audio[chunk_start:best_split]
        if len(chunk) >= min_samples:
            chunks.append(chunk)
        chunk_start = best_split

    return chunks


def load_file_quality_filter(csv_path, min_avg_score=0.6):
    """Load quality grades and return set of (source, filename) that pass file-level filter.

    A file passes if its average quality score across all segments >= min_avg_score.
    """
    file_scores = {}  # (source, filename) -> [scores]
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["source"], row["file"])
            if key not in file_scores:
                file_scores[key] = []
            file_scores[key].append(float(row["quality_score"]))

    allowed = set()
    for (source, fname), scores in file_scores.items():
        if np.mean(scores) >= min_avg_score:
            allowed.add((source, fname))

    return allowed


def process_dataset(name, audio_dir, tokenizer, output_dir, max_duration=30.0,
                    file_patterns=("*.wav",), quality_filter=None):
    """Process a denoised dataset: load, segment (long chunks), tokenize, save."""
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for pattern in file_patterns:
        files.extend(audio_dir.rglob(pattern))
    files = sorted(f for f in files if f.stat().st_size > 0)

    if not files:
        print(f"  No audio files found in {audio_dir}")
        return {"n_files": 0, "n_chunks": 0, "n_tokens": 0, "n_failed": 0,
                "total_duration": 0.0, "chunk_durations": []}

    stats = {"n_files": 0, "n_chunks": 0, "n_tokens": 0, "n_failed": 0,
             "total_duration": 0.0, "chunk_durations": []}
    target_sr = tokenizer.sample_rate
    chunk_idx = 0

    for audio_path in tqdm(files, desc=f"  {name}"):
        try:
            # File-level quality filter
            if quality_filter is not None:
                if (name, audio_path.name) not in quality_filter:
                    continue

            audio, sr = sf.read(str(audio_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Denoised files should already be at target_sr, but check
            if sr != target_sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            stats["total_duration"] += len(audio) / target_sr
            stats["n_files"] += 1

            # Segment into long chunks
            chunks = segment_audio_long(audio, target_sr, max_duration=max_duration)

            for chunk in chunks:
                import torch
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                codes, z = tokenizer.encode(chunk_tensor)
                tokens = tokenizer.codes_to_sequence(codes)

                if len(tokens) > 2:
                    out_path = output_dir / f"{name}_{chunk_idx:06d}.npy"
                    np.save(out_path, tokens)
                    stats["n_chunks"] += 1
                    stats["n_tokens"] += len(tokens)
                    stats["chunk_durations"].append(len(chunk) / target_sr)
                    chunk_idx += 1

        except Exception as e:
            stats["n_failed"] += 1
            tqdm.write(f"    FAILED {audio_path.name}: {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Tokenize denoised audio (long chunks)")
    parser.add_argument("--codec-path", type=str, default="models/codec.pth")
    parser.add_argument("--denoised-dir", type=str, default="data/denoised",
                        help="Directory with denoised WAVs")
    parser.add_argument("--output-dir", type=str, default="data/tokenized/denoised_4cb")
    parser.add_argument("--n-codebooks", type=int, default=4)
    parser.add_argument("--max-duration", type=float, default=30.0,
                        help="Max chunk duration in seconds")
    parser.add_argument("--quality-csv", type=str, default=None,
                        help="Path to audio_quality_grades.csv for file-level filtering")
    parser.add_argument("--min-quality-score", type=float, default=0.6,
                        help="Min avg quality score for file-level filter (default: 0.6 = B grade)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

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

    # Load quality filter if specified
    quality_filter = None
    if args.quality_csv:
        print(f"Loading file-level quality filter from {args.quality_csv} "
              f"(min avg score: {args.min_quality_score})")
        quality_filter = load_file_quality_filter(args.quality_csv, args.min_quality_score)
        print(f"  {len(quality_filter)} files pass quality filter")

    # Datasets — point to denoised directories
    datasets = [
        ("dswp", f"{args.denoised_dir}/dswp"),
        ("watkins", f"{args.denoised_dir}/watkins"),
        ("esp_orcas", f"{args.denoised_dir}/esp_orcas"),
        ("orcasound", f"{args.denoised_dir}/orcasound"),
        ("mbari", f"{args.denoised_dir}/mbari"),
        ("dori_orca", f"{args.denoised_dir}/dori_orca"),
        ("dori_orca_full", f"{args.denoised_dir}/dori_orca_full"),
        ("humpback_tsujii", f"{args.denoised_dir}/humpback_tsujii"),
        ("kw_pei", f"{args.denoised_dir}/kw_pei"),
        ("right_whale", f"{args.denoised_dir}/right_whale"),
    ]

    all_stats = {}
    total_tokens = 0
    total_chunks = 0
    all_durations = []

    for name, audio_dir in datasets:
        if not Path(audio_dir).exists():
            print(f"\nSkipping {name}: {audio_dir} not found")
            continue

        print(f"\n=== {name} ===")
        stats = process_dataset(
            name, audio_dir, tokenizer, output_dir,
            max_duration=args.max_duration,
            quality_filter=quality_filter,
        )
        all_stats[name] = stats
        total_tokens += stats["n_tokens"]
        total_chunks += stats["n_chunks"]
        all_durations.extend(stats["chunk_durations"])

        print(f"  Files: {stats['n_files']}, Chunks: {stats['n_chunks']}, "
              f"Tokens: {stats['n_tokens']:,}")
        if stats["chunk_durations"]:
            avg_dur = np.mean(stats["chunk_durations"])
            print(f"  Avg chunk duration: {avg_dur:.1f}s")
        if stats["n_failed"]:
            print(f"  Failed: {stats['n_failed']}")

    print(f"\n{'='*50}")
    print(f"TOTAL: {total_chunks} chunks, {total_tokens:,} tokens")
    if all_durations:
        print(f"Avg chunk duration: {np.mean(all_durations):.1f}s "
              f"(min {np.min(all_durations):.1f}s, max {np.max(all_durations):.1f}s)")
    print(f"Token files in: {output_dir}")

    # Save metadata
    meta = {
        "codec_path": args.codec_path,
        "n_codebooks": args.n_codebooks,
        "sample_rate": tokenizer.sample_rate,
        "tokens_per_second": tokenizer.tokens_per_second,
        "vocab_size": tokenizer.vocab_size,
        "max_chunk_duration": args.max_duration,
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "avg_chunk_duration": float(np.mean(all_durations)) if all_durations else 0,
        "denoised": True,
        "datasets": {k: {kk: vv for kk, vv in v.items() if kk != "chunk_durations"}
                     for k, v in all_stats.items()},
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
