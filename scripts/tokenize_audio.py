#!/usr/bin/env python3
"""Batch-tokenize DSWP audio files using LAC codec."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Tokenize audio files with LAC codec")
    parser.add_argument("--audio-dir", type=str, default="data/raw/dswp", help="Directory of WAV files")
    parser.add_argument("--output-dir", type=str, default="data/tokenized/dswp", help="Output directory for .npy files")
    parser.add_argument("--codec-path", type=str, default="models/codec.pth", help="Path to LAC codec weights")
    parser.add_argument("--n-codebooks", type=int, default=1, help="Number of codebooks to use (1=coarse only)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from src.tokenizer.audio_tokenizer import AudioTokenizer

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading AudioTokenizer from {args.codec_path}...")
    tokenizer = AudioTokenizer(
        codec_path=args.codec_path,
        device=args.device,
        n_codebooks=args.n_codebooks,
    )
    print(f"  Sample rate: {tokenizer.sample_rate}")
    print(f"  Tokens/sec: {tokenizer.tokens_per_second:.1f}")
    print(f"  Codebooks: {args.n_codebooks}")

    wav_files = sorted(audio_dir.glob("*.wav"))
    print(f"\nFound {len(wav_files)} WAV files in {audio_dir}")

    stats = {
        "n_files": 0,
        "n_failed": 0,
        "total_tokens": 0,
        "total_duration_sec": 0.0,
        "token_lengths": [],
    }

    for wav_path in tqdm(wav_files, desc="Tokenizing"):
        try:
            codes, z = tokenizer.encode_file(wav_path)
            seq = tokenizer.codes_to_sequence(codes)

            out_path = output_dir / f"{wav_path.stem}.npy"
            np.save(out_path, seq)

            audio, sr = sf.read(str(wav_path), dtype="float32")
            duration = len(audio) / sr

            stats["n_files"] += 1
            stats["total_tokens"] += len(seq)
            stats["total_duration_sec"] += duration
            stats["token_lengths"].append(len(seq))

        except Exception as e:
            stats["n_failed"] += 1
            tqdm.write(f"  FAILED {wav_path.name}: {e}")

    # Summary
    print(f"\n=== Tokenization Summary ===")
    print(f"Files processed: {stats['n_files']}")
    print(f"Files failed:    {stats['n_failed']}")
    print(f"Total tokens:    {stats['total_tokens']:,}")
    print(f"Total duration:  {stats['total_duration_sec']:.1f}s ({stats['total_duration_sec']/60:.1f} min)")
    if stats["token_lengths"]:
        lengths = np.array(stats["token_lengths"])
        print(f"Tokens per file: mean={lengths.mean():.0f}, min={lengths.min()}, max={lengths.max()}, median={np.median(lengths):.0f}")
        print(f"Tokens/sec:      {stats['total_tokens'] / stats['total_duration_sec']:.1f}")

    # Save metadata
    meta = {
        "codec_path": args.codec_path,
        "n_codebooks": args.n_codebooks,
        "sample_rate": tokenizer.sample_rate,
        "tokens_per_second": tokenizer.tokens_per_second,
        "vocab_size": tokenizer.vocab_size,
        "n_files": stats["n_files"],
        "total_tokens": stats["total_tokens"],
        "total_duration_sec": stats["total_duration_sec"],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
