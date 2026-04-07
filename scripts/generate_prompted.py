#!/usr/bin/env python3
"""Generate whale audio continuations from real whale prompts.

Picks the best whale vocalization segments from tokenized data,
feeds them as prompts to the model, and generates continuations
up to the full context length. Outputs both the prompt audio and
the full (prompt + continuation) audio for comparison.
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import soundfile as sf

from src.model.transformer import CausalTransformer
from src.tokenizer.audio_tokenizer import AudioTokenizer


def score_segment(path: Path, n_codebooks: int = 4) -> dict:
    """Score a tokenized segment for whale vocalization quality.

    Uses CB0 entropy and unique ratio as proxies for melodic,
    non-repetitive whale song content.
    """
    tokens = np.load(path)
    n_tokens = len(tokens)

    if n_tokens < n_codebooks * 20:
        return None

    # Extract CB0 tokens (every n_codebooks-th token starting at 0)
    cb0 = tokens[::n_codebooks]

    # Entropy of CB0 (higher = more melodic variation)
    _, counts = np.unique(cb0, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # Unique ratio (higher = less repetitive)
    unique_ratio = len(counts) / len(cb0)

    # Penalize very short segments
    length_bonus = min(1.0, n_tokens / 2000)

    score = entropy * unique_ratio * length_bonus

    return {
        "path": path,
        "n_tokens": n_tokens,
        "entropy": entropy,
        "unique_ratio": unique_ratio,
        "score": score,
    }


def pick_best_segments(token_dir: str, n: int = 5, n_codebooks: int = 4,
                       min_tokens: int = 400) -> list[dict]:
    """Scan tokenized directory and return the top N segments by quality."""
    token_dir = Path(token_dir)
    files = sorted(token_dir.glob("*.npy"))
    print(f"Scanning {len(files)} files for best whale segments...")

    # Sample a manageable subset if too many files
    if len(files) > 10000:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(files), size=10000, replace=False)
        files = [files[i] for i in sorted(indices)]
        print(f"  (sampled 10,000 files for scoring)")

    scored = []
    for f in files:
        info = score_segment(f, n_codebooks)
        if info and info["n_tokens"] >= min_tokens:
            scored.append(info)

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate: pick from different source files (different hour/station)
    # Group by prefix (e.g., sanctsound_hi01_01)
    selected = []
    seen_prefixes = set()
    for s in scored:
        prefix = "_".join(s["path"].stem.rsplit("_", 1)[0].split("_")[:3])
        if prefix not in seen_prefixes:
            selected.append(s)
            seen_prefixes.add(prefix)
        if len(selected) >= n:
            break

    # Fall back if not enough unique prefixes
    if len(selected) < n:
        for s in scored:
            if s not in selected:
                selected.append(s)
            if len(selected) >= n:
                break

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Generate whale audio continuations from real prompts"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="runs/audio_medium_sanctsound_humpback_4cb/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--token-dir", type=str,
                        default="data/tokenized/sanctsound_humpback_4cb",
                        help="Directory with tokenized .npy files")
    parser.add_argument("--codec-path", type=str, default="models/codec.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--prompt-tokens", type=int, default=512,
                        help="Number of tokens to use as prompt (rest is generated)")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as checkpoint dir)")
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt["config"]
    model = CausalTransformer(config).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    max_seq_len = config.max_seq_len
    n_codebooks = 4
    sep_token = n_codebooks * 1024 + 2  # 4098

    print(f"Model: {model.count_parameters():,} params, max_seq_len={max_seq_len}")
    print(f"Checkpoint: val_loss={ckpt.get('val_loss', '?')}, step={ckpt.get('step', '?')}")

    # Load tokenizer for decoding (always on CPU to avoid VRAM contention)
    print("Loading codec...")
    tokenizer = AudioTokenizer(
        codec_path=args.codec_path, device="cpu", n_codebooks=n_codebooks
    )

    # Pick best segments
    segments = pick_best_segments(
        args.token_dir, n=args.n_samples, n_codebooks=n_codebooks,
        min_tokens=args.prompt_tokens + 100,
    )

    if not segments:
        print("No suitable segments found!")
        return

    # Output directory
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / "prompted"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {len(segments)} prompted continuations → {out_dir}/")
    print(f"Prompt: {args.prompt_tokens} tokens, generating up to {max_seq_len} total\n")

    for i, seg in enumerate(segments):
        tokens = np.load(seg["path"])
        prompt_len = min(args.prompt_tokens, len(tokens))
        prompt = tokens[:prompt_len]

        max_new = max_seq_len - prompt_len
        print(f"[{i}] {seg['path'].name}")
        print(f"    score={seg['score']:.2f}, entropy={seg['entropy']:.1f}, "
              f"unique={seg['unique_ratio']:.2f}, tokens={seg['n_tokens']}")
        print(f"    prompt={prompt_len} tokens, generating {max_new} new tokens...")

        # Generate
        prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=args.device).unsqueeze(0)
        with torch.no_grad():
            generated = model.generate(
                prompt_tensor,
                max_new_tokens=max_new,
                temperature=args.temperature,
                top_k=args.top_k,
                eos_token_id=-1,
            )
        full_tokens = generated[0].cpu().numpy()

        # Decode prompt-only audio
        prompt_audio = tokenizer.decode_tokens_to_audio(
            prompt, n_codebooks=n_codebooks, sep_token=sep_token
        )

        # Decode full (prompt + continuation) audio
        full_audio = tokenizer.decode_tokens_to_audio(
            full_tokens, n_codebooks=n_codebooks, sep_token=sep_token
        )

        # Save
        prompt_path = out_dir / f"prompt_{i}.wav"
        full_path = out_dir / f"continued_{i}.wav"
        sf.write(str(prompt_path), prompt_audio, tokenizer.sample_rate)
        sf.write(str(full_path), full_audio, tokenizer.sample_rate)

        prompt_dur = len(prompt_audio) / tokenizer.sample_rate
        full_dur = len(full_audio) / tokenizer.sample_rate
        print(f"    → prompt: {prompt_path.name} ({prompt_dur:.1f}s)")
        print(f"    → full:   {full_path.name} ({full_dur:.1f}s)")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
