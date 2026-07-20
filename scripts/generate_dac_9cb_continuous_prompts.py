#!/usr/bin/env python3
"""Generate samples using continuous (non-sparse) prompts.

Selects prompts based on audio density rather than detector score.
"""

import argparse
import csv
from pathlib import Path
import json
import random
import numpy as np
import torch
import soundfile as sf

from src.model.transformer import CausalTransformer
from src.tokenizer.dac_tokenizer import DACTokenizer

N_CB = 9
SEP_TOKEN = 9218
SEP_GAP_TOKEN = 9217
TOKENS_PER_SEC = 86.133
INTERLEAVED_PER_SEC = TOKENS_PER_SEC * N_CB

def interleave_2d(codes_2d: np.ndarray) -> np.ndarray:
    n_cb, T = codes_2d.shape
    offsets = np.arange(n_cb).reshape(n_cb, 1) * 1024
    return (codes_2d + offsets).T.reshape(-1).astype(np.int32)

def compute_audio_density(audio: np.ndarray, sr: int, window_size: int = 2048) -> float:
    """Compute fraction of audio with significant energy."""
    energy = np.array([
        np.sqrt(np.mean(audio[i:i+window_size]**2))
        for i in range(0, len(audio), window_size)
    ])
    threshold = np.max(energy) * 0.01 if np.max(energy) > 0 else 1e-6
    return np.mean(energy > threshold)

def pick_continuous_prompts(token_dir: Path, scores_csv: Path, n: int = 3, min_density: float = 0.5) -> list[dict]:
    """Select top N prompts by audio density (continuous audio)."""
    tokenizer = DACTokenizer(device="cpu", n_codebooks=N_CB)
    prompt_tokens = int(round(4.0 * INTERLEAVED_PER_SEC / N_CB)) * N_CB

    rows = []
    with open(scores_csv) as f:
        for row in csv.DictReader(f):
            npy_file = row["npy_file"]
            path = token_dir / npy_file

            if not path.exists():
                continue

            codes_2d = np.load(str(path))
            tokens_1d = interleave_2d(codes_2d)

            if len(tokens_1d) < prompt_tokens + N_CB:
                continue

            prompt_tokens_subset = tokens_1d[:prompt_tokens]

            try:
                audio = tokenizer.decode_tokens_to_audio(
                    prompt_tokens_subset, n_codebooks=N_CB, sep_token=SEP_TOKEN
                )
                density = compute_audio_density(audio, tokenizer.sample_rate)
            except:
                continue

            det_score = float(row["detector_score"]) if row["detector_score"] else 0.0

            rows.append({
                "npy_file": npy_file,
                "detector_score": det_score,
                "path": path,
                "density": density,
            })

    # Sort by density (prefer continuous audio)
    rows.sort(key=lambda r: r["density"], reverse=True)

    # Select top N with diversity
    selected, seen = [], set()
    for r in rows:
        if r["density"] < min_density:
            continue
        prefix = "_".join(r["npy_file"].split("_")[:3])
        if prefix not in seen:
            selected.append(r)
            seen.add(prefix)
        if len(selected) >= n:
            break

    if len(selected) < n:
        for r in rows:
            if r not in selected:
                prefix = "_".join(r["npy_file"].split("_")[:3])
                if prefix not in seen:
                    selected.append(r)
                    seen.add(prefix)
            if len(selected) >= n:
                break

    return selected[:n]

def generate_with_sep_stopping(
    model: CausalTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.85,
    top_k: int = 80,
    top_p: float = 0.0,
    device: str = "cuda",
    seed: int = None,
) -> torch.Tensor:
    """Generate tokens with explicit CUDA generator."""
    model.eval()

    # Use explicit CUDA generator for proper stochasticity
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    prompt = prompt[:, :model.config.max_seq_len]
    with torch.no_grad():
        result = model.forward(prompt)
        past_kv = result.get("past_kv", None)
        logits = result["logits"][:, -1, :]

    generated = []
    for step in range(max_new_tokens):
        with torch.no_grad():
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = logits / temperature

                if top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[:, [-1]]] = -float("inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    scaled[indices_to_remove] = -float("inf")

                probs = scaled.softmax(dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=generator)

        token_val = next_token.item()
        generated.append(next_token)

        if token_val in (SEP_TOKEN, SEP_GAP_TOKEN):
            break

        with torch.no_grad():
            result = model.forward(next_token, past_kv=past_kv)
            past_kv = result.get("past_kv", None)
            logits = result["logits"][:, -1, :]

    if generated:
        input_ids = torch.cat([prompt] + generated, dim=1)
    else:
        input_ids = prompt

    return input_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--token-dir", default="data/tokenized/sanctsound_humpback_dac")
    parser.add_argument("--scores-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=3)
    parser.add_argument("--min-density", type=float, default=0.5, help="Minimum audio density (0-1)")
    args = parser.parse_args()

    token_dir = Path(args.token_dir)
    scores_csv = Path(args.scores_csv) if args.scores_csv else token_dir / "chunk_scores.csv"
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "comparison_continuous"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {ckpt_path}...")
    ckpt = torch.load(str(ckpt_path), map_location=args.device, weights_only=False)
    config = ckpt["config"]
    model = CausalTransformer(config).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} params")

    # Load tokenizer
    print("Loading DAC tokenizer...")
    tokenizer = DACTokenizer(device="cpu", n_codebooks=N_CB)

    # Select continuous prompts
    prompt_tokens = int(round(4.0 * INTERLEAVED_PER_SEC / N_CB)) * N_CB
    prompt_tokens = min(prompt_tokens, config.max_seq_len // 2)
    max_new = min(int(10 * INTERLEAVED_PER_SEC), config.max_seq_len - prompt_tokens)

    print(f"\nPrompt: {prompt_tokens} tokens ({prompt_tokens / INTERLEAVED_PER_SEC:.1f}s)")
    print(f"Max generation: {max_new} tokens (~{max_new / INTERLEAVED_PER_SEC:.1f}s)")

    print(f"\nSelecting top {args.n_prompts} continuous prompts (density > {args.min_density:.0%})...")
    prompts = pick_continuous_prompts(token_dir, scores_csv, n=args.n_prompts, min_density=args.min_density)

    if not prompts:
        print(f"No prompts found with density > {args.min_density:.0%}")
        return

    for p in prompts:
        print(f"  {p['npy_file']}: density={p['density']:.1%}, det={p['detector_score']:.3f}")

    # Hyperparameters
    configs = [
        {"temp": 0.70, "top_k": 40, "name": "conservative", "seed": 42},
        {"temp": 0.85, "top_k": 80, "name": "balanced", "seed": 43},
        {"temp": 1.00, "top_k": 120, "name": "diverse", "seed": 44},
    ]

    results_summary = []

    print(f"\nGenerating → {out_dir}/\n")
    for i, seg in enumerate(prompts):
        npy_path = seg["path"]
        codes_2d = np.load(str(npy_path))
        tokens_1d = interleave_2d(codes_2d)
        prompt = tokens_1d[:prompt_tokens]
        stem = npy_path.stem

        print(f"[{i}] {npy_path.name}")
        print(f"     Audio density: {seg['density']:.1%}, detector: {seg['detector_score']:.3f}")

        # Generate with each configuration
        for cfg in configs:
            print(f"     {cfg['name']:12s} (T={cfg['temp']:.2f}, top_k={cfg['top_k']:3d})...", end=" ", flush=True)

            prompt_t = torch.tensor(prompt, dtype=torch.long, device=args.device).unsqueeze(0)
            with torch.no_grad():
                generated = generate_with_sep_stopping(
                    model, prompt_t, max_new,
                    temperature=cfg["temp"],
                    top_k=cfg["top_k"],
                    top_p=0.0,
                    device=args.device,
                    seed=cfg["seed"],
                )

            full_tokens = generated[0].cpu().numpy()
            gen_tokens = full_tokens[len(prompt):]

            # Decode
            prompt_audio = tokenizer.decode_tokens_to_audio(
                prompt, n_codebooks=N_CB, sep_token=SEP_TOKEN
            )
            full_audio = tokenizer.decode_tokens_to_audio(
                full_tokens, n_codebooks=N_CB, sep_token=SEP_TOKEN
            )

            sr = tokenizer.sample_rate
            prompt_dur = len(prompt_audio) / sr
            full_dur = len(full_audio) / sr
            gen_dur = full_dur - prompt_dur

            # Save
            cfg_str = f"{cfg['temp']:.2f}_{cfg['top_k']}"
            full_path = out_dir / f"full_{i:02d}_{stem}_T{cfg_str}.wav"
            sf.write(str(full_path), full_audio, sr)

            print(f"gen={gen_dur:.1f}s → {full_path.name}")

            results_summary.append({
                "prompt_idx": i,
                "prompt_file": stem,
                "audio_density": seg["density"],
                "detector_score": seg["detector_score"],
                "temperature": cfg["temp"],
                "top_k": cfg["top_k"],
                "config_name": cfg["name"],
                "generated_duration": gen_dur,
                "total_duration": full_dur,
                "generated_tokens": len(gen_tokens),
            })

    # Save summary
    summary_path = out_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved to {summary_path.name}")

    print(f"\n✓ Generated {len(results_summary)} samples in {out_dir}/")


if __name__ == "__main__":
    main()
