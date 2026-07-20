#!/usr/bin/env python3
"""Generate audio samples with varying temperature/top_k to compare quality.

FIXED VERSION: Ensures stochastic sampling by resetting RNG state before each generation.

Usage:
    PYTHONPATH=. python3 scripts/generate_dac_9cb_comparison_fixed.py \
      --checkpoint runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/best_model_step109000.pt
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
    """Convert (9, T) codes array to 1D interleaved sequence with CB offsets."""
    n_cb, T = codes_2d.shape
    offsets = np.arange(n_cb).reshape(n_cb, 1) * 1024
    return (codes_2d + offsets).T.reshape(-1).astype(np.int32)


def pick_best_prompts(
    token_dir: Path,
    scores_csv: Path,
    n: int = 5,
    min_detector: float = 0.8,
) -> list[dict]:
    """Return top N diverse high-quality chunks."""
    rows = []
    with open(scores_csv) as f:
        for row in csv.DictReader(f):
            det = float(row["detector_score"]) if row["detector_score"] else 0.0
            rows.append({
                "npy_file": row["npy_file"],
                "flac_name": row["flac_name"],
                "detector_score": det,
                "whale_cv": float(row["whale_cv"]),
                "energy_ratio": float(row["energy_ratio"]),
                "path": token_dir / row["npy_file"],
            })

    rows.sort(key=lambda r: r["detector_score"] * 0.7 + r["whale_cv"] * 0.3, reverse=True)

    selected, seen = [], set()
    for r in rows:
        if r["detector_score"] < min_detector:
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
    """Generate tokens, stopping at SEP tokens.

    Args:
        seed: Optional random seed for reproducibility within a run (set to None for stochastic)
    """
    model.eval()

    # Create explicit CUDA generator with seed
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
    first_tokens_log = []
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
        if step < 10:
            first_tokens_log.append(token_val)

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

    return input_ids, first_tokens_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--token-dir", default="data/tokenized/sanctsound_humpback_dac")
    parser.add_argument("--scores-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=3)
    args = parser.parse_args()

    token_dir = Path(args.token_dir)
    scores_csv = Path(args.scores_csv) if args.scores_csv else token_dir / "chunk_scores.csv"
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "comparison_fixed"
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

    # Select prompts
    prompt_tokens = int(round(4.0 * INTERLEAVED_PER_SEC / N_CB)) * N_CB
    prompt_tokens = min(prompt_tokens, config.max_seq_len // 2)
    max_new = min(int(10 * INTERLEAVED_PER_SEC), config.max_seq_len - prompt_tokens)

    print(f"\nPrompt: {prompt_tokens} tokens ({prompt_tokens / INTERLEAVED_PER_SEC:.1f}s)")
    print(f"Max generation: {max_new} tokens (~{max_new / INTERLEAVED_PER_SEC:.1f}s)")

    print(f"\nSelecting top {args.n_prompts} prompts...")
    prompts = pick_best_prompts(token_dir, scores_csv, n=args.n_prompts)

    # Hyperparameters to test
    configs = [
        {"temp": 0.70, "top_k": 40, "name": "conservative", "seed": 42},
        {"temp": 0.85, "top_k": 80, "name": "balanced", "seed": 43},
        {"temp": 1.00, "top_k": 120, "name": "diverse", "seed": 44},
    ]

    results_summary = []

    print(f"\nGenerating → {out_dir}/\n")
    for i, seg in enumerate(prompts):
        npy_path = seg["path"]
        if not npy_path.exists():
            print(f"[{i}] SKIP {npy_path.name}")
            continue

        codes_2d = np.load(str(npy_path))
        tokens_1d = interleave_2d(codes_2d)

        if len(tokens_1d) < prompt_tokens + N_CB:
            print(f"[{i}] SKIP {npy_path.name} (too short)")
            continue

        prompt = tokens_1d[:prompt_tokens]
        stem = npy_path.stem

        print(f"[{i}] {npy_path.name}")
        print(f"     Quality: det={seg['detector_score']:.3f}, cv={seg['whale_cv']:.2f}")

        # Generate with each configuration
        for cfg in configs:
            print(f"     {cfg['name']:12s} (T={cfg['temp']:.2f}, top_k={cfg['top_k']:3d})...", end=" ", flush=True)

            prompt_t = torch.tensor(prompt, dtype=torch.long, device=args.device).unsqueeze(0)

            # NOTE: Using torch.Generator with explicit seed to properly control stochasticity
            with torch.no_grad():
                generated, first_tokens = generate_with_sep_stopping(
                    model, prompt_t, max_new,
                    temperature=cfg["temp"],
                    top_k=cfg["top_k"],
                    top_p=0.0,
                    device=args.device,
                    seed=cfg["seed"],  # Use different seed for each config
                )

            print(f"[first10: {first_tokens}]", end=" ", flush=True)
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
                "detector_score": seg["detector_score"],
                "temperature": cfg["temp"],
                "top_k": cfg["top_k"],
                "config_name": cfg["name"],
                "generated_duration": gen_dur,
                "total_duration": full_dur,
                "generated_tokens": len(gen_tokens),
                "seed": cfg["seed"],
            })

    # Save summary
    summary_path = out_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved to {summary_path.name}")

    print(f"\n✓ Generated {len(results_summary)} samples in {out_dir}/")


if __name__ == "__main__":
    main()
