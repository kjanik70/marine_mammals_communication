#!/usr/bin/env python3
"""Generate using longer continuous chunks (by file size heuristic).

Uses file size as proxy for continuous audio - larger files = more content.
"""

import argparse
import csv
from pathlib import Path
import json
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

def pick_long_chunks(token_dir: Path, scores_csv: Path, n: int = 3, min_size_mb: float = 0.05) -> list[dict]:
    """Select longest chunks (by file size) for continuous audio."""
    rows = []
    with open(scores_csv) as f:
        for row in csv.DictReader(f):
            npy_file = row["npy_file"].strip()
            if not npy_file:
                continue

            path = token_dir / npy_file

            if not path.exists() or path.is_dir():
                continue

            try:
                # File size in MB
                file_size_mb = path.stat().st_size / (1024 * 1024)

                if file_size_mb < min_size_mb:
                    continue

                codes_2d = np.load(str(path))
            except Exception as e:
                continue
            tokens_1d = interleave_2d(codes_2d)

            det_score = float(row["detector_score"]) if row["detector_score"] else 0.0

            rows.append({
                "npy_file": npy_file,
                "detector_score": det_score,
                "path": path,
                "file_size_mb": file_size_mb,
                "tokens": len(tokens_1d),
            })

    # Sort by file size (longer = more continuous)
    rows.sort(key=lambda r: r["file_size_mb"], reverse=True)

    # Select top N with diversity
    selected, seen = [], set()
    for r in rows:
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
    device: str = "cuda",
    seed: int = None,
) -> torch.Tensor:
    """Generate tokens with explicit CUDA generator."""
    model.eval()

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
    args = parser.parse_args()

    token_dir = Path(args.token_dir)
    scores_csv = Path(args.scores_csv) if args.scores_csv else token_dir / "chunk_scores.csv"
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "comparison_long_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model...")
    ckpt = torch.load(str(ckpt_path), map_location=args.device, weights_only=False)
    config = ckpt["config"]
    model = CausalTransformer(config).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} params")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = DACTokenizer(device="cpu", n_codebooks=N_CB)

    prompt_tokens = int(round(4.0 * INTERLEAVED_PER_SEC / N_CB)) * N_CB
    prompt_tokens = min(prompt_tokens, config.max_seq_len // 2)
    max_new = min(int(10 * INTERLEAVED_PER_SEC), config.max_seq_len - prompt_tokens)

    print(f"Prompt: {prompt_tokens} tokens ({prompt_tokens / INTERLEAVED_PER_SEC:.1f}s)")
    print(f"Max generation: {max_new} tokens (~{max_new / INTERLEAVED_PER_SEC:.1f}s)\n")

    print(f"Selecting longest chunks (continuous audio)...")
    prompts = pick_long_chunks(token_dir, scores_csv, n=args.n_prompts)

    if not prompts:
        print("No suitable prompts found!")
        return

    for p in prompts:
        print(f"  {p['npy_file']}: {p['file_size_mb']:.2f}MB ({p['tokens']:,} tokens)")

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

        # Generate with each configuration
        for cfg in configs:
            print(f"     {cfg['name']:12s} (T={cfg['temp']:.2f}, top_k={cfg['top_k']:3d})...", end=" ", flush=True)

            prompt_t = torch.tensor(prompt, dtype=torch.long, device=args.device).unsqueeze(0)
            with torch.no_grad():
                generated = generate_with_sep_stopping(
                    model, prompt_t, max_new,
                    temperature=cfg["temp"],
                    top_k=cfg["top_k"],
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
                "file_size_mb": seg["file_size_mb"],
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
