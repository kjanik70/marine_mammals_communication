#!/usr/bin/env python3
"""Generate with token logging to see what's actually being sampled."""

import torch
import numpy as np
from pathlib import Path
import csv
import json

from src.model.transformer import CausalTransformer
from src.tokenizer.dac_tokenizer import DACTokenizer

N_CB = 9
SEP_TOKEN = 9218
SEP_GAP_TOKEN = 9217
INTERLEAVED_PER_SEC = 86.133 * N_CB

def interleave_2d(codes_2d: np.ndarray) -> np.ndarray:
    n_cb, T = codes_2d.shape
    offsets = np.arange(n_cb).reshape(n_cb, 1) * 1024
    return (codes_2d + offsets).T.reshape(-1).astype(np.int32)

def pick_best_prompts(token_dir: Path, scores_csv: Path, n: int = 3, min_detector: float = 0.8) -> list[dict]:
    rows = []
    with open(scores_csv) as f:
        for row in csv.DictReader(f):
            det = float(row["detector_score"]) if row["detector_score"] else 0.0
            rows.append({
                "npy_file": row["npy_file"],
                "detector_score": det,
                "whale_cv": float(row["whale_cv"]),
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
    return selected[:n]

def generate_with_sep_stopping_logging(
    model: CausalTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.85,
    top_k: int = 80,
    seed: int = None,
    device: str = "cuda",
) -> tuple:
    """Generate tokens with logging."""
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

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
                next_token = torch.multinomial(probs, num_samples=1)

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

    return input_ids, [g.item() for g in generated]

# Main
ckpt_path = Path("runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/best_model_step109000.pt")
token_dir = Path("data/tokenized/sanctsound_humpback_dac")
scores_csv = token_dir / "chunk_scores.csv"
out_dir = Path("runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/comparison_token_logged")
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading model...")
ckpt = torch.load(str(ckpt_path), map_location="cuda", weights_only=False)
config = ckpt["config"]
model = CausalTransformer(config).to("cuda")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Loading tokenizer...")
tokenizer = DACTokenizer(device="cpu", n_codebooks=N_CB)

# Setup
prompt_tokens = int(round(4.0 * INTERLEAVED_PER_SEC / N_CB)) * N_CB
max_new = min(int(10 * INTERLEAVED_PER_SEC), config.max_seq_len - prompt_tokens)

print(f"Prompt: {prompt_tokens} tokens")
print(f"Max generation: {max_new} tokens\n")

prompts = pick_best_prompts(token_dir, scores_csv, n=1)

configs = [
    {"temp": 0.70, "top_k": 40, "name": "conservative", "seed": 42},
    {"temp": 0.85, "top_k": 80, "name": "balanced", "seed": 43},
    {"temp": 1.00, "top_k": 120, "name": "diverse", "seed": 44},
]

token_log = {}

for i, seg in enumerate(prompts):
    npy_path = seg["path"]
    codes_2d = np.load(str(npy_path))
    tokens_1d = interleave_2d(codes_2d)
    prompt = tokens_1d[:prompt_tokens]
    stem = npy_path.stem

    print(f"[{i}] {npy_path.name}\n")

    for cfg in configs:
        print(f"  {cfg['name']:12s} (seed={cfg['seed']})...", end=" ", flush=True)

        prompt_t = torch.tensor(prompt, dtype=torch.long, device="cuda").unsqueeze(0)
        with torch.no_grad():
            generated, gen_tokens = generate_with_sep_stopping_logging(
                model, prompt_t, max_new,
                temperature=cfg["temp"],
                top_k=cfg["top_k"],
                seed=cfg["seed"],
                device="cuda",
            )

        print(f"generated {len(gen_tokens)} tokens")
        token_log[f"{cfg['name']}_{cfg['seed']}"] = {
            "temperature": cfg["temp"],
            "top_k": cfg["top_k"],
            "seed": cfg["seed"],
            "first_20_tokens": gen_tokens[:20],
            "total_tokens": len(gen_tokens),
            "all_tokens": gen_tokens,
        }

# Save token log
log_path = out_dir / "token_log.json"
with open(log_path, "w") as f:
    # Convert to serializable format
    serializable = {}
    for key, val in token_log.items():
        serializable[key] = {
            "temperature": val["temperature"],
            "top_k": val["top_k"],
            "seed": val["seed"],
            "first_20_tokens": val["first_20_tokens"],
            "total_tokens": val["total_tokens"],
        }
    json.dump(serializable, f, indent=2)

print(f"\nToken log saved to {log_path}")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

tokens_by_seed = {val["seed"]: val["all_tokens"] for val in token_log.values()}
seeds = sorted(tokens_by_seed.keys())

for i, seed1 in enumerate(seeds):
    for seed2 in seeds[i+1:]:
        tokens1 = tokens_by_seed[seed1]
        tokens2 = tokens_by_seed[seed2]
        diffs = sum(1 for a, b in zip(tokens1, tokens2) if a != b)
        print(f"Seed {seed1} vs {seed2}: {diffs} differences in {len(tokens1)} tokens")
