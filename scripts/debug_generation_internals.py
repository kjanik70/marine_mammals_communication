#!/usr/bin/env python3
"""Debug what happens inside the generation function."""

import torch
import numpy as np
from pathlib import Path
import csv

from src.model.transformer import CausalTransformer
from src.tokenizer.dac_tokenizer import DACTokenizer

N_CB = 9
SEP_TOKEN = 9218
INTERLEAVED_PER_SEC = 86.133 * 9

def interleave_2d(codes_2d: np.ndarray) -> np.ndarray:
    n_cb, T = codes_2d.shape
    offsets = np.arange(n_cb).reshape(n_cb, 1) * 1024
    return (codes_2d + offsets).T.reshape(-1).astype(np.int32)

def generate_debug(
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    device: str = "cuda",
):
    """Generate with detailed debugging output."""
    print(f"\n  [DEBUG] Starting generation: T={temperature}, top_k={top_k}, seed={seed}")

    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model.eval()
    prompt = prompt[:, :model.config.max_seq_len]

    with torch.no_grad():
        result = model.forward(prompt)
        past_kv = result.get("past_kv", None)
        logits = result["logits"][:, -1, :]

    generated = []
    token_samples = []

    for step in range(min(5, max_new_tokens)):  # Only show first 5 steps
        with torch.no_grad():
            # Get current logits
            current_logits = logits.clone()

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
        token_samples.append(token_val)

        # Debug output
        top5_logits, top5_idx = torch.topk(current_logits[0], 5)
        print(f"    Step {step}: token={token_val}, top5_logits={top5_logits.tolist()[:3]}, "
              f"top5_idx={top5_idx.tolist()[:3]}")

        if token_val == SEP_TOKEN:
            break

        with torch.no_grad():
            result = model.forward(next_token, past_kv=past_kv)
            past_kv = result.get("past_kv", None)
            logits = result["logits"][:, -1, :]

    return token_samples

# Load model and tokenizer
print("Loading model...")
ckpt = torch.load("runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/best_model_step109000.pt",
                  map_location="cuda", weights_only=False)
config = ckpt["config"]
model = CausalTransformer(config).to("cuda")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

token_dir = Path("data/tokenized/sanctsound_humpback_dac")
scores_csv = token_dir / "chunk_scores.csv"

print("Loading sample...")
with open(scores_csv) as f:
    for row in csv.DictReader(f):
        npy_file = row["npy_file"]
        path = token_dir / npy_file
        if path.exists():
            codes_2d = np.load(str(path))
            tokens_1d = interleave_2d(codes_2d)
            break

prompt_tokens = int(round(4.0 * INTERLEAVED_PER_SEC / 9)) * 9
prompt = tokens_1d[:prompt_tokens]

prompt_t = torch.tensor(prompt, dtype=torch.long, device="cuda").unsqueeze(0)

print("\n" + "="*80)
print("GENERATION DEBUG")
print("="*80)

configs = [
    {"temp": 0.70, "top_k": 40, "seed": 42},
    {"temp": 0.85, "top_k": 80, "seed": 43},
    {"temp": 1.00, "top_k": 120, "seed": 44},
]

results = {}
for cfg in configs:
    tokens = generate_debug(model, prompt_t, 1000, cfg["temp"], cfg["top_k"], cfg["seed"], "cuda")
    results[cfg["seed"]] = tokens
    print(f"  → First 10 tokens: {tokens[:10]}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

seeds = list(results.keys())
seqs = list(results.values())

if all(seq == seqs[0] for seq in seqs):
    print("\n⚠️  ALL SEQUENCES STILL IDENTICAL!")
    print("\nPossible causes:")
    print("  1. The model is somehow caching logits")
    print("  2. The forward pass is not being called correctly")
    print("  3. There's something in the model that deterministically maps prompt → tokens")
    print("  4. The multinomial is not actually being called")
else:
    print("\n✓ Sequences are different")
    for i, seq in enumerate(seqs):
        diffs = sum(1 for a, b in zip(seqs[0], seq) if a != b) if i > 0 else 0
        print(f"  Config {i}: {diffs} differences from config 0")
