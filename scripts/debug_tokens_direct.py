#!/usr/bin/env python3
"""Debug script - writes directly to file."""

import torch
import numpy as np
from pathlib import Path
import csv
import sys

from src.model.transformer import CausalTransformer
from src.tokenizer.dac_tokenizer import DACTokenizer

N_CB = 9
SEP_TOKEN = 9218
INTERLEAVED_PER_SEC = 86.133 * 9

def interleave_2d(codes_2d: np.ndarray) -> np.ndarray:
    n_cb, T = codes_2d.shape
    offsets = np.arange(n_cb).reshape(n_cb, 1) * 1024
    return (codes_2d + offsets).T.reshape(-1).astype(np.int32)

def gen_with_seed(model, prompt, max_new, temp, top_k, device, seed):
    """Generate with specific seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.eval()
    prompt = prompt[:, :model.config.max_seq_len]

    with torch.no_grad():
        result = model.forward(prompt)
        past_kv = result.get("past_kv", None)
        logits = result["logits"][:, -1, :]

    generated = []
    for step in range(max_new):
        with torch.no_grad():
            if temp == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = logits / temp
                if top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[:, [-1]]] = -float("inf")
                probs = scaled.softmax(dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

        generated.append(next_token.item())
        if next_token.item() == SEP_TOKEN:
            break

        with torch.no_grad():
            result = model.forward(next_token, past_kv=past_kv)
            past_kv = result.get("past_kv", None)
            logits = result["logits"][:, -1, :]

    return generated

out = open("/tmp/debug_output.txt", "w")

out.write("Loading model...\n")
out.flush()

ckpt = torch.load("runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/best_model_step109000.pt", map_location="cuda", weights_only=False)
config = ckpt["config"]
model = CausalTransformer(config).to("cuda")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

token_dir = Path("data/tokenized/sanctsound_humpback_dac")
scores_csv = token_dir / "chunk_scores.csv"

out.write("Loading sample...\n")
out.flush()

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
max_new = 1000

out.write(f"Prompt length: {len(prompt)}\n")
out.write(f"Max generation: {max_new}\n\n")
out.flush()

prompt_t = torch.tensor(prompt, dtype=torch.long, device="cuda").unsqueeze(0)

# Generate 3 times with different seeds
configs = [
    {"temp": 0.70, "top_k": 40, "name": "conservative", "seed": 42},
    {"temp": 0.85, "top_k": 80, "name": "balanced", "seed": 43},
    {"temp": 1.00, "top_k": 120, "name": "diverse", "seed": 44},
]

results = {}
for cfg in configs:
    out.write(f"Generating {cfg['name']} (T={cfg['temp']}, seed={cfg['seed']})...\n")
    out.flush()

    tokens = gen_with_seed(model, prompt_t, max_new, cfg["temp"], cfg["top_k"], "cuda", cfg["seed"])
    results[cfg["name"]] = tokens

    out.write(f"  Generated {len(tokens)} tokens\n")
    out.write(f"  First 30: {tokens[:30]}\n\n")
    out.flush()

out.write("=" * 80 + "\n")
out.write("COMPARISON\n")
out.write("=" * 80 + "\n\n")

# Check if all are identical
names = list(results.keys())
seqs = list(results.values())

if all(seq == seqs[0] for seq in seqs):
    out.write("⚠️  ALL SEQUENCES ARE IDENTICAL!\n")
    out.write(f"Length: {len(seqs[0])}\n")
else:
    out.write("✓ Sequences are different\n")
    for i, name in enumerate(names):
        for j in range(i+1, len(names)):
            other_name = names[j]
            diffs = sum(1 for a, b in zip(seqs[i], seqs[j]) if a != b)
            total = min(len(seqs[i]), len(seqs[j]))
            pct = 100.0 * diffs / total if total > 0 else 0
            out.write(f"  {name} vs {other_name}: {diffs}/{total} diffs ({pct:.1f}%)\n")

out.close()
print("Output written to /tmp/debug_output.txt")
