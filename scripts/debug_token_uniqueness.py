#!/usr/bin/env python3
"""Debug script to check if different temperature settings produce different tokens."""

import torch
import numpy as np
from pathlib import Path
import csv

from src.model.transformer import CausalTransformer
from src.tokenizer.dac_tokenizer import DACTokenizer

N_CB = 9
SEP_TOKEN = 9218
SEP_GAP_TOKEN = 9217
TOKENS_PER_SEC = 86.133
INTERLEAVED_PER_SEC = TOKENS_PER_SEC * N_CB


def interleave_2d(codes_2d: np.ndarray) -> np.ndarray:
    """Convert (9, T) codes array to 1D interleaved sequence."""
    n_cb, T = codes_2d.shape
    offsets = np.arange(n_cb).reshape(n_cb, 1) * 1024
    return (codes_2d + offsets).T.reshape(-1).astype(np.int32)


def generate_with_sep_stopping_verbose(
    model: CausalTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.85,
    top_k: int = 80,
    top_p: float = 0.0,
    device: str = "cuda",
) -> tuple:
    """Generate tokens, stopping at SEP tokens. Return tokens and sequence of choices."""
    model.eval()

    prompt = prompt[:, :model.config.max_seq_len]
    with torch.no_grad():
        result = model.forward(prompt)
        past_kv = result.get("past_kv", None)
        logits = result["logits"][:, -1, :]

    generated = []
    choice_log = []

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
                next_token = torch.multinomial(probs, num_samples=1)

        token_val = next_token.item()
        generated.append(next_token)
        choice_log.append(token_val)

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

    return input_ids, choice_log


def main():
    ckpt_path = Path("runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/best_model_step109000.pt")
    token_dir = Path("data/tokenized/sanctsound_humpback_dac")
    scores_csv = token_dir / "chunk_scores.csv"

    print("Loading model...")
    ckpt = torch.load(str(ckpt_path), map_location="cuda", weights_only=False)
    config = ckpt["config"]
    model = CausalTransformer(config).to("cuda")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Loading tokenizer...")
    tokenizer = DACTokenizer(device="cpu", n_codebooks=N_CB)

    # Load one sample
    print("Loading first sample...")
    with open(scores_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            npy_file = row["npy_file"]
            path = token_dir / npy_file
            if path.exists():
                codes_2d = np.load(str(path))
                tokens_1d = interleave_2d(codes_2d)
                break

    prompt_tokens = int(round(4.0 * INTERLEAVED_PER_SEC / N_CB)) * N_CB
    prompt_tokens = min(prompt_tokens, config.max_seq_len // 2)
    max_new = min(int(10 * INTERLEAVED_PER_SEC), config.max_seq_len - prompt_tokens)

    if len(tokens_1d) < prompt_tokens + N_CB:
        print(f"Sample too short: {len(tokens_1d)} tokens")
        return

    prompt = tokens_1d[:prompt_tokens]
    print(f"Prompt: {len(prompt)} tokens")
    print(f"Max generation: {max_new} tokens\n")

    # Test 3 configs
    configs = [
        {"temp": 0.70, "top_k": 40, "name": "conservative"},
        {"temp": 0.85, "top_k": 80, "name": "balanced"},
        {"temp": 1.00, "top_k": 120, "name": "diverse"},
    ]

    all_sequences = {}
    for cfg in configs:
        print(f"Generating {cfg['name']} (T={cfg['temp']}, top_k={cfg['top_k']})...", end=" ", flush=True)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        prompt_t = torch.tensor(prompt, dtype=torch.long, device="cuda").unsqueeze(0)
        with torch.no_grad():
            generated, choices = generate_with_sep_stopping_verbose(
                model, prompt_t, max_new,
                temperature=cfg["temp"],
                top_k=cfg["top_k"],
                top_p=0.0,
                device="cuda",
            )

        generated_tokens = generated[0][len(prompt):].cpu().numpy()
        all_sequences[cfg["name"]] = choices
        print(f"generated {len(choices)} tokens")

    # Compare sequences
    print("\n" + "="*80)
    print("TOKEN SEQUENCE COMPARISON")
    print("="*80)

    seqs = list(all_sequences.values())
    names = list(all_sequences.keys())

    # Check if all are identical
    all_identical = all(seq == seqs[0] for seq in seqs)
    if all_identical:
        print("\n⚠️  ALL SEQUENCES ARE IDENTICAL!")
        print(f"   Length: {len(seqs[0])} tokens")
        print(f"   First 20 tokens: {seqs[0][:20]}")
        print("\n   This indicates a randomness/seeding issue. The multinomial sampling")
        print("   is producing the same results regardless of temperature/top_k.")
    else:
        print("\n✓ Sequences are different across temperature settings")

        for i, name in enumerate(names):
            other_names = [n for j, n in enumerate(names) if i != j]
            for other_name in other_names:
                diffs = sum(1 for a, b in zip(seqs[i], seqs[j]) if a != b)
                total = min(len(seqs[i]), len(seqs[j]))
                pct = 100.0 * diffs / total if total > 0 else 0
                print(f"   {name} vs {other_name}: {diffs}/{total} differences ({pct:.1f}%)")

    # Show first few tokens of each
    print("\n" + "="*80)
    print("FIRST 30 GENERATED TOKENS")
    print("="*80)
    for name in names:
        seq = all_sequences[name][:30]
        print(f"  {name:<12s}: {seq}")


if __name__ == "__main__":
    main()
