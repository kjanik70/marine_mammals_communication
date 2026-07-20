#!/usr/bin/env python3
"""Generate from 128K model with 45-second whale song prompts."""

import argparse
import json
from pathlib import Path
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

def generate_with_sep_stopping(
    model: CausalTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.85,
    top_k: int = 80,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate tokens with SEP token stopping."""
    model.eval()

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

    return input_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--token-dir", default="data/tokenized/sanctsound_humpback_dac")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompts", nargs="+", required=True)
    args = parser.parse_args()

    token_dir = Path(args.token_dir)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)
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

    # Use full prompt length (up to context window)
    prompt_tokens = config.max_seq_len // 2
    max_new = min(int(10 * INTERLEAVED_PER_SEC), config.max_seq_len - prompt_tokens)

    print(f"Max prompt: {prompt_tokens} tokens")
    print(f"Max generation: {max_new} tokens (~{max_new / INTERLEAVED_PER_SEC:.1f}s)\n")

    results_summary = []

    print(f"Generating → {out_dir}/\n")
    for i, prompt_file in enumerate(args.prompts):
        npy_path = token_dir / prompt_file
        if not npy_path.exists():
            print(f"[{i}] {prompt_file} - NOT FOUND")
            continue

        codes_2d = np.load(str(npy_path))
        tokens_1d = interleave_2d(codes_2d)
        prompt = tokens_1d[:prompt_tokens]
        stem = npy_path.stem

        print(f"[{i}] {npy_path.name}")
        print(f"     Prompt: {len(prompt):,} tokens (~{len(prompt) / INTERLEAVED_PER_SEC:.1f}s)")

        prompt_t = torch.tensor(prompt, dtype=torch.long, device=args.device).unsqueeze(0)
        with torch.no_grad():
            generated = generate_with_sep_stopping(
                model, prompt_t, max_new,
                temperature=0.85,
                top_k=80,
                device=args.device,
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
        full_path = out_dir / f"full_{i:02d}_{stem}_128k.wav"
        sf.write(str(full_path), full_audio, sr)

        print(f"     Generation: {gen_dur:.1f}s (from {len(gen_tokens):,} tokens)")
        print(f"     Total: {full_dur:.1f}s → {full_path.name}")

        results_summary.append({
            "prompt_idx": i,
            "prompt_file": stem,
            "prompt_duration": prompt_dur,
            "generated_duration": gen_dur,
            "total_duration": full_dur,
            "prompt_tokens": len(prompt),
            "generated_tokens": len(gen_tokens),
        })

    # Save summary
    summary_path = out_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved to {summary_path.name}")
    print(f"✓ Generated {len(results_summary)} samples in {out_dir}/")


if __name__ == "__main__":
    main()
