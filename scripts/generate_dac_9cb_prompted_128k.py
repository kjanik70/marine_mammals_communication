#!/usr/bin/env python3
"""Generate audio from 128K NSA+MoE model, stopping at SEP tokens.

Stops generation when encountering SEP tokens (9217/9218) to respect natural boundaries.

Usage:
    PYTHONPATH=. python3 scripts/generate_dac_9cb_prompted_128k.py \
      --checkpoint runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/best_model_step109000.pt
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from src.model.transformer import CausalTransformer
from src.model.config import get_config
from src.tokenizer.dac_tokenizer import DACTokenizer

N_CB = 9
SEP_TOKEN = 9218          # adjacent boundary
SEP_GAP_TOKEN = 9217      # gap boundary
TOKENS_PER_SEC = 86.133   # 44100 / 512 per codebook
INTERLEAVED_PER_SEC = TOKENS_PER_SEC * N_CB  # ~775


def interleave_2d(codes_2d: np.ndarray) -> np.ndarray:
    """Convert (9, T) codes array to 1D interleaved sequence with CB offsets."""
    n_cb, T = codes_2d.shape
    offsets = np.arange(n_cb).reshape(n_cb, 1) * 1024  # (9,1)
    return (codes_2d + offsets).T.reshape(-1).astype(np.int32)


def pick_best_prompts(
    token_dir: Path,
    scores_csv: Path,
    n: int = 5,
    min_detector: float = 0.8,
) -> list[dict]:
    """Return top N diverse high-quality chunks from chunk_scores.csv."""
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

    # Combined score: detector_score dominates, whale_cv breaks ties
    rows.sort(key=lambda r: r["detector_score"] * 0.7 + r["whale_cv"] * 0.3, reverse=True)

    # Keep top entries with diversity across station+deployment
    selected, seen = [], set()
    for r in rows:
        if r["detector_score"] < min_detector:
            continue
        # e.g. "sanctsound_hi04_02" from "sanctsound_hi04_02_000101.npy"
        prefix = "_".join(r["npy_file"].split("_")[:3])
        if prefix not in seen:
            selected.append(r)
            seen.add(prefix)
        if len(selected) >= n:
            break

    # Fall back to lower detector threshold if needed
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
) -> torch.Tensor:
    """Generate tokens, stopping when encountering SEP tokens (9217 or 9218)."""
    model.eval()

    # Prefill with prompt
    prompt = prompt[:, :model.config.max_seq_len]
    with torch.no_grad():
        result = model.forward(prompt)
        past_kv = result.get("past_kv", None)
        logits = result["logits"][:, -1, :]

    generated = []
    for step in range(max_new_tokens):
        # Sample next token
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

        # Stop if we hit a SEP token
        if token_val in (SEP_TOKEN, SEP_GAP_TOKEN):
            print(f"     → Stopped at step {step+1} (token {token_val})")
            break

        # Forward single token with KV cache
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
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--token-dir", default="data/tokenized/sanctsound_humpback_dac")
    parser.add_argument("--scores-csv", default=None,
                        help="chunk_scores.csv path (default: token-dir/chunk_scores.csv)")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: <checkpoint-dir>/prompted_128k)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--prompt-seconds", type=float, default=4.0,
                        help="Seconds of real audio to use as prompt")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--top-p", type=float, default=0.0,
                        help="Nucleus sampling p (0 = disabled)")
    args = parser.parse_args()

    token_dir = Path(args.token_dir)
    scores_csv = Path(args.scores_csv) if args.scores_csv else token_dir / "chunk_scores.csv"
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "prompted_128k"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    print(f"Loading model from {ckpt_path}...")
    ckpt = torch.load(str(ckpt_path), map_location=args.device, weights_only=False)

    # Handle both checkpoint formats: with/without config
    if "config" in ckpt:
        config = ckpt["config"]
        model = CausalTransformer(config).to(args.device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # Load model weights only - use config from model weights file
        config = ckpt  # assume checkpoint is the full dict
        model = CausalTransformer(config).to(args.device)
        model.load_state_dict(ckpt)

    model.eval()
    max_seq_len = model.config.max_seq_len
    print(f"  {sum(p.numel() for p in model.parameters()):,} params  "
          f"max_seq_len={max_seq_len}  "
          f"val_loss={ckpt.get('val_loss', '?'):.4f}  step={ckpt.get('step', '?')}")

    # ---- Load DAC tokenizer (CPU to avoid VRAM contention) ----
    print("Loading DAC tokenizer...")
    tokenizer = DACTokenizer(device="cpu", n_codebooks=N_CB)
    print(f"  DAC {tokenizer.sample_rate} Hz, {tokenizer.hop_length} hop, "
          f"~{tokenizer.tokens_per_second:.1f} tokens/sec per codebook")

    # ---- Select prompts ----
    prompt_tokens = int(round(args.prompt_seconds * INTERLEAVED_PER_SEC / N_CB)) * N_CB
    prompt_tokens = min(prompt_tokens, max_seq_len // 2)
    prompt_sec_actual = prompt_tokens / INTERLEAVED_PER_SEC

    # Cap generation to 10 seconds worth of tokens (reasonable for generation time)
    max_new = min(int(10 * INTERLEAVED_PER_SEC), max_seq_len - prompt_tokens)
    gen_sec_max = max_new / INTERLEAVED_PER_SEC

    print(f"\nPrompt: {prompt_tokens} tokens ({prompt_sec_actual:.1f}s)  "
          f"Max generation: {max_new} tokens (~{gen_sec_max:.1f}s max)  "
          f"(stops at SEP tokens 9217/9218)")

    print(f"\nSelecting top {args.n_samples} prompts from {scores_csv}...")
    prompts = pick_best_prompts(token_dir, scores_csv, n=args.n_samples)
    if not prompts:
        print("No suitable prompts found!")
        return

    for p in prompts:
        print(f"  {p['npy_file']:50s} det={p['detector_score']:.3f}  "
              f"cv={p['whale_cv']:.2f}  er={p['energy_ratio']:.2f}")

    # ---- Generate ----
    print(f"\nGenerating → {out_dir}/")
    for i, seg in enumerate(prompts):
        npy_path = seg["path"]
        if not npy_path.exists():
            print(f"[{i}] SKIP {npy_path.name} (file not found)")
            continue

        # Load and interleave 2D codes → 1D tokens
        codes_2d = np.load(str(npy_path))  # (9, T)
        tokens_1d = interleave_2d(codes_2d)

        if len(tokens_1d) < prompt_tokens + N_CB:
            print(f"[{i}] SKIP {npy_path.name} (too short: {len(tokens_1d)} tokens)")
            continue

        prompt = tokens_1d[:prompt_tokens]
        actual_new = min(max_new, max_seq_len - len(prompt))

        print(f"\n[{i}] {npy_path.name}")
        print(f"     chunk has {len(tokens_1d)} tokens ({len(tokens_1d)/INTERLEAVED_PER_SEC:.1f}s)")
        print(f"     prompt={len(prompt)} tokens, max generate {actual_new} new tokens...")

        # Generate on GPU, stopping at SEP tokens
        prompt_t = torch.tensor(prompt, dtype=torch.long, device=args.device).unsqueeze(0)
        with torch.no_grad():
            generated = generate_with_sep_stopping(
                model,
                prompt_t,
                max_new_tokens=actual_new,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else 80,
                top_p=args.top_p,
                device=args.device,
            )
        full_tokens = generated[0].cpu().numpy()
        gen_tokens = full_tokens[len(prompt):]

        print(f"     generated {len(gen_tokens)} tokens")

        # Decode prompt-only and full sequence to audio on CPU
        print("     decoding prompt audio...")
        prompt_audio = tokenizer.decode_tokens_to_audio(
            prompt, n_codebooks=N_CB, sep_token=SEP_TOKEN
        )

        print("     decoding full audio (prompt + continuation)...")
        full_audio = tokenizer.decode_tokens_to_audio(
            full_tokens, n_codebooks=N_CB, sep_token=SEP_TOKEN
        )

        sr = tokenizer.sample_rate
        prompt_dur = len(prompt_audio) / sr
        full_dur = len(full_audio) / sr
        gen_dur = full_dur - prompt_dur

        stem = npy_path.stem
        prompt_path = out_dir / f"prompt_{i:02d}_{stem}.wav"
        full_path = out_dir / f"full_{i:02d}_{stem}.wav"

        sf.write(str(prompt_path), prompt_audio, sr)
        sf.write(str(full_path), full_audio, sr)

        print(f"     prompt ({prompt_dur:.1f}s) → {prompt_path.name}")
        print(f"     full   ({full_dur:.1f}s = {prompt_dur:.1f}s prompt + {gen_dur:.1f}s gen) → {full_path.name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
