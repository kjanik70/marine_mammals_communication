#!/usr/bin/env python3
"""Generate audio samples from all trained models for comparison."""

import argparse
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import yaml

from src.model.config import get_config
from src.model.transformer import CausalTransformer
from src.tokenizer.audio_tokenizer import AudioTokenizer


def load_model(run_dir, device="cuda"):
    """Load best model from a training run."""
    run_dir = Path(run_dir)
    best_path = run_dir / "best_model.pt"
    if not best_path.exists():
        return None, None

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = CausalTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("val_loss", float("inf"))


def detect_n_codebooks(run_dir):
    """Detect number of codebooks from run directory name or config."""
    name = Path(run_dir).name
    if "4cb" in name:
        return 4
    # Check for saved config
    config_path = Path(run_dir) / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("data", {}).get("n_codebooks", 1)
    return 1


@torch.no_grad()
def generate_audio(model, tokenizer, n_samples=5, max_tokens=200,
                   temperature=0.9, top_k=50, device="cuda",
                   n_codebooks=1, sep_token=None):
    """Generate unconditional audio samples."""
    vocab_size = model.config.vocab_size
    # Pick a random start token from the first codebook range
    max_start = min(1025, vocab_size - 1)
    samples = []
    for i in range(n_samples):
        start_token = torch.randint(1, max_start + 1, (1, 1), device=device)
        generated = model.generate(
            start_token, max_new_tokens=max_tokens,
            temperature=temperature, top_k=top_k, eos_token_id=-1,
        )
        tokens = generated[0].cpu().numpy()
        audio_np = tokenizer.decode_tokens_to_audio(
            tokens, n_codebooks=n_codebooks, sep_token=sep_token,
        )
        samples.append(audio_np)

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec-path", default="models/codec.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    # Find all training runs
    runs_dir = Path("runs")
    run_dirs = sorted(runs_dir.glob("audio_*"))

    print(f"Found {len(run_dirs)} audio training runs\n")

    # Cache tokenizers by n_codebooks to avoid reloading
    tokenizers = {}

    for run_dir in run_dirs:
        print(f"=== {run_dir.name} ===")
        model, val_loss = load_model(run_dir, args.device)
        if model is None:
            print("  No best_model.pt found, skipping")
            continue

        n_cb = detect_n_codebooks(run_dir)
        sep_token = n_cb * 1024 + 2 if n_cb > 1 else None

        if n_cb not in tokenizers:
            tokenizers[n_cb] = AudioTokenizer(
                codec_path=args.codec_path, device=args.device, n_codebooks=n_cb
            )
        tokenizer = tokenizers[n_cb]

        n_params = model.count_parameters()
        print(f"  Params: {n_params:,}, Val_loss: {val_loss:.4f}, Codebooks: {n_cb}")

        # Generate
        gen_dir = run_dir / "generated"
        gen_dir.mkdir(exist_ok=True)

        samples = generate_audio(
            model, tokenizer,
            n_samples=args.n_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=args.device,
            n_codebooks=n_cb,
            sep_token=sep_token,
        )

        for i, audio in enumerate(samples):
            out_path = gen_dir / f"sample_{i}.wav"
            sf.write(str(out_path), audio, tokenizer.sample_rate)
            dur = len(audio) / tokenizer.sample_rate
            print(f"  Generated: {out_path.name} ({dur:.1f}s)")

        print()


if __name__ == "__main__":
    main()
