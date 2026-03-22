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


@torch.no_grad()
def generate_audio(model, tokenizer, n_samples=5, max_tokens=200,
                   temperature=0.9, top_k=50, device="cuda"):
    """Generate unconditional audio samples."""
    samples = []
    for i in range(n_samples):
        # Start with a random token
        start_token = torch.randint(1, 1025, (1, 1), device=device)
        generated = model.generate(
            start_token, max_new_tokens=max_tokens,
            temperature=temperature, top_k=top_k, eos_token_id=-1,
        )
        tokens = generated[0].cpu().numpy()

        # Decode to audio
        codes = tokens - 1  # remove PAD offset
        codes = np.clip(codes, 0, 1023)
        codes_tensor = torch.tensor(codes, dtype=torch.long, device=device)
        codes_tensor = codes_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, T)

        # Pad to 14 codebooks
        n_codebooks = 14
        full_codes = torch.zeros(1, n_codebooks, codes_tensor.shape[-1],
                                 dtype=torch.long, device=device)
        full_codes[:, 0, :] = codes_tensor[:, 0, :]

        z = tokenizer.codec.quantizer.from_codes(full_codes)[0]
        audio = tokenizer.codec.decode(z)["audio"]
        audio_np = audio.squeeze().cpu().numpy()
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

    tokenizer = AudioTokenizer(
        codec_path=args.codec_path, device=args.device, n_codebooks=1
    )

    # Find all training runs
    runs_dir = Path("runs")
    run_dirs = sorted(runs_dir.glob("audio_*"))

    print(f"Found {len(run_dirs)} audio training runs\n")

    for run_dir in run_dirs:
        print(f"=== {run_dir.name} ===")
        model, val_loss = load_model(run_dir, args.device)
        if model is None:
            print("  No best_model.pt found, skipping")
            continue

        n_params = model.count_parameters()
        print(f"  Params: {n_params:,}, Best val_loss: {val_loss:.4f}")

        # Generate
        gen_dir = run_dir / "generated"
        gen_dir.mkdir(exist_ok=True)

        samples = generate_audio(
            model, tokenizer,
            n_samples=args.n_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=args.device,
        )

        for i, audio in enumerate(samples):
            out_path = gen_dir / f"sample_{i}.wav"
            sf.write(str(out_path), audio, tokenizer.sample_rate)
            dur = len(audio) / tokenizer.sample_rate
            print(f"  Generated: {out_path.name} ({dur:.1f}s)")

        print()


if __name__ == "__main__":
    main()
