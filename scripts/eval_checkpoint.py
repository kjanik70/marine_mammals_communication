#!/usr/bin/env python3
"""Evaluate a model checkpoint on a specific token dataset.

Usage:
    PYTHONPATH=. python3 scripts/eval_checkpoint.py \
        runs/audio_small_all_combined_4cb/best_model.pt \
        --token-dir data/tokenized/all_4cb_ab \
        --split val

    # Run all cross-evaluations and save summary:
    PYTHONPATH=. python3 scripts/eval_checkpoint.py --run-all

Compares checkpoints on a fixed val set to determine whether adding
new data (e.g., SanctSound) helps or hurts on the original distribution.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import AudioTokenDataset
from src.model.config import get_config
from src.model.transformer import CausalTransformer


def evaluate_checkpoint(checkpoint_path, token_dir, split="val", device="cuda",
                        batch_size=8, max_seq_len=1024, vocab_size=4099,
                        sep_token=4098):
    """Evaluate a checkpoint on a dataset. Returns dict with results."""
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["config"]
    model = CausalTransformer(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load dataset with same concat/split logic as train.py
    ds = AudioTokenDataset(
        token_dir,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        concat=True,
        sep_token=sep_token,
        augment=False,
    )

    # Reproduce the same 80/20 split as train.py
    n_total = len(ds)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train

    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=g).tolist()

    if split == "val":
        eval_ds = torch.utils.data.Subset(ds, indices[n_train:])
        n_eval = n_val
    else:
        eval_ds = torch.utils.data.Subset(ds, indices[:n_train])
        n_eval = n_train

    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Evaluate
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.autocast(device, dtype=torch.bfloat16):
                output = model(input_ids, attention_mask=attention_mask, targets=targets)

            total_loss += output["loss"].item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = float(np.exp(avg_loss))

    return {
        "checkpoint": str(checkpoint_path),
        "token_dir": str(token_dir),
        "split": split,
        "val_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 1),
        "n_windows": n_eval,
        "checkpoint_step": ckpt.get("step", None),
        "checkpoint_val_loss": ckpt.get("val_loss", None),
    }


def run_all(device="cuda"):
    """Run all cross-evaluations and save results."""
    output_dir = Path("runs/eval_cross")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        "all_4cb_ab": "runs/audio_small_all_4cb_ab/best_model.pt",
        "denoised_4cb": "runs/audio_small_denoised_4cb/best_model.pt",
        "clean_combined_4cb": "runs/audio_small_clean_combined_4cb/best_model.pt",
    }

    datasets = {
        "all_4cb_ab": "data/tokenized/all_4cb_ab",
        "denoised_4cb": "data/tokenized/denoised_4cb",
        "sanctsound_4cb": "data/tokenized/sanctsound_4cb",
    }

    # Verify all paths exist
    for name, path in checkpoints.items():
        if not Path(path).exists():
            print(f"WARNING: checkpoint {name} not found at {path}, skipping")
    for name, path in datasets.items():
        if not Path(path).exists():
            print(f"WARNING: dataset {name} not found at {path}, skipping")

    results = []
    for ckpt_name, ckpt_path in checkpoints.items():
        if not Path(ckpt_path).exists():
            continue
        for ds_name, ds_path in datasets.items():
            if not Path(ds_path).exists():
                continue
            print(f"\n{'='*60}")
            print(f"Model: {ckpt_name} -> Dataset: {ds_name}")
            print(f"{'='*60}")

            result = evaluate_checkpoint(ckpt_path, ds_path, device=device)
            result["model_name"] = ckpt_name
            result["dataset_name"] = ds_name
            results.append(result)

            print(f"  Val loss: {result['val_loss']:.4f}  Perplexity: {result['perplexity']:.1f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"cross_eval_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n\n{'='*70}")
    print(f"CROSS-EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Dataset':<20} {'Val Loss':>10} {'Perplexity':>12}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['model_name']:<25} {r['dataset_name']:<20} {r['val_loss']:>10.4f} {r['perplexity']:>12.1f}")
    print(f"{'='*70}")
    print(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on a dataset")
    parser.add_argument("checkpoint", type=str, nargs="?", default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--token-dir", type=str, default=None,
                        help="Token directory to evaluate on")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Which split to evaluate (default: val)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=4099)
    parser.add_argument("--sep-token", type=int, default=4098)
    parser.add_argument("--run-all", action="store_true",
                        help="Run all cross-evaluations")
    args = parser.parse_args()

    if args.run_all:
        run_all(device=args.device)
        return

    if not args.checkpoint or not args.token_dir:
        parser.error("checkpoint and --token-dir are required (or use --run-all)")

    print(f"Loading checkpoint: {args.checkpoint}")
    result = evaluate_checkpoint(
        args.checkpoint, args.token_dir, split=args.split,
        device=args.device, batch_size=args.batch_size,
        max_seq_len=args.max_seq_len, vocab_size=args.vocab_size,
        sep_token=args.sep_token,
    )

    print(f"\n{'='*50}")
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Dataset:    {result['token_dir']} ({result['split']})")
    print(f"Val loss:   {result['val_loss']:.4f}")
    print(f"Perplexity: {result['perplexity']:.1f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
