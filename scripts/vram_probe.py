#!/usr/bin/env python3
"""Probe max model size × context length on this GPU.

Runs a forward + backward pass with dummy data at each combo.
Reports peak VRAM usage and whether it fits.
"""

import gc
import sys

import torch

from src.model.config import PRESETS, get_config
from src.model.transformer import CausalTransformer


def probe(preset_name, seq_len, batch_size, vocab_size=4099, grad_accum=1):
    """Try a forward+backward pass. Returns (success, peak_mb)."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        cfg = get_config(preset_name, vocab_size=vocab_size, max_seq_len=seq_len)
        model = CausalTransformer(cfg).cuda()
        model.train()

        # Count params
        n_params = sum(p.numel() for p in model.parameters())

        # Dummy data
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device="cuda")
        targets = torch.randint(1, vocab_size, (batch_size, seq_len), device="cuda")
        mask = torch.ones(batch_size, seq_len, device="cuda")

        # Optimizer (matching real training — AdamW)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Forward + backward with mixed precision (matching real training)
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids, attention_mask=mask, targets=targets)
            loss = output["loss"] / grad_accum

        loss.backward()
        optimizer.step()

        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Cleanup
        del model, input_ids, targets, mask, output, loss
        gc.collect()
        torch.cuda.empty_cache()

        return True, peak_mb, n_params

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return False, -1, -1
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        return False, -1, -1


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    print(f"Total VRAM: {total_mb:.0f} MB")
    print()

    presets = ["small", "medium", "large", "xlarge"]
    seq_lens = [1024, 2048, 4096, 8192]
    batch_sizes = [1, 2, 4, 8]

    print(f"{'Preset':<8} {'Params':>8} {'SeqLen':>6} {'Batch':>5} {'GradAcc':>7} {'EffBS':>5} {'VRAM MB':>8} {'Status':>8}")
    print("-" * 70)

    for preset in presets:
        for seq_len in seq_lens:
            # Find max batch size that fits
            best_bs = 0
            best_vram = 0
            best_params = 0

            for bs in batch_sizes:
                ok, vram, params = probe(preset, seq_len, bs)
                if ok:
                    best_bs = bs
                    best_vram = vram
                    best_params = params
                else:
                    break

            if best_bs == 0:
                # Try batch_size=1
                ok, vram, params = probe(preset, seq_len, 1)
                if ok:
                    # Determine grad_accum to reach effective BS=8
                    grad_accum = 8
                    print(f"{preset:<8} {params/1e6:>7.1f}M {seq_len:>6} {1:>5} {grad_accum:>7} {grad_accum:>5} {vram:>7.0f} {'OK':>8}")
                else:
                    print(f"{preset:<8} {'?':>8} {seq_len:>6} {'-':>5} {'-':>7} {'-':>5} {'OOM':>8} {'FAIL':>8}")
            else:
                # Calculate grad_accum to reach effective BS=8
                grad_accum = max(1, 8 // best_bs)
                eff_bs = best_bs * grad_accum
                print(f"{preset:<8} {best_params/1e6:>7.1f}M {seq_len:>6} {best_bs:>5} {grad_accum:>7} {eff_bs:>5} {best_vram:>7.0f} {'OK':>8}")

        print()


if __name__ == "__main__":
    main()
