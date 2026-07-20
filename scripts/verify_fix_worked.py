#!/usr/bin/env python3
"""Verify that the seeding fix produced different audio files."""

import json
from pathlib import Path
import numpy as np
import soundfile as sf


def analyze_directory(comp_dir: Path, label: str) -> dict:
    """Analyze a comparison directory."""
    summary_path = comp_dir / "generation_summary.json"

    if not summary_path.exists():
        print(f"  {label}: {summary_path.name} not found")
        return {}

    with open(summary_path) as f:
        summary = json.load(f)

    print(f"\n{label}:")
    print(f"  {'Prompt':<5} {'Config':<12} {'File Size':>12} {'RMS':>8} {'Peak':>8}")
    print(f"  {'-'*60}")

    stats = {}
    for entry in summary:
        filename = f"full_{entry['prompt_idx']:02d}_{entry['prompt_file']}_T{entry['temperature']:.2f}_{entry['top_k']}.wav"
        path = comp_dir / filename

        if not path.exists():
            continue

        audio, sr = sf.read(str(path))
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        file_size = path.stat().st_size
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))

        config = entry['config_name']
        prompt_idx = entry['prompt_idx']

        if prompt_idx not in stats:
            stats[prompt_idx] = {}
        stats[prompt_idx][config] = {
            "size": file_size,
            "rms": rms,
            "peak": peak,
        }

        print(f"  [{prompt_idx}] {config:<12} {file_size:>12,d} {rms:>8.4f} {peak:>8.4f}")

    return stats


def main():
    original_dir = Path("runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/comparison")
    fixed_dir = Path("runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/comparison_fixed")

    print("=" * 80)
    print("VERIFICATION: Fixed Generation vs Original")
    print("=" * 80)

    original_stats = analyze_directory(original_dir, "Original (Broken)")
    fixed_stats = analyze_directory(fixed_dir, "Fixed")

    if not fixed_stats:
        print("\n⚠️  Fixed samples not yet generated")
        return

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Check if fixed samples are different within each prompt
    for prompt_idx in sorted(fixed_stats.keys()):
        fixed_config_stats = fixed_stats[prompt_idx]
        configs = list(fixed_config_stats.keys())

        if len(configs) < 2:
            continue

        print(f"\nPrompt {prompt_idx}:")
        all_same = True
        for i, config1 in enumerate(configs):
            for config2 in configs[i + 1 :]:
                size1 = fixed_config_stats[config1]["size"]
                size2 = fixed_config_stats[config2]["size"]
                rms1 = fixed_config_stats[config1]["rms"]
                rms2 = fixed_config_stats[config2]["rms"]

                size_diff = size1 != size2
                rms_diff = abs(rms1 - rms2) > 1e-5

                status = "✓ Different" if (size_diff or rms_diff) else "⚠️  Identical"
                all_same = all_same and not (size_diff or rms_diff)

                print(f"  {config1} vs {config2}: {status}")
                if size_diff or rms_diff:
                    print(f"      Size: {size1} vs {size2} | RMS: {rms1:.4f} vs {rms2:.4f}")

        if all_same:
            print(f"  ⚠️  ALL CONFIGS IDENTICAL FOR PROMPT {prompt_idx} — Fix didn't work!")
        else:
            print(f"  ✓ Configs are different — Fix likely worked!")

    # Compare fixed vs original
    print("\n" + "=" * 80)
    print("FIXED vs ORIGINAL")
    print("=" * 80)

    if original_stats and fixed_stats:
        for prompt_idx in original_stats:
            if prompt_idx not in fixed_stats:
                continue
            orig = original_stats[prompt_idx]
            fixed = fixed_stats[prompt_idx]

            print(f"\nPrompt {prompt_idx}:")
            for config in orig:
                if config in fixed:
                    orig_size = orig[config]["size"]
                    fixed_size = fixed[config]["size"]
                    size_same = orig_size == fixed_size

                    print(
                        f"  {config}: "
                        f"Original {orig_size:,} → Fixed {fixed_size:,} "
                        f"({'same' if size_same else 'different'})"
                    )


if __name__ == "__main__":
    main()
