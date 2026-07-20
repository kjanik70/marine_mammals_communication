#!/usr/bin/env python3
"""Simple audio property analysis of generated samples."""

import json
from pathlib import Path
import numpy as np
import soundfile as sf


def main():
    comp_dir = Path("runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/comparison")

    # Load summary
    with open(comp_dir / "generation_summary.json") as f:
        summary = json.load(f)

    print("\n" + "="*100)
    print("GENERATED AUDIO PROPERTIES")
    print("="*100 + "\n")

    for i, entry in enumerate(summary):
        filename = f"full_{entry['prompt_idx']:02d}_{entry['prompt_file']}_T{entry['temperature']:.2f}_{entry['top_k']}.wav"
        path = comp_dir / filename

        if not path.exists():
            print(f"[{i}] MISSING: {filename}")
            continue

        audio, sr = sf.read(str(path))
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Basic stats
        duration = len(audio) / sr
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))

        # Generate part (estimate: last 10 seconds worth)
        gen_samples = int(10.0 * sr)
        if len(audio) > gen_samples:
            gen_audio = audio[-gen_samples:]
            gen_rms = np.sqrt(np.mean(gen_audio ** 2))
            gen_peak = np.max(np.abs(gen_audio))
        else:
            gen_rms = rms
            gen_peak = peak

        # Count zero-crossings as proxy for activity
        zero_crossings = int(np.sum(np.abs(np.diff(np.sign(audio)))))

        config = entry['config_name']
        temp = entry['temperature']
        print(f"[{entry['prompt_idx']}] {config:<12s} (T={temp:.2f}, top_k={entry['top_k']:>3d}) | " +
              f"Duration: {duration:.2f}s | RMS: {rms:.4f} | Peak: {peak:.4f} | " +
              f"Gen RMS: {gen_rms:.4f} | Zero-crossings: {zero_crossings:>5d}")

    print("\n" + "="*100)
    print("OBSERVATIONS")
    print("="*100)
    print("""
If all samples have identical RMS and peak levels per prompt, the model may be generating
the same audio regardless of temperature/top_k settings. This could indicate:

  1. Model is in evaluation mode but not sampling stochastically (always taking argmax)
  2. Generated tokens are identical even with different sampling parameters
  3. Decoder is deterministic (unlikely for audio codecs)

Check: Are the audio files different byte-for-byte?
""")

    # Quick file uniqueness check
    print("FILE SIZE COMPARISON:")
    for i in range(3):
        samples_for_prompt = [s for s in summary if s['prompt_idx'] == i]
        print(f"\n  Prompt {i}: {samples_for_prompt[0]['prompt_file']}")
        for s in samples_for_prompt:
            filename = f"full_{s['prompt_idx']:02d}_{s['prompt_file']}_T{s['temperature']:.2f}_{s['top_k']}.wav"
            path = comp_dir / filename
            if path.exists():
                size = path.stat().st_size
                print(f"    {s['config_name']:<12s}: {size:>10,d} bytes")


if __name__ == "__main__":
    main()
