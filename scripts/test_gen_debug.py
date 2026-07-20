#!/usr/bin/env python3
"""Debug generation from 109000 checkpoint - just one sample, minimal output."""

import torch
import numpy as np
from pathlib import Path

from src.model.transformer import CausalTransformer
from src.tokenizer.dac_tokenizer import DACTokenizer

N_CB = 9
SEP_TOKEN = 9218
SEP_GAP_TOKEN = 9217
INTERLEAVED_PER_SEC = 86.133 * N_CB

def interleave_2d(codes_2d: np.ndarray) -> np.ndarray:
    """Convert (9, T) codes array to 1D interleaved sequence."""
    n_cb, T = codes_2d.shape
    offsets = np.arange(n_cb).reshape(n_cb, 1) * 1024
    return (codes_2d + offsets).T.reshape(-1).astype(np.int32)

# Load model
ckpt_path = Path("runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/best_model_step109000.pt")
print(f"Loading model from {ckpt_path}...")

ckpt = torch.load(str(ckpt_path), map_location="cuda", weights_only=False)
print(f"Checkpoint keys: {ckpt.keys()}")
print(f"Checkpoint type: {type(ckpt)}")

# Try to load model
try:
    if "config" in ckpt:
        config = ckpt["config"]
        model = CausalTransformer(config).to("cuda")
        model.load_state_dict(ckpt["model_state_dict"])
        print("Loaded with config from checkpoint")
    else:
        print("ERROR: No 'config' key in checkpoint!")
        print(f"Available keys: {list(ckpt.keys())}")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()

# Check GPU memory
print(f"\nGPU Memory after model load:")
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

model.eval()

# Load one small sample
token_dir = Path("data/tokenized/sanctsound_humpback_dac")
sample_file = list(token_dir.glob("*.npy"))[0]
print(f"\nLoading sample: {sample_file.name}")
codes_2d = np.load(str(sample_file))
tokens_1d = interleave_2d(codes_2d)
print(f"  Total tokens: {len(tokens_1d)} ({len(tokens_1d) / INTERLEAVED_PER_SEC:.1f}s)")

# Use very short prompt
prompt_tokens = 512  # Just 512 tokens (~0.66 seconds)
prompt = tokens_1d[:prompt_tokens]
max_new = 256  # Generate only 256 tokens

print(f"\nPrompt: {len(prompt)} tokens")
print(f"Max generation: {max_new} tokens")

# Try generation
print(f"\nGPU Memory before generation:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

prompt_t = torch.tensor(prompt, dtype=torch.long, device="cuda").unsqueeze(0)

try:
    with torch.no_grad():
        print("Starting generation...")
        generated = model.generate(
            prompt_t,
            max_new_tokens=max_new,
            temperature=0.85,
            top_k=80,
            eos_token_id=-1,
        )
    print(f"Generated {generated.shape[1] - len(prompt)} tokens")

    print(f"\nGPU Memory after generation:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    print("SUCCESS!")
except Exception as e:
    print(f"ERROR during generation: {e}")
    import traceback
    traceback.print_exc()
    print(f"\nGPU Memory after error:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

