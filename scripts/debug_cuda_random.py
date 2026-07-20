#!/usr/bin/env python3
"""Debug CUDA random number generation."""

import torch
import numpy as np

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A")

# Test 1: CPU multinomial with different seeds
print("\n" + "="*80)
print("TEST 1: CPU Multinomial with Manual Seeds")
print("="*80)

for seed in [42, 43, 44]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    probs = torch.softmax(logits, dim=0)

    samples = []
    for _ in range(10):
        sample = torch.multinomial(probs, num_samples=1)
        samples.append(sample.item())

    print(f"Seed {seed}: {samples}")

# Test 2: CUDA multinomial with different seeds
if torch.cuda.is_available():
    print("\n" + "="*80)
    print("TEST 2: CUDA Multinomial with Manual Seeds")
    print("="*80)

    for seed in [42, 43, 44]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device="cuda")
        probs = torch.softmax(logits, dim=0)

        samples = []
        for _ in range(10):
            sample = torch.multinomial(probs, num_samples=1)
            samples.append(sample.item())

        print(f"Seed {seed}: {samples}")

# Test 3: Check if deterministic algorithms are enabled
print("\n" + "="*80)
print("TEST 3: Determinism Settings")
print("="*80)
print("torch.backends.cudnn.deterministic:", torch.backends.cudnn.deterministic)
print("torch.backends.cudnn.benchmark:", torch.backends.cudnn.benchmark)

# Test 4: Try with torch.Generator
if torch.cuda.is_available():
    print("\n" + "="*80)
    print("TEST 4: CUDA Multinomial with torch.Generator")
    print("="*80)

    for seed in [42, 43, 44]:
        generator = torch.Generator(device="cuda").manual_seed(seed)

        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device="cuda")
        probs = torch.softmax(logits, dim=0)

        samples = []
        for _ in range(10):
            sample = torch.multinomial(probs, num_samples=1, generator=generator)
            samples.append(sample.item())

        print(f"Seed {seed}: {samples}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
If TEST 2 produces identical results across seeds, then torch.manual_seed()
is not properly seeding CUDA operations. The solution is to use torch.Generator
with the generator= parameter in multinomial(), as shown in TEST 4.
""")
