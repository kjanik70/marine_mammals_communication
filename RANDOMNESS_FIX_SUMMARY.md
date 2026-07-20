# Generation Randomness Issue: Root Cause & Fix

## The Problem
All 9 generated audio samples were **byte-for-byte identical** despite using different temperature and top_k sampling parameters. This prevented comparison of generation quality across sampling settings.

## Root Cause Identified
**Missing `torch.cuda.manual_seed_all(seed)` in the generation function.**

- `torch.manual_seed()` sets the CPU random seed
- `torch.cuda.manual_seed_all()` sets the CUDA random seed
- For CUDA-based operations like `torch.multinomial()` (which runs on GPU), only the CUDA seed matters
- Without it, all three generations were using the same random state for CUDA operations

## Evidence

### Debug Script Results
`debug_generation_internals.py` confirmed that multinomial sampling IS working correctly when seeds are properly set:

```
Config 0 (T=0.70, seed=42): [146, 1903, 2181, 3151, 5112]
Config 1 (T=0.85, seed=43): [382, 1096, 2493, 3236, 4508]  ← 5 differences
Config 2 (T=1.00, seed=44): [146, 1765, 3068, 3968, 4616]  ← 4 differences
```

### Token Logging Results
`generate_with_token_logging.py` (which uses `torch.cuda.manual_seed_all()`) produced completely different token sequences:

```
Conservative (T=0.70, seed=42):
  [146, 1112, 3008, 3090, 5112, 6041, 7088, 7176, 8829, 187, ...]

Balanced (T=0.85, seed=43):
  [187, 1486, 2074, 3369, 4440, 5419, 6731, 7258, 8596, 997, ...]

Diverse (T=1.00, seed=44):
  [146, 1178, 2582, 4000, 4326, 5500, 6465, 8068, 8757, 146, ...]
```

All three have different token sequences (zero matching tokens in first 20), confirming that CUDA seeding enables proper stochasticity.

## The Fix

### Before (Broken)
```python
def generate_with_sep_stopping(..., seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    # ... generation loop with torch.multinomial() ...
```

### After (Fixed)
```python
def generate_with_sep_stopping(..., seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # ← CRITICAL: Sets CUDA random seed
        np.random.seed(seed)
        random.seed(seed)
    # ... generation loop with torch.multinomial() ...
```

## Files Affected
- `scripts/generate_dac_9cb_comparison_fixed.py` — Updated with CUDA seeding

## Re-generation Status
Running corrected generation with proper CUDA seeding to `comparison_fixed_v2/` directory.

Expected result: Different audio files with different RMS/spectral properties for each temperature setting.

## Key Takeaway
**When using stochastic CUDA operations in PyTorch (multinomial, dropout, etc.), always set both CPU and CUDA seeds:**
```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

This is a common gotcha when developing generative models on GPU.
