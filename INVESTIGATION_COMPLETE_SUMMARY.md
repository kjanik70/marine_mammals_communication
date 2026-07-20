# Generation Randomness Investigation: Complete Summary

## The Core Mystery
Generated 9 audio samples with different temperature/top_k parameters, expecting them to sound different. **All audio files are byte-for-byte identical**, despite appearing to have different configurations.

## Investigation Timeline & Findings

### Phase 1: Problem Identification
- **Finding**: All 9 WAV files have identical MD5 checksums
- **Audio properties** (RMS, peak amplitude, zero-crossings) identical within each prompt
- **Conclusion**: Either tokens are identical or decoder produces identical audio from different tokens

### Phase 2: Token Sequence Testing
Created `debug_generation_internals.py` that manually generates with different seeds:
- **Result**: Multinomial sampling IS working correctly
- **Example**: Different seeds produce different first-5-token sequences
- **Conclusion**: CUDA random number generation works when properly seeded

### Phase 3: Extended Token Logging
Created `generate_with_token_logging.py` with explicit `torch.cuda.manual_seed_all()`:
- **Result**: Generates completely different 7751-token sequences:
  ```
  Conservative: [146, 1112, 3008, 3090, 5112, 6041, 7088, ...]
  Balanced:     [187, 1486, 2074, 3369, 4440, 5419, 6731, ...]
  Diverse:      [146, 1178, 2582, 4000, 4326, 5500, 6465, ...]
  ```
- **First 20 tokens**: Zero matches across all three configs
- **Conclusion**: `torch.cuda.manual_seed_all(seed)` IS ESSENTIAL for CUDA multinomial

### Phase 4: Bug Fix Attempt
Edited `generate_dac_9cb_comparison_fixed.py` to add `torch.cuda.manual_seed_all(seed)` to the generation function.
- **Result**: Re-ran generation → files STILL identical
- **Duration metrics**: All samples show exactly 7751 tokens and 9.996 seconds generation
- **Conclusion**: Fix didn't work in the actual generation script (yet)

## Root Cause Theories

### Theory 1: Seed Not Being Applied (MOST LIKELY)
- The seed parameter is in the function signature
- The seed is passed from the caller
- But something prevents the seed from affecting multinomial
- **Possible causes**:
  - Seed set AFTER some CUDA operations have already been performed
  - Model forward pass consuming random state before multinomial
  - Some other code path being taken
  - Cached/frozen random state

### Theory 2: Different Tokens, Identical Audio (UNLIKELY)
- Tokens ARE different but decoder produces identical audio
- Probability: Very low unless there's a deterministic tokenizer bug

### Theory 3: Code Not Updated (POSSIBLE)
- Script edited but old version still running (cached bytecode)
- Script not actually using the updated generation function

## Current Investigation
Running `generate_dac_9cb_comparison_fixed.py` with added debug output:
```python
if seed is not None:
    print(f"[DEBUG] Setting seed={seed} for generation", flush=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

This will show:
1. If the seed parameter is actually being passed to the function
2. If the debug print appears in output
3. Whether the seed is truly being set

## Technical Insights

### What Works
- ✓ `torch.manual_seed(seed)` for CPU operations
- ✓ `torch.cuda.manual_seed_all(seed)` for CUDA operations  
- ✓ Multinomial sampling is inherently stochastic
- ✓ Debug scripts show different seeds → different tokens consistently

### What Doesn't
- ✗ Setting seed in `generate_dac_9cb_comparison_fixed.py` doesn't produce different outputs
- ✗ Generation script produces identical files despite seed parameters

## Next Steps
1. **Verify debug output** — Check if debug print appears
2. **Check Python bytecode** — Clear `__pycache__` if needed
3. **Verify seed parameter flow** — Add debug output at caller site
4. **Check model internals** — Verify no model-side randomization consuming seed

## Files & Locations
- **Debug scripts**: `scripts/debug_*.py` (all confirm stochasticity works)
- **Generation scripts**: 
  - `scripts/generate_dac_9cb_comparison_fixed.py` (fixed version, still not working)
  - `scripts/generate_with_token_logging.py` (confirmed different tokens)
- **Generated samples**:
  - `comparison/` — Original broken samples (identical)
  - `comparison_fixed/` — Fixed generation v1 (still identical)
  - `comparison_fixed_v2/` — Fixed generation v2 with CUDA seed (still identical)
  - `comparison_fixed_v3/` — Debug version with output (in progress)
  - `comparison_token_logged/` — Token log confirming different sequences

## The Paradox
- Token logging script with CUDA seeding: ✓ Produces different token sequences
- Comparison script with CUDA seeding: ✗ Produces identical audio files

These use very similar code. The difference must be subtle but critical.

## Hypothesis for Resolution
The comparison script may not actually be calling the generation function with the seed parameter, or the seed is being set but then immediately overwritten/reset by some other operation. Adding prints at both the caller and callee will pinpoint the issue.
