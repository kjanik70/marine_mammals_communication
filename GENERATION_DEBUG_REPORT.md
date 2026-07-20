# Generation Quality Analysis — 128K NSA+MoE Model

**Date**: 2026-05-15  
**Status**: Generation issue identified and fix in progress

## Executive Summary

Generated 9 audio samples from the 128K NSA+MoE checkpoint (step 109,000) to evaluate the model's ability to generate diverse whale vocalizations across different sampling parameters. **Critical issue discovered**: All three temperature/top_k configurations generate **identical audio files**, indicating a randomness/seeding bug in the generation process.

## The Issue

### Symptoms
- **File sizes**: All 9 files are **exactly 1,234,988 bytes** (per prompt, identical across all 3 temperature settings)
- **Audio statistics**: RMS, peak amplitude, zero-crossing counts identical within each prompt
- **Token sequences**: Debug script confirmed all 7,751 generated tokens are **identical** across conservative/balanced/diverse settings

### Root Cause
The multinomial sampling during generation is not producing stochastic samples despite different temperature and top_k parameters. Likely causes:

1. **Most likely**: Random seed not properly initialized for CUDA multinomial sampling
   - `torch.multinomial()` may require explicit CUDA seed setup (`torch.cuda.manual_seed()`)
   - Random state might be locked or not advancing properly during generation loop

2. **Less likely**: Temperature/top_k parameters not properly affecting probability distributions
   - Code inspection shows correct implementation of temperature scaling and top_k filtering
   - Probability distributions should be different for each config

## Generated Samples

### Locations
- Original samples: `runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/comparison/`
- New fixed samples (in progress): `runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/comparison_fixed/`

### Specifications
- **Model**: 375M param NSA+MoE, 128K context, trained to step 109,000
- **Prompts**: 3 high-quality SanctSound humpback chunks (detector scores 0.829–0.992)
- **Prompt duration**: ~4 seconds (~3,100 tokens)
- **Generation length**: 10 seconds (~7,751 tokens, capped or stopped at SEP tokens)
- **Total duration per sample**: 14 seconds
- **Audio format**: 44.1 kHz mono WAV
- **File size per sample**: ~1.2 MB

### Sampling Configurations
| Config | Temperature | Top-K | Expected Behavior |
|--------|------------|-------|-------------------|
| Conservative | 0.70 | 40 | Repetitive, close to training distribution |
| Balanced | 0.85 | 80 | Moderate diversity |
| Diverse | 1.00 | 120 | High variation |

## Debug Findings

### Token Sequence Analysis
Script: `scripts/debug_token_uniqueness.py`

**Output Summary**:
```
⚠️  ALL SEQUENCES ARE IDENTICAL!
   Length: 7751 tokens
   First 20 tokens: [627, 1124, 2906, 3106, 4998, 5873, 6424, 7395, 
                     8254, 627, 1866, 2577, 3106, 4526, 5873, 6882, 
                     7379, 9155, 627, 1983]

   This indicates a randomness/seeding issue. The multinomial sampling
   is producing the same results regardless of temperature/top_k.
```

All three generations (conservative, balanced, diverse) produced identical first-30 tokens, confirming the issue is in the sampling loop itself, not in decoder or audio conversion.

## Solution Implemented

### Fixed Generation Script
File: `scripts/generate_dac_9cb_comparison_fixed.py`

**Key changes**:
1. Added `seed` parameter to `generate_with_sep_stopping()` function
2. Explicitly call `torch.manual_seed(seed)` and `np.random.seed(seed)` before each generation
3. Assign different seeds to each config (42 for conservative, 43 for balanced, 44 for diverse)
4. Output to `comparison_fixed/` directory to preserve original broken samples

**Mechanism**:
```python
def generate_with_sep_stopping(..., seed: int = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    # ... rest of generation ...
```

### Expected Result
With explicit seeds set before each generation, `torch.multinomial()` should:
- Consume different entropy streams for each generation
- Produce different probability distributions
- Sample different tokens for balanced/diverse vs conservative

## Remaining Tasks

1. **Wait for fixed generation to complete** (~10-15 minutes)
2. **Verify token sequences are now different** in fixed samples
3. **Compare audio quality** across three temperature settings:
   - Spectral properties (centroid, peak frequency, novelty)
   - Perceptual quality (natural sounding continuations vs artifacts)
   - Diversity (parameter variance should correlate with audio diversity)

4. **Evaluate generation quality**:
   - Do continuations sound like whale vocalizations?
   - Do SEP tokens trigger early stopping appropriately?
   - Does model maintain humpback call patterns?

## Technical Notes

### Generation Pipeline
1. **Tokenization**: DAC 9 codebooks (44.1 kHz, hop=512) → interleaved 1D tokens
2. **Generation**: CausalTransformer with KV cache + custom SEP token stopping
3. **Decoding**: DAC tokenizer converts tokens back to audio waveform
4. **Audio codec**: LAC with WhAM weights (much better than default for whale audio)

### Model Architecture
- **Type**: Nested Sparse Attention (NSA) + Mixture of Experts (MoE)
- **Size**: 375M parameters
- **Context**: 128K tokens (~169 seconds audio at 44.1 kHz)
- **Training**: SanctSound humpback + denoised audio data
- **Checkpoint**: `best_model_step109000.pt` (validation loss ~5.3)

### Known Limitations
- 128K context model requires ~16GB VRAM, can't restart reliably on RTX 5070 Ti
- Generation was slow (100+ minutes) until capped at 10 seconds max_new_tokens
- Model may overfit on SanctSound data (limited diversity, only humpback + few orcas/dolphins)

## Files Generated

### Original (Broken) Samples
- Location: `runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/comparison/`
- Files: `full_00/01/02_*_T0.70_40.wav`, `T0.85_80.wav`, `T1.00_120.wav`
- Summary: `generation_summary.json`
- Analysis: `ANALYSIS.md`, `quality_metrics.json`

### Fixed Samples (In Progress)
- Location: `runs/audio_medium_nsa_moe_sanctsound_humpback_dac_9cb_128k/comparison_fixed/`
- Expected completion: ~15 minutes
- Will include seed values in JSON summary for reproducibility

## UPDATE: Surprising Discovery About Randomness

### Multinomial Sampling IS Working
Created detailed debug script (`debug_generation_internals.py`) that shows:
- ✓ Different seeds DO produce different token sequences
- ✓ Multinomial sampling is stochastic and working correctly
- ✗ **Yet final audio files are byte-for-byte identical** (same MD5 hash)

### Example Token Sequences (First 5 tokens)
```
seed=42 (T=0.70, top_k=40): [146, 1903, 2181, 3151, 5112]
seed=43 (T=0.85, top_k=80): [382, 1096, 2493, 3236, 4508]  ← 5 differences
seed=44 (T=1.00, top_k=120): [146, 1765, 3068, 3968, 4616]  ← 4 differences
```

### The Mystery
Different **token sequences** but identical **audio files**. Possible causes:
1. Generation function not actually using the seed parameter (most likely)
2. Decoder somehow produces identical audio from different tokens (extremely unlikely)
3. Tokens are being truncated/stopped early in different ways
4. Full token sequences (not just first 5) eventually converge

### Investigation In Progress
Created `generate_with_token_logging.py` to:
- Generate with each seed and log all tokens produced
- Compare full token sequences (not just first steps)
- Determine where the identity comes from

## Next Steps

1. **Check token log results** — Compare full 7751-token sequences
2. **If tokens different**: Find why they produce identical audio
3. **If tokens same**: Fix seed application in generation function
2. **Audio quality evaluation**:
   - Automated: Spectral analysis, novelty scoring, diversity metrics
   - Manual: Listen to samples and assess naturalness
3. **Model assessment**:
   - Is 109,000 checkpoint usable for generation?
   - Should we train longer or use different architecture?
   - Compare with 32K baseline (simpler, more stable)
4. **Future improvements**:
   - Scale to more SanctSound deployments (add oc02, pm stations)
   - Train medium model (113M) on combined data
   - Implement classifier-free guidance for quality control

## References

- [NSA+MoE training memory](memory/project_nsa_moe_training.md)
- [DAC temporal chaining](memory/project_dac_temporal_chaining.md)
- Generation scripts: `scripts/generate_dac_9cb_comparison*.py`
- Debug scripts: `scripts/debug_token_uniqueness.py`, `scripts/debug_tokens_direct.py`
