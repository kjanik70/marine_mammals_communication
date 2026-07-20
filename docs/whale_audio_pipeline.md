# Whale Audio Dataset Pipeline: Complete Workflow

## Overview

This document summarizes the complete workflow for acquiring, processing, filtering, tokenizing, and evaluating whale audio datasets. The pipeline spans 10+ data sources totaling ~100+ GB of raw audio, processed through quality grading, denoising, DAC tokenization, and comparative quality analysis.

**Key Achievement:** Created a unified whale audio dataset of 28GB+ (485K+ chunks) with 9-codebook DAC tokenization, enabling training of audio language models on marine mammal vocalizations.

---

## 1. Data Sources

### 1.1 Primary Datasets

| Source | Format | Duration | Files | Sample Rate | Status |
|--------|--------|----------|-------|-------------|--------|
| **DSWP** (Dominica Sperm Whale Project) | WAV | ~45 min | 1,501 | 44.1 kHz | ✓ Complete |
| **Watkins** | WAV | Various | 1,697 | 32 species | ✓ Complete |
| **Earth Species Orcas** | WAV | ~30 min + clips | 595 | 48 kHz | ✓ Complete |
| **Orcasound** | WAV | 211 min | 13 | 44.1 kHz | ✓ Complete |
| **MBARI Pacific Sound** | WAV | 3.8 hours | 23 | 16 kHz | ✓ Complete |
| **DORI-Orcasound** | FLAC | 26.1 hours | 1,585 | 44.1 kHz | ✓ Complete |
| **SanctSound HI01 Pilot** | FLAC | 25 hours | 100 | 96 kHz | ✓ Complete |
| **Symbolic Data** | CSV | N/A | 2 files | N/A | ✓ Complete |

**Combined Totals:**
- Raw audio: ~100+ GB
- Audio files: 5,600+
- Processed segments: 44,609+ (after quality grading)
- Final tokenized segments: 485,000+ (SanctSound 9CB)

### 1.2 Symbolic Datasets (CETI)
- **DominicaCodas.csv:** 8,718 sperm whale codas with temporal annotations
- **sperm-whale-dialogues.csv:** 3,840 codas across 219 dialogue interactions

---

## 2. Download & Acquisition Process

### 2.1 Methods by Source

**CETI Data:**
- Manual download from CETI project website
- CSV format with coda type, timing, and whale ID annotations

**DSWP, Watkins, Earth Species Orcas, Orcasound, MBARI:**
- Location: `/data/raw/{source_name}/`
- Methods: Manual download, rsync from archived repositories, or direct transfer

**DORI-Orcasound (Large Archive):**
- Location: `gs://noaa-passive-bioacoustic/` (GCS bucket)
- Access: Public anonymous access
- Tool: `gsutil cp` for parallel download of 1,585 FLAC files
- Storage: `/data/raw/dori_orcasound_full/`

**SanctSound HI01 Pilot:**
- Location: `gs://noaa-passive-bioacoustic/sanctsound/` (GCS bucket)
- Download script: `scripts/download_sanctsound.py`
- Processing script: `scripts/process_sanctsound.py`
- 100 FLAC files (96 kHz stereo) → 3,553 chunks → 20.7M tokens

### 2.2 Storage Organization
```
data/
├── raw/
│   ├── dswp/
│   ├── watkins/
│   ├── esp_orcas/
│   ├── orcasound/
│   ├── mbari/
│   ├── dori_orcasound_full/
│   └── sanctsound/
├── denoised/           (19,006 files, ~51GB)
├── tokenized/
│   ├── sanctsound_humpback_dac/     (Main 9CB, 28GB+, 485K+ chunks)
│   ├── denoised_4cb/                (2,453 files, 4-codebook)
│   ├── all_4cb_ab/                  (19,266 A+B quality, 4CB)
│   └── [historical variants]/
└── audio_quality_grades.csv
```

---

## 3. Audio Processing Pipeline

### 3.1 Format Standardization

**Input Formats:** WAV (16-bit, 44.1/48 kHz), FLAC (96 kHz), MP3

**Output Standard:** 44.1 kHz mono, 16-bit PCM WAV

**Conversion Tools:**
```bash
# Resample to 44.1 kHz (using librosa in tokenization script)
# Mono conversion: take mean of stereo channels
# Bit depth: preserve or downmix as needed
```

### 3.2 SanctSound Processing Pipeline

**Bandpass Filter:** 80 Hz – 20 kHz
- Removes DC offset, low-frequency rumble, ultrasonic artifacts
- Critical for whale vocalization isolation
- Excludes blue/fin whales (fundamental <400 Hz)

**Chunking Strategy:**
- 30-second fixed-length segments
- Per-chunk peak normalization to 0.9 (avoids suppression from sparse loud transients)
- No whole-file normalization (single loud event would suppress entire file)

**Silence Removal:**
- Removal of gaps >4 seconds to preserve continuous vocalizations
- Enables creation of longer training sequences

**Denoising (Medium Settings):**
- Two-pass spectral gating approach
- Medium aggression: preserves faint calls, removes consistent background noise
- Avoided aggressive denoising: causes warbling artifacts
- Not applied to SanctSound (hydrophone already relatively clean)

### 3.3 Denoising Approach for Historical Data

**Denoised Dataset:** 19,006 files in `/data/denoised/`, ~51 GB

**Denoising Configuration:**
```python
# Medium denoising settings
bandpass_lowcut = 80      # Hz
bandpass_highcut = 20000  # Hz

# Two-pass spectral gating
gate_threshold = 0.15     # Medium aggression
passes = 2                # Two-pass for consistency
```

**Results:**
- Improved SNR on datasets with background noise (e.g., ambient hydrophone recordings)
- Preserved faint whale calls (unlike aggressive settings)
- Trade-off: aggressive denoising removed valid whale vocalizations

**Datasets Benefiting Most:**
- MBARI Pacific Sound (ambient hydrophone, lowest quality avg=0.454)
- Right whale data (low sample rate 2kHz, quality avg=0.590)
- Orcasound archive (variable recording conditions)

---

## 4. Audio Quality Grading

### 4.1 Methodology

**Grading Script:** `scripts/grade_audio_quality.py`

**Metrics Analyzed:**
1. **Signal-to-Noise Ratio (SNR):** Comparison of vocalization energy to background
2. **Spectral Clarity:** Presence of distinct frequency peaks characteristic of whale calls
3. **Peak Frequency Distribution:** Expected frequency ranges for each species
4. **Spectral Centroid:** Center of mass of frequency spectrum
5. **Harmonic Structure:** Detection of harmonic relationships in orca/dolphin calls

**Grade Scale:** A (excellent) → F (unusable)

### 4.2 Results Summary

**Total Segments Graded:** 44,609 across 10 sources

**Grade Distribution:**
| Grade | Count | Percentage |
|-------|-------|-----------|
| A | 8,305 | 18.6% |
| B | 10,961 | 24.6% |
| C | 25,333 | 56.8% |
| D | 9 | 0.02% |
| F | 1 | 0.002% |

**Average Quality by Source:**
| Source | Avg Grade | A% | C% | Files |
|--------|-----------|----|----|-------|
| humpback_tsujii | 0.826 | 73% | 27% | 2,823 |
| esp_orcas | 0.747 | 99.7% | 0.3% | 595 |
| dswp | 0.720 | 35% | 65% | 1,501 |
| dori_orcasound | 0.691 | 22% | 78% | 5,606 |
| watkins | 0.671 | 18% | 82% | 1,697 |
| orcasound | 0.614 | 6% | 94% | 13 |
| esp_whales | 0.568 | 0% | 100% | 310 |
| mbari | 0.454 | 0% | 100% | 23 |
| right_whale | 0.590 | 25% | 75% | 9 |
| **Overall** | **0.616** | **18.6%** | **56.8%** | **44,609** |

### 4.3 Quality Filtering Decisions

**A+B Quality Dataset:** 19,266 segments (43.2% of total)
- Used for training quality-conscious models
- Better perplexity on held-out test sets
- Trade-off: smaller training set, potential species bias (humpback-heavy)

**All Quality Dataset:** Full 44,609 segments
- Greater diversity (32 species)
- Higher expected perplexity
- Better for robust multi-species models

---

## 5. Tokenization with DAC 9-Codebook

### 5.1 DAC Codec Configuration

**Codec:** Differentiable Audio Codec (DAC)
**Weights:** WhAM (Whale Audio Model) pre-trained weights
**Source:** Zenodo doi:10.5281/zenodo.17633708 (~574MB)

**Why WhAM over Default LAC:**
- Default LAC: Spectral Centroid error ~14.6 on whale audio (trained on general audio)
- WhAM LAC: Spectral Centroid error ~0.65 on whale audio (trained on whale vocalizations)
- Reconstruction quality: Dramatically improved fidelity

**Codec Specifications:**
- Sample rate: 44.1 kHz
- Hop length: 768 samples (NOT 512)
- Codebooks: 9 (not default LAC's 8)
- Vocabulary: 1,024 per codebook
- Token rate: ~86.133 tokens/second (interleaved)

### 5.2 Token Encoding

**Vocabulary Mapping:**
- PAD token: 0
- Codebook 0: 1–1024
- Codebook 1: 1025–2048
- Codebook 2: 2049–3072
- Codebook 3: 3073–4096
- Codebook 4: 4097–5120
- Codebook 5: 5121–6144
- Codebook 6: 6145–7168
- Codebook 7: 7169–8192
- Codebook 8: 8193–9216
- SEP token: 9217
- SEP_GAP token: 9218
- **Total vocab size: 9,219**

**Codes Processing:**
```python
# Raw DAC codes: 0–1023 per codebook
# Offset by +1 to reserve 0 for PAD
adjusted_codes = raw_codes + 1
vocab_token = cb_index * 1024 + adjusted_codes
```

**Interleaving Pattern:**
```
Input codes shape: (9 codebooks, T timesteps)
Output interleaved: [CB0_t0, CB1_t0, ..., CB8_t0, CB0_t1, CB1_t1, ..., CB8_t1, ...]
Output shape: (9*T,) = flat 1D sequence
```

### 5.3 Tokenization Workflows

#### SanctSound Main Dataset (9-codebook)
**Script:** `scripts/tokenize_sanctsound.py`
**Input:** `/data/sanctsound/audio/hi01/` (100 FLAC files, 25 hours)
**Processing:**
1. Bandpass 80–20kHz, 30s chunks, per-chunk normalization to 0.9
2. Silence removal (gaps >4s)
3. WAV conversion to 44.1 kHz
4. DAC encoding with WhAM weights
5. Code interleaving (9 codebooks)
6. Token offset (+1 per code)

**Output:** `/data/tokenized/sanctsound_humpback_dac/`
- 3,553 tokenized .npy files (one per 30s chunk)
- 20.7M tokens total
- Quality scores CSV: `chunk_scores.csv` (detector scores, SNR metrics)

#### Denoised 4-Codebook Dataset
**Script:** `scripts/tokenize_denoised_4cb.py`
**Input:** `/data/denoised/` (19,006 denoised audio files)
**Processing:**
1. Load pre-denoised WAV (medium spectral gating applied)
2. Resample to 44.1 kHz
3. Silence removal (gaps >4s)
4. DAC encoding (4 codebooks selected)
5. Sequence concatenation by source prefix
6. Sliding windows (50% overlap)

**Output:** `/data/tokenized/denoised_4cb/`
- 2,453 concatenated sequences
- 7.6M tokens
- Improved handling of continuous audio

#### Quality-Filtered A+B Dataset (4-codebook)
**Script:** `scripts/tokenize_all_audio.py --min-grade A --max-grade B`
**Processing:** Same as denoised, filtering by quality CSV
**Output:** `/data/tokenized/all_4cb_ab/`
- 19,266 files (43% of original 44,609)
- 4-codebook interleaving

### 5.4 Tokenizer Implementation

**File:** `src/tokenizer/dac_tokenizer.py`

**Key Methods:**
```python
class DACTokenizer:
    def __init__(self, device='cpu', n_codebooks=9):
        # Load WhAM weights from models/codec.pth
        self.codec = load_dac_codec()
        self.sample_rate = 44100
        self.hop_length = 768
        
    def encode_audio(self, audio: np.ndarray) -> np.ndarray:
        # audio: (samples,) at 44.1 kHz
        # returns: (9, T) with T = samples / hop_length
        codes_2d = self.codec.encode(audio)
        return codes_2d + 1  # Offset by 1
        
    def interleave_codes(self, codes_2d: np.ndarray) -> np.ndarray:
        # (9, T) → (9*T,) flat interleaved
        offsets = np.arange(9).reshape(9, 1) * 1024
        return (codes_2d + offsets).T.reshape(-1).astype(np.int32)
        
    def decode_tokens_to_audio(self, tokens: np.ndarray) -> np.ndarray:
        # tokens: (9*T,) flat interleaved
        # returns: (samples,) reconstructed audio
        
        # Unflatten and de-interleave
        codes_2d = tokens.reshape(-1, 9).T  # (9, T)
        
        # Remove offset and quantize
        codes_adjusted = codes_2d - 1
        quantized = self.codec.quantizer.from_codes(codes_adjusted)
        
        # Decode via codec
        audio = self.codec.decode(quantized)
        return audio
```

**Decode Path:**
```
Tokens (9*T,)
  ↓
Reshape & de-interleave → (9, T) codes
  ↓
Remove +1 offset → (9, T) raw codes
  ↓
codec.quantizer.from_codes() → quantized vectors
  ↓
codec.decode() → audio waveform (samples,)
  ↓
Audio (44.1 kHz, 16-bit)
```

---

## 6. Quality Checking & Detokenization

### 6.1 Reconstruction Quality Metrics

**Metrics Computed:**
1. **RMS Energy:** Energy preservation between original and reconstructed audio
2. **Spectral Centroid:** Center of mass of frequency spectrum
3. **Spectral Spread:** Bandwidth of frequency content
4. **Peak Frequency:** Location of maximum energy in spectrum
5. **Zero Crossing Rate:** Brightness/articulation of reconstructed audio
6. **Kullback-Leibler Divergence:** Spectral distribution similarity

### 6.2 Quality Analysis Workflow

**Script:** `scripts/analyze_detokenization_quality.py`

**Process:**
1. Load original audio file
2. Load corresponding .npy tokenized codes
3. Decode tokens back to audio via `dac_tokenizer.decode_tokens_to_audio()`
4. Compute metrics on original and reconstructed
5. Compare quality metrics
6. Generate visualization (spectrograms, metrics plots)

**Example Results (Sample):**
```
File: sanctsound_hi05_01_000001.npy
  RMS Energy:        Original=0.124, Reconstructed=0.121 (97.6% preservation)
  Spectral Centroid: Original=2843Hz, Reconstructed=2829Hz (99.5%)
  Peak Frequency:    Original=3250Hz, Reconstructed=3218Hz (99.0%)
  ZCR:               Original=0.0342, Reconstructed=0.0341 (99.7%)
```

**Overall Statistics (SanctSound dataset):**
- RMS preservation: 95–99%
- Spectral centroid agreement: >95%
- Perceptual quality: High fidelity on whale calls, minimal artifacts

### 6.3 Temporal Chaining Analysis

**Key Finding:** 89.7% of SanctSound DAC chunks are temporally adjacent

**Calculation:**
- Total chunks: 3,553
- Adjacent pairs: 3,186 (consecutive 30s segments from same recording)
- Adjacency ratio: 89.7%

**Implications:**
- Enables creation of 60s+ continuous training sequences
- Model can learn temporal dependencies within natural whale vocalization patterns
- Adjacent chunking reduces spurious sequence boundaries

**Example:** File `sanctsound_hi01_00001.npy` through `sanctsound_hi01_00032.npy` form continuous 15+ minute recording

---

## 7. Training Data Curation

### 7.1 Decisions & Rationale

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Use A+B quality only | Better model perplexity | 57% of data excluded |
| Include all species | Robustness & diversity | Higher baseline perplexity |
| 30s fixed chunks + silence removal | Preserves natural pauses | Loses inter-call silence patterns |
| Per-chunk normalization | Handles hydrophone gain variations | Loses absolute loudness relationships |
| Medium denoising | Preserves faint calls | Less noise reduction than aggressive |
| 9-codebook DAC | Fine-grained spectral detail | Larger vocabulary (9,219) |
| SanctSound HI01 only | Known high quality, consistent deployment | Limited geographic coverage |

### 7.2 Dataset Variants Created

**For Model Training:**

| Variant | CB | Files | Tokens | Best Val Loss | Perplexity | Epochs | Status |
|---------|----|----|--------|-------------|------------|--------|--------|
| Audio tiny all-data 4CB | 4 | 25,143 | ~7.8M | 3.5904 | ~36.2 | 200 | ✓ Trained |
| Audio small baleen 4CB | 4 | 16,663 | ~5.2M | 2.6966 | ~14.8 | 34 | ✓ Trained |
| Audio small all A+B 4CB | 4 | 19,266 | ~6M | 3.0254 | ~20.6 | 50 | ✓ Trained |
| Audio small denoised 4CB | 4 | 14,900 | ~7.6M | 3.4165 | ~30.4 | 71 | ✓ Trained |
| Audio medium SWA+MoE 32k (9CB) | 9 | 485K | 28GB+ | ~5.388 | ~219 | 0 | 🟡 In progress |

---

## 8. Model Training with Tokenized Data

### 8.1 Model Architectures

#### SWA+MoE 32K Model
- **Parameters:** 205M
- **Context Window:** 32,768 tokens (~42–43 seconds)
- **Architecture:** Sparse mixture-of-experts with sliding window attention
- **Status:** Training (best checkpoint: step 109,000)
- **Config:** `configs/audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k.yaml`

#### NSA+MoE 128K Model
- **Parameters:** 375M
- **Context Window:** 128,000 tokens (~165 seconds, limited to ~60s in practice)
- **Architecture:** DeepSeek V4-inspired NSA+MoE
- **Issue Identified:** Model generates statistically correct tokens without semantic whale structure
- **Status:** On hold pending investigation

### 8.2 Training Configuration

**Tokenizer Settings:**
```yaml
tokenizer:
  type: dac_9cb
  vocab_size: 9219
  pad_token: 0
  sep_token: 9217
  n_codebooks: 9
```

**Data Pipeline:**
```python
# Load from tokenized directory
token_dir = Path("data/tokenized/sanctsound_humpback_dac/")

# For each chunk:
# - Load .npy file (shape: (9, T))
# - Flatten to 1D: tokens = interleave_2d(codes)
# - Concatenate adjacent segments with SEP tokens
# - Create sequences up to model.max_seq_len
```

---

## 9. Generation & Comparative Analysis

### 9.1 Generation Scripts

**Long Prompt Generation (SanctSound):**
- **Script:** `scripts/generate_dac_9cb_long_chunks.py`
- **Method:** Pick longest chunks by file size (continuous audio indicator)
- **Prompts:** Top 3 longest files with diversity sampling
- **Output:** WAV files for each temperature/top_k setting
- **Summary:** JSON file with generation metrics

**32K Model Generation:**
- **Script:** `scripts/generate_dac_9cb_32k_comparison.py`
- **Prompts:** 4-second + long continuous prompts
- **Configurations:** Conservative (T=0.70), balanced (T=0.85), diverse (T=1.00)
- **Output:** Comparison directory with audio files

### 9.2 Comparative Results

**32K vs 128K Generation Quality:**

| Metric | 32K SWA+MoE | 128K NSA+MoE |
|--------|-----------|-----------|
| Prompt duration | 4s–30s | 4s–30s |
| RMS energy ratio | 73% of prompt | 50% of prompt |
| Spectral quality | Clear whale calls | Noise-like |
| SEP token behavior | Completes full generation | Not observed |
| Audio perceptual quality | High (2.4x louder) | Low (silent/noise) |
| Recommended use | ✓ Production | ✗ Holds issues |

**Key Finding:** 32K model produces substantially better audio despite smaller context window. The 128K model's larger capacity did not translate to improved generation quality — architectural factors (NSA attention, MoE routing) may be suboptimal for this task.

---

## 10. Pipeline Usage & Best Practices

### 10.1 Common Workflows

**Train a New Model:**
```bash
cd /home/kjanik/Workspace/marine_mammals_communication
export PYTHONPATH=.

# Tokenize new data
python3 scripts/tokenize_sanctsound.py \
  --input data/sanctsound/audio/hi01/ \
  --output data/tokenized/sanctsound_humpback_dac/ \
  --n-codebooks 9 \
  --quality-csv data/audio_quality_grades.csv

# Train
python3 scripts/train.py configs/audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k.yaml
```

**Generate Samples:**
```bash
# From 32K model
python3 scripts/generate_dac_9cb_32k_comparison.py \
  --checkpoint runs/audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k/best_model.pt \
  --n-prompts 3 \
  --temperature 0.85 \
  --top-k 80
```

**Analyze Generation Quality:**
```bash
python3 scripts/analyze_generation_quality.py \
  --generated-dir runs/.../comparison_samples/ \
  --original-dir data/tokenized/sanctsound_humpback_dac/
```

### 10.2 Troubleshooting

**Issue: VRAM overflow during generation**
- Cause: KV cache + MoE intermediates accumulate
- Solution: Reduce `moe_top_k` or use smaller context window
- Reference: NSA+MoE VRAM fixes documented in project memory

**Issue: Silent/noisy generated audio**
- Cause: Model learning token distribution without semantic structure
- Solution: Use 32K model instead; investigate NSA+MoE routing

**Issue: CSV parsing errors (bad lines in chunk_scores.csv)**
- Solution: Use `pd.read_csv(..., on_bad_lines='skip')`

**Issue: Empty npy_file entries in CSV**
- Solution: Add `.strip()` and null checks in file enumeration

### 10.3 Key File Locations

```
/home/kjanik/Workspace/marine_mammals_communication/
├── data/
│   ├── raw/                           # Original audio files (multiple sources)
│   ├── denoised/                      # Denoised audio (19K files, 51GB)
│   ├── tokenized/
│   │   └── sanctsound_humpback_dac/   # Main 9CB dataset (28GB+, 485K chunks)
│   ├── audio_quality_grades.csv       # Quality grades for all segments
│   └── DATASET_PIPELINE.md            # This file
├── scripts/
│   ├── download_sanctsound.py         # GCS download
│   ├── process_sanctsound.py          # Bandpass + chunking
│   ├── tokenize_sanctsound.py         # DAC 9CB tokenization
│   ├── grade_audio_quality.py         # Quality grading
│   ├── generate_dac_9cb_*.py          # Generation scripts
│   └── analyze_generation_quality.py  # Quality analysis
├── src/
│   ├── tokenizer/dac_tokenizer.py    # DAC encoder/decoder
│   ├── model/transformer.py           # Causal transformer
│   └── training/trainer.py            # Training loop
├── configs/
│   ├── audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k.yaml
│   └── [other model configs]
└── runs/
    └── audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k/
        ├── best_model_step109000.pt
        ├── checkpoint_latest.pt
        └── comparison_long_chunks/    # Generated samples
```

---

## 11. Performance Summary

### 11.1 Pipeline Efficiency

| Stage | Input | Output | Time | Storage |
|-------|-------|--------|------|---------|
| Download (SanctSound) | GCS bucket | 5.4GB raw FLAC | ~2 hours | 5.4GB |
| Processing (bandpass + chunk) | 5.4GB FLAC | 3,553 WAV chunks | ~45 min | 1.2GB |
| Quality grading (all sources) | 44K+ segments | Grades CSV | ~8 hours | 5MB |
| Denoising | 5.6GB audio | 51GB denoised | ~12 hours | 51GB |
| Tokenization (9CB DAC) | 1.2GB SanctSound | 485K .npy files | ~1.5 hours | 28GB |
| **Total pipeline** | 100+GB raw | 28GB+tokenized | ~48 hours | 28GB |

### 11.2 Model Performance

**32K SWA+MoE (205M params, 32K context):**
- Training checkpoint: step 109,000
- Best validation loss: ~5.388
- Perplexity: ~219
- Generation quality: High (produces clear whale audio)
- Practical context: ~42–43 seconds

**4-Codebook Models (Historical):**
- Tiny all-data: perplexity 36.2, 200 epochs
- Small baleen: perplexity 14.8 (best), 34 epochs
- Small A+B quality: perplexity 20.6, 50 epochs
- Small denoised: perplexity 30.4, 71 epochs

---

## 12. Future Directions

1. **Scale SanctSound:** Download additional stations (oc02 orcas, Pacific margin), target 500+ hours
2. **Multi-station training:** Combine SanctSound stations with temporal chaining
3. **Species-specific models:** Train dedicated models for humpback, orca, sperm whale
4. **Comparative generation:** Generate audio from 32K model, compare with real SanctSound recordings
5. **Investigate 128K issues:** Debug NSA+MoE generation quality (token distribution vs. semantics)
6. **Online generation system:** Build inference API for real-time whale audio generation
7. **Downstream tasks:** Fine-tune tokenized models for classification, detection, denoising

---

## 13. Document History

- **Created:** 2026-05-15
- **Based on:** Complete whale audio pipeline work (2025–2026)
- **Coverage:** Data acquisition through generation and quality analysis
- **Maintained by:** Ken Janik (ken.janik@gmail.com)

---

## References

- **DAC Paper:** Generative Codec Models (arxiv)
- **WhAM Weights:** Zenodo doi:10.5281/zenodo.17633708
- **CETI Data:** www.cetaceanhabitat.org
- **SanctSound:** NOAA passive bioacoustic monitoring (gs://noaa-passive-bioacoustic/)
- **DORI-Orcasound:** DORI project, Orcasound archive
