# Marine Mammal Communication LLM

An autoregressive language model (GPT-style) for learning the sequential and conversational structure of marine mammal communication. The project focuses on sperm whale codas but extends to toothed cetaceans, baleen whales, and multi-species models.

The goal is not just to generate realistic whale sounds — it is to **understand communicative patterns**: what codas follow what, how whales take turns, and what combinatorial structure exists in their dialogues.

## Background

Project CETI's [2024 Nature paper](https://doi.org/10.1038/s41467-024-47221-8) showed that sperm whale codas have **combinatorial phonetic structure** (rhythm, tempo, rubato, ornamentation), and their dialogue dataset captures multi-whale conversations with up to 9 identified whales exchanging codas over minutes to hours.

[WhAM (NeurIPS 2025)](https://github.com/project-ceti/wham) published a BERT-style masked token model for whale audio, using the **LAC audio codec** for tokenization. We reuse their trained codec weights for audio tokenization but build an **autoregressive** model to capture sequential and conversational structure that a bidirectional model cannot.

## Two-Track Approach

### Track 1 — Symbolic (fast, interpretable)

Tokenizes CETI coda annotations directly: each coda becomes a token encoding its type, with special tokens for whale identity, pauses, and turn-taking. Trains on coda sequences and multi-whale dialogues.

### Track 2 — Audio (richer, produces audio)

Uses WhAM's LAC codec to tokenize raw audio into discrete codes. Trains on audio token sequences from multiple species. Generates actual audio output.

Both tracks use the **same GPT-style causal transformer** — only the vocabulary and input data differ.

## Results

### Symbolic Models

| Model | Data | Params | Perplexity | Top-1 Acc | Top-5 Acc |
|-------|------|--------|-----------|-----------|-----------|
| Coda sequences (tiny) | 8,718 codas | 6.3M | 12.61 | 59.4% | 86.0% |
| Dialogues (tiny) | 219 multi-whale dialogues | 6.3M | 3.40 | 59.6% | 95.0% |

### Audio Models — 1 Codebook

| Model | Data | Segments | Params | Val Loss | Perplexity |
|-------|------|----------|--------|----------|------------|
| DSWP-only (small) | Sperm whale codas | 1,501 | 34M | 1.72 | 5.6 |
| All species (small) | 5 sources, 32 species | 5,995 | 34M | 2.83 | 17.0 |
| Sperm whale (small) | DSWP + Watkins | 2,176 | 34M | 2.99 | 19.9 |
| Toothed cetaceans (small) | Odontoceti | 10,462 | 34M | 2.68 | 14.5 |
| **Baleen whales (small)** | **Mysticeti** | **29,560** | **34M** | **1.01** | **2.7** |
| **All species (tiny)** | **9 sources** | **39,394** | **6.6M** | **1.63** | **5.1** |

### Audio Models — 4 Codebooks + Sequence Concatenation

These models use 4 interleaved LAC codebooks for richer audio representation and concatenate segments from the same source with SEP tokens to learn cross-vocalization patterns. Sequences use sliding windows (50% overlap) up to 1024 tokens.

| Model | Data | Windows | Params | Val Loss | Perplexity | Notes |
|-------|------|---------|--------|----------|------------|-------|
| All species (tiny, 4CB) | 9 sources | 25,143 | 7.3M | 3.59 | 36.2 | 200 epochs, ~194 min |
| Baleen whales (small, 4CB) | Mysticeti | 16,663 | 35.7M | 2.70 | 14.8 | Early stopped epoch 34, ~107 min |
| All species A+B (small, 4CB) | 10 sources, quality filtered | 19,266 | 35.7M | 3.03 | 20.6 | Early stopped epoch 50, ~217 min |
| **Denoised (small, 4CB)** | **7 sources, denoised + 30s chunks** | **14,900** | **35.7M** | **3.42** | **30.4** | **Early stopped epoch 71, no epoch-12 overfit** |

> **Note**: Val loss is not directly comparable between 1CB and 4CB models — the 4CB vocabulary is 4x larger (4099 vs 1026), making per-token prediction harder. The real comparison is in generated audio quality: 4CB captures finer spectral detail that 1CB misses.

### Audio Models — SanctSound Humpback (Large-Scale, 4CB LAC)

Trained on ~3.2B tokens from 497K SanctSound Hawaii humpback files (Pipeline D, LAC codec). These models use longer context windows (2048–4096 tokens) and lazy-loading datasets to handle the scale.

| Model | Context | Params | Batch | Val Loss | Perplexity | Notes |
|-------|---------|--------|-------|----------|------------|-------|
| Medium 4096 | 4096 | 116M | 4 (eff 8) | 4.6989 | ~110 | Killed after epoch 0 (plateaued) |
| Medium 2048 | 2048 | 116M | 8 (eff 16) | 4.6432 | ~104 | Killed after epoch 1 (plateaued) |
| **Large 4096** | **4096** | **273M** | **16** | **4.6080** | **~100** | **76% through epoch 0, still improving** |

Key observations:
- **Shorter context converges faster**: The 2048 model reached 4.80 val loss in 15K steps vs the 4096 model needing ~200K steps for the same level, thanks to 2x more windows and larger effective batch.
- **Larger model beats smaller**: The large model (273M) surpassed both medium models (116M) by step ~15K and continued improving.
- **Gradient checkpointing enables large batches**: The large model fits batch_size=16 at 11GB with gradient checkpointing, vs the medium model needing batch_size=4 at 10.6GB without it.
- **Prompted generation works**: Feeding 5s of real whale audio as a prompt produces more coherent continuations than unconditional generation from random tokens.

### Audio Models — SanctSound Humpback (Large-Scale, DAC 9CB)

Re-tokenized with Descript Audio Codec (9 codebooks, ~10.4B interleaved tokens from 488K files). DAC has better reconstruction quality than LAC 4CB (xcorr 0.50 vs 0.14) and finer temporal resolution (86.1 vs 57.4 tokens/sec). Uses SWA+MoE architecture with 10K token context (13.2s of audio per window).

| Model | Context | Params | Batch | Val Loss | Perplexity | Notes |
|-------|---------|--------|-------|----------|------------|-------|
| Medium SWA+MoE | 10240 | 199M | 1 (eff 8) | — | — | Completed, superseded by large |
| Large SWA+MoE | 10240 | 479M | 1 (eff 8) | 5.3141 | ~203 | Stopped at step 656K; superseded by 32K medium |
| **Medium SWA+MoE 32K** | **32768** | **375M** | **1 (eff 8)** | **—** | **—** | **16 experts, SWA=2048, gradient checkpointing (~9.7 GB); config ready** |

Key observations:
- **Chunked SWA eliminates O(T²) mask**: Training uses `is_causal=True` per chunk instead of a `(T×T)` float mask. Saves 400MB VRAM at 10K context, enables Flash Attention kernel.
- **8-bit Adam + gradient checkpointing**: Fits the 479M-param model in 13.1GB VRAM on RTX 5070 Ti (16GB), leaving 3GB headroom.
- **Higher perplexity expected**: Vocab is 9219 (9×1024 + PAD + SEP) vs 4099 for 4CB LAC. The model predicts finer-grained tokens across 9 codebooks.
- **DAC 9CB generation quality**: Three bugs were fixed in the generation pipeline (extra +1 offset, wrong codec used for decoding, KV cache RoPE offset in SWA layers). After fixes: 0% codebook violations, coherent audio output.
- **32K context (42.3s) with gradient checkpointing**: Medium 32K config uses 16 experts (375M total params) and SWA window of 2048 tokens (~2.6s local context). Gradient checkpointing reduces peak VRAM to ~9.7 GB (batch=1) vs ~18 GB without — fits comfortably in 16 GB.

### Audio Quality Grading

Each raw audio segment is graded A–F based on signal quality metrics (spectral flatness, peak-to-RMS ratio, energy variance, RMS energy). The "A+B" model above uses only segments graded A or B — filtering out noisy/silent segments (mostly right_whale at 2kHz and ambient MBARI hydrophone recordings).

| Source | Segments | Avg Score | Grade A | Grade B | Grade C |
|--------|----------|-----------|---------|---------|---------|
| humpback_tsujii | 216 | 0.826 | 73% | 27% | 0% |
| esp_orcas | 594 | 0.747 | 0% | 100% | 0% |
| watkins | 2,909 | 0.703 | 9% | 86% | 5% |
| dswp | 1,500 | 0.662 | 0% | 75% | 24% |
| dori_orca_full | 5,215 | 0.640 | 0% | 66% | 34% |
| orcasound | 989 | 0.640 | 0% | 64% | 36% |
| right_whale | 27,932 | 0.590 | 0% | 25% | 75% |
| **Total** | **44,609** | **0.616** | **1%** | **42%** | **57%** |

Quality histograms are generated in `data/quality_histograms/`.

### Orca Call Detector

A binary CNN classifier (`OrcaDetectorCNN`) that detects orca vocalizations in 3-second log-mel spectrogram windows. Trained from scratch because existing tools either required incompatible Python versions (orcAI requires 3.11) or had no publicly available weights (OrcaHello, ORCA-SPOT).

**Architecture** (`src/detector/orca_detector.py`): 4-block CNN (32→64→128→256 channels, stride-2 convolutions), global average pooling, two-layer MLP head. ~1M parameters. Input: `(1, 128, T)` log-mel spectrogram normalized to [0, 1] (128 mel bins, n_fft=2048, hop=512, f_min=50 Hz, f_max=20 kHz, top_db=80).

**Training** (`scripts/train_orca_detector.py`):

| Source | Label | Windows | Notes |
|--------|-------|---------|-------|
| DORI-Orcasound (FLAC) | Positive | ~15K | 60s FLACs, 15 windows each |
| ESP Orcas (WAV) | Positive | ~2K | Short orca call clips |
| KW Prince Edward Islands | Positive | ~40 | 14-min recording |
| Orcasound SRKW | Positive | ~9K | 10-min WAV files |
| MBARI ambient hydrophone | Negative | ~4K | 16 kHz ambient ocean |
| Watkins (non-orca species) | Negative | ~500 | Hard negatives: other marine mammals |
| **Total** | | **30,418** | **26,415 pos / 4,003 neg** |

Training used `WeightedRandomSampler` to balance classes, time/frequency masking augmentation, cosine LR decay (3e-4 → 0), early stopping (patience 6 epochs). **Best val_loss: 0.0002, val_acc: 100%, early stopped at epoch 15.**

**Domain shift finding**: The detector trained on high-SNR curated recordings scores 1.0 on DORI/ESP/SRKW clips, but ~0 on SanctSound chunks — passive acoustic hydrophone data has much lower SNR. The signal is present but below the threshold the model learned from clean training data.

**Fine-tuning for SanctSound** (`scripts/finetune_orca_detector_sanctsound.py`): Adapts the detector to low-SNR hydrophone data using in-domain examples:
- **Positives**: windows within annotation timestamps ±30s from OC02/OC01 FLACs
- **Negatives**: windows >180s from any annotation in the same FLACs (same noise floor, no whale calls)
- Fine-tunes from `models/orca_detector.pt` at LR=5e-5, outputs `models/orca_detector_ft.pt`

### Codec Reconstruction Quality (LAC vs DAC)

We evaluated roundtrip reconstruction quality (original → encode → decode → compare) to understand how much information the audio tokenization preserves. Tests used 10 files from DSWP, humpback, orcasound, and ESP orcas datasets, preprocessed with the same pipeline used for tokenization (bandpass 80 Hz–20 kHz + peak normalization to 0.9).

**LAC (WhAM weights)** — 44.1 kHz, hop=768, 14 codebooks, ~57 tokens/sec per codebook:

| Codebooks | SC | SNR (dB) | MCD | xcorr | Total tok/s |
|-----------|--------|----------|------|-------|-------------|
| 1 | 0.64 | 0.4 | 19.7 | 0.30 | 57 |
| **4 (current)** | **0.81** | **-2.0** | **48.7** | **0.14** | **230** |
| 8 | 0.65 | -0.2 | 15.0 | 0.33 | 459 |
| 14 (all) | 0.54 | -0.1 | 14.2 | 0.40 | 804 |

**DAC (Descript Audio Codec, 44 kHz)** — 44.1 kHz, hop=512, 9 codebooks, ~86 tokens/sec per codebook:

| Codebooks | SC | SNR (dB) | MCD | xcorr | Total tok/s |
|-----------|--------|----------|------|-------|-------------|
| 1 | 0.86 | -2.3 | 44.6 | 0.16 | 86 |
| 4 | 0.85 | -2.4 | 44.1 | 0.19 | 345 |
| 9 (all) | 0.51 | 0.8 | 14.8 | 0.50 | 775 |

Metrics: SC = Spectral Convergence (lower = better), SNR = Signal-to-Noise Ratio (higher = better), MCD = Mel Cepstral Distortion (lower = better), xcorr = cross-correlation (higher = better).

**Key findings:**

- **LAC 4CB reconstruction is poor** (xcorr=0.14, negative SNR). The first 4 of 14 RVQ codebooks capture only coarse audio structure — fine spectral detail is lost. This is the representation our current models train on.
- **DAC at full codebooks is significantly better** (xcorr=0.50, positive SNR) at a similar total token rate (775 vs 804 tok/s). DAC's 9 codebooks distribute information more evenly than LAC's 14.
- **LAC 4CB has a "dead zone"** — quality is actually worse than 1CB, likely because the interleaved codebook tokens create dependencies the model must learn to reconstruct.
- **Neither codec is great on whale audio**. Even at full codebooks, xcorr peaks at 0.50 (DAC) / 0.40 (LAC). Human speech codecs are optimized for 300 Hz–8 kHz; whale vocalizations span 80 Hz–25 kHz with different spectral characteristics.
- **Switching to DAC would require re-tokenizing** all 497K files (~3.2B tokens) and retraining. The LAC WhAM weights were specifically trained on whale audio, which may provide advantages not captured by these metrics.

Audio comparison files for listening tests are saved in `runs/codec_comparison_processed/`.

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 5070 Ti, 16GB VRAM)
- ~10 GB free disk space for data and models

### 1. Clone and install

```bash
git clone https://github.com/<your-user>/marine_mammals_communication.git
cd marine_mammals_communication
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the audio codec (required for Track 2):

```bash
pip install lac@git+https://github.com/hugofloresgarcia/lac.git
pip install descript-audiotools@git+https://github.com/hugofloresgarcia/audiotools.git
```

Optional (for notebooks):

```bash
pip install ipykernel ipywidgets jupyter
```

### 2. Download codec weights

Download `codec.pth` from the WhAM Zenodo release:

```bash
mkdir -p models
# Download from: https://zenodo.org/records/17633708
# Extract codec.pth and place it in models/
```

> **Important**: You must use the WhAM-trained weights (`codec.pth` from Zenodo), not the default LAC weights. The default LAC weights produce extremely poor reconstructions on whale audio (spectral convergence 14.6 vs 0.65 with WhAM weights).

### 3. Download datasets

```bash
export PYTHONPATH=.

# Core data: CETI annotations + DSWP audio + codec weights pointer
python3 scripts/download_data.py

# Additional datasets: MBARI, HuggingFace whale sounds, DORI-Orcasound
python3 scripts/download_more_data.py
```

For the extended datasets (see [Data Sources](#data-sources) below for manual downloads):

```bash
# Watkins Marine Mammal Sound Database
# Download from: https://cis.whoi.edu/science/B/whalesounds/
# Extract audio into data/raw/watkins/audio/<Species_Name>/

# Earth Species Project Orcas
# Download from HuggingFace: https://huggingface.co/datasets/earthspecies/orcas
# Place in data/raw/esp_orcas/audio/

# Orcasound (hydrophone recordings)
# Download from: https://www.orcasound.net/data/
# Place in data/raw/orcasound/

# Humpback Whale Songs (Tsujii et al.)
# Download from Zenodo: https://zenodo.org/records/14862938
# Place WAV files in data/raw/humpback_zenodo/

# Right Whale Upcalls (NOAA/Cornell)
# Download from Kaggle: https://www.kaggle.com/c/whale-detection-challenge/data
# Extract to data/raw/right_whale/v1/

# DORI-Orcasound (orca hydrophone FLAC files)
# Download from HuggingFace: https://huggingface.co/datasets/DORI-SRKW/DORI-Orcasound
# Place FLAC files in data/raw/dori_orcasound/

# Killer Whale Prince Edward Islands
# Download from Zenodo: https://zenodo.org/records/7712582
# Place WAV in data/raw/kw_pei/
```

#### SanctSound (NOAA passive acoustic monitoring)

Requires `google-cloud-storage`:

```bash
pip install google-cloud-storage

# Download FLAC files from a SanctSound station (anonymous GCS access)
# HI01 = Hawaii humpback, deployment 1
python3 scripts/download_sanctsound.py --station hi01 --deployment 1 --max-files 100

# Other stations of interest:
# python3 scripts/download_sanctsound.py --station oc02 --deployment 1  # Olympic Coast orcas
# python3 scripts/download_sanctsound.py --station pm05 --deployment 1  # Pacific humpback
```

Files are saved to `data/sanctsound/audio/<station>/`.

### 4. Process audio

There are three processing pipelines depending on the data source:

#### Pipeline A: Raw tokenization (short clips)

For pre-segmented datasets (DSWP, Watkins, ESP Orcas, etc.) where each file is a single vocalization:

```bash
export PYTHONPATH=.

# 1 codebook
python3 scripts/tokenize_all_audio.py --n-codebooks 1   # → data/tokenized/all/

# 4 codebooks (richer representation)
python3 scripts/tokenize_all_audio.py --n-codebooks 4   # → data/tokenized/all_4cb/

# Organize by species group
python3 scripts/organize_species.py --n-codebooks 4      # → data/tokenized/{sperm_whale,toothed,baleen}_4cb/
```

#### Pipeline B: Denoised + long-chunk tokenization

For existing datasets that benefit from denoising. Applies medium denoising (bandpass 400 Hz–20 kHz + two-pass spectral gating + loudness normalization), then segments into 30s chunks preserving natural pauses:

```bash
export PYTHONPATH=.

# Step 1: Denoise all raw datasets → data/denoised/
python3 scripts/denoise_all_audio.py

# Step 2: Tokenize denoised audio in 30s chunks → data/tokenized/denoised_4cb/
python3 scripts/tokenize_denoised_audio.py --codec-path models/codec.pth --n-codebooks 4

# Optional: apply file-level quality filter (only include files with avg grade >= B)
python3 scripts/tokenize_denoised_audio.py --codec-path models/codec.pth --n-codebooks 4 \
    --quality-csv data/audio_quality_grades.csv --min-quality-score 0.6
```

#### Pipeline C: SanctSound pilot (passive acoustic monitoring)

For continuous hydrophone recordings. These have very low SNR — spectral gating removes faint whale calls, so we use bandpass-only with per-chunk peak normalization:

```bash
export PYTHONPATH=.

# Process all downloaded FLAC files for a station
python3 scripts/process_sanctsound.py --station hi01 --device cuda
# → data/tokenized/sanctsound_4cb/

# Process all stations at once (omit --station)
python3 scripts/process_sanctsound.py --device cuda
```

The pipeline per file:
1. Load FLAC, resample to 44,100 Hz
2. Bandpass filter 80 Hz – 20 kHz (removes ocean ambient noise)
3. Segment into ≤30s chunks (adaptive silence detection)
4. Per-chunk peak normalization to 0.9
5. Tokenize with LAC codec (4 codebooks)

#### Pipeline D: SanctSound Hawaii humpback (large-scale, detection-guided)

Builds on Pipeline C with three key improvements: (1) skips the 5-second test tone at the start of each FLAC, (2) uses NOAA detection annotations to process only high-detection hours (>80% humpback), and (3) applies a whale-band variability filter to keep only chunks with actual vocalizations. Processes one deployment at a time, streaming FLACs from GCS and deleting them after tokenization to manage disk space.

Supports both LAC (4CB) and DAC (9CB) codecs via `--codec` flag:

```bash
export PYTHONPATH=.

# Process with LAC 4CB (original codec)
python3 scripts/process_sanctsound_humpback.py
# → data/tokenized/sanctsound_humpback_4cb/

# Process with DAC 9CB (better reconstruction, finer temporal resolution)
python3 scripts/process_sanctsound_humpback.py --codec dac --save-2d
# → data/tokenized/sanctsound_humpback_dac/

# With Google humpback detector for chunk-level scoring
python3 scripts/process_sanctsound_humpback.py --codec dac --save-2d --use-detector
# Scores saved to chunk_scores.csv, filterable at training time via min_detector_score

# Process a specific station and deployment
python3 scripts/process_sanctsound_humpback.py --codec dac --save-2d --station hi04 --deployment 2

# Dry run (list qualifying FLACs without downloading)
python3 scripts/process_sanctsound_humpback.py --station hi05 --dry-run
```

The pipeline per file:
1. Load FLAC, convert to mono, resample to 44,100 Hz (chunked to limit memory)
2. Skip first 5 seconds (test tone present in all SanctSound recordings)
3. Bandpass filter 80 Hz – 20 kHz
4. Segment into ≤30s chunks, remove silence >4s
5. Per-chunk peak normalization to 0.9
6. **Whale-band variability filter**: compute coefficient of variation of RMS energy in 200–4000 Hz band (0.5s frames). Keep chunks with CV > 0.8 (whale songs ~1.5–3.5, ocean noise ~0.3–0.5)
7. Loudness normalization to -20 LUFS (**no spectral gating** — it removes faint whale calls)
8. Optional: Google humpback detector scoring (TF Hub, CPU-only, parallel threaded)
9. Tokenize with LAC 4CB or DAC 9CB codec
10. Save as 1D interleaved (LAC) or 2D `(n_codebooks, T)` arrays (DAC with `--save-2d`)

**DAC 9CB specifics:**
- Saves as 2D `(9, T)` numpy arrays with +1 PAD offset already applied
- Interleaving at training time: add `cb_index * 1024` per codebook (no extra +1)
- Vocab: CB0=1–1024, CB1=1025–2048, ..., CB8=8193–9216, PAD=0, SEP=9218, vocab_size=9219
- ~86.1 tokens/sec per codebook, ~775 interleaved tokens/sec
- `chunk_scores.csv` sidecar with heuristic scores (whale_cv, energy_ratio, whale_rms) and optional detector_score for training-time filtering

**Key lessons learned from SanctSound processing:**
- **No spectral gating**: Standard noise reduction (spectral gating) destroys faint whale calls in low-SNR hydrophone data. Bandpass + loudness normalization preserves them.
- **Per-chunk normalization**: Hydrophone recordings have sparse loud transients (boat passes, snapping shrimp) that suppress the entire file if normalized globally. Per-chunk normalization ensures each 30s chunk uses the full dynamic range.
- **Test tone**: Every SanctSound FLAC begins with a ~5s calibration tone that must be skipped.
- **Detection-guided selection**: Processing all hours wastes compute on empty ocean. Using NOAA's hourly detection annotations (>80% humpback proportion) focuses on hours with confirmed whale presence.
- **Whale-band variability filter**: Even within high-detection hours, many 30s chunks contain only ambient noise. The CV filter provides a cheap, effective way to keep only chunks with actual vocalizations — no ML detector needed.
- **Stream-and-delete**: Each FLAC is ~5.4 GB (96 kHz, 15 min). Downloading all at once is infeasible. Process one deployment at a time, delete FLACs after tokenization.
- **Done-file tracking**: `.done_{station}_{dep}.txt` files track which FLACs have been processed, enabling clean restarts without duplicate token creation.

**Stations processed** (4 stations, 10 deployments):

| Station | Deployments | FLACs | Token files | Notes |
|---------|------------|-------|-------------|-------|
| HI05 | 01 | ~135 | 6,937 | Smallest, used for pipeline validation |
| HI01 | 01, 02, 03 | 1,144 | 154,449 | |
| HI03 | 01, 03 | 562 | 95,689 | 02 has 0 qualifying FLACs |
| HI04 | 01, 02, 03 | 981 | 240,369 | |
| **Total** | | **~2,822** | **497,444** | **~3.2B tokens** |

**Token-level quality grading** (sampled 2,000 files per deployment, 17,245 total):

| Station | Files | Avg Score | A% | B% | C% | D+F% |
|---------|------:|----------:|---:|---:|---:|-----:|
| HI01 | 154,449 | 0.709 | 4.5 | 94.9 | 0.5 | 0.0 |
| HI03 | 95,689 | 0.719 | 3.4 | 96.1 | 0.5 | 0.0 |
| HI04 | 240,369 | 0.720 | 4.5 | 95.1 | 0.4 | 0.0 |
| HI05 | 6,937 | 0.713 | 0.5 | 99.5 | 0.0 | 0.0 |

Grades are computed from token-level metrics (CB0 entropy, unique token ratio, consecutive repeat ratio, codebook range usage). 95–99.5% of chunks score grade B across all stations — the whale-band CV filter effectively rejects ambient noise, leaving only vocalization-rich content. Zero D/F grades.

#### Pipeline E: SanctSound Orca (annotation-guided, DAC 9CB)

Downloads only the FLAC windows that overlap with manual orca call annotations, extracts those time ranges (±2s buffer), applies bandpass + optional spectral gating, and tokenizes with DAC 9CB. Uses precise start/end timestamps with ecotype labels (SRKW, NR, Transient, Unknown) — no ML detector needed.

```bash
export PYTHONPATH=.

# Process all OC stations (oc01–oc04) with annotation-guided download
python3 scripts/process_sanctsound_orca.py
# → data/tokenized/sanctsound_orca_dac/

# Single station
python3 scripts/process_sanctsound_orca.py --station oc02

# With two-pass spectral gating (experimental; can remove faint calls)
python3 scripts/process_sanctsound_orca.py --station oc01 --denoise

# Dry run (list qualifying FLACs without downloading)
python3 scripts/process_sanctsound_orca.py --station oc01 --dry-run
```

The pipeline per FLAC:
1. Check overlap with orca annotation CSV → skip if none
2. Download FLAC from GCS to tmp dir
3. Load → mono → resample to 44,100 Hz
4. Extract annotated time ranges + 2s buffer
5. Bandpass 80 Hz – 20 kHz
6. Optional: two-pass spectral gating (`--denoise`)
7. Segment into ≤30s chunks (remove >4s silence)
8. Per-chunk: peak normalize → orca band CV/energy filter → loudness normalize
9. Tokenize with DAC 9CB (save as 2D `(9, T)` .npy)
10. Save scores + ecotype label to `chunk_scores.csv`, then delete FLAC

**Stations processed** (4 stations, 11 deployments):

| Station | Location | Deployments | Token files | Notes |
|---------|----------|-------------|-------------|-------|
| OC01 | Olympic Coast NMS | 1, 3 | — | Southern Resident KW |
| OC02 | Olympic Coast NMS | 1, 2, 4, 5 | — | SRKW + Northern Resident |
| OC03 | Olympic Coast NMS | 2, 3, 4 | — | Mixed ecotypes |
| OC04 | Olympic Coast NMS | 2, 4 | — | Transient KW |
| **Total** | | **11** | **8,657** | **~190M tokens, ~68 hours** |

Scores are written to `chunk_scores.csv` alongside the token files for training-time filtering by ecotype, orca band CV, or energy ratio.

### 5. Train models

```bash
export PYTHONPATH=.

# === Track 1: Symbolic ===
# Individual coda sequences
python3 scripts/train.py configs/symbolic_tiny.yaml

# Multi-whale dialogues
python3 scripts/train.py configs/symbolic_tiny_dialogue.yaml

# === Track 2: Audio (1 codebook) ===
# Single-species (DSWP sperm whale only)
python3 scripts/train.py configs/audio_small.yaml

# All species
python3 scripts/train.py configs/audio_tiny_all.yaml

# Species-group models
python3 scripts/train.py configs/audio_small_sperm.yaml
python3 scripts/train.py configs/audio_small_toothed.yaml
python3 scripts/train.py configs/audio_small_baleen.yaml

# === Track 2: Audio (4 codebooks + sequence concatenation) ===
python3 scripts/train.py configs/audio_tiny_all_4cb.yaml
python3 scripts/train.py configs/audio_small_baleen_4cb.yaml

# === Track 2: Audio (quality-filtered A+B) ===
# First grade all audio segments:
python3 scripts/grade_audio_quality.py
# Tokenize only A+B quality segments:
python3 scripts/tokenize_all_audio.py --n-codebooks 4 \
    --quality-csv data/audio_quality_grades.csv --min-grade B
# Train:
python3 scripts/train.py configs/audio_small_all_4cb_ab.yaml

# === Track 2: Audio (denoised long-chunk) ===
# Uses Pipeline B output (data/tokenized/denoised_4cb/)
python3 scripts/train.py configs/audio_small_denoised_4cb.yaml

# === Track 2: Audio (SanctSound humpback, Pipeline D) ===
# Uses data/tokenized/sanctsound_humpback_4cb/ (~3.2B tokens, 497K files)
python3 scripts/train.py configs/audio_medium_sanctsound_humpback_4cb.yaml
python3 scripts/train.py configs/audio_large_sanctsound_humpback_4cb.yaml

# Or run all 1CB models sequentially:
bash scripts/train_all.sh
```

### 6. Generate audio samples

```bash
export PYTHONPATH=.

# Unconditional generation (all trained models)
python3 scripts/generate_all.py --n-samples 5 --max-tokens 300 --temperature 0.9

# Prompted generation — LAC 4CB models (feed real whale audio, model continues)
python3 scripts/generate_prompted.py \
    --checkpoint runs/audio_medium_sanctsound_humpback_4cb/best_model.pt \
    --token-dir data/tokenized/sanctsound_humpback_4cb \
    --prompt-tokens 1150 --temperature 0.85 --top-k 80
```

Generated WAV files are saved to `runs/<model>/generated/` (unconditional) and `runs/<model>/prompted/` (prompted).

#### DAC 9CB generation

DAC 9CB models use the Descript Audio Codec (9 codebooks, hop=512, ~86.1 tokens/sec per codebook). These require the `DACTokenizer` for decoding — **not** the LAC `AudioTokenizer`. The tokenized files are 2D arrays of shape `(9, T)` with the +1 PAD offset already applied.

```python
import numpy as np
import torch
import soundfile as sf
from src.model.transformer import CausalTransformer
from src.tokenizer.dac_tokenizer import DACTokenizer

N_CB = 9
SEP = 9218  # 9*1024 + 2

# Load model
ckpt = torch.load("runs/<dac_9cb_run>/best_model.pt", map_location="cuda", weights_only=False)
model = CausalTransformer(ckpt["config"]).cuda()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load DAC codec (not LAC!)
tokenizer = DACTokenizer(device="cpu", n_codebooks=N_CB)

# Load and interleave a tokenized file
raw = np.load("data/tokenized/sanctsound_humpback_dac/some_file.npy")  # (9, T)
offsets = np.arange(N_CB).reshape(N_CB, 1) * 1024
tokens_1d = (raw + offsets).T.reshape(-1).astype(np.int32)

# Use first ~5s as prompt (86.1 * 9 ≈ 775 tokens/sec)
prompt = tokens_1d[:3870]
prompt_t = torch.tensor(prompt, dtype=torch.long, device="cuda").unsqueeze(0)

# Generate continuation
with torch.no_grad():
    generated = model.generate(
        prompt_t,
        max_new_tokens=ckpt["config"].max_seq_len - len(prompt),
        temperature=0.85,
        top_k=80,
        eos_token_id=-1,
    )

# Decode to audio using DACTokenizer
full_tokens = generated[0].cpu().numpy()
audio = tokenizer.decode_tokens_to_audio(full_tokens, n_codebooks=N_CB, sep_token=SEP)
sf.write("output.wav", audio, tokenizer.sample_rate)
```

**Important notes for DAC 9CB:**
- Use `DACTokenizer`, not `AudioTokenizer` (LAC). DAC and LAC are different codecs with different codebook structures.
- The 2D `.npy` files already have the +1 PAD offset. Interleave by adding `cb_index * 1024` — do **not** add an extra +1.
- Vocab layout: CB0=1–1024, CB1=1025–2048, ..., CB8=8193–9216, PAD=0, SEP=9218, vocab_size=9219.
- DAC sample rate is 44,100 Hz with hop_length=512 (~86.1 tokens/sec per codebook, ~775 interleaved tokens/sec).

### 7. Evaluate symbolic models

```bash
export PYTHONPATH=.
python3 scripts/evaluate.py runs/symbolic_tiny_coda/best_model.pt --dataset-type coda
python3 scripts/evaluate.py runs/symbolic_tiny_dialogue/best_model.pt --dataset-type dialogue
```

## Data Sources

### CETI Annotations (Symbolic Track)

| Source | Content | Size |
|--------|---------|------|
| [Project CETI sw-combinatoriality](https://github.com/Project-CETI/sw-combinatoriality) | DominicaCodas.csv: 8,718 annotated codas with type, ICI, duration, whale ID | ~1 MB |
| [Project CETI sw-combinatoriality](https://github.com/Project-CETI/sw-combinatoriality) | sperm-whale-dialogues.csv: 3,840 codas across 219 multi-whale conversations | ~500 KB |

### Audio Datasets (Audio Track)

| Source | Species | Files | Duration | Sample Rate | Segments |
|--------|---------|-------|----------|-------------|----------|
| [DSWP](https://huggingface.co/datasets/orrp/DSWP) (HuggingFace) | Sperm whale | 1,501 | ~45 min | 44.1 kHz | 1,501 |
| [Watkins](https://cis.whoi.edu/science/B/whalesounds/) | 32 species | 1,697 | ~5 hrs | varies | ~1,700 |
| [Earth Species Orcas](https://huggingface.co/datasets/earthspecies/orcas) | Orca | 595 + 1 | ~35 min | 44.1 kHz | ~600 |
| [Orcasound](https://www.orcasound.net/data/) | Sperm whale, Orca | 13 | ~211 min | varies | ~2,500 |
| [MBARI Pacific Sound](https://registry.opendata.aws/pacific-sound/) | Various (hydrophone) | 23 | 3.8 hrs | 16 kHz | ~200 |
| [DORI-Orcasound](https://huggingface.co/datasets/DORI-SRKW/DORI-Orcasound) | Orca (SRKW) | 1,585 | ~26 hrs | 44.1 kHz | ~5,200 |
| [Humpback Songs (Tsujii)](https://zenodo.org/records/14862938) | Humpback whale | 6 | 60 min | 44.1 kHz | ~700 |
| [Right Whale Upcalls](https://www.kaggle.com/c/whale-detection-challenge) | Right whale | 12,000 | ~100 hrs | 2 kHz | ~24,000 |
| [KW Prince Edward Islands](https://zenodo.org/records/7712582) | Killer whale | 1 | 14 min | 96 kHz | ~170 |

**Total**: ~44,600 graded segments across 10 sources. ~19,300 pass A+B quality filter.

### NOAA SanctSound (Passive Acoustic Monitoring)

[SanctSound](https://sanctsound.ioos.us/) is a NOAA program that deployed hydrophones across U.S. National Marine Sanctuaries. The data is publicly available on Google Cloud Storage (`noaa-passive-bioacoustic` bucket, anonymous access).

Processed via Pipeline D (detection-guided, whale-band CV filter):

| Station | Location | Deployments | FLACs processed | Token files | Tokens |
|---------|----------|-------------|-----------------|-------------|--------|
| HI01 | Hilo, Hawaii | 01, 02, 03 | 1,144 | 154,449 | ~1.0B |
| HI03 | Maui, Hawaii | 01, 03 | 562 | 95,689 | ~610M |
| HI04 | Hawaii (west) | 01, 02, 03 | 981 | 240,369 | ~1.5B |
| HI05 | Kona, Hawaii | 01 | ~135 | 6,937 | ~44M |
| **Total** | | **10** | **~2,822** | **497,444** | **~3.2B** |

**DAC 9CB re-tokenization** (Pipeline D with `--codec dac --save-2d`):

| Station | Token files (DAC) | Tokens (interleaved) | Notes |
|---------|-------------------|---------------------|-------|
| HI01 | 146,083 | ~3.1B | |
| HI03 | 97,758 | ~2.1B | |
| HI04 | 235,875 | ~5.1B | |
| HI05 | 6,088 | ~130M | |
| **Total** | **485,804** | **~10.4B** | **avg ~21K tokens/chunk** |

The full SanctSound dataset contains ~96,000 FLAC files across 31 stations (~28 TB). Future expansion targets: OC01-04 (Olympic Coast orcas, ~643 hours annotated), PM stations (Pacific humpback).

**Species frequency compatibility**: Humpback (80–8,000 Hz), orca (1–25 kHz), and dolphins (2–150 kHz) work well with the LAC codec's 400 Hz+ range. Blue whale (10–100 Hz) and fin whale (15–30 Hz) are below the bandpass and cannot be used.

### Species Taxonomy

Audio data is organized into species groups for group-specific models:

**Toothed cetaceans (Odontoceti)**: Sperm whale, Killer whale, Atlantic Spotted Dolphin, Bottlenose Dolphin, Clymene Dolphin, Common Dolphin, Fraser's Dolphin, Risso's Dolphin, Pantropical Spotted Dolphin, Rough-Toothed Dolphin, Spinner Dolphin, Striped Dolphin, White-beaked Dolphin, White-sided Dolphin, Long-Finned Pilot Whale, Short-Finned Pilot Whale, False Killer Whale, Melon-Headed Whale, Narwhal, Beluga.

**Baleen whales (Mysticeti)**: Humpback Whale, Fin Whale, Bowhead Whale, Minke Whale, Northern Right Whale, Southern Right Whale.

## Audio Tokenization

Audio is tokenized using WhAM's trained LAC (Learned Audio Codec):

- **Sample rate**: 44,100 Hz
- **Hop length**: 768 samples
- **Token rate**: ~57.4 tokens/sec
- **Codebooks**: 14 via RVQ (residual vector quantization)
- **1CB mode**: First codebook only, vocab 1,026 (1,024 codes + PAD + offset)
- **4CB mode**: 4 codebooks interleaved, vocab 4,099 (4×1024 codes + PAD + offset + SEP)

Three processing pipelines handle different data types:

- **Pipeline A** (raw tokenization): Segments short clips (0.3–5s) directly. For pre-segmented datasets.
- **Pipeline B** (denoised long-chunk): Denoises with bandpass + spectral gating, segments into 30s chunks preserving natural pauses. For existing datasets.
- **Pipeline C** (SanctSound pilot): Bandpass-only (no spectral gating — it removes faint whale calls on low-SNR hydrophone data), per-chunk peak normalization, 30s chunks. For passive acoustic monitoring recordings.
- **Pipeline D** (SanctSound large-scale): Detection-guided selection (NOAA annotations, >80% humpback), whale-band variability filter (CV > 0.8), test-tone skip, stream-and-delete FLAC handling. Produces ~3.2B tokens from 4 Hawaii stations.

### Multi-Codebook (4CB) Tokenization

The LAC codec produces 14 codebooks via RVQ — codebook 0 captures coarse audio structure, subsequent codebooks add finer spectral detail. Using 4 codebooks gives significantly richer representation:

- **Interleaving**: `[cb1_t1, cb2_t1, cb3_t1, cb4_t1, cb1_t2, cb2_t2, ...]` — each timestep produces 4 tokens
- **Offsets**: CB0 = tokens 1–1025, CB1 = 1025–2049, CB2 = 2049–3073, CB3 = 3073–4097
- **Special tokens**: PAD = 0, SEP = 4098 (used between concatenated segments)
- **Sequence concatenation**: Segments from the same source are concatenated with SEP tokens, creating longer training sequences that span multiple vocalizations. This lets the model learn what sounds follow other sounds across clip boundaries.
- **Sliding windows**: Concatenated sequences are split into windows of up to 1024 tokens with 50% overlap.

## Model Architecture

GPT-style causal transformer decoder with:

- Token embedding + RoPE positional encoding
- Causal self-attention (Flash Attention via `scaled_dot_product_attention`)
- Sliding Window Attention (SWA) — optional, configurable window size and SWA:Full ratio
- Sparse Mixture of Experts (MoE) — optional, replaces all dense FFN layers with routed experts
- SwiGLU feed-forward network
- bf16 mixed precision training
- AdamW + cosine annealing with warmup

### Dense Presets

| Preset | Layers | Heads | d_model | d_ff | Params |
|--------|--------|-------|---------|------|--------|
| tiny | 6 | 4 | 256 | 1,024 | ~6.6M |
| small | 8 | 8 | 512 | 2,048 | ~34M |
| medium | 12 | 12 | 768 | 3,072 | ~113M |
| large | 16 | 16 | 1,024 | 4,096 | ~200M |
| xlarge | 24 | 16 | 1,280 | 5,120 | ~350M |

### SWA + MoE Presets

These presets replace most attention layers with **Sliding Window Attention** (local context, O(T×W) instead of O(T²)) and all FFN layers with **Mixture of Experts** (multiple smaller expert FFNs with top-K routing). This increases model capacity (total parameters) without proportional increase in per-token compute.

| Preset | Layers | Heads | d_model | Experts | expert_d_ff | Top-K | SWA Window | Full Attn Every | Total Params |
|--------|--------|-------|---------|---------|-------------|-------|------------|-----------------|--------------|
| small_swa_moe | 8 | 8 | 512 | 8 | 512 | 2 | 512 | 5th layer | ~59M |
| medium_swa_moe | 12 | 12 | 768 | 8 | 768 | 2 | 1,024 | 5th layer | ~199M |
| large_swa_moe | 16 | 16 | 1,024 | 8 | 1,024 | 2 | 1,024 | 5th layer | ~471M |

> **32K config overrides**: The `audio_medium_swa_moe_..._32k` config uses `n_experts=16` (375M total params) and `swa_window_size=2048` (~2.6s local attention at 9CB token rate). Both are set in the YAML — preset defaults are overridden per-run.

**Layer pattern** (example: `large_swa_moe`, 16 layers, `full_attention_every_n=5`):

| Layers | Attention | FFN |
|--------|-----------|-----|
| 0–3 | SWA (window=1024) | MoE (8 experts, top-2) |
| 4 | Full causal | MoE (8 experts, top-2) |
| 5–8 | SWA | MoE |
| 9 | Full causal | MoE |
| 10–13 | SWA | MoE |
| 14 | Full causal | MoE |
| 15 | SWA | MoE |

### SWA + MoE Configuration

All parameters are configurable via YAML config or `get_config()` overrides:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `swa_window_size` | 0 (disabled) | Sliding window size in tokens. 0 = all layers use full attention. |
| `full_attention_every_n` | 0 (disabled) | Every N-th layer uses full attention, rest use SWA. e.g. 5 = 4 SWA per 1 full, 7 = 6 SWA per 1 full. |
| `n_experts` | 1 (dense) | Number of experts per MoE layer. 1 = standard dense FFN. >1 = MoE on all FFN layers. |
| `moe_top_k` | 2 | Number of experts activated per token. |
| `expert_d_ff` | 0 (use d_ff) | Expert intermediate dimension. 0 = same size as `d_ff`. Smaller values = more experts with less per-expert capacity. |
| `moe_aux_weight` | 0.01 | Weight of the load-balancing auxiliary loss (prevents expert collapse). |

**VRAM usage** (batch_size=1, vocab_size=1026):

| Preset | seq_len=4096 | seq_len=8192 |
|--------|-------------|-------------|
| small_swa_moe (59M) | 2.0 GB | 4.9 GB |
| medium_swa_moe (199M) | 4.5 GB | 9.4 GB |
| large_swa_moe (471M) | 11.7 GB | OOM (16GB GPU) |

## Training Configurations

All configs are in `configs/`. Key configs:

| Config | Track | Model | Data | Key Settings |
|--------|-------|-------|------|--------------|
| `symbolic_tiny.yaml` | Symbolic | tiny | Individual coda sequences | LR 3e-4, batch 32, seq_len 128 |
| `symbolic_tiny_dialogue.yaml` | Symbolic | tiny | Multi-whale dialogues | LR 3e-4, batch 32, seq_len 256 |
| `audio_small.yaml` | Audio 1CB | small | DSWP only | LR 3e-4, batch 16, seq_len 512 |
| `audio_tiny_all.yaml` | Audio 1CB | tiny | All species (augmented) | LR 5e-4, batch 64, dropout 0.15 |
| `audio_small_all_aug.yaml` | Audio 1CB | small | All species (augmented) | LR 2e-4, batch 32, dropout 0.15 |
| `audio_small_sperm.yaml` | Audio 1CB | small | Sperm whale | LR 3e-4, batch 16, seq_len 512 |
| `audio_small_toothed.yaml` | Audio 1CB | small | Toothed cetaceans | LR 3e-4, batch 16, seq_len 512 |
| `audio_small_baleen.yaml` | Audio 1CB | small | Baleen whales | LR 3e-4, batch 16, seq_len 512 |
| `audio_tiny_all_4cb.yaml` | Audio 4CB | tiny | All species (4CB + concat) | LR 5e-4, batch 32, seq_len 1024, vocab 4099 |
| `audio_small_baleen_4cb.yaml` | Audio 4CB | small | Baleen whales (4CB + concat) | LR 2e-4, batch 8, seq_len 1024, vocab 4099 |
| `audio_small_all_4cb_ab.yaml` | Audio 4CB | small | All species, A+B quality filtered | LR 2e-4, batch 8, seq_len 1024, vocab 4099 |
| `audio_small_denoised_4cb.yaml` | Audio 4CB | small | Denoised long-chunk (30s) | LR 2e-4, batch 8, seq_len 1024, vocab 4099 |
| `audio_medium_sanctsound_humpback_4cb.yaml` | Audio 4CB | medium | SanctSound humpback (~3.2B tokens) | LR 2e-4, batch 2, grad_accum 4, seq_len 4096, vocab 4099 |
| `audio_large_sanctsound_humpback_4cb.yaml` | Audio 4CB | large | SanctSound humpback (~3.2B tokens) | LR 1.5e-4, batch 8, seq_len 4096, vocab 4099 |
| `audio_large_swa_moe_sanctsound_humpback_dac_9cb_10k.yaml` | Audio DAC 9CB | large_swa_moe | SanctSound humpback (~10.4B tokens) | LR 1e-4, batch 1 (eff 8), seq_len 10240, vocab 9219, SWA+MoE |
| `audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k.yaml` | Audio DAC 9CB | medium_swa_moe | SanctSound humpback (~10.4B tokens) | LR 1e-4, batch 1 (eff 8), seq_len 32768, 16 experts, SWA=2048, grad checkpointing |

Token-level augmentation (for audio track): random token noise (±1-3), token masking, and time stretching.

> **Checkpoint management**: The trainer keeps the N most recent step checkpoints (`save_top_k`, default 2) plus a separate `best_model.pt` that is always retained. Periodic saves (`save_interval`) rotate by step number — oldest checkpoint is deleted when the limit is exceeded. This prevents disk exhaustion during long runs.

## Project Structure

```
marine_mammals_communication/
├── configs/                          # YAML training configurations
│   ├── symbolic_tiny.yaml            # Symbolic coda sequences
│   ├── symbolic_tiny_dialogue.yaml   # Symbolic multi-whale dialogues
│   ├── audio_small.yaml              # Audio, DSWP-only (1CB)
│   ├── audio_tiny_all.yaml           # Audio, all species (tiny, 1CB)
│   ├── audio_small_all_aug.yaml      # Audio, all species (small + aug, 1CB)
│   ├── audio_small_sperm.yaml        # Audio, sperm whale group (1CB)
│   ├── audio_small_toothed.yaml      # Audio, toothed cetaceans (1CB)
│   ├── audio_small_baleen.yaml       # Audio, baleen whales (1CB)
│   ├── audio_tiny_all_4cb.yaml       # Audio, all species (tiny, 4CB + concat)
│   ├── audio_small_baleen_4cb.yaml   # Audio, baleen whales (small, 4CB + concat)
│   ├── audio_small_all_4cb_ab.yaml   # Audio, all species A+B quality (small, 4CB)
│   ├── audio_small_denoised_4cb.yaml # Audio, denoised long-chunk (small, 4CB)
│   ├── audio_medium_sanctsound_humpback_4cb.yaml # SanctSound humpback (medium, 4CB)
│   ├── audio_large_sanctsound_humpback_4cb.yaml  # SanctSound humpback (large, 4CB)
│   ├── audio_large_swa_moe_sanctsound_humpback_dac_9cb_10k.yaml # SanctSound humpback (large SWA+MoE, DAC 9CB)
│   └── audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k.yaml # SanctSound humpback (medium SWA+MoE 32K, DAC 9CB)
├── data/
│   ├── raw/                          # Downloaded datasets (not in git)
│   │   ├── ceti/                     # CETI annotation CSVs
│   │   ├── dswp/                     # DSWP sperm whale audio
│   │   ├── watkins/                  # Watkins Marine Mammal Sound Database
│   │   ├── esp_orcas/                # Earth Species Project orca calls
│   │   ├── orcasound/                # Orcasound hydrophone recordings
│   │   ├── mbari/                    # MBARI Pacific Sound segments
│   │   ├── dori_orcasound/           # DORI-Orcasound orca (small clips)
│   │   ├── dori_orcasound_full/     # DORI-Orcasound orca (1,585 FLAC, 26 hrs)
│   │   ├── humpback_zenodo/          # Humpback whale songs (Tsujii)
│   │   ├── right_whale/              # Right whale upcalls (Kaggle)
│   │   └── kw_pei/                   # Killer whale Prince Edward Islands
│   ├── denoised/                     # Denoised WAVs (not in git)
│   │   ├── dswp/                     # Denoised DSWP
│   │   ├── watkins/                  # Denoised Watkins
│   │   └── .../                      # (10 sources total)
│   ├── sanctsound/                   # SanctSound data (not in git)
│   │   ├── audio/hi01/               # Downloaded FLAC files
│   │   └── detections/               # Detection annotation CSVs
│   └── tokenized/                    # Tokenized .npy files (not in git)
│       ├── all/                      # All species combined (1CB)
│       ├── sperm_whale/              # Sperm whale tokens (1CB)
│       ├── toothed/                  # Toothed cetacean tokens (1CB)
│       ├── baleen/                   # Baleen whale tokens (1CB)
│       ├── all_4cb/                  # All species combined (4CB)
│       ├── all_4cb_ab/              # All species, A+B quality filtered (4CB)
│       ├── baleen_4cb/               # Baleen whale tokens (4CB)
│       ├── denoised_4cb/             # Denoised long-chunk tokens (4CB)
│       ├── sanctsound_4cb/           # SanctSound pilot tokens (4CB)
│       ├── sanctsound_humpback_4cb/ # SanctSound humpback tokens (4CB, ~497K files)
│       ├── sanctsound_humpback_dac/ # SanctSound humpback tokens (DAC 9CB, ~488K files)
│       └── sanctsound_orca_dac/    # SanctSound orca tokens (DAC 9CB, 8,657 files, ~190M tokens)
├── models/
│   ├── codec.pth                     # WhAM LAC codec weights (not in git)
│   ├── orca_detector.pt              # Trained OrcaDetectorCNN (not in git)
│   └── orca_detector_ft.pt           # Fine-tuned on SanctSound (not in git)
├── runs/                             # Training outputs (not in git)
│   ├── symbolic_tiny_coda/           # Symbolic coda model
│   ├── symbolic_tiny_dialogue/       # Symbolic dialogue model
│   ├── audio_small_dswp/             # Audio DSWP-only (1CB)
│   ├── audio_small_all/              # Audio all-species (small, 1CB)
│   ├── audio_tiny_all/               # Audio all-species (tiny, 1CB)
│   ├── audio_small_sperm/            # Audio sperm whale (1CB)
│   ├── audio_small_toothed/          # Audio toothed cetaceans (1CB)
│   ├── audio_small_baleen/           # Audio baleen whales (1CB)
│   ├── audio_tiny_all_4cb/           # Audio all-species (tiny, 4CB)
│   ├── audio_small_baleen_4cb/       # Audio baleen whales (small, 4CB)
│   ├── audio_small_all_4cb_ab/      # Audio all-species A+B quality (small, 4CB)
│   ├── audio_small_denoised_4cb/    # Audio denoised long-chunk (small, 4CB)
│   ├── audio_medium_sanctsound_humpback_4cb/ # SanctSound humpback (medium, 4CB)
│   ├── audio_large_sanctsound_humpback_4cb/ # SanctSound humpback (large, 4CB)
│   ├── codec_quality/                 # Codec reconstruction evaluation output
│   └── codec_comparison_processed/    # LAC vs DAC comparison audio files
├── scripts/
│   ├── download_data.py              # Download CETI + DSWP + codec pointer
│   ├── download_more_data.py         # Download MBARI + HuggingFace datasets
│   ├── tokenize_audio.py             # Tokenize DSWP audio (first codebook)
│   ├── tokenize_all_audio.py         # Tokenize all datasets into one dir
│   ├── organize_species.py           # Tokenize by species group
│   ├── train.py                      # Training CLI entry point
│   ├── train_all.sh                  # Train all configs sequentially
│   ├── evaluate.py                   # Evaluate symbolic models
│   ├── generate_all.py               # Generate audio from all trained models
│   ├── grade_audio_quality.py        # Grade audio quality (A-F) per segment
│   ├── denoise_all_audio.py          # Batch denoise all datasets (Pipeline B)
│   ├── denoise_medium.py             # Medium denoising functions
│   ├── tokenize_denoised_audio.py    # Tokenize denoised audio, 30s chunks (Pipeline B)
│   ├── download_sanctsound.py        # Download SanctSound FLAC files from GCS
│   ├── process_sanctsound.py         # SanctSound: bandpass + tokenize (Pipeline C)
│   ├── process_sanctsound_humpback.py # SanctSound large-scale humpback (Pipeline D)
│   ├── generate_prompted.py          # Generate continuations from real whale prompts
│   ├── eval_codec_quality.py         # LAC roundtrip reconstruction quality test
│   ├── eval_codec_comparison.py      # LAC vs DAC codec comparison
│   ├── process_sanctsound_orca.py    # SanctSound orca: annotation-guided Pipeline E
│   ├── train_orca_detector.py        # Train binary OrcaDetectorCNN
│   └── finetune_orca_detector_sanctsound.py # Fine-tune detector on SanctSound low-SNR data
├── src/
│   ├── data/
│   │   ├── symbolic_tokenizer.py     # CETI annotations → tokens
│   │   ├── dialogue_builder.py       # Reconstruct multi-whale dialogues
│   │   └── dataset.py                # PyTorch Dataset classes
│   ├── tokenizer/
│   │   ├── audio_tokenizer.py        # LAC codec wrapper (encode/decode)
│   │   └── dac_tokenizer.py          # DAC codec wrapper (encode/decode, 9 codebooks)
│   ├── model/
│   │   ├── config.py                 # Model size presets (tiny→xlarge)
│   │   └── transformer.py            # Causal transformer decoder
│   ├── detector/
│   │   └── orca_detector.py          # OrcaDetectorCNN + OrcaDetector inference wrapper
│   ├── training/
│   │   └── trainer.py                # Training loop (AdamW, cosine LR, bf16)
│   └── evaluation/
│       ├── metrics.py                # Perplexity, accuracy, sequence analysis
│       ├── visualize.py              # Training curves, coda distributions
│       ├── audio_player.py           # Audio playback utilities
│       └── round_trip.py             # Encode→decode quality evaluation
├── notebooks/                        # Jupyter notebooks (exploratory)
├── pyproject.toml                    # Project dependencies
├── .gitignore
└── README.md
```

## Full Reproduction

To reproduce all results from scratch:

```bash
# 1. Setup
git clone <repo-url> && cd marine_mammals_communication
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[audio-codec,dev]"

# 2. Download codec weights from Zenodo
#    https://zenodo.org/records/17633708 → extract codec.pth → models/

# 3. Download data
export PYTHONPATH=.
python3 scripts/download_data.py
python3 scripts/download_more_data.py
# + manual downloads listed above for Watkins, Orcasound, Humpback, Right Whale, etc.

# 4. Pipeline A: Tokenize raw audio
python3 scripts/tokenize_all_audio.py --n-codebooks 1   # → data/tokenized/all/
python3 scripts/tokenize_all_audio.py --n-codebooks 4   # → data/tokenized/all_4cb/
python3 scripts/organize_species.py --n-codebooks 4      # → data/tokenized/{..._4cb}/

# 5. Train 1CB + 4CB models
bash scripts/train_all.sh
python3 scripts/train.py configs/audio_tiny_all_4cb.yaml
python3 scripts/train.py configs/audio_small_baleen_4cb.yaml

# 6. Grade audio quality + train quality-filtered model
python3 scripts/grade_audio_quality.py
python3 scripts/tokenize_all_audio.py --n-codebooks 4 \
    --quality-csv data/audio_quality_grades.csv --min-grade B
python3 scripts/train.py configs/audio_small_all_4cb_ab.yaml

# 7. Pipeline B: Denoise + long-chunk tokenize + train
python3 scripts/denoise_all_audio.py                      # → data/denoised/
python3 scripts/tokenize_denoised_audio.py \
    --codec-path models/codec.pth --n-codebooks 4 \
    --quality-csv data/audio_quality_grades.csv           # → data/tokenized/denoised_4cb/
python3 scripts/train.py configs/audio_small_denoised_4cb.yaml

# 8. Pipeline C: SanctSound pilot (passive acoustic monitoring)
pip install google-cloud-storage
python3 scripts/download_sanctsound.py --station hi01 --deployment 1
python3 scripts/process_sanctsound.py --station hi01 --device cuda
# → data/tokenized/sanctsound_4cb/

# 9. Pipeline D: SanctSound large-scale humpback (~3.2B tokens)
python3 scripts/process_sanctsound_humpback.py
# → data/tokenized/sanctsound_humpback_4cb/
python3 scripts/train.py configs/audio_medium_sanctsound_humpback_4cb.yaml

# 10. Pipeline D (DAC 9CB re-tokenization) + train medium SWA+MoE 32K
python3 scripts/process_sanctsound_humpback.py --codec dac --save-2d
# → data/tokenized/sanctsound_humpback_dac/
python3 scripts/train.py configs/audio_medium_swa_moe_sanctsound_humpback_dac_9cb_32k.yaml

# 11. Pipeline E: SanctSound orca (Olympic Coast, 11 deployments)
python3 scripts/process_sanctsound_orca.py
# → data/tokenized/sanctsound_orca_dac/

# 12. Orca detector: train from scratch, then fine-tune on SanctSound
python3 scripts/train_orca_detector.py
python3 scripts/finetune_orca_detector_sanctsound.py

# 13. Generate audio samples
python3 scripts/generate_all.py --n-samples 5 --max-tokens 300

# 14. Evaluate symbolic models
python3 scripts/evaluate.py runs/symbolic_tiny_coda/best_model.pt --dataset-type coda
python3 scripts/evaluate.py runs/symbolic_tiny_dialogue/best_model.pt --dataset-type dialogue

# 15. Evaluate codec reconstruction quality
python3 scripts/eval_codec_quality.py --device cpu   # LAC only, various codebook counts
python3 scripts/eval_codec_comparison.py --device cpu # LAC vs DAC comparison
```

## Future Enhancements

### Temporal Chaining for Longer Context Training

The `chunk_scores.csv` sidecar file preserves each chunk's position within its source FLAC via `chunk_idx_in_flac`. Analysis shows **89.7% of tokenized chunks are temporally adjacent** — consecutive 30s windows from the same recording that both passed the heuristic filters. This enables chaining adjacent chunks into longer training examples without re-downloading or re-tokenizing:

| Context Length | Adjacent Chunks | Available Examples |
|---|---|---|
| 60s | 2 | ~249,000 |
| 90s | 3 | ~200,000 |
| 120s | 4 | ~166,000 |
| 150s | 5 | ~139,000 |
| 180s | 6 | ~118,000 |

The longest consecutive run is **124 chunks (62 minutes)** of unbroken humpback song. At 60s context with DAC 9CB (86.1 T/s), each example would be ~5,162 tokens per codebook. Implementation: at dataset loading time, group by `flac_name`, sort by `chunk_idx_in_flac`, find consecutive runs, and concatenate `.npy` arrays with optional SEP tokens. This would let the model learn longer-range song structure — humpback songs repeat themes over 10–30 minutes.

### Additional Targets

- Species-specific vs multi-species model comparison
- Hierarchical codebook modeling (predict coarse codebooks first, then refine)
- DolphinGemma integration (Google's dolphin communication model)

### Additional Data Sources

The NOAA GCS bucket (`gs://noaa-passive-bioacoustic`, anonymous access) contains **20+ TB** of passive acoustic data beyond the Hawaii stations already processed. All use the same FLAC format compatible with the existing pipeline.

**SanctSound — Other Sanctuaries** (same format, just change `--station`):

| Sanctuary | Stations | Est. Size | Key Species |
|-----------|----------|-----------|-------------|
| Stellwagen Bank NMS (MA) | sb01-03 | ~7 TB | Humpback, right whale, dolphins, fin |
| Olympic Coast NMS (WA) | oc01-04 | ~1.4 TB | Killer whale, humpback, fin, blue |
| Gray's Reef NMS (GA) | gr01-03 | ~1.7 TB | Right whale, humpback, dolphins |
| Channel Islands NMS (CA) | ci01-05 | ~2.9 TB | Humpback, fin, blue |
| Monterey Bay NMS (CA) | mb01-03 | ~1.3 TB | Humpback, blue, fin |
| Florida Keys NMS | fk01-04 | ~2 TB | Dolphins, fin, right whale |
| Papahanaumokuakea (HI) | pm01,02,05 | ~77+ GB | Humpback (GoogleAI detections available) |

**Other GCS Bucket Programs:**

| Program | Prefix | Est. Size | Species |
|---------|--------|-----------|---------|
| SWFSC CCES + PASCAL | `swfsc/audio/` | Multi-TB | Blue, fin, humpback, beaked (35 Pacific survey deployments) |
| Navy MBARC LMR | `navy/audio/` | Multi-TB | Arctic bowhead, SoCal blue/fin/humpback (59 deployments) |
| NEFSC | `nefsc/audio/` | ~760 GB | Right whale, humpback (Atlantic coast) |
| SEFSC Gulf of Mexico | `sefsc/audio/gomex/` | ~184 GB | Sperm whale, beaked, Bryde's |
| PIFSC 200kHz | `pifsc/audio/pipan_200/` | Multi-TB | Humpback, sperm whale (Hawaii HARP) |
| NRS (Noise Reference) | `nrs/audio/` | ~3.6 TB | Various (12 stations, multi-year) |
| DCLDE 2027 Killer Whales | `dclde/` | ~1.3 TB | Orca from 8 sources (annotated, CC-BY-4.0) |

**External Sources:**

| Source | Access | Est. Size | Notes |
|--------|--------|-----------|-------|
| MBARI Pacific Sound | AWS S3 `pacific-sound-16khz` (anonymous) | ~16 TB | 12 continuous years, 16kHz — needs resample or native-rate DAC |
| Ocean Networks Canada | `data.oceannetworks.ca` (free account) | Multi-TB | Strait of Georgia orca/humpback hydrophones |

**Priority order:** OC02 (orca+humpback) → PM01/02 (more Hawaii humpback) → SB01-03 (Stellwagen, 7 TB) → DCLDE 2027 killer whales (annotated).

## References

- Sharma, P. et al. "An automatic approach for learning sperm whale codas using a large audio recording dataset." *Nature Communications* 15, 3194 (2024). [doi:10.1038/s41467-024-47221-8](https://doi.org/10.1038/s41467-024-47221-8)
- Flores Garcia, H. et al. "WhAM: Whale Audio Masking for marine mammal audio synthesis." *NeurIPS 2025*. [Zenodo: 10.5281/zenodo.17633708](https://doi.org/10.5281/zenodo.17633708)
- Watkins Marine Mammal Sound Database. Woods Hole Oceanographic Institution. [whalesounds](https://cis.whoi.edu/science/B/whalesounds/)
- MBARI Pacific Sound. Monterey Bay Aquarium Research Institute. [AWS Open Data](https://registry.opendata.aws/pacific-sound/)

## License

The code in this repository is available under the MIT License. The datasets have their own licenses — see each data source for details. DSWP is CC-BY-4.0.
