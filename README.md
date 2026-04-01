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

```bash
export PYTHONPATH=.

# Process all Hawaii stations (hi01, hi03, hi04, hi05)
python3 scripts/process_sanctsound_humpback.py

# Process a specific station and deployment
python3 scripts/process_sanctsound_humpback.py --station hi04 --deployment 2

# Dry run (list qualifying FLACs without downloading)
python3 scripts/process_sanctsound_humpback.py --station hi05 --dry-run
# → data/tokenized/sanctsound_humpback_4cb/
```

The pipeline per file:
1. Load FLAC, convert to mono, resample to 44,100 Hz
2. Skip first 5 seconds (test tone present in all SanctSound recordings)
3. Bandpass filter 80 Hz – 20 kHz
4. Segment into ≤30s chunks, remove silence >4s
5. Per-chunk peak normalization to 0.9
6. **Whale-band variability filter**: compute coefficient of variation of RMS energy in 200–4000 Hz band (0.5s frames). Keep chunks with CV > 0.8 (whale songs ~1.5–3.5, ocean noise ~0.3–0.5)
7. Loudness normalization to -20 LUFS (**no spectral gating** — it removes faint whale calls)
8. Tokenize with LAC codec (4 codebooks)

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
python3 scripts/generate_all.py --n-samples 5 --max-tokens 300 --temperature 0.9
```

Generated WAV files are saved to `runs/<model>/generated/`.

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

The full SanctSound dataset contains ~25,000 FLAC files across 31 stations. Future expansion targets: OC02 (Olympic Coast orcas), PM stations (Pacific humpback/orca).

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
- bf16 mixed precision training
- AdamW + cosine annealing with warmup

### Size Presets

| Preset | Layers | Heads | d_model | d_ff | Params |
|--------|--------|-------|---------|------|--------|
| tiny | 6 | 4 | 256 | 1,024 | ~6.6M |
| small | 8 | 8 | 512 | 2,048 | ~34M |
| medium | 12 | 12 | 768 | 3,072 | ~113M |
| large | 16 | 16 | 1,024 | 4,096 | ~200M |
| xlarge | 24 | 16 | 1,280 | 5,120 | ~350M |

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

Token-level augmentation (for audio track): random token noise (±1-3), token masking, and time stretching.

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
│   └── audio_large_sanctsound_humpback_4cb.yaml  # SanctSound humpback (large, 4CB)
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
│       └── sanctsound_humpback_4cb/ # SanctSound humpback tokens (4CB, ~497K files)
├── models/
│   └── codec.pth                     # WhAM LAC codec weights (not in git)
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
│   └── audio_medium_sanctsound_humpback_4cb/ # SanctSound humpback (medium, 4CB)
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
│   └── process_sanctsound_humpback.py # SanctSound large-scale humpback (Pipeline D)
├── src/
│   ├── data/
│   │   ├── symbolic_tokenizer.py     # CETI annotations → tokens
│   │   ├── dialogue_builder.py       # Reconstruct multi-whale dialogues
│   │   └── dataset.py                # PyTorch Dataset classes
│   ├── tokenizer/
│   │   └── audio_tokenizer.py        # LAC codec wrapper (encode/decode)
│   ├── model/
│   │   ├── config.py                 # Model size presets (tiny→xlarge)
│   │   └── transformer.py            # Causal transformer decoder
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

# 10. Generate audio samples
python3 scripts/generate_all.py --n-samples 5 --max-tokens 300

# 11. Evaluate symbolic models
python3 scripts/evaluate.py runs/symbolic_tiny_coda/best_model.pt --dataset-type coda
python3 scripts/evaluate.py runs/symbolic_tiny_dialogue/best_model.pt --dataset-type dialogue
```

## References

- Sharma, P. et al. "An automatic approach for learning sperm whale codas using a large audio recording dataset." *Nature Communications* 15, 3194 (2024). [doi:10.1038/s41467-024-47221-8](https://doi.org/10.1038/s41467-024-47221-8)
- Flores Garcia, H. et al. "WhAM: Whale Audio Masking for marine mammal audio synthesis." *NeurIPS 2025*. [Zenodo: 10.5281/zenodo.17633708](https://doi.org/10.5281/zenodo.17633708)
- Watkins Marine Mammal Sound Database. Woods Hole Oceanographic Institution. [whalesounds](https://cis.whoi.edu/science/B/whalesounds/)
- MBARI Pacific Sound. Monterey Bay Aquarium Research Institute. [AWS Open Data](https://registry.opendata.aws/pacific-sound/)

## License

The code in this repository is available under the MIT License. The datasets have their own licenses — see each data source for details. DSWP is CC-BY-4.0.
