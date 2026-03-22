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
| **Baleen whales (small, 4CB)** | **Mysticeti** | **16,663** | **35.7M** | **2.70** | **14.8** | **Early stopped epoch 34, ~107 min** |

> **Note**: Val loss is not directly comparable between 1CB and 4CB models — the 4CB vocabulary is 4x larger (4099 vs 1026), making per-token prediction harder. The real comparison is in generated audio quality: 4CB captures finer spectral detail that 1CB misses.

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

### 4. Tokenize audio

```bash
export PYTHONPATH=.

# === 1 Codebook (original) ===
# Tokenize all datasets into data/tokenized/all/
python3 scripts/tokenize_all_audio.py --n-codebooks 1

# Organize by species group (toothed/baleen/sperm_whale)
python3 scripts/organize_species.py --n-codebooks 1

# === 4 Codebooks (richer representation) ===
# Tokenize all datasets into data/tokenized/all_4cb/
python3 scripts/tokenize_all_audio.py --n-codebooks 4

# Organize by species group into *_4cb/ directories
python3 scripts/organize_species.py --n-codebooks 4
```

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
| [MBARI Pacific Sound](https://registry.opendata.aws/pacific-sound/) | Various (hydrophone) | 4 | 40 min | 16 kHz | ~200 |
| [DORI-Orcasound](https://huggingface.co/datasets/DORI-SRKW/DORI-Orcasound) | Orca (SRKW) | 1,585 | ~26 hrs | varies | ~4,200 |
| [Humpback Songs (Tsujii)](https://zenodo.org/records/14862938) | Humpback whale | 6 | 60 min | 44.1 kHz | ~700 |
| [Right Whale Upcalls](https://www.kaggle.com/c/whale-detection-challenge) | Right whale | 12,000 | ~100 hrs | 2 kHz | ~24,000 |
| [KW Prince Edward Islands](https://zenodo.org/records/7712582) | Killer whale | 1 | 14 min | 96 kHz | ~170 |

**Total**: ~39,400 tokenized segments, ~3.2M tokens.

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

Long recordings are segmented using energy-based silence detection into clips of 0.3–5 seconds before tokenization.

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
│   └── audio_small_baleen_4cb.yaml   # Audio, baleen whales (small, 4CB + concat)
├── data/
│   ├── raw/                          # Downloaded datasets (not in git)
│   │   ├── ceti/                     # CETI annotation CSVs
│   │   ├── dswp/                     # DSWP sperm whale audio
│   │   ├── watkins/                  # Watkins Marine Mammal Sound Database
│   │   ├── esp_orcas/                # Earth Species Project orca calls
│   │   ├── orcasound/                # Orcasound hydrophone recordings
│   │   ├── mbari/                    # MBARI Pacific Sound segments
│   │   ├── dori_orcasound/           # DORI-Orcasound orca FLAC files
│   │   ├── humpback_zenodo/          # Humpback whale songs (Tsujii)
│   │   ├── right_whale/              # Right whale upcalls (Kaggle)
│   │   └── kw_pei/                   # Killer whale Prince Edward Islands
│   └── tokenized/                    # Tokenized .npy files (not in git)
│       ├── all/                      # All species combined (1CB)
│       ├── sperm_whale/              # Sperm whale tokens (1CB)
│       ├── toothed/                  # Toothed cetacean tokens (1CB)
│       ├── baleen/                   # Baleen whale tokens (1CB)
│       ├── all_4cb/                  # All species combined (4CB)
│       └── baleen_4cb/               # Baleen whale tokens (4CB)
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
│   └── audio_small_baleen_4cb/       # Audio baleen whales (small, 4CB)
├── scripts/
│   ├── download_data.py              # Download CETI + DSWP + codec pointer
│   ├── download_more_data.py         # Download MBARI + HuggingFace datasets
│   ├── tokenize_audio.py             # Tokenize DSWP audio (first codebook)
│   ├── tokenize_all_audio.py         # Tokenize all datasets into one dir
│   ├── organize_species.py           # Tokenize by species group
│   ├── train.py                      # Training CLI entry point
│   ├── train_all.sh                  # Train all configs sequentially
│   ├── evaluate.py                   # Evaluate symbolic models
│   └── generate_all.py               # Generate audio from all trained models
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

# 4. Tokenize (1 codebook)
python3 scripts/tokenize_all_audio.py --n-codebooks 1   # → data/tokenized/all/
python3 scripts/organize_species.py --n-codebooks 1      # → data/tokenized/{sperm_whale,toothed,baleen}/

# 5. Tokenize (4 codebooks)
python3 scripts/tokenize_all_audio.py --n-codebooks 4   # → data/tokenized/all_4cb/
python3 scripts/organize_species.py --n-codebooks 4      # → data/tokenized/{sperm_whale_4cb,toothed_4cb,baleen_4cb}/

# 6. Train 1CB models
bash scripts/train_all.sh

# 7. Train 4CB models
python3 scripts/train.py configs/audio_tiny_all_4cb.yaml
python3 scripts/train.py configs/audio_small_baleen_4cb.yaml

# 8. Generate audio samples
python3 scripts/generate_all.py --n-samples 5 --max-tokens 300

# 9. Evaluate symbolic models
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
