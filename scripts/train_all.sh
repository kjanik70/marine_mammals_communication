#!/bin/bash
# Train all model configurations sequentially
set -e
export PYTHONPATH=.

echo "========================================"
echo "Training all model configurations"
echo "========================================"

# 1. Tiny model on all data (with augmentation) — fastest, baseline
echo -e "\n>>> [1/5] Tiny model on all data (augmented)"
python3 scripts/train.py configs/audio_tiny_all.yaml

# 2. Small model on all data (with augmentation + lower LR)
echo -e "\n>>> [2/5] Small model on all data (augmented + low LR)"
python3 scripts/train.py configs/audio_small_all_aug.yaml

# 3. Sperm whale specialist model
echo -e "\n>>> [3/5] Small model on sperm whale data"
python3 scripts/train.py configs/audio_small_sperm.yaml

# 4. Toothed cetacean model
echo -e "\n>>> [4/5] Small model on toothed cetaceans"
python3 scripts/train.py configs/audio_small_toothed.yaml

# 5. Baleen whale model
echo -e "\n>>> [5/5] Small model on baleen whales"
python3 scripts/train.py configs/audio_small_baleen.yaml

echo -e "\n========================================"
echo "All training runs complete!"
echo "========================================"
echo "Results in:"
echo "  runs/audio_tiny_all/"
echo "  runs/audio_small_all_aug/"
echo "  runs/audio_small_sperm/"
echo "  runs/audio_small_toothed/"
echo "  runs/audio_small_baleen/"
