#!/usr/bin/env python3
"""Download all datasets for the marine mammal communication project."""

import os
import urllib.request
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
CETI_DIR = DATA_RAW / "ceti"
DSWP_DIR = DATA_RAW / "dswp"

CETI_BASE_URL = "https://raw.githubusercontent.com/Project-CETI/sw-combinatoriality/main/data"
CETI_FILES = [
    "DominicaCodas.csv",
    "sperm-whale-dialogues.csv",
    "mean_codas.p",
    "ornaments.p",
    "rhythms.p",
    "tempos-dict.p",
]


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return
    print(f"  Downloading: {dest.name}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Done: {dest.name} ({dest.stat().st_size:,} bytes)")


def download_ceti_annotations() -> None:
    print("\n=== Downloading CETI annotation data ===")
    CETI_DIR.mkdir(parents=True, exist_ok=True)
    for filename in CETI_FILES:
        url = f"{CETI_BASE_URL}/{filename}"
        download_file(url, CETI_DIR / filename)


def download_dswp_audio() -> None:
    print("\n=== Downloading DSWP audio dataset from HuggingFace ===")
    DSWP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        return

    print("  Loading dataset orrp/DSWP from HuggingFace...")
    ds = load_dataset("orrp/DSWP", trust_remote_code=True)
    print(f"  Dataset loaded: {ds}")

    # Save audio files
    audio_dir = DSWP_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)

    for split_name in ds:
        split = ds[split_name]
        print(f"  Processing split '{split_name}': {len(split)} samples")
        for i, sample in enumerate(split):
            audio = sample.get("audio")
            if audio is None:
                continue
            # Save as WAV
            import soundfile as sf
            import numpy as np

            array = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            out_path = audio_dir / f"{split_name}_{i:05d}.wav"
            if not out_path.exists():
                sf.write(str(out_path), array, sr)

        print(f"  Saved {len(split)} audio files to {audio_dir}")


def download_codec_weights() -> None:
    """Download WhAM's LAC codec weights from Zenodo."""
    print("\n=== Downloading LAC codec weights ===")
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    codec_path = models_dir / "codec.pth"
    if codec_path.exists():
        print(f"  Already exists: {codec_path}")
        return

    print("  NOTE: codec.pth must be downloaded from https://zenodo.org/records/17633708")
    print("  Download the archive, extract codec.pth, and place it in models/")
    print("  (The Zenodo archive contains: codec.pth, coarse.pth, c2f.pth)")


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")

    download_ceti_annotations()
    download_dswp_audio()
    download_codec_weights()

    print("\n=== Done ===")
