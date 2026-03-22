#!/usr/bin/env python3
"""Download additional marine mammal audio data.

Sources:
1. MBARI Pacific Sound — 10-min segments from multiple months (correct S3 key format)
2. HuggingFace — WhaleSounds, marine_ocean_mammal_sound, DORI-Orcasound
"""

import io
import struct
import wave
from pathlib import Path

import numpy as np


def download_mbari_segments(output_dir, n_segments=20):
    """Download MBARI Pacific Sound segments from S3.

    Keys are: YYYY/MM/MARS-YYYYMMDDTHHMMSSZ-16kHz.wav (each ~4.1GB = 24 hrs).
    We download 10-min chunks from the middle of different days.
    """
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        print("  pip install boto3 first")
        return 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED),
                       region_name="us-west-2")
    bucket = "pacific-sound-16khz"
    sr = 16000
    segment_duration = 600  # 10 minutes
    segment_bytes = segment_duration * sr * 2  # 16-bit audio

    # Sample from different seasons/months for diversity
    # Whale activity varies by season: humpbacks peak Oct-Feb, blue whales Aug-Nov
    dates_keys = [
        # Humpback season
        ("20151015", "2015/10/MARS-20151015T000000Z-16kHz.wav"),
        ("20151115", "2015/11/MARS-20151115T000000Z-16kHz.wav"),
        ("20151215", "2015/12/MARS-20151215T000000Z-16kHz.wav"),
        ("20160115", "2016/01/MARS-20160115T000000Z-16kHz.wav"),
        ("20160215", "2016/02/MARS-20160215T000000Z-16kHz.wav"),
        # Blue whale season
        ("20150815", "2015/08/MARS-20150815T000000Z-16kHz.wav"),
        ("20150915", "2015/09/MARS-20150915T000000Z-16kHz.wav"),
        ("20151001", "2015/10/MARS-20151001T000000Z-16kHz.wav"),
        # Different years
        ("20160815", "2016/08/MARS-20160815T000000Z-16kHz.wav"),
        ("20161015", "2016/10/MARS-20161015T000000Z-16kHz.wav"),
        ("20161115", "2016/11/MARS-20161115T000000Z-16kHz.wav"),
        ("20170115", "2017/01/MARS-20170115T000000Z-16kHz.wav"),
        ("20170315", "2017/03/MARS-20170315T000000Z-16kHz.wav"),
        ("20170815", "2017/08/MARS-20170815T000000Z-16kHz.wav"),
        ("20171015", "2017/10/MARS-20171015T000000Z-16kHz.wav"),
        ("20180115", "2018/01/MARS-20180115T000000Z-16kHz.wav"),
        ("20180815", "2018/08/MARS-20180815T000000Z-16kHz.wav"),
        ("20181015", "2018/10/MARS-20181015T000000Z-16kHz.wav"),
        ("20190115", "2019/01/MARS-20190115T000000Z-16kHz.wav"),
        ("20190815", "2019/08/MARS-20190815T000000Z-16kHz.wav"),
    ]

    downloaded = 0
    for date, key in dates_keys[:n_segments]:
        out_path = output_dir / f"MARS-{date}-10min.wav"

        if out_path.exists():
            print(f"  Already exists: {out_path.name}")
            downloaded += 1
            continue

        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            file_size = head["ContentLength"]

            # Sample from different offsets within the day for variety
            # Offset into hours 6-18 (daytime for better chance of bio activity)
            import random
            random.seed(hash(date))
            hour_offset = random.randint(6, 17)
            start_offset = max(44, hour_offset * 3600 * sr * 2)
            end_offset = min(start_offset + segment_bytes, file_size - 1)

            resp = s3.get_object(
                Bucket=bucket, Key=key,
                Range=f"bytes={start_offset}-{end_offset}"
            )
            raw_data = resp["Body"].read()

            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(raw_data)

            print(f"  Downloaded: {out_path.name} ({len(raw_data) / 1e6:.1f} MB)")
            downloaded += 1

        except Exception as e:
            print(f"  Failed {key}: {e}")

    return downloaded


def download_whale_sounds_hf(output_dir):
    """Download WhaleSounds dataset from HuggingFace (monster-monash/WhaleSounds)."""
    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError:
        print("  pip install datasets soundfile first")
        return 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("*.wav"))
    if len(existing) > 10:
        print(f"  Already have {len(existing)} files in {output_dir}")
        return len(existing)

    try:
        print("  Loading monster-monash/WhaleSounds...")
        ds = load_dataset("monster-monash/WhaleSounds", split="train",
                          trust_remote_code=True)
        count = 0
        for i, item in enumerate(ds):
            if "audio" in item:
                audio_data = item["audio"]
                arr = np.array(audio_data["array"], dtype=np.float32)
                sr = audio_data["sampling_rate"]
                label = item.get("label", "unknown")
                out_path = output_dir / f"whale_{label}_{i:05d}.wav"
                sf.write(str(out_path), arr, sr)
                count += 1
        print(f"  Downloaded {count} files")
        return count
    except Exception as e:
        print(f"  Failed: {e}")
        return 0


def download_marine_ocean_mammal_hf(output_dir):
    """Download ardavey/marine_ocean_mammal_sound from HuggingFace."""
    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError:
        return 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("*.wav"))
    if len(existing) > 10:
        print(f"  Already have {len(existing)} files in {output_dir}")
        return len(existing)

    try:
        print("  Loading ardavey/marine_ocean_mammal_sound...")
        ds = load_dataset("ardavey/marine_ocean_mammal_sound", split="train",
                          trust_remote_code=True)
        count = 0
        for i, item in enumerate(ds):
            if "audio" in item:
                audio_data = item["audio"]
                arr = np.array(audio_data["array"], dtype=np.float32)
                sr = audio_data["sampling_rate"]
                label = item.get("label", item.get("class", "unknown"))
                out_path = output_dir / f"marine_{label}_{i:05d}.wav"
                sf.write(str(out_path), arr, sr)
                count += 1
        print(f"  Downloaded {count} files")
        return count
    except Exception as e:
        print(f"  Failed: {e}")
        return 0


def download_dori_orcasound_hf(output_dir):
    """Download DORI-SRKW/DORI-Orcasound dataset."""
    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError:
        return 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("*.wav"))
    if len(existing) > 10:
        print(f"  Already have {len(existing)} files in {output_dir}")
        return len(existing)

    try:
        print("  Loading DORI-SRKW/DORI-Orcasound...")
        ds = load_dataset("DORI-SRKW/DORI-Orcasound", split="train",
                          trust_remote_code=True)
        count = 0
        for i, item in enumerate(ds):
            if "audio" in item:
                audio_data = item["audio"]
                arr = np.array(audio_data["array"], dtype=np.float32)
                sr = audio_data["sampling_rate"]
                out_path = output_dir / f"dori_orca_{i:05d}.wav"
                sf.write(str(out_path), arr, sr)
                count += 1
        print(f"  Downloaded {count} files")
        return count
    except Exception as e:
        print(f"  Failed: {e}")
        return 0


def main():
    print("=== Downloading additional marine mammal audio ===\n")

    # 1. More MBARI
    print("--- MBARI Pacific Sound (20 x 10-min segments from different months/years) ---")
    n = download_mbari_segments("data/raw/mbari", n_segments=20)
    print(f"  Total MBARI files: {n}\n")

    # 2. HuggingFace: WhaleSounds
    print("--- HuggingFace: monster-monash/WhaleSounds ---")
    download_whale_sounds_hf("data/raw/hf_whale_sounds")

    # 3. HuggingFace: marine ocean mammal sound
    print("\n--- HuggingFace: ardavey/marine_ocean_mammal_sound ---")
    download_marine_ocean_mammal_hf("data/raw/hf_marine_mammal")

    # 4. HuggingFace: DORI Orcasound (orca vocalizations)
    print("\n--- HuggingFace: DORI-SRKW/DORI-Orcasound ---")
    download_dori_orcasound_hf("data/raw/hf_dori_orcasound")

    print("\n=== Download complete ===")
    # Print summary
    for d in ["data/raw/mbari", "data/raw/hf_whale_sounds",
              "data/raw/hf_marine_mammal", "data/raw/hf_dori_orcasound"]:
        p = Path(d)
        if p.exists():
            n = len(list(p.rglob("*.wav")))
            print(f"  {d}: {n} files")


if __name__ == "__main__":
    main()
