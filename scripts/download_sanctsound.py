#!/usr/bin/env python3
"""Download SanctSound audio files from Google Cloud Storage.

Downloads FLAC files for specified stations/deployments, optionally filtering
by detection timestamps to avoid downloading silent recordings.
"""

import argparse
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

from google.cloud import storage
from tqdm import tqdm


def parse_flac_timestamp(filename):
    """Extract datetime from FLAC filename like SanctSound_HI01_01_671129638_20181115T000002Z.flac"""
    parts = filename.replace('.flac', '').split('_')
    for part in parts:
        if len(part) >= 15 and 'T' in part:
            try:
                return datetime.strptime(part.rstrip('Z'), '%Y%m%dT%H%M%S')
            except ValueError:
                continue
    return None


def download_deployment(station, deployment_num, output_dir, max_files=None,
                        skip_existing=True):
    """Download FLAC files from a SanctSound deployment."""
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket("noaa-passive-bioacoustic")

    deployment_name = f"sanctsound_{station}_{deployment_num:02d}"
    prefix = f"sanctsound/audio/{station}/{deployment_name}/audio/"

    out_dir = Path(output_dir) / station
    out_dir.mkdir(parents=True, exist_ok=True)

    # List all FLAC files
    print(f"Listing files in {prefix}...")
    blobs = sorted(bucket.list_blobs(prefix=prefix), key=lambda b: b.name)
    flac_blobs = [b for b in blobs if b.name.endswith('.flac')]
    print(f"  Found {len(flac_blobs)} FLAC files")

    if max_files:
        flac_blobs = flac_blobs[:max_files]
        print(f"  Downloading first {max_files} files")

    downloaded = 0
    skipped = 0
    total_bytes = 0

    for blob in tqdm(flac_blobs, desc=f"  {station}/{deployment_num:02d}"):
        fname = blob.name.split('/')[-1]
        local_path = out_dir / fname

        if skip_existing and local_path.exists() and local_path.stat().st_size > 0:
            skipped += 1
            continue

        blob.download_to_filename(str(local_path))
        downloaded += 1
        total_bytes += blob.size

    print(f"  Downloaded: {downloaded}, Skipped: {skipped}, "
          f"Size: {total_bytes/1e9:.1f} GB")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download SanctSound audio")
    parser.add_argument("--station", type=str, default="hi01",
                        help="Station code (e.g., hi01, oc02, ci05)")
    parser.add_argument("--deployment", type=int, default=1,
                        help="Deployment number")
    parser.add_argument("--output-dir", type=str,
                        default="data/sanctsound/audio")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max number of files to download")
    args = parser.parse_args()

    download_deployment(
        args.station, args.deployment, args.output_dir,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
