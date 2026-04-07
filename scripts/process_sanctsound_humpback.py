#!/usr/bin/env python3
"""Download, process, and tokenize SanctSound Hawaii humpback whale data.

Unified pipeline: download FLACs → process → tokenize → delete FLACs.
Processes one deployment at a time to manage disk space.

Pipeline per file:
1. Load FLAC, convert to mono
2. Resample to 44100 Hz
3. Skip first 5 seconds (test tone)
4. Bandpass filter (80 Hz – 20 kHz)
5. Segment into ≤30s chunks, remove >4s silence
6. Per-chunk peak normalization to 0.9
7. Whale-band variability filter (keep chunks with vocalizations)
8. Loudness normalization to -20 LUFS (no spectral gating — Pipeline B)
9. Tokenize with LAC codec (4 codebooks)
10. Save as .npy token files

Usage:
    # Process all stations and deployments:
    PYTHONPATH=. python3 scripts/process_sanctsound_humpback.py

    # Process specific station:
    PYTHONPATH=. python3 scripts/process_sanctsound_humpback.py --station hi05

    # Dry run (list files, no download):
    PYTHONPATH=. python3 scripts/process_sanctsound_humpback.py --station hi05 --dry-run
"""

import argparse
import csv
import io
import json
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfilt
from tqdm import tqdm


# --- Audio processing functions ---

def bandpass_audio(audio, sr, low_hz=80, high_hz=20000):
    """Bandpass filter."""
    nyq = sr / 2
    low = min(low_hz / nyq, 0.95)
    high = min(high_hz / nyq, 0.95)
    if low < high:
        sos = butter(5, [low, high], btype='band', output='sos')
        audio = sosfilt(sos, audio).astype(np.float32)
    return audio


def loudness_normalize(audio, sr, target_lufs=-20.0):
    """Loudness normalization (ITU-R BS.1770)."""
    meter = pyln.Meter(sr)
    current_loudness = meter.integrated_loudness(audio)
    if np.isfinite(current_loudness):
        audio = pyln.normalize.loudness(audio, current_loudness, target_lufs)
    else:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.5 / peak)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def whale_band_score(chunk, sr):
    """Score a chunk for whale vocalization presence.

    Computes coefficient of variation of RMS energy in the 200-4000 Hz band.
    Whale songs produce high CV (variable energy), ocean noise is steady (low CV).

    Returns:
        float: CV score. Whale songs typically 1.5-3.5, ocean noise 0.3-0.5.
    """
    nyq = sr / 2
    sos = butter(5, [200 / nyq, 4000 / nyq], btype='band', output='sos')
    whale_band = sosfilt(sos, chunk).astype(np.float32)
    frame_len = int(0.5 * sr)
    n_frames = len(whale_band) // frame_len
    if n_frames < 3:
        return 0.0
    rms = np.array([
        np.sqrt(np.mean(whale_band[j * frame_len:(j + 1) * frame_len] ** 2))
        for j in range(n_frames)
    ])
    return float(np.std(rms) / max(np.mean(rms), 1e-10))


def whale_energy_ratio(chunk, sr, whale_low=200, whale_high=4000):
    """Fraction of spectral energy in whale frequency band.

    Whale songs concentrate 60-90% of energy in 200-4000 Hz.
    Broadband ocean noise is typically 20-40%.

    Returns:
        float: Ratio of whale-band energy to total energy (0-1).
    """
    S = np.abs(librosa.stft(chunk, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    whale_mask = (freqs >= whale_low) & (freqs <= whale_high)
    whale_energy = np.sum(S[whale_mask, :] ** 2)
    total_energy = np.sum(S ** 2) + 1e-10
    return float(whale_energy / total_energy)


def whale_band_rms(chunk, sr, low=200, high=4000):
    """RMS energy in whale frequency band.

    Rejects near-silence chunks even if CV is high (e.g., a single click
    in 30s of silence can have high CV but low overall energy).

    Returns:
        float: RMS energy in whale band.
    """
    nyq = sr / 2
    sos = butter(5, [low / nyq, high / nyq], btype='band', output='sos')
    whale_band = sosfilt(sos, chunk).astype(np.float32)
    return float(np.sqrt(np.mean(whale_band ** 2)))


def segment_audio_long(audio, sr, max_duration=30.0, min_duration=2.0,
                       max_silence=4.0, replacement_silence=0.5):
    """Segment into ≤30s chunks, removing long silence, keeping short pauses."""
    duration = len(audio) / sr
    if duration < min_duration:
        return []
    if duration <= max_duration:
        return [audio]

    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    n_frames = max(1, (len(audio) - frame_length) // hop_length)
    energy = np.array([
        np.sqrt(np.mean(audio[i * hop_length:i * hop_length + frame_length] ** 2))
        for i in range(n_frames)
    ])

    silence_threshold = max(np.percentile(energy, 25), 1e-6)
    is_silent = energy < silence_threshold

    # Find silence regions
    silence_regions = []
    in_silence = False
    start = 0
    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            start = i * hop_length
            in_silence = True
        elif not silent and in_silence:
            end = i * hop_length
            dur = (end - start) / sr
            silence_regions.append((start, end, dur))
            in_silence = False
    if in_silence:
        end = len(energy) * hop_length
        dur = (end - start) / sr
        silence_regions.append((start, end, dur))

    # Remove long silence
    long_silences = [r for r in silence_regions if r[2] > max_silence]
    if long_silences:
        replacement_samples = int(replacement_silence * sr)
        pieces = []
        prev_end = 0
        for s, e, d in sorted(long_silences):
            if s > prev_end:
                pieces.append(audio[prev_end:s])
            pieces.append(np.zeros(replacement_samples, dtype=audio.dtype))
            prev_end = e
        if prev_end < len(audio):
            pieces.append(audio[prev_end:])
        if pieces:
            audio = np.concatenate(pieces)

    if len(audio) / sr <= max_duration:
        if len(audio) / sr >= min_duration:
            return [audio]
        return []

    # Recompute silence for splitting
    n_frames2 = max(1, (len(audio) - frame_length) // hop_length)
    energy2 = np.array([
        np.sqrt(np.mean(audio[i * hop_length:i * hop_length + frame_length] ** 2))
        for i in range(n_frames2)
    ])
    is_silent2 = energy2 < silence_threshold
    silence_regions2 = []
    in_silence2 = False
    for i, silent in enumerate(is_silent2):
        if silent and not in_silence2:
            start = i * hop_length
            in_silence2 = True
        elif not silent and in_silence2:
            end = i * hop_length
            dur = (end - start) / sr
            silence_regions2.append((start, end, dur))
            in_silence2 = False

    max_samples = int(max_duration * sr)
    min_samples = int(min_duration * sr)
    split_candidates = [(s + e) // 2 for s, e, d in silence_regions2 if d >= 0.1]

    chunks = []
    chunk_start = 0
    total = len(audio)
    while chunk_start < total:
        remaining = total - chunk_start
        if remaining <= max_samples:
            if remaining >= min_samples:
                chunks.append(audio[chunk_start:])
            break
        chunk_end_max = chunk_start + max_samples
        best_split = None
        for sp in reversed(split_candidates):
            if chunk_start + min_samples <= sp <= chunk_end_max:
                best_split = sp
                break
        if best_split is None:
            best_split = chunk_end_max
        chunk = audio[chunk_start:best_split]
        if len(chunk) >= min_samples:
            chunks.append(chunk)
        chunk_start = best_split

    return chunks


# --- Whale song detector (Pass 2) ---

def load_humpback_detector():
    """Load Google's humpback whale detection model from TF Hub.

    Requires: pip install tensorflow-cpu tensorflow-hub
    Model expects 10 kHz mono audio via the 'score' signature.
    Returns the score callable directly.

    Forces TF to CPU-only so it doesn't conflict with PyTorch's GPU usage
    (TF 2.21 doesn't support Blackwell sm_120 anyway).
    """
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    import tensorflow_hub as hub
    model = hub.load("https://tfhub.dev/google/humpback_whale/1")
    return model.signatures["score"]


def score_chunk_whale(chunk, sr, score_fn, target_sr=10000):
    """Score a chunk for humpback whale presence using Google's detector.

    Args:
        chunk: audio array at `sr` sample rate
        sr: source sample rate (44100)
        score_fn: the 'score' signature from the TF Hub model
        target_sr: model's expected sample rate (10000 Hz)

    Returns:
        float: mean detection score (0-1). Higher = more whale.
    """
    import tensorflow as tf

    # Resample to 10 kHz
    resampled = librosa.resample(chunk, orig_sr=sr, target_sr=target_sr)

    # Score signature expects (batch, samples, 1) float32
    waveform = tf.constant(resampled, dtype=tf.float32)
    waveform = tf.reshape(waveform, [1, -1, 1])

    # Run detection via score signature (handles mel spectrogram internally)
    # context_step_samples controls the hop between scoring windows
    # Using the full context width (39124) for non-overlapping windows
    result = score_fn(
        waveform=waveform,
        context_step_samples=tf.constant(39124, dtype=tf.int64),
    )
    scores = result["scores"]  # (batch, frames, 1)

    return float(tf.reduce_mean(scores).numpy())


# --- Detection data functions ---

def load_detection_hours(det_dir, station, min_detection=0.8):
    """Load high-detection hours from GoogleAI humpback detection CSVs.

    Returns set of datetime-hour strings like '2018-11-15T10' that have
    detection proportion > min_detection.
    """
    station_dir = Path(det_dir) / station
    if not station_dir.exists():
        return set()

    csvs = sorted(station_dir.glob("*humpbackwhale*.csv"))
    high_hours = set()

    for csv_path in csvs:
        with open(csv_path) as f:
            content = f.read()
        reader = csv.reader(io.StringIO(content))
        header = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                ts = row[0].strip('"').rstrip('zZ')
                prop = float(row[1].strip('"'))
                if prop >= min_detection:
                    # Parse to get hour key: '2018-11-15T10'
                    dt = datetime.fromisoformat(ts)
                    hour_key = dt.strftime('%Y-%m-%dT%H')
                    high_hours.add(hour_key)
            except (ValueError, IndexError):
                continue

    return high_hours


def parse_flac_timestamp(filename):
    """Extract datetime from FLAC filename.

    Handles two SanctSound timestamp formats:
    - Deployment 01: ...20181115T000002Z.flac  (YYYYMMDDTHHMMSSz)
    - Deployment 02+: ...191201000002.flac     (YYMMDDHHMMSS)
    """
    parts = filename.replace('.flac', '').split('_')
    for part in parts:
        # Format 1: 20181115T000002Z
        if len(part) >= 15 and 'T' in part:
            try:
                return datetime.strptime(part.rstrip('Z'), '%Y%m%dT%H%M%S')
            except ValueError:
                continue
        # Format 2: 191201000002 (YYMMDDHHMMSS, 12 digits)
        if len(part) == 12 and part.isdigit():
            try:
                return datetime.strptime(part, '%y%m%d%H%M%S')
            except ValueError:
                continue
    return None


# --- Download functions ---

def list_deployment_flacs(station, deployment_num):
    """List FLAC blobs in a GCS deployment."""
    from google.cloud import storage
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket("noaa-passive-bioacoustic")
    station_upper = station.upper()
    deployment_name = f"sanctsound_{station}_{deployment_num:02d}"
    prefix = f"sanctsound/audio/{station}/{deployment_name}/audio/"
    blobs = sorted(bucket.list_blobs(prefix=prefix), key=lambda b: b.name)
    return [b for b in blobs if b.name.endswith('.flac')]


def download_blob(blob, local_path):
    """Download a single blob to local path."""
    blob.download_to_filename(str(local_path))


# --- Main processing ---

def _resample_chunked(audio, orig_sr, target_sr, chunk_duration=60.0):
    """Resample audio in chunks to limit peak memory usage.

    librosa.resample on a full 15-min 96kHz file uses ~11GB RAM.
    Processing in 60s chunks keeps peak usage under ~1GB.
    """
    if orig_sr == target_sr:
        return audio

    chunk_samples = int(chunk_duration * orig_sr)
    parts = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        resampled = librosa.resample(chunk, orig_sr=orig_sr, target_sr=target_sr)
        parts.append(resampled)
    return np.concatenate(parts)


def preprocess_flac_cpu(flac_path, target_sr=44100, skip_seconds=5.0,
                        whale_cv_threshold=0.8, energy_ratio_threshold=0.0,
                        min_whale_rms=0.0):
    """CPU-only preprocessing: load → bandpass → segment → filter → loudness normalize.

    Two-pass heuristic filtering:
    1. Whale-band CV (existing): rejects steady-state noise
    2. Spectrogram energy ratio: rejects broadband noise without whale-band concentration
    3. Minimum whale-band RMS: rejects near-silence chunks

    Returns list of audio chunks (ready for GPU tokenization) and stats dict.
    """
    audio, sr = sf.read(str(flac_path), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    file_duration = len(audio) / sr

    # Skip test tone
    skip_samples = int(skip_seconds * sr)
    if len(audio) > skip_samples:
        audio = audio[skip_samples:]

    # Resample in chunks to limit memory (~1GB peak per chunk vs ~11GB for full file)
    if sr != target_sr:
        audio = _resample_chunked(audio, sr, target_sr)
        sr = target_sr

    # Bandpass
    audio = bandpass_audio(audio, sr)

    # Segment
    chunks = segment_audio_long(audio, sr)

    ready_chunks = []
    n_filtered = 0

    for chunk in chunks:
        # Peak normalize
        peak = np.max(np.abs(chunk))
        if peak > 0:
            chunk = chunk * (0.9 / peak)
        chunk = np.clip(chunk, -1.0, 1.0).astype(np.float32)

        # Whale-band variability filter
        cv = whale_band_score(chunk, sr)
        if cv < whale_cv_threshold:
            n_filtered += 1
            continue

        # Spectrogram energy ratio filter
        if energy_ratio_threshold > 0:
            er = whale_energy_ratio(chunk, sr)
            if er < energy_ratio_threshold:
                n_filtered += 1
                continue

        # Minimum whale-band energy filter
        if min_whale_rms > 0:
            wb_rms = whale_band_rms(chunk, sr)
            if wb_rms < min_whale_rms:
                n_filtered += 1
                continue

        # Loudness normalize (Pipeline B — no spectral gating)
        chunk = loudness_normalize(chunk, sr)
        ready_chunks.append(chunk)

    stats = {
        "file_duration": file_duration,
        "n_chunks_total": len(chunks),
        "n_chunks_filtered": n_filtered,
        "n_chunks_kept": len(ready_chunks),
    }
    return ready_chunks, stats


def _download_and_preprocess(args):
    """Worker function for parallel processing. Downloads FLAC, preprocesses, deletes.

    Returns (fname, ready_chunks, stats) or (fname, None, error_msg).
    """
    blob_name, bucket_name, local_path, whale_cv_threshold = args
    from google.cloud import storage
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    local_path = Path(local_path)
    try:
        # Download
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))

        # Preprocess (CPU-only)
        ready_chunks, stats = preprocess_flac_cpu(
            local_path, whale_cv_threshold=whale_cv_threshold,
        )

        # Delete FLAC
        if local_path.exists():
            local_path.unlink()

        fname = local_path.name
        return (fname, ready_chunks, stats)

    except Exception as e:
        # Clean up on failure
        if local_path.exists():
            local_path.unlink()
        return (local_path.name, None, str(e))


def process_flac_file(flac_path, tokenizer, target_sr=44100,
                      skip_seconds=5.0, whale_cv_threshold=0.8):
    """Process a single FLAC: skip test tone → bandpass → segment → filter → tokenize.

    Returns list of token arrays and stats dict.
    """
    import torch

    ready_chunks, stats = preprocess_flac_cpu(
        flac_path, target_sr=target_sr,
        skip_seconds=skip_seconds, whale_cv_threshold=whale_cv_threshold,
    )

    all_tokens = []
    total_tokens = 0

    for chunk in ready_chunks:
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        codes, z = tokenizer.encode(chunk_tensor)
        tokens = tokenizer.codes_to_sequence(codes)

        if len(tokens) > 2:
            all_tokens.append(tokens)
            total_tokens += len(tokens)

    stats["n_chunks_kept"] = len(all_tokens)
    stats["n_tokens"] = total_tokens
    return all_tokens, stats


def process_deployment(station, deployment_num, output_dir, det_dir,
                       tokenizer, whale_cv_threshold=0.8,
                       energy_ratio_threshold=0.0, min_whale_rms=0.0,
                       detector=None, detector_threshold=0.5,
                       min_detection=0.8, tmp_dir=None, dry_run=False,
                       max_files=None, n_workers=None,
                       save_2d=False):
    """Download, process, and tokenize a single deployment.

    Two-pass filtering:
    - Pass 1 (heuristic): CV + energy ratio + min RMS (in preprocess_flac_cpu)
    - Pass 2 (detector): Google humpback detector on survivors (optional)

    Supports saving as 2D (n_codebooks, T) arrays for hierarchical models.
    """
    import torch

    station_upper = station.upper()
    dep_str = f"{station_upper}_{deployment_num:02d}"
    print(f"\n{'='*60}")
    print(f"Deployment: {dep_str}")
    print(f"{'='*60}")

    # Load detection hours
    high_hours = load_detection_hours(det_dir, station, min_detection)
    print(f"  High-detection hours (>{min_detection*100:.0f}%): {len(high_hours)}")

    if not high_hours:
        print("  No detection data found, skipping")
        return {}

    # List FLACs
    print("  Listing GCS files...")
    all_blobs = list_deployment_flacs(station, deployment_num)
    print(f"  Total FLACs in deployment: {len(all_blobs)}")

    # Filter by detection hours
    filtered_blobs = []
    for blob in all_blobs:
        fname = blob.name.split('/')[-1]
        dt = parse_flac_timestamp(fname)
        if dt:
            hour_key = dt.strftime('%Y-%m-%dT%H')
            if hour_key in high_hours:
                filtered_blobs.append(blob)

    print(f"  FLACs in high-detection hours: {len(filtered_blobs)}")

    if max_files:
        filtered_blobs = filtered_blobs[:max_files]
        print(f"  Limited to first {max_files} files")

    if dry_run:
        est_gb = len(filtered_blobs) * 54 / 1000  # ~54 MB per file
        print(f"  Dry run: would download {len(filtered_blobs)} files (~{est_gb:.1f} GB)")
        return {"deployment": dep_str, "n_flacs": len(filtered_blobs), "est_gb": est_gb}

    # Set up temporary download directory
    if tmp_dir is None:
        tmp_dir = Path(f"data/sanctsound/tmp_{station}_{deployment_num:02d}")
    else:
        tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count existing tokens for this deployment to set chunk_idx offset
    existing = list(output_dir.glob(f"sanctsound_{station}_{deployment_num:02d}_*.npy"))
    chunk_idx = len(existing)
    if existing:
        print(f"  Found {len(existing)} existing tokens, resuming from idx {chunk_idx}")

    # Track processed FLACs to skip on restart (prevents duplicate tokens)
    done_file = output_dir / f".done_{station}_{deployment_num:02d}.txt"
    done_flacs = set()
    if done_file.exists():
        done_flacs = set(done_file.read_text().strip().split('\n'))
        done_flacs.discard('')
        print(f"  {len(done_flacs)} FLACs already processed, skipping them")

    dep_stats = {
        "deployment": dep_str,
        "n_flacs_available": len(filtered_blobs),
        "n_flacs_processed": 0,
        "n_flacs_failed": 0,
        "file_duration": 0,
        "n_chunks_total": 0,
        "n_chunks_filtered": 0,
        "n_chunks_kept": 0,
        "n_tokens": 0,
    }

    # Serial processing: download → preprocess → tokenize → delete
    for blob in tqdm(filtered_blobs, desc=f"  {dep_str}"):
        fname = blob.name.split('/')[-1]

        # Skip already-processed FLACs
        if fname in done_flacs:
            continue

        local_path = tmp_dir / fname

        # Download
        if not local_path.exists():
            try:
                download_blob(blob, local_path)
            except Exception as e:
                tqdm.write(f"    Download failed {fname}: {e}")
                dep_stats["n_flacs_failed"] += 1
                continue

        # CPU preprocessing + GPU tokenization
        try:
            ready_chunks, stats = preprocess_flac_cpu(
                local_path, whale_cv_threshold=whale_cv_threshold,
                energy_ratio_threshold=energy_ratio_threshold,
                min_whale_rms=min_whale_rms,
            )

            dep_stats["n_flacs_processed"] += 1
            dep_stats["file_duration"] += stats["file_duration"]
            dep_stats["n_chunks_total"] += stats["n_chunks_total"]
            dep_stats["n_chunks_filtered"] += stats["n_chunks_filtered"]

            # Pass 2: Google humpback detector (optional)
            if detector is not None:
                before_det = len(ready_chunks)
                ready_chunks = [
                    c for c in ready_chunks
                    if score_chunk_whale(c, 44100, detector) >= detector_threshold
                ]
                dep_stats["n_chunks_filtered"] += before_det - len(ready_chunks)

            # GPU tokenization
            for chunk in ready_chunks:
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                codes, z = tokenizer.encode(chunk_tensor)

                if save_2d:
                    # Save as 2D (n_codebooks, T) for hierarchical models
                    codes_np = codes[0].cpu().numpy().astype(np.int32)
                    if codes_np.shape[1] > 2:
                        out_path = output_dir / f"sanctsound_{station}_{deployment_num:02d}_{chunk_idx:06d}.npy"
                        np.save(out_path, codes_np)
                        chunk_idx += 1
                        dep_stats["n_chunks_kept"] += 1
                        dep_stats["n_tokens"] += codes_np.shape[1]
                else:
                    # Save as 1D interleaved (legacy format)
                    tokens = tokenizer.codes_to_sequence(codes)
                    if len(tokens) > 2:
                        out_path = output_dir / f"sanctsound_{station}_{deployment_num:02d}_{chunk_idx:06d}.npy"
                        np.save(out_path, tokens)
                        chunk_idx += 1
                        dep_stats["n_chunks_kept"] += 1
                        dep_stats["n_tokens"] += len(tokens)

            # Mark FLAC as done (append to done file)
            with open(done_file, 'a') as f:
                f.write(fname + '\n')

        except Exception as e:
            tqdm.write(f"    Process failed {fname}: {e}")
            dep_stats["n_flacs_failed"] += 1

        # Delete FLAC after processing
        if local_path.exists():
            local_path.unlink()

    # Clean up tmp dir
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Print summary
    kept = dep_stats["n_chunks_kept"]
    total = dep_stats["n_chunks_total"]
    pct = 100 * kept / max(total, 1)
    print(f"\n  {dep_str} summary:")
    print(f"    FLACs processed: {dep_stats['n_flacs_processed']}")
    print(f"    Duration: {dep_stats['file_duration']/3600:.1f} hours")
    print(f"    Chunks: {kept}/{total} kept ({pct:.0f}%)")
    print(f"    Tokens: {dep_stats['n_tokens']:,}")

    return dep_stats


def main():
    parser = argparse.ArgumentParser(
        description="Download and process SanctSound Hawaii humpback whale data")
    parser.add_argument("--station", type=str, default=None,
                        help="Process only this station (e.g., hi05). Default: all Hawaii stations")
    parser.add_argument("--deployment", type=int, default=None,
                        help="Process only this deployment number")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto from codec choice)")
    parser.add_argument("--det-dir", type=str,
                        default="data/sanctsound/detections")
    parser.add_argument("--codec", choices=["lac", "dac"], default="lac",
                        help="Audio codec: lac (WhAM) or dac (Descript Audio Codec)")
    parser.add_argument("--codec-path", type=str, default="models/codec.pth",
                        help="Path to LAC weights (only used with --codec lac)")
    parser.add_argument("--n-codebooks", type=int, default=None,
                        help="Number of codebooks (default: 4 for LAC, 9 for DAC)")
    parser.add_argument("--save-2d", action="store_true",
                        help="Save as 2D (n_codebooks, T) arrays for hierarchical models")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min-detection", type=float, default=0.8,
                        help="Minimum humpback detection proportion (default: 0.8)")
    parser.add_argument("--whale-cv-threshold", type=float, default=0.8,
                        help="Minimum whale-band CV score to keep a chunk (default: 0.8)")
    parser.add_argument("--energy-ratio-threshold", type=float, default=0.0,
                        help="Minimum whale-band energy ratio (0=disabled, try 0.4)")
    parser.add_argument("--min-whale-rms", type=float, default=0.0,
                        help="Minimum whale-band RMS energy (0=disabled, try 0.01)")
    parser.add_argument("--use-detector", action="store_true",
                        help="Use Google humpback detector as Pass 2 filter")
    parser.add_argument("--detector-threshold", type=float, default=0.5,
                        help="Minimum detector score to keep a chunk (default: 0.5)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max FLAC files per deployment (for testing)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for download+preprocessing (default: cpu_count)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files and estimate sizes without downloading")
    args = parser.parse_args()

    # Define stations and deployments
    all_stations = {
        "hi05": [1],          # Smallest, ~100% detection — process first
        "hi01": [1, 2, 3],
        "hi03": [1, 2, 3],
        "hi04": [1, 2, 3],
    }

    if args.station:
        if args.station not in all_stations:
            print(f"Unknown station: {args.station}. Available: {list(all_stations.keys())}")
            return
        stations = {args.station: all_stations[args.station]}
    else:
        stations = all_stations

    if args.deployment is not None:
        stations = {s: [args.deployment] for s in stations}

    # Set defaults based on codec choice
    if args.n_codebooks is None:
        args.n_codebooks = 9 if args.codec == "dac" else 4
    if args.output_dir is None:
        if args.codec == "dac":
            args.output_dir = "data/tokenized/sanctsound_humpback_dac"
        else:
            args.output_dir = "data/tokenized/sanctsound_4cb"

    # Load tokenizer (unless dry run)
    tokenizer = None
    if not args.dry_run:
        if args.codec == "dac":
            from src.tokenizer.dac_tokenizer import DACTokenizer
            print(f"Loading DACTokenizer ({args.n_codebooks} codebooks)...")
            tokenizer = DACTokenizer(
                device=args.device,
                n_codebooks=args.n_codebooks,
            )
        else:
            from src.tokenizer.audio_tokenizer import AudioTokenizer
            print(f"Loading AudioTokenizer from {args.codec_path}...")
            tokenizer = AudioTokenizer(
                codec_path=args.codec_path,
                device=args.device,
                n_codebooks=args.n_codebooks,
            )
        print(f"  Codec: {args.codec.upper()}, Sample rate: {tokenizer.sample_rate}, "
              f"Tokens/sec: {tokenizer.tokens_per_second:.1f}, Codebooks: {args.n_codebooks}")

    # Load Google humpback detector (Pass 2)
    detector = None
    if args.use_detector and not args.dry_run:
        print("Loading Google humpback whale detector...")
        detector = load_humpback_detector()
        print(f"  Detector loaded, threshold: {args.detector_threshold}")

    # Process each station/deployment
    all_stats = []
    for station, deployments in stations.items():
        for dep_num in deployments:
            stats = process_deployment(
                station, dep_num,
                output_dir=args.output_dir,
                det_dir=args.det_dir,
                tokenizer=tokenizer,
                whale_cv_threshold=args.whale_cv_threshold,
                energy_ratio_threshold=args.energy_ratio_threshold,
                min_whale_rms=args.min_whale_rms,
                detector=detector,
                detector_threshold=args.detector_threshold,
                min_detection=args.min_detection,
                dry_run=args.dry_run,
                max_files=args.max_files,
                n_workers=args.workers,
                save_2d=args.save_2d,
            )
            all_stats.append(stats)

    # Print overall summary
    if not args.dry_run and all_stats:
        total_tokens = sum(s.get("n_tokens", 0) for s in all_stats)
        total_kept = sum(s.get("n_chunks_kept", 0) for s in all_stats)
        total_chunks = sum(s.get("n_chunks_total", 0) for s in all_stats)
        total_hours = sum(s.get("file_duration", 0) for s in all_stats) / 3600

        print(f"\n{'='*60}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"Total audio processed: {total_hours:.1f} hours")
        print(f"Chunks kept: {total_kept}/{total_chunks} "
              f"({100*total_kept/max(total_chunks,1):.0f}%)")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Output: {args.output_dir}")

        # Save metadata
        output_dir = Path(args.output_dir)
        pipeline_steps = "skip_test_tone → bandpass → segment → normalize → loudness_norm"
        if args.energy_ratio_threshold > 0 or args.min_whale_rms > 0:
            pipeline_steps += " → heuristic_filter"
        if args.use_detector:
            pipeline_steps += " → humpback_detector"
        pipeline_steps += " → tokenize"
        if args.save_2d:
            pipeline_steps += " (2D)"

        meta = {
            "source": "sanctsound_hawaii_humpback",
            "pipeline": pipeline_steps,
            "codec": args.codec,
            "whale_cv_threshold": args.whale_cv_threshold,
            "energy_ratio_threshold": args.energy_ratio_threshold,
            "min_whale_rms": args.min_whale_rms,
            "use_detector": args.use_detector,
            "detector_threshold": args.detector_threshold if args.use_detector else None,
            "min_detection": args.min_detection,
            "n_codebooks": args.n_codebooks,
            "save_2d": args.save_2d,
            "total_hours": round(total_hours, 1),
            "total_tokens": total_tokens,
            "total_chunks_kept": total_kept,
            "total_chunks_total": total_chunks,
            "deployments": all_stats,
        }
        if args.codec == "lac":
            meta["codec_path"] = args.codec_path
        if tokenizer:
            meta["sample_rate"] = tokenizer.sample_rate
            meta["tokens_per_second"] = tokenizer.tokens_per_second
            meta["vocab_size"] = tokenizer.vocab_size

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata saved to {output_dir / 'metadata.json'}")

    elif args.dry_run:
        total_flacs = sum(s.get("n_flacs", 0) for s in all_stats)
        total_gb = sum(s.get("est_gb", 0) for s in all_stats)
        print(f"\nDry run total: {total_flacs} FLACs, ~{total_gb:.0f} GB")


if __name__ == "__main__":
    main()
