#!/usr/bin/env python3
"""Download, process, and tokenize SanctSound orca (killer whale) data.

Pipeline per FLAC:
1. Check overlap with orca annotations → skip if none
2. Download FLAC to tmp dir
3. Load → mono → resample to 44100 Hz (chunked)
4. Extract only annotated time ranges (+ 2s buffer)
5. Bandpass 80–20 kHz
6. [optional] Two-pass spectral gating (--denoise)
7. Segment into ≤30s chunks (remove >4s silence)
8. Per-chunk: peak normalize → orca band CV/energy filter → loudness normalize
9. Tokenize with DAC 9CB (save as 2D (9, T) .npy)
10. Save scores + ecotype to chunk_scores.csv
11. Delete FLAC

Detection format: precise start/end timestamps with ecotype (SR/NR/Transient/Unknown).
Manual annotations are the detector — no Google TF Hub model needed.

Usage:
    PYTHONPATH=. python3 scripts/process_sanctsound_orca.py
    PYTHONPATH=. python3 scripts/process_sanctsound_orca.py --station oc01
    PYTHONPATH=. python3 scripts/process_sanctsound_orca.py --station oc01 --denoise
    PYTHONPATH=. python3 scripts/process_sanctsound_orca.py --station oc01 --dry-run
"""

import argparse
import csv
import ctypes
import gc
import json
import multiprocessing as mp
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfilt
from tqdm import tqdm


# Stations and deployments that have killerwhale annotation CSVs
ORCA_STATIONS = {
    "oc01": [1, 3],
    "oc02": [1, 2, 4, 5],
    "oc03": [2, 3, 4],
    "oc04": [2, 4],
}

# Assumed FLAC duration (SanctSound archives in 15-minute files)
FLAC_DURATION_S = 900.0


# --- Audio processing utilities ---

def bandpass_audio(audio, sr, low_hz=80, high_hz=20000):
    nyq = sr / 2
    low = min(low_hz / nyq, 0.95)
    high = min(high_hz / nyq, 0.95)
    if low < high:
        sos = butter(5, [low, high], btype='band', output='sos')
        audio = sosfilt(sos, audio).astype(np.float32)
    return audio


def loudness_normalize(audio, sr, target_lufs=-20.0):
    meter = pyln.Meter(sr)
    current_loudness = meter.integrated_loudness(audio)
    if np.isfinite(current_loudness):
        audio = pyln.normalize.loudness(audio, current_loudness, target_lufs)
    else:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.5 / peak)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def orca_band_score(chunk, sr, low=500, high=15000):
    """CV of RMS energy in orca vocalization band (500–15000 Hz)."""
    nyq = sr / 2
    sos = butter(5, [low / nyq, min(high / nyq, 0.95)], btype='band', output='sos')
    band = sosfilt(sos, chunk).astype(np.float32)
    frame_len = int(0.5 * sr)
    n_frames = len(band) // frame_len
    if n_frames < 3:
        return 0.0
    rms = np.array([
        np.sqrt(np.mean(band[j * frame_len:(j + 1) * frame_len] ** 2))
        for j in range(n_frames)
    ])
    return float(np.std(rms) / max(np.mean(rms), 1e-10))


def orca_energy_ratio(chunk, sr, low=500, high=15000):
    """Fraction of spectral energy in orca band (500–15000 Hz)."""
    S = np.abs(librosa.stft(chunk, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = (freqs >= low) & (freqs <= high)
    orca_energy = np.sum(S[mask, :] ** 2)
    total_energy = np.sum(S ** 2) + 1e-10
    return float(orca_energy / total_energy)


def orca_band_rms(chunk, sr, low=500, high=15000):
    """RMS energy in orca frequency band."""
    nyq = sr / 2
    sos = butter(5, [low / nyq, min(high / nyq, 0.95)], btype='band', output='sos')
    band = sosfilt(sos, chunk).astype(np.float32)
    return float(np.sqrt(np.mean(band ** 2)))


def spectral_gate(audio, sr, stationary_prop=0.90, nonstationary_prop=0.75):
    """Two-pass spectral gating (mirrors medium denoising pipeline).

    Pass 1 targets constant background hiss/hum; pass 2 is gentler for
    variable noise. Works well on annotated call regions (high SNR).
    """
    import noisereduce as nr
    audio = nr.reduce_noise(
        y=audio, sr=sr, stationary=True,
        prop_decrease=stationary_prop,
        n_fft=2048, freq_mask_smooth_hz=250, time_mask_smooth_ms=60,
    ).astype(np.float32)
    audio = nr.reduce_noise(
        y=audio, sr=sr, stationary=False,
        prop_decrease=nonstationary_prop,
        n_fft=2048, freq_mask_smooth_hz=200, time_mask_smooth_ms=50,
    ).astype(np.float32)
    return audio


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

    silence_regions = []
    in_silence = False
    start = 0
    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            start = i * hop_length
            in_silence = True
        elif not silent and in_silence:
            end = i * hop_length
            silence_regions.append((start, end, (end - start) / sr))
            in_silence = False
    if in_silence:
        end = len(energy) * hop_length
        silence_regions.append((start, end, (end - start) / sr))

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
            silence_regions2.append((start, end, (end - start) / sr))
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


def _resample_chunked(audio, orig_sr, target_sr, chunk_duration=60.0):
    if orig_sr == target_sr:
        return audio
    chunk_samples = int(chunk_duration * orig_sr)
    parts = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        resampled = librosa.resample(chunk, orig_sr=orig_sr, target_sr=target_sr)
        parts.append(resampled)
    return np.concatenate(parts)


# --- Annotation loading ---

def _normalize_ecotype(raw):
    """Normalize ecotype strings to canonical form (SR, NR, Transient, Unknown)."""
    upper = raw.strip().upper()
    if upper in ('SR', 'SOUTHERN RESIDENT'):
        return 'SR'
    if upper in ('NR', 'NORTHERN RESIDENT'):
        return 'NR'
    if upper == 'TRANSIENT':
        return 'Transient'
    if not upper or upper.startswith('UNK'):
        return 'Unknown'
    return raw.strip()


def _parse_iso(ts):
    """Parse ISO timestamp string to naive UTC datetime."""
    return datetime.fromisoformat(ts.rstrip('Zz').replace('+00:00', ''))


def load_orca_annotations(det_dir, station, deployment_num):
    """Load orca annotations for a station/deployment from killerwhale CSV.

    Returns list of (start_dt, end_dt, ecotype) tuples (naive UTC datetimes).
    """
    det_path = (Path(det_dir) / station /
                f"SanctSound_{station.upper()}_{deployment_num:02d}_killerwhale.csv")
    if not det_path.exists():
        return []

    annotations = []
    with open(det_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start_dt = _parse_iso(row['ISOStartTime'])
                end_dt = _parse_iso(row['ISOEndTime'])
                raw = row.get('Ecotype', '').strip()
                ecotype = _normalize_ecotype(raw)
                annotations.append((start_dt, end_dt, ecotype))
            except (ValueError, KeyError):
                continue
    return annotations


def overlapping_annotations(flac_dt, annotations, flac_duration_s=FLAC_DURATION_S):
    """Return annotations that overlap this FLAC's time window."""
    flac_end = flac_dt + timedelta(seconds=flac_duration_s)
    return [(s, e, ec) for s, e, ec in annotations if s < flac_end and e > flac_dt]


# --- FLAC timestamp parsing ---

def parse_flac_timestamp(filename):
    """Extract datetime from FLAC filename (naive UTC).

    Handles two SanctSound timestamp formats:
    - Deployment 01: ...20181115T000002Z.flac  (YYYYMMDDTHHMMSSz)
    - Deployment 02+: ...191201000002.flac     (YYMMDDHHMMSS)
    """
    parts = filename.replace('.flac', '').split('_')
    for part in parts:
        if len(part) >= 15 and 'T' in part:
            try:
                return datetime.strptime(part.rstrip('Z'), '%Y%m%dT%H%M%S')
            except ValueError:
                continue
        if len(part) == 12 and part.isdigit():
            try:
                return datetime.strptime(part, '%y%m%d%H%M%S')
            except ValueError:
                continue
    return None


# --- GCS download ---

def list_deployment_flacs(station, deployment_num):
    from google.cloud import storage
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket("noaa-passive-bioacoustic")
    deployment_name = f"sanctsound_{station}_{deployment_num:02d}"
    prefix = f"sanctsound/audio/{station}/{deployment_name}/audio/"
    blobs = sorted(bucket.list_blobs(prefix=prefix), key=lambda b: b.name)
    return [b for b in blobs if b.name.endswith('.flac')]


def download_blob(blob, local_path):
    blob.download_to_filename(str(local_path))


# --- CPU preprocessing ---

def preprocess_flac_orca_cpu(flac_path, annotations_iso, flac_dt_iso,
                              target_sr=44100, orca_cv_threshold=0.8,
                              energy_ratio_threshold=0.0, min_orca_rms=0.0,
                              annotation_buffer_s=2.0, denoise=False):
    """Load only annotated regions from FLAC, segment, filter, loudness normalize.

    Uses soundfile range-reads to avoid loading multi-hour files entirely into RAM.

    annotations_iso: list of (start_iso, end_iso, ecotype) strings
    flac_dt_iso: ISO string of FLAC recording start time

    Returns:
        ready_chunks: list of audio arrays
        chunk_metadata: list of dicts with chunk_idx_in_region, orca_cv, energy_ratio,
                        orca_rms, ecotype, ann_start, ann_end
        stats: dict
    """
    flac_dt = _parse_iso(flac_dt_iso)
    annotations = [(_parse_iso(s), _parse_iso(e), ec) for s, e, ec in annotations_iso]

    with sf.SoundFile(str(flac_path)) as f:
        sr = f.samplerate
        n_frames = f.frames
    file_duration = n_frames / sr

    ready_chunks = []
    chunk_metadata = []
    n_total = 0
    n_filtered = 0

    for ann_start, ann_end, ecotype in annotations:
        rel_start = max(0.0, (ann_start - flac_dt).total_seconds() - annotation_buffer_s)
        rel_end = min(file_duration, (ann_end - flac_dt).total_seconds() + annotation_buffer_s)

        if rel_end <= rel_start:
            continue

        start_frame = int(rel_start * sr)
        end_frame = min(int(rel_end * sr), n_frames)
        region_raw, _ = sf.read(str(flac_path), start=start_frame, stop=end_frame,
                                dtype='float32', always_2d=False)
        if region_raw.ndim > 1:
            region_raw = region_raw.mean(axis=1)

        if sr != target_sr:
            region_raw = _resample_chunked(region_raw, sr, target_sr)

        region = bandpass_audio(region_raw, target_sr)
        del region_raw

        if denoise:
            region = spectral_gate(region, target_sr)

        chunks = segment_audio_long(region, target_sr)
        n_total += len(chunks)

        for i, chunk in enumerate(chunks):
            peak = np.max(np.abs(chunk))
            if peak > 0:
                chunk = chunk * (0.9 / peak)
            chunk = np.clip(chunk, -1.0, 1.0).astype(np.float32)

            cv = orca_band_score(chunk, target_sr)
            if cv < orca_cv_threshold:
                n_filtered += 1
                continue

            er = orca_energy_ratio(chunk, target_sr)
            if energy_ratio_threshold > 0 and er < energy_ratio_threshold:
                n_filtered += 1
                continue

            rms = orca_band_rms(chunk, target_sr)
            if min_orca_rms > 0 and rms < min_orca_rms:
                n_filtered += 1
                continue

            chunk = loudness_normalize(chunk, target_sr)
            ready_chunks.append(chunk)
            chunk_metadata.append({
                'chunk_idx_in_region': i,
                'orca_cv': round(cv, 4),
                'energy_ratio': round(er, 4),
                'orca_rms': round(rms, 6),
                'ecotype': ecotype,
                'ann_start': ann_start.isoformat(),
                'ann_end': ann_end.isoformat(),
            })

    stats = {
        'file_duration': file_duration,
        'n_chunks_total': n_total,
        'n_chunks_filtered': n_filtered,
        'n_chunks_kept': len(ready_chunks),
    }
    return ready_chunks, chunk_metadata, stats


def _preprocess_worker(flac_path, out_dir, kwargs):
    ready_chunks, chunk_metadata, stats = preprocess_flac_orca_cpu(str(flac_path), **kwargs)
    chunk_paths = []
    for i, chunk in enumerate(ready_chunks):
        p = Path(out_dir) / f"chunk_{i:04d}.npy"
        np.save(p, chunk)
        chunk_paths.append(str(p))
    meta = {'chunk_paths': chunk_paths, 'chunk_metadata': chunk_metadata, 'stats': stats}
    with open(Path(out_dir) / 'meta.json', 'w') as f:
        json.dump(meta, f)


def preprocess_flac_subprocess(flac_path, **kwargs):
    """Run preprocessing in a subprocess to isolate peak memory (~10–15 GB for 96kHz FLACs)."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix='flac_prep_')
    try:
        p = mp.Process(target=_preprocess_worker, args=(flac_path, tmp, kwargs))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Preprocessing subprocess exited with code {p.exitcode}")
        with open(Path(tmp) / 'meta.json') as f:
            meta = json.load(f)
        ready_chunks = [np.load(cp) for cp in meta['chunk_paths']]
        return ready_chunks, meta['chunk_metadata'], meta['stats']
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- Parallel worker (pool-based) ---

_worker_tokenizer = None


def _worker_init(n_codebooks):
    """Load DAC tokenizer once per worker process."""
    global _worker_tokenizer
    import warnings
    warnings.filterwarnings('ignore')
    from src.tokenizer.dac_tokenizer import DACTokenizer
    _worker_tokenizer = DACTokenizer(device='cpu', n_codebooks=n_codebooks)


def _flac_stem(fname):
    """Extract timestamp part of FLAC filename for collision-free output naming."""
    parts = fname.replace('.flac', '').split('_')
    for part in parts:
        if (len(part) >= 15 and 'T' in part) or (len(part) == 12 and part.isdigit()):
            return part.rstrip('Z')
    import hashlib
    return hashlib.md5(fname.encode()).hexdigest()[:12]


def _process_flac_task(task):
    """Pool worker: download one FLAC → preprocess → tokenize → save .npy files.

    Each worker process runs this function for many FLACs. Because the worker
    is already an isolated process (via Pool), no nested subprocess is needed
    for memory isolation — memory is freed when the task returns.

    Returns a result dict with stats, score_rows, and token counts.
    """
    import hashlib
    import tempfile
    import torch
    from google.cloud import storage

    global _worker_tokenizer

    blob_name = task['blob_name']
    fname = blob_name.split('/')[-1]
    flac_stem = task['flac_stem']
    output_dir = Path(task['output_dir'])
    station = task['station']
    dep_num = task['dep_num']

    tmp_dir = Path(tempfile.mkdtemp(prefix='orca_flac_'))
    local_path = tmp_dir / fname

    try:
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket('noaa-passive-bioacoustic')
        bucket.blob(blob_name).download_to_filename(str(local_path))

        ready_chunks, chunk_metadata, stats = preprocess_flac_orca_cpu(
            local_path,
            annotations_iso=task['annotations_iso'],
            flac_dt_iso=task['flac_dt_iso'],
            orca_cv_threshold=task['orca_cv_threshold'],
            energy_ratio_threshold=task['energy_ratio_threshold'],
            min_orca_rms=task['min_orca_rms'],
            annotation_buffer_s=task['annotation_buffer_s'],
            denoise=task['denoise'],
        )

        npy_names = []
        n_tokens = 0
        for i, chunk in enumerate(ready_chunks):
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                codes, z = _worker_tokenizer.encode(chunk_tensor)
            del chunk_tensor, z
            codes_np = codes[0].cpu().numpy().astype(np.int32)
            del codes

            if codes_np.shape[1] > 2:
                npy_name = f"sanctsound_{station}_{dep_num:02d}_{flac_stem}_{i:04d}.npy"
                np.save(output_dir / npy_name, codes_np)
                npy_names.append(npy_name)
                n_tokens += codes_np.shape[1]
            else:
                npy_names.append(None)

        score_rows = []
        for j, meta in enumerate(chunk_metadata):
            npy_name = npy_names[j] if j < len(npy_names) else None
            score_rows.append({
                'npy_file': npy_name or '',
                'flac_name': fname,
                **meta,
                'tokenized': 'yes' if npy_name else 'no',
            })

        return {
            'fname': fname,
            'stats': stats,
            'score_rows': score_rows,
            'n_tokens': n_tokens,
            'error': None,
        }

    except Exception as e:
        return {'fname': fname, 'stats': {}, 'score_rows': [], 'n_tokens': 0, 'error': str(e)}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()
        ctypes.CDLL('libc.so.6').malloc_trim(0)


# --- Score CSV ---

def append_score_rows(score_path, rows):
    write_header = not score_path.exists()
    with open(score_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'npy_file', 'flac_name', 'chunk_idx_in_region',
            'orca_cv', 'energy_ratio', 'orca_rms',
            'ecotype', 'ann_start', 'ann_end', 'tokenized'])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# --- Deployment processing ---

def process_deployment(station, deployment_num, output_dir, det_dir, tokenizer,
                       orca_cv_threshold=0.8, energy_ratio_threshold=0.0,
                       min_orca_rms=0.0, annotation_buffer_s=2.0, denoise=False,
                       tmp_dir=None, dry_run=False, max_files=None,
                       max_flacs_per_run=None, n_workers=1, n_codebooks=9):
    dep_str = f"{station.upper()}_{deployment_num:02d}"
    print(f"\n{'='*60}")
    print(f"Deployment: {dep_str}")
    print(f"{'='*60}")

    annotations = load_orca_annotations(det_dir, station, deployment_num)
    print(f"  Orca annotation windows: {len(annotations)}")
    if not annotations:
        print("  No annotation data found, skipping")
        return {}

    print("  Listing GCS files...")
    all_blobs = list_deployment_flacs(station, deployment_num)
    print(f"  Total FLACs in deployment: {len(all_blobs)}")

    # Filter to FLACs that overlap at least one annotation window
    filtered = []
    for blob in all_blobs:
        fname = blob.name.split('/')[-1]
        flac_dt = parse_flac_timestamp(fname)
        if flac_dt and overlapping_annotations(flac_dt, annotations):
            filtered.append((blob, flac_dt))

    print(f"  FLACs overlapping annotations: {len(filtered)}")

    if max_files:
        filtered = filtered[:max_files]
        print(f"  Limited to first {max_files} files")

    if dry_run:
        est_gb = len(filtered) * 54 / 1000
        print(f"  Dry run: would download {len(filtered)} files (~{est_gb:.1f} GB)")
        return {"deployment": dep_str, "n_flacs": len(filtered), "est_gb": est_gb}

    if tmp_dir is None:
        tmp_dir = Path(f"data/sanctsound/tmp_{station}_{deployment_num:02d}")
    else:
        tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob(f"sanctsound_{station}_{deployment_num:02d}_*.npy"))
    chunk_idx = len(existing)
    if existing:
        print(f"  Found {len(existing)} existing tokens, resuming from idx {chunk_idx}")

    done_file = output_dir / f".done_{station}_{deployment_num:02d}.txt"
    done_flacs = set()
    if done_file.exists():
        done_flacs = set(done_file.read_text().strip().split('\n'))
        done_flacs.discard('')
        print(f"  {len(done_flacs)} FLACs already processed, skipping")

    dep_stats = {
        "deployment": dep_str,
        "n_flacs_available": len(filtered),
        "n_flacs_processed": 0,
        "n_flacs_failed": 0,
        "file_duration": 0.0,
        "n_chunks_total": 0,
        "n_chunks_filtered": 0,
        "n_chunks_tokenized": 0,
        "n_tokens": 0,
    }

    score_path = output_dir / "chunk_scores.csv"

    if n_workers > 1:
        # --- Parallel path ---
        tasks = []
        for blob, flac_dt in filtered:
            fname = blob.name.split('/')[-1]
            if fname in done_flacs:
                continue
            if max_flacs_per_run and len(tasks) >= max_flacs_per_run:
                break
            flac_anns = overlapping_annotations(flac_dt, annotations)
            tasks.append({
                'blob_name': blob.name,
                'flac_dt_iso': flac_dt.isoformat(),
                'annotations_iso': [(s.isoformat(), e.isoformat(), ec) for s, e, ec in flac_anns],
                'output_dir': str(output_dir),
                'station': station,
                'dep_num': deployment_num,
                'flac_stem': _flac_stem(fname),
                'orca_cv_threshold': orca_cv_threshold,
                'energy_ratio_threshold': energy_ratio_threshold,
                'min_orca_rms': min_orca_rms,
                'annotation_buffer_s': annotation_buffer_s,
                'denoise': denoise,
                'n_codebooks': n_codebooks,
            })

        print(f"  Processing {len(tasks)} FLACs with {n_workers} workers...")
        with mp.Pool(processes=n_workers, initializer=_worker_init,
                     initargs=(n_codebooks,), maxtasksperchild=10) as pool:
            for result in tqdm(pool.imap_unordered(_process_flac_task, tasks),
                               total=len(tasks), desc=f"  {dep_str}"):
                if result['error']:
                    tqdm.write(f"    Failed {result['fname']}: {result['error']}")
                    dep_stats['n_flacs_failed'] += 1
                else:
                    s = result['stats']
                    dep_stats['n_flacs_processed'] += 1
                    dep_stats['file_duration'] += s.get('file_duration', 0)
                    dep_stats['n_chunks_total'] += s.get('n_chunks_total', 0)
                    dep_stats['n_chunks_filtered'] += s.get('n_chunks_filtered', 0)
                    n_tok = len([r for r in result['score_rows'] if r['tokenized'] == 'yes'])
                    dep_stats['n_chunks_tokenized'] += n_tok
                    dep_stats['n_tokens'] += result['n_tokens']
                    if result['score_rows']:
                        append_score_rows(score_path, result['score_rows'])
                    with open(done_file, 'a') as f:
                        f.write(result['fname'] + '\n')

    else:
        # --- Serial path ---
        import torch
        n_processed_this_run = 0

        for blob, flac_dt in tqdm(filtered, desc=f"  {dep_str}"):
            fname = blob.name.split('/')[-1]

            if fname in done_flacs:
                continue

            if max_flacs_per_run and n_processed_this_run >= max_flacs_per_run:
                tqdm.write(f"  Reached {max_flacs_per_run} FLACs this run, exiting for restart...")
                break

            rss_mb = int(open(f'/proc/{os.getpid()}/status').read().split('VmRSS:')[1].split('kB')[0].strip()) // 1024
            if rss_mb > 8000 and n_processed_this_run > 0:
                tqdm.write(f"  RSS={rss_mb}MB > 8GB, exiting for memory reset...")
                break

            local_path = tmp_dir / fname

            if not local_path.exists():
                try:
                    download_blob(blob, local_path)
                except Exception as e:
                    tqdm.write(f"    Download failed {fname}: {e}")
                    dep_stats["n_flacs_failed"] += 1
                    continue

            try:
                flac_anns = overlapping_annotations(flac_dt, annotations)
                annotations_iso = [(s.isoformat(), e.isoformat(), ec) for s, e, ec in flac_anns]

                ready_chunks, chunk_metadata, stats = preprocess_flac_subprocess(
                    local_path,
                    annotations_iso=annotations_iso,
                    flac_dt_iso=flac_dt.isoformat(),
                    orca_cv_threshold=orca_cv_threshold,
                    energy_ratio_threshold=energy_ratio_threshold,
                    min_orca_rms=min_orca_rms,
                    annotation_buffer_s=annotation_buffer_s,
                    denoise=denoise,
                )

                dep_stats["n_flacs_processed"] += 1
                dep_stats["file_duration"] += stats["file_duration"]
                dep_stats["n_chunks_total"] += stats["n_chunks_total"]
                dep_stats["n_chunks_filtered"] += stats["n_chunks_filtered"]

                npy_names = []
                for j, chunk in enumerate(ready_chunks):
                    chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    codes, z = tokenizer.encode(chunk_tensor)
                    del chunk_tensor, z
                    codes_np = codes[0].cpu().numpy().astype(np.int32)
                    del codes

                    if codes_np.shape[1] > 2:
                        npy_name = f"sanctsound_{station}_{deployment_num:02d}_{chunk_idx:06d}.npy"
                        np.save(output_dir / npy_name, codes_np)
                        npy_names.append(npy_name)
                        chunk_idx += 1
                        dep_stats["n_chunks_tokenized"] += 1
                        dep_stats["n_tokens"] += codes_np.shape[1]
                    else:
                        npy_names.append(None)

                score_rows = []
                for j, meta in enumerate(chunk_metadata):
                    npy_name = npy_names[j] if j < len(npy_names) else None
                    score_rows.append({
                        'npy_file': npy_name or '',
                        'flac_name': fname,
                        **meta,
                        'tokenized': 'yes' if npy_name else 'no',
                    })
                if score_rows:
                    append_score_rows(score_path, score_rows)

                with open(done_file, 'a') as f:
                    f.write(fname + '\n')
                n_processed_this_run += 1

            except Exception as e:
                tqdm.write(f"    Process failed {fname}: {e}")
                dep_stats["n_flacs_failed"] += 1

            torch.cuda.empty_cache()
            gc.collect()
            ctypes.CDLL('libc.so.6').malloc_trim(0)

            if local_path.exists():
                local_path.unlink()

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    tokenized = dep_stats["n_chunks_tokenized"]
    total = dep_stats["n_chunks_total"]
    pct = 100 * tokenized / max(total, 1)
    print(f"\n  {dep_str} summary:")
    print(f"    FLACs processed: {dep_stats['n_flacs_processed']}")
    print(f"    Duration: {dep_stats['file_duration']/3600:.1f} hours")
    print(f"    Chunks kept / total: {tokenized}/{total} ({pct:.0f}%)")
    print(f"    Tokens: {dep_stats['n_tokens']:,}")

    return dep_stats


def main():
    parser = argparse.ArgumentParser(
        description="Download and process SanctSound orca data → DAC 9CB tokens")
    parser.add_argument("--station", type=str, default=None,
                        help="Process only this station (e.g., oc01). Default: all orca stations")
    parser.add_argument("--deployment", type=int, default=None,
                        help="Process only this deployment number")
    parser.add_argument("--output-dir", type=str, default="data/tokenized/sanctsound_orca_dac")
    parser.add_argument("--det-dir", type=str, default="data/sanctsound/detections")
    parser.add_argument("--n-codebooks", type=int, default=9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--orca-cv-threshold", type=float, default=0.8,
                        help="Minimum orca-band CV score to keep a chunk (default: 0.8)")
    parser.add_argument("--energy-ratio-threshold", type=float, default=0.0,
                        help="Minimum orca-band energy ratio (0=disabled)")
    parser.add_argument("--min-orca-rms", type=float, default=0.0,
                        help="Minimum orca-band RMS (0=disabled)")
    parser.add_argument("--annotation-buffer", type=float, default=2.0,
                        help="Seconds of audio to include before/after each annotation (default: 2.0)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max FLAC files per deployment (for testing)")
    parser.add_argument("--denoise", action="store_true",
                        help="Apply two-pass spectral gating to each annotation region before tokenization")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (default: 1). Each worker loads its own "
                             "CPU DAC tokenizer. Start with 8-12; each worker uses ~2-4 GB RAM.")
    parser.add_argument("--max-flacs-per-run", type=int, default=None,
                        help="Exit after processing this many FLACs (for periodic restart)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files and estimate sizes without downloading")
    args = parser.parse_args()

    if args.station:
        if args.station not in ORCA_STATIONS:
            print(f"Unknown station: {args.station}. Available: {list(ORCA_STATIONS.keys())}")
            return
        stations = {args.station: ORCA_STATIONS[args.station]}
    else:
        stations = ORCA_STATIONS

    if args.deployment is not None:
        stations = {s: [args.deployment] for s in stations}

    tokenizer = None
    if not args.dry_run and args.workers == 1:
        from src.tokenizer.dac_tokenizer import DACTokenizer
        print(f"Loading DACTokenizer ({args.n_codebooks} codebooks)...")
        tokenizer = DACTokenizer(device=args.device, n_codebooks=args.n_codebooks)
        print(f"  Sample rate: {tokenizer.sample_rate}, "
              f"Tokens/sec: {tokenizer.tokens_per_second:.1f}, "
              f"Codebooks: {args.n_codebooks}")
    elif not args.dry_run:
        print(f"Parallel mode: {args.workers} workers, each loads DAC on CPU at startup")

    all_stats = []
    for station, deployments in stations.items():
        for dep_num in deployments:
            stats = process_deployment(
                station, dep_num,
                output_dir=args.output_dir,
                det_dir=args.det_dir,
                tokenizer=tokenizer,
                orca_cv_threshold=args.orca_cv_threshold,
                energy_ratio_threshold=args.energy_ratio_threshold,
                min_orca_rms=args.min_orca_rms,
                annotation_buffer_s=args.annotation_buffer,
                denoise=args.denoise,
                dry_run=args.dry_run,
                max_files=args.max_files,
                max_flacs_per_run=args.max_flacs_per_run,
                n_workers=args.workers,
                n_codebooks=args.n_codebooks,
            )
            all_stats.append(stats)

    if not args.dry_run and all_stats:
        total_tokens = sum(s.get("n_tokens", 0) for s in all_stats)
        total_tokenized = sum(s.get("n_chunks_tokenized", 0) for s in all_stats)
        total_chunks = sum(s.get("n_chunks_total", 0) for s in all_stats)
        total_hours = sum(s.get("file_duration", 0) for s in all_stats) / 3600

        print(f"\n{'='*60}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"Total audio processed: {total_hours:.1f} hours")
        print(f"Chunks tokenized: {total_tokenized}/{total_chunks} "
              f"({100*total_tokenized/max(total_chunks,1):.0f}%)")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Output: {args.output_dir}")

        output_dir = Path(args.output_dir)
        denoise_step = " → spectral_gate" if args.denoise else ""
        meta = {
            "source": "sanctsound_orca",
            "pipeline": (f"annotations_overlap → extract_regions → bandpass{denoise_step}"
                         " → segment → normalize → loudness_norm → dac_9cb_2d"),
            "denoise": args.denoise,
            "codec": "dac",
            "n_codebooks": args.n_codebooks,
            "orca_cv_threshold": args.orca_cv_threshold,
            "energy_ratio_threshold": args.energy_ratio_threshold,
            "min_orca_rms": args.min_orca_rms,
            "annotation_buffer_s": args.annotation_buffer,
            "total_hours": round(total_hours, 1),
            "total_tokens": total_tokens,
            "total_chunks_tokenized": total_tokenized,
            "total_chunks_total": total_chunks,
            "deployments": all_stats,
        }
        if tokenizer:
            meta["sample_rate"] = tokenizer.sample_rate
            meta["tokens_per_second"] = tokenizer.tokens_per_second

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata saved to {output_dir / 'metadata.json'}")

    elif args.dry_run:
        total_flacs = sum(s.get("n_flacs", 0) for s in all_stats)
        total_gb = sum(s.get("est_gb", 0) for s in all_stats)
        print(f"\nDry run total: {total_flacs} FLACs, ~{total_gb:.0f} GB")


if __name__ == "__main__":
    main()
