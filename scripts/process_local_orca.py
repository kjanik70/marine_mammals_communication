#!/usr/bin/env python3
"""Process local orca audio files → DAC 9CB tokens.

For curated orca datasets that don't need SanctSound annotation lookup.
Same quality filtering and tokenization pipeline as process_sanctsound_orca.py.

Pipeline per file:
1. Load → mono → resample to 44100 Hz
2. Bandpass 80–20 kHz
3. [optional] two-pass spectral gating
4. Segment into ≤30s chunks (remove >4s silence)
5. Per-chunk: peak normalize → orca CV/energy filter → loudness normalize
6. Tokenize with DAC 9CB → save .npy

Usage:
    PYTHONPATH=. python3 scripts/process_local_orca.py \\
        --input-dir data/raw/dori_orcasound_full/data \\
        --source-name dori_orca

    PYTHONPATH=. python3 scripts/process_local_orca.py \\
        --input-dir data/raw/orcasound \\
        --source-name orcasound_srkw \\
        --glob "2017*SRKW*.wav"

    PYTHONPATH=. python3 scripts/process_local_orca.py \\
        --input-dir data/raw/esp_orcas/audio \\
        --source-name esp_orca \\
        --orca-cv-threshold 0.3
"""

import argparse
import csv
import ctypes
import gc
import glob as glob_module
import json
import os
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfilt
from tqdm import tqdm


TARGET_SR = 44100
AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.ogg', '.aiff', '.aif'}


# --- Audio utilities (mirrors process_sanctsound_orca.py) ---

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
    S = np.abs(librosa.stft(chunk, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = (freqs >= low) & (freqs <= high)
    orca_energy = np.sum(S[mask, :] ** 2)
    total_energy = np.sum(S ** 2) + 1e-10
    return float(orca_energy / total_energy)


def spectral_gate(audio, sr, stationary_prop=0.90, nonstationary_prop=0.75):
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


def segment_audio(audio, sr, max_duration=30.0, min_duration=2.0, max_silence=4.0):
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
    in_silence, start = False, 0
    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            start, in_silence = i * hop_length, True
        elif not silent and in_silence:
            silence_regions.append((start, i * hop_length, (i * hop_length - start) / sr))
            in_silence = False
    if in_silence:
        silence_regions.append((start, n_frames * hop_length, (n_frames * hop_length - start) / sr))

    long_silences = [r for r in silence_regions if r[2] > max_silence]
    if long_silences:
        replacement_samples = int(0.5 * sr)
        pieces, prev_end = [], 0
        for s, e, _ in sorted(long_silences):
            if s > prev_end:
                pieces.append(audio[prev_end:s])
            pieces.append(np.zeros(replacement_samples, dtype=audio.dtype))
            prev_end = e
        if prev_end < len(audio):
            pieces.append(audio[prev_end:])
        if pieces:
            audio = np.concatenate(pieces)

    if len(audio) / sr <= max_duration:
        return [audio] if len(audio) / sr >= min_duration else []

    max_samples = int(max_duration * sr)
    min_samples = int(min_duration * sr)
    chunks, chunk_start = [], 0
    total = len(audio)
    while chunk_start < total:
        remaining = total - chunk_start
        if remaining <= max_samples:
            if remaining >= min_samples:
                chunks.append(audio[chunk_start:])
            break
        chunk_end = chunk_start + max_samples
        chunk = audio[chunk_start:chunk_end]
        if len(chunk) >= min_samples:
            chunks.append(chunk)
        chunk_start = chunk_end
    return chunks


def resample_chunked(audio, orig_sr, target_sr, chunk_duration=60.0):
    if orig_sr == target_sr:
        return audio
    chunk_samples = int(chunk_duration * orig_sr)
    parts = []
    for start in range(0, len(audio), chunk_samples):
        parts.append(librosa.resample(audio[start:start + chunk_samples],
                                      orig_sr=orig_sr, target_sr=target_sr))
    return np.concatenate(parts)


# --- Per-file processing ---

def process_file(file_path, tokenizer, output_dir, source_name, chunk_idx,
                 orca_cv_threshold, energy_ratio_threshold, min_orca_rms,
                 denoise, score_path):
    audio, sr = sf.read(str(file_path), dtype='float32', always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != TARGET_SR:
        audio = resample_chunked(audio, sr, TARGET_SR)

    audio = bandpass_audio(audio, TARGET_SR)

    if denoise:
        audio = spectral_gate(audio, TARGET_SR)

    chunks = segment_audio(audio, TARGET_SR)
    del audio
    gc.collect()

    import torch
    n_kept, score_rows, npy_names = 0, [], []
    for i, chunk in enumerate(chunks):
        peak = np.max(np.abs(chunk))
        if peak > 0:
            chunk = chunk * (0.9 / peak)
        chunk = np.clip(chunk, -1.0, 1.0).astype(np.float32)

        cv = orca_band_score(chunk, TARGET_SR)
        if cv < orca_cv_threshold:
            score_rows.append({'npy_file': '', 'source_file': file_path.name,
                               'chunk_idx': i, 'orca_cv': round(cv, 4),
                               'energy_ratio': 0.0, 'orca_rms': 0.0, 'tokenized': 'no'})
            continue

        er = orca_energy_ratio(chunk, TARGET_SR)
        if energy_ratio_threshold > 0 and er < energy_ratio_threshold:
            score_rows.append({'npy_file': '', 'source_file': file_path.name,
                               'chunk_idx': i, 'orca_cv': round(cv, 4),
                               'energy_ratio': round(er, 4), 'orca_rms': 0.0, 'tokenized': 'no'})
            continue

        rms_val = float(np.sqrt(np.mean(
            sosfilt(butter(5, [500 / (TARGET_SR / 2), min(15000 / (TARGET_SR / 2), 0.95)],
                          btype='band', output='sos'), chunk).astype(np.float32) ** 2
        )))
        if min_orca_rms > 0 and rms_val < min_orca_rms:
            score_rows.append({'npy_file': '', 'source_file': file_path.name,
                               'chunk_idx': i, 'orca_cv': round(cv, 4),
                               'energy_ratio': round(er, 4),
                               'orca_rms': round(rms_val, 6), 'tokenized': 'no'})
            continue

        chunk = loudness_normalize(chunk, TARGET_SR)

        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            codes, z = tokenizer.encode(chunk_tensor)
        del chunk_tensor, z
        codes_np = codes[0].cpu().numpy().astype(np.int32)
        del codes

        if codes_np.shape[1] > 2:
            npy_name = f"{source_name}_{chunk_idx:06d}.npy"
            np.save(output_dir / npy_name, codes_np)
            score_rows.append({'npy_file': npy_name, 'source_file': file_path.name,
                               'chunk_idx': i, 'orca_cv': round(cv, 4),
                               'energy_ratio': round(er, 4),
                               'orca_rms': round(rms_val, 6), 'tokenized': 'yes'})
            chunk_idx += 1
            n_kept += 1

    if score_rows:
        write_header = not score_path.exists()
        with open(score_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'npy_file', 'source_file', 'chunk_idx',
                'orca_cv', 'energy_ratio', 'orca_rms', 'tokenized'])
            if write_header:
                writer.writeheader()
            writer.writerows(score_rows)

    return chunk_idx, n_kept, len(chunks)


def main():
    parser = argparse.ArgumentParser(description='Process local orca audio → DAC 9CB tokens')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--source-name', required=True,
                        help='Prefix for output NPY files (e.g. dori_orca, esp_orca)')
    parser.add_argument('--glob', default=None,
                        help='Filename glob pattern within input-dir (default: all audio files)')
    parser.add_argument('--output-dir', default='data/tokenized/sanctsound_orca_dac')
    parser.add_argument('--n-codebooks', type=int, default=9)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--orca-cv-threshold', type=float, default=0.8)
    parser.add_argument('--energy-ratio-threshold', type=float, default=0.0)
    parser.add_argument('--min-orca-rms', type=float, default=0.0)
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.glob:
        files = sorted(input_dir.glob(args.glob))
    else:
        files = sorted(f for f in input_dir.iterdir()
                       if f.suffix.lower() in AUDIO_EXTS)

    print(f"Source: {args.source_name} ({len(files)} files from {input_dir})")

    done_file = output_dir / f'.done_{args.source_name}.txt'
    done = set()
    if done_file.exists():
        done = set(done_file.read_text().strip().split('\n'))
        done.discard('')
        print(f"  {len(done)} files already processed, skipping")

    existing = list(output_dir.glob(f'{args.source_name}_*.npy'))
    chunk_idx = len(existing)
    if existing:
        print(f"  Found {chunk_idx} existing tokens, resuming from idx {chunk_idx}")

    if args.dry_run:
        pending = [f for f in files if f.name not in done]
        print(f"  Dry run: would process {len(pending)} files")
        return

    from src.tokenizer.dac_tokenizer import DACTokenizer
    print(f"Loading DACTokenizer ({args.n_codebooks} codebooks, device={args.device})...")
    tokenizer = DACTokenizer(device=args.device, n_codebooks=args.n_codebooks)

    score_path = output_dir / 'chunk_scores.csv'
    n_files, n_total_chunks, n_kept_chunks, n_tokens = 0, 0, 0, 0

    for file_path in tqdm(files, desc=args.source_name):
        if file_path.name in done:
            continue
        try:
            chunk_idx, n_kept, n_total = process_file(
                file_path, tokenizer, output_dir, args.source_name, chunk_idx,
                orca_cv_threshold=args.orca_cv_threshold,
                energy_ratio_threshold=args.energy_ratio_threshold,
                min_orca_rms=args.min_orca_rms,
                denoise=args.denoise,
                score_path=score_path,
            )
            n_files += 1
            n_total_chunks += n_total
            n_kept_chunks += n_kept
            n_tokens += n_kept  # each kept chunk → codes_np.shape[1] tokens; approximate here
            with open(done_file, 'a') as f:
                f.write(file_path.name + '\n')
        except Exception as e:
            tqdm.write(f"  Failed {file_path.name}: {e}")
        finally:
            gc.collect()
            ctypes.CDLL('libc.so.6').malloc_trim(0)

    pct = 100 * n_kept_chunks / max(n_total_chunks, 1)
    print(f"\n{args.source_name} summary:")
    print(f"  Files processed: {n_files}")
    print(f"  Chunks kept / total: {n_kept_chunks}/{n_total_chunks} ({pct:.0f}%)")
    print(f"  Output NPY files: {chunk_idx}")


if __name__ == '__main__':
    main()
