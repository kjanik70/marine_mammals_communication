#!/usr/bin/env python3
"""Process SanctSound FLAC files with denoising: bandpass, segment, normalize, denoise, tokenize.

Pipeline per file:
1. Load FLAC, convert to mono
2. Resample to 44100 Hz (from 48/96 kHz)
3. Bandpass filter (80 Hz – 20 kHz)
4. Remove >4s low-energy patches, segment into ≤30s chunks
5. Per-chunk peak normalization to 0.9
6. Two-pass spectral gating (denoise after normalization — SNR is much
   better per-chunk than whole-file on hydrophone recordings)
7. Loudness normalization to -20 LUFS
8. Tokenize with LAC codec (4 codebooks)
9. Save as .npy token files

Key difference from process_sanctsound.py: denoising is applied AFTER
per-chunk normalization. On whole-file hydrophone audio, spectral gating
removes faint whale calls. But after peak-normalizing each 30s chunk,
the whale calls are at a reasonable level and spectral gating can
distinguish them from background noise.
"""

import argparse
import json
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfilt
from tqdm import tqdm


def bandpass_audio(audio, sr, low_hz=80, high_hz=20000):
    """Bandpass filter to remove ocean ambient noise outside whale frequency range."""
    nyq = sr / 2
    low = min(low_hz / nyq, 0.95)
    high = min(high_hz / nyq, 0.95)
    if low < high:
        sos = butter(5, [low, high], btype='band', output='sos')
        audio = sosfilt(sos, audio).astype(np.float32)
    return audio


def spectral_gate(audio, sr):
    """Two-pass spectral gating. Returns denoised audio (no loudness norm)."""
    # First pass: stationary noise reduction (90%)
    audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=True,
        prop_decrease=0.90,
        n_fft=min(2048, len(audio)),
        freq_mask_smooth_hz=250,
        time_mask_smooth_ms=60,
    )

    # Second pass: non-stationary (75%)
    audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,
        prop_decrease=0.75,
        n_fft=min(2048, len(audio)),
        freq_mask_smooth_hz=200,
        time_mask_smooth_ms=50,
    )
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


def segment_audio_long(audio, sr, max_duration=30.0, min_duration=2.0,
                       max_silence=4.0, replacement_silence=0.5,
                       silence_threshold=None):
    """Segment into ≤30s chunks, removing long silence, keeping short pauses.

    If silence_threshold is None, uses adaptive threshold (P25 of energy).
    """
    duration = len(audio) / sr
    if duration < min_duration:
        return []

    if duration <= max_duration:
        return [audio]

    # Compute energy
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    n_frames = max(1, (len(audio) - frame_length) // hop_length)
    energy = np.array([
        np.sqrt(np.mean(audio[i * hop_length:i * hop_length + frame_length] ** 2))
        for i in range(n_frames)
    ])

    # Adaptive silence threshold: use P25 of energy if not specified
    if silence_threshold is None:
        silence_threshold = max(np.percentile(energy, 25), 1e-6)

    # Find silence regions
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
            # Recompute silence regions
            n_frames = max(1, (len(audio) - frame_length) // hop_length)
            energy = np.array([
                np.sqrt(np.mean(audio[i * hop_length:i * hop_length + frame_length] ** 2))
                for i in range(n_frames)
            ])
            is_silent = energy < silence_threshold
            silence_regions = []
            in_silence = False
            for i, silent in enumerate(is_silent):
                if silent and not in_silence:
                    start = i * hop_length
                    in_silence = True
                elif not silent and in_silence:
                    end = i * hop_length
                    dur = (end - start) / sr
                    silence_regions.append((start, end, dur))
                    in_silence = False

    if len(audio) / sr <= max_duration:
        if len(audio) / sr >= min_duration:
            return [audio]
        return []

    # Split at silence boundaries
    max_samples = int(max_duration * sr)
    min_samples = int(min_duration * sr)
    split_candidates = [(s + e) // 2 for s, e, d in silence_regions if d >= 0.1]

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


def process_flac_file(flac_path, tokenizer, target_sr=44100,
                      min_energy_retention=0.2):
    """Process a single FLAC file: bandpass → segment → normalize → denoise → tokenize.

    Chunks where spectral gating removes >80% of energy (retention < 0.2)
    are filtered out as mostly ocean noise with no whale vocalizations.

    Returns list of token arrays and stats dict.
    """
    import torch

    audio, sr = sf.read(str(flac_path), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    file_duration = len(audio) / sr

    # Resample to target SR
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Bandpass filter
    audio_clean = bandpass_audio(audio, sr)

    # Remove >4s silence and segment into ≤30s chunks
    chunks = segment_audio_long(audio_clean, sr)

    all_tokens = []
    total_tokens = 0
    n_filtered = 0
    peak_target = 0.9

    for chunk in chunks:
        # Per-chunk peak normalization — boosts SNR before denoising
        peak = np.max(np.abs(chunk))
        if peak > 0:
            chunk = chunk * (peak_target / peak)
        chunk = np.clip(chunk, -1.0, 1.0).astype(np.float32)

        rms_before = float(np.sqrt(np.mean(chunk ** 2)))

        # Spectral gating (no loudness norm yet — need RMS for filtering)
        chunk_gated = spectral_gate(chunk, sr)

        # Filter: if spectral gating removed most energy, chunk is just noise
        rms_after = float(np.sqrt(np.mean(chunk_gated ** 2)))
        retention = rms_after / max(rms_before, 1e-10)
        if retention < min_energy_retention:
            n_filtered += 1
            continue

        # Loudness normalize the gated audio
        chunk = loudness_normalize(chunk_gated, sr)

        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        codes, z = tokenizer.encode(chunk_tensor)
        tokens = tokenizer.codes_to_sequence(codes)

        if len(tokens) > 2:
            all_tokens.append(tokens)
            total_tokens += len(tokens)

    stats = {
        "file_duration": file_duration,
        "processed_duration": sum(len(c) / sr for c in chunks),
        "n_chunks_total": len(chunks),
        "n_chunks_filtered": n_filtered,
        "n_chunks": len(all_tokens),
        "n_tokens": total_tokens,
    }
    return all_tokens, stats


def main():
    parser = argparse.ArgumentParser(
        description="Process SanctSound FLAC files with denoising")
    parser.add_argument("--input-dir", type=str,
                        default="data/sanctsound/audio",
                        help="Directory with downloaded FLAC files")
    parser.add_argument("--output-dir", type=str,
                        default="data/tokenized/sanctsound_denoised_4cb")
    parser.add_argument("--codec-path", type=str, default="models/codec.pth")
    parser.add_argument("--n-codebooks", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--station", type=str, default=None,
                        help="Process only this station (e.g., hi01)")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--min-energy-retention", type=float, default=0.2,
                        help="Filter chunks where spectral gating retains less than this fraction of energy (default: 0.2)")
    args = parser.parse_args()

    from src.tokenizer.audio_tokenizer import AudioTokenizer

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading AudioTokenizer from {args.codec_path}...")
    tokenizer = AudioTokenizer(
        codec_path=args.codec_path,
        device=args.device,
        n_codebooks=args.n_codebooks,
    )
    print(f"  Sample rate: {tokenizer.sample_rate}, "
          f"Tokens/sec: {tokenizer.tokens_per_second:.1f}")

    # Find FLAC files
    if args.station:
        search_dir = input_dir / args.station
    else:
        search_dir = input_dir

    flac_files = sorted(search_dir.rglob("*.flac"))
    if args.max_files:
        flac_files = flac_files[:args.max_files]

    if not flac_files:
        print(f"No FLAC files found in {search_dir}")
        return

    print(f"\nProcessing {len(flac_files)} FLAC files from {search_dir}")
    print(f"Pipeline: bandpass → segment → normalize → denoise → filter → tokenize")
    print(f"Energy retention threshold: {args.min_energy_retention}")

    total_stats = {
        "file_duration": 0, "processed_duration": 0,
        "n_chunks_total": 0, "n_chunks_filtered": 0,
        "n_chunks": 0, "n_tokens": 0,
        "n_files_processed": 0, "n_files_with_output": 0,
    }
    chunk_idx = 0

    for flac_path in tqdm(flac_files, desc="Processing"):
        try:
            token_arrays, stats = process_flac_file(flac_path, tokenizer,
                                                     min_energy_retention=args.min_energy_retention)

            total_stats["file_duration"] += stats["file_duration"]
            total_stats["processed_duration"] += stats["processed_duration"]
            total_stats["n_chunks_total"] += stats["n_chunks_total"]
            total_stats["n_chunks_filtered"] += stats["n_chunks_filtered"]
            total_stats["n_chunks"] += stats["n_chunks"]
            total_stats["n_tokens"] += stats["n_tokens"]
            total_stats["n_files_processed"] += 1

            if token_arrays:
                total_stats["n_files_with_output"] += 1

            # Save token arrays
            station = flac_path.parent.name
            for tokens in token_arrays:
                out_path = output_dir / f"sanctsound_{station}_{chunk_idx:06d}.npy"
                np.save(out_path, tokens)
                chunk_idx += 1

        except Exception as e:
            tqdm.write(f"  FAILED {flac_path.name}: {e}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Files processed: {total_stats['n_files_processed']}")
    print(f"Files with output: {total_stats['n_files_with_output']}")
    print(f"Total file duration: {total_stats['file_duration']/3600:.1f} hours")
    print(f"Processed duration: {total_stats['processed_duration']/3600:.1f} hours "
          f"({100*total_stats['processed_duration']/max(1,total_stats['file_duration']):.1f}%)")
    print(f"Chunks total: {total_stats['n_chunks_total']}")
    print(f"Chunks filtered (noise): {total_stats['n_chunks_filtered']} "
          f"({100*total_stats['n_chunks_filtered']/max(1,total_stats['n_chunks_total']):.0f}%)")
    print(f"Chunks kept: {total_stats['n_chunks']}")
    print(f"Tokens: {total_stats['n_tokens']:,}")
    print(f"Output: {output_dir}")

    # Save metadata
    meta = {
        "source": "sanctsound_denoised_filtered",
        "pipeline": "bandpass → segment → normalize → denoise → filter → tokenize",
        "min_energy_retention": args.min_energy_retention,
        "codec_path": args.codec_path,
        "n_codebooks": args.n_codebooks,
        "sample_rate": tokenizer.sample_rate,
        "tokens_per_second": tokenizer.tokens_per_second,
        "vocab_size": tokenizer.vocab_size,
        **total_stats,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
