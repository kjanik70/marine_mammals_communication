#!/usr/bin/env python3
"""Grade audio quality of all raw datasets.

Computes signal quality metrics per segment and assigns grades (A-F).
Generates histograms by data source and species.
"""

import argparse
import csv
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm

from scripts.tokenize_all_audio import segment_audio


# ── Species mapping ──────────────────────────────────────────────────────────

def get_species_for_file(source: str, filepath: Path) -> str:
    """Determine species from source name and file path."""
    if source == "dswp":
        return "Sperm Whale"
    if source == "esp_orcas":
        return "Orca"
    if source == "humpback_tsujii":
        return "Humpback Whale"
    if source == "right_whale":
        return "Right Whale"
    if source == "kw_pei":
        return "Killer Whale"
    if source == "mbari":
        return "Unknown (hydrophone)"
    if source == "watkins":
        # Species from subdirectory name: data/raw/watkins/audio/<Species_Name>/file.wav
        parts = filepath.parts
        for i, p in enumerate(parts):
            if p == "audio" and i + 1 < len(parts):
                return parts[i + 1].replace("_", " ")
        return "Unknown"
    if source == "orcasound":
        name = filepath.name.lower()
        if "sperm" in name or "yukusam" in name:
            return "Sperm Whale"
        if "srkw" in name or "orca" in name:
            return "Orca"
        return "Unknown (orcasound)"
    if source == "dori_orca":
        return "Orca"
    return "Unknown"


# ── Quality metrics ──────────────────────────────────────────────────────────

def compute_quality_metrics(audio: np.ndarray, sr: int) -> dict:
    """Compute signal quality metrics for an audio segment."""
    # RMS energy
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # Peak-to-RMS ratio (crest factor)
    peak = float(np.max(np.abs(audio)))
    peak_to_rms = peak / (rms + 1e-10)

    # Short-time energy envelope for dynamics
    frame_length = min(int(0.025 * sr), len(audio))
    hop_length = max(int(0.010 * sr), 1)
    if frame_length < 2:
        energy_var = 0.0
    else:
        n_frames = max(1, (len(audio) - frame_length) // hop_length)
        energy = np.array([
            np.sqrt(np.mean(audio[i * hop_length:i * hop_length + frame_length] ** 2))
            for i in range(n_frames)
        ])
        energy_var = float(np.var(energy))

    # Spectral features via librosa
    # Use a reasonable n_fft for the segment length
    n_fft = min(2048, len(audio))
    if n_fft < 64:
        return {
            "spectral_flatness": 1.0,
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "rms_energy": rms,
            "peak_to_rms": peak_to_rms,
            "energy_variance": energy_var,
        }

    S = np.abs(librosa.stft(audio, n_fft=n_fft)) ** 2

    # Spectral flatness (mean across frames)
    flatness = librosa.feature.spectral_flatness(S=S)
    spec_flatness = float(np.mean(flatness))

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    spec_centroid = float(np.mean(centroid))

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    spec_bandwidth = float(np.mean(bandwidth))

    return {
        "spectral_flatness": spec_flatness,
        "spectral_centroid": spec_centroid,
        "spectral_bandwidth": spec_bandwidth,
        "rms_energy": rms,
        "peak_to_rms": peak_to_rms,
        "energy_variance": energy_var,
    }


def compute_quality_score(metrics: dict) -> float:
    """Compute composite quality score from metrics (0-1, higher = better)."""
    # Tonality: low spectral flatness = tonal = good
    tonality = 1.0 - min(metrics["spectral_flatness"], 1.0)

    # Dynamics: high energy variance = dynamic vocalizations
    # Normalize with a soft clip (typical range 0-0.01 for good signals)
    dynamics = min(metrics["energy_variance"] / 0.005, 1.0)

    # Crest factor: high peak-to-RMS = transient signals (clicks, calls)
    # Typical range: 2-20 for vocalizations, 1-3 for noise
    crest = min((metrics["peak_to_rms"] - 1.0) / 10.0, 1.0)
    crest = max(crest, 0.0)

    # Loudness: very quiet = likely silence/noise floor
    # Normalize so ~0.01 RMS = score 1.0
    loudness = min(metrics["rms_energy"] / 0.01, 1.0)

    score = 0.35 * tonality + 0.25 * dynamics + 0.20 * crest + 0.20 * loudness
    return max(0.0, min(1.0, score))


def score_to_grade(score: float) -> str:
    """Map composite score to letter grade."""
    if score >= 0.8:
        return "A"
    if score >= 0.6:
        return "B"
    if score >= 0.4:
        return "C"
    if score >= 0.2:
        return "D"
    return "F"


# ── Dataset processing ───────────────────────────────────────────────────────

def process_source(source: str, audio_dir: str, target_sr: int = 44100,
                   max_duration: float = 5.0) -> list[dict]:
    """Process all audio files from a source, return list of graded segments."""
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        return []

    files = []
    for pat in ("*.wav", "*.flac"):
        files.extend(audio_dir.rglob(pat))
    files = sorted(f for f in files if f.stat().st_size > 0)

    results = []
    for audio_path in tqdm(files, desc=f"  {source}"):
        try:
            audio, sr = sf.read(str(audio_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr < 2000:
                continue

            # Resample to common rate for consistent metric comparison
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            segments = segment_audio(audio, target_sr, max_duration=max_duration)
            species = get_species_for_file(source, audio_path)

            for seg_idx, seg in enumerate(segments):
                metrics = compute_quality_metrics(seg, target_sr)
                score = compute_quality_score(metrics)
                grade = score_to_grade(score)

                results.append({
                    "source": source,
                    "species": species,
                    "file": audio_path.name,
                    "segment_idx": seg_idx,
                    "duration_s": round(len(seg) / target_sr, 3),
                    **{k: round(v, 6) for k, v in metrics.items()},
                    "quality_score": round(score, 4),
                    "grade": grade,
                })
        except Exception as e:
            tqdm.write(f"    FAILED {audio_path.name}: {e}")

    return results


# ── Histogram generation ─────────────────────────────────────────────────────

GRADE_ORDER = ["A", "B", "C", "D", "F"]
GRADE_COLORS = {"A": "#2ecc71", "B": "#3498db", "C": "#f39c12", "D": "#e67e22", "F": "#e74c3c"}


def plot_quality_by_source(rows: list[dict], output_dir: Path):
    """Bar chart of grade distribution per source."""
    sources = sorted(set(r["source"] for r in rows))
    grade_counts = {s: {g: 0 for g in GRADE_ORDER} for s in sources}
    for r in rows:
        grade_counts[r["source"]][r["grade"]] += 1

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(sources))
    width = 0.15

    for i, grade in enumerate(GRADE_ORDER):
        counts = [grade_counts[s][grade] for s in sources]
        ax.bar(x + i * width, counts, width, label=f"Grade {grade}",
               color=GRADE_COLORS[grade])

    ax.set_xlabel("Data Source")
    ax.set_ylabel("Number of Segments")
    ax.set_title("Audio Quality Grade Distribution by Source")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(sources, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_by_source.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir / 'quality_by_source.png'}")


def plot_quality_by_species(rows: list[dict], output_dir: Path, top_n: int = 15):
    """Bar chart of grade distribution per species (top N by count)."""
    species_counts = {}
    for r in rows:
        species_counts[r["species"]] = species_counts.get(r["species"], 0) + 1

    top_species = sorted(species_counts, key=species_counts.get, reverse=True)[:top_n]

    grade_counts = {s: {g: 0 for g in GRADE_ORDER} for s in top_species}
    for r in rows:
        if r["species"] in grade_counts:
            grade_counts[r["species"]][r["grade"]] += 1

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(top_species))
    width = 0.15

    for i, grade in enumerate(GRADE_ORDER):
        counts = [grade_counts[s][grade] for s in top_species]
        ax.bar(x + i * width, counts, width, label=f"Grade {grade}",
               color=GRADE_COLORS[grade])

    ax.set_xlabel("Species")
    ax.set_ylabel("Number of Segments")
    ax.set_title(f"Audio Quality Grade Distribution by Species (Top {top_n})")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(top_species, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_by_species.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir / 'quality_by_species.png'}")


def plot_metrics_by_source(rows: list[dict], output_dir: Path):
    """Box plots of key metrics by source."""
    sources = sorted(set(r["source"] for r in rows))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    metrics = [
        ("spectral_flatness", "Spectral Flatness (lower = more tonal)"),
        ("energy_variance", "Energy Variance (higher = more dynamic)"),
        ("peak_to_rms", "Peak-to-RMS Ratio (higher = more transient)"),
        ("quality_score", "Composite Quality Score"),
    ]

    for ax, (metric, title) in zip(axes.flat, metrics):
        data = [[r[metric] for r in rows if r["source"] == s] for s in sources]
        bp = ax.boxplot(data, tick_labels=sources, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#3498db")
            patch.set_alpha(0.6)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Audio Quality Metrics by Source", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_by_source.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir / 'metrics_by_source.png'}")


def print_summary(rows: list[dict]):
    """Print summary table of grades by source."""
    sources = sorted(set(r["source"] for r in rows))

    print(f"\n{'Source':<20} {'Total':>6} {'A':>5} {'B':>5} {'C':>5} {'D':>5} {'F':>5} {'Avg':>6}")
    print("-" * 73)

    for source in sources:
        source_rows = [r for r in rows if r["source"] == source]
        total = len(source_rows)
        grades = {g: sum(1 for r in source_rows if r["grade"] == g) for g in GRADE_ORDER}
        avg_score = np.mean([r["quality_score"] for r in source_rows])
        print(f"{source:<20} {total:>6} {grades['A']:>5} {grades['B']:>5} "
              f"{grades['C']:>5} {grades['D']:>5} {grades['F']:>5} {avg_score:>6.3f}")

    total = len(rows)
    grades = {g: sum(1 for r in rows if r["grade"] == g) for g in GRADE_ORDER}
    avg_score = np.mean([r["quality_score"] for r in rows])
    print("-" * 73)
    print(f"{'TOTAL':<20} {total:>6} {grades['A']:>5} {grades['B']:>5} "
          f"{grades['C']:>5} {grades['D']:>5} {grades['F']:>5} {avg_score:>6.3f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grade audio quality of all datasets")
    parser.add_argument("--output-csv", default="data/audio_quality_grades.csv")
    parser.add_argument("--hist-dir", default="data/quality_histograms")
    parser.add_argument("--max-duration", type=float, default=5.0)
    parser.add_argument("--target-sr", type=int, default=44100)
    args = parser.parse_args()

    datasets = [
        ("dswp", "data/raw/dswp"),
        ("watkins", "data/raw/watkins/audio"),
        ("esp_orcas", "data/raw/esp_orcas/audio"),
        ("orcasound", "data/raw/orcasound"),
        ("mbari", "data/raw/mbari"),
        ("dori_orca", "data/raw/dori_orcasound"),
        ("humpback_tsujii", "data/raw/humpback_zenodo"),
        ("kw_pei", "data/raw/kw_pei"),
        ("right_whale", "data/raw/right_whale/v1"),
    ]

    all_rows = []
    for source, audio_dir in datasets:
        if not Path(audio_dir).exists():
            print(f"Skipping {source}: {audio_dir} not found")
            continue
        print(f"\n=== {source} ===")
        rows = process_source(source, audio_dir, target_sr=args.target_sr,
                              max_duration=args.max_duration)
        all_rows.extend(rows)
        n = len(rows)
        if n > 0:
            avg = np.mean([r["quality_score"] for r in rows])
            print(f"  {n} segments, avg quality: {avg:.3f}")

    if not all_rows:
        print("No segments found!")
        return

    # Save CSV
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {csv_path}")

    # Print summary
    print_summary(all_rows)

    # Generate histograms
    hist_dir = Path(args.hist_dir)
    hist_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating histograms in {hist_dir}/")
    plot_quality_by_source(all_rows, hist_dir)
    plot_quality_by_species(all_rows, hist_dir)
    plot_metrics_by_source(all_rows, hist_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
