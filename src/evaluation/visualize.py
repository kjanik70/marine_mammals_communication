"""Visualization utilities for marine mammal communication."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    log_file: str | Path,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot training and validation loss curves from a JSONL log file."""
    import json

    entries = []
    with open(log_file) as f:
        for line in f:
            entries.append(json.loads(line))

    steps = [e["step"] for e in entries]
    train_loss = [e["train_loss"] for e in entries]
    val_entries = [(e["step"], e["val_loss"]) for e in entries if "val_loss" in e]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(steps, train_loss, label="Train Loss", alpha=0.7)
    if val_entries:
        val_steps, val_loss = zip(*val_entries)
        ax.plot(val_steps, val_loss, "ro-", label="Val Loss", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_coda_distribution(
    real_dist: dict,
    generated_dist: Optional[dict] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot coda type distribution (real vs generated)."""
    types = sorted(real_dist.keys())
    real_vals = [real_dist.get(t, 0) for t in types]

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    x = np.arange(len(types))
    width = 0.35

    ax.bar(x - width / 2, real_vals, width, label="Real Data", color="steelblue")

    if generated_dist:
        gen_vals = [generated_dist.get(t, 0) for t in types]
        ax.bar(x + width / 2, gen_vals, width, label="Generated", color="coral")

    ax.set_xlabel("Coda Type")
    ax.set_ylabel("Proportion")
    ax.set_title("Coda Type Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_waveform(
    audio: np.ndarray,
    sr: int,
    title: str = "Waveform",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot audio waveform."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    t = np.arange(len(audio)) / sr
    ax.plot(t, audio, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_spectrogram(
    audio: np.ndarray,
    sr: int,
    title: str = "Spectrogram",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot mel spectrogram of audio."""
    import librosa
    import librosa.display

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(title)
    fig.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sr: int,
    title: str = "Original vs Reconstructed",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Side-by-side waveform and spectrogram comparison."""
    import librosa
    import librosa.display

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Waveforms
    t_orig = np.arange(len(original)) / sr
    axes[0, 0].plot(t_orig, original, linewidth=0.5)
    axes[0, 0].set_title("Original Waveform")
    axes[0, 0].set_xlabel("Time (s)")

    t_recon = np.arange(len(reconstructed)) / sr
    axes[0, 1].plot(t_recon, reconstructed, linewidth=0.5, color="coral")
    axes[0, 1].set_title("Reconstructed Waveform")
    axes[0, 1].set_xlabel("Time (s)")

    # Spectrograms
    for i, (audio, label) in enumerate([(original, "Original"), (reconstructed, "Reconstructed")]):
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=axes[1, i])
        axes[1, i].set_title(f"{label} Spectrogram")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
