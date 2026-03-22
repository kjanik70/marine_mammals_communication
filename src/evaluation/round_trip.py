"""End-to-end round-trip evaluation for audio token models."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def audio_round_trip(
    audio: np.ndarray,
    sr: int,
    tokenizer,
    device: str = "cuda",
) -> dict:
    """Encode audio to tokens and decode back. Measure reconstruction quality.

    Args:
        audio: Input audio array (1D, float32)
        sr: Sample rate
        tokenizer: AudioTokenizer with encode/decode methods
        device: Device for computation

    Returns:
        dict with: reconstructed_audio, original_audio, sr, metrics
    """
    import torch

    # Encode
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    audio_tensor = audio_tensor.to(device)

    tokens, z = tokenizer.encode(audio_tensor)

    # Decode
    recon_tensor = tokenizer.decode(z)
    recon = recon_tensor.squeeze().cpu().numpy()

    # Compute metrics
    min_len = min(len(audio), len(recon))
    metrics = compute_audio_metrics(audio[:min_len], recon[:min_len], sr)

    return {
        "original_audio": audio,
        "reconstructed_audio": recon,
        "tokens": tokens.cpu().numpy(),
        "sr": sr,
        "metrics": metrics,
    }


def compute_audio_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sr: int,
) -> dict:
    """Compute audio quality metrics between original and reconstructed."""
    import librosa

    # MSE
    mse = np.mean((original - reconstructed) ** 2)

    # Spectral convergence
    S_orig = np.abs(librosa.stft(original))
    S_recon = np.abs(librosa.stft(reconstructed))
    min_t = min(S_orig.shape[1], S_recon.shape[1])
    S_orig = S_orig[:, :min_t]
    S_recon = S_recon[:, :min_t]

    spectral_convergence = np.linalg.norm(S_orig - S_recon) / (np.linalg.norm(S_orig) + 1e-8)

    # Log spectral distance
    S_orig_db = librosa.amplitude_to_db(S_orig + 1e-8)
    S_recon_db = librosa.amplitude_to_db(S_recon + 1e-8)
    log_spectral_distance = np.sqrt(np.mean((S_orig_db - S_recon_db) ** 2))

    # Mel cepstral distortion
    mfcc_orig = librosa.feature.mfcc(y=original, sr=sr, n_mfcc=13)
    mfcc_recon = librosa.feature.mfcc(y=reconstructed, sr=sr, n_mfcc=13)
    min_t = min(mfcc_orig.shape[1], mfcc_recon.shape[1])
    mcd = np.sqrt(2 * np.mean((mfcc_orig[:, :min_t] - mfcc_recon[:, :min_t]) ** 2))

    return {
        "mse": float(mse),
        "spectral_convergence": float(spectral_convergence),
        "log_spectral_distance": float(log_spectral_distance),
        "mel_cepstral_distortion": float(mcd),
    }
