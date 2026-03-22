"""Audio playback and file I/O utilities."""

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


def save_audio(
    audio: np.ndarray,
    sr: int,
    path: str | Path,
) -> None:
    """Save audio as WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def load_audio(
    path: str | Path,
    target_sr: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """Load audio file, optionally resample."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono

    if target_sr and target_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr
