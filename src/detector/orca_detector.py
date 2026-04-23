"""OrcaDetector: binary orca-call classifier on 3-second log-mel spectrogram windows.

Model: 4-block CNN with strided convolutions, global average pool, linear head.
Input: (B, 1, 128, T) log-mel spectrogram, normalised to roughly [0, 1].
Output: scalar logit per sample (use sigmoid for probability).

Inference entry points
----------------------
OrcaDetector.score(audio_1d)        → float  probability for a ≤3 s clip
OrcaDetector.scan(audio_1d, hop_s)  → [(t0, t1, prob), ...]
OrcaDetector.has_orca(audio_1d)     → bool
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio


SR = 44100
WIN_SAMPLES = 3 * SR          # 132 300 samples per window
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
F_MIN = 50
F_MAX = 20_000
TOP_DB = 80.0
# Normalisation: AmplitudeToDB output ∈ [-TOP_DB, 0]; shift and scale to [0, 1]
SPEC_SHIFT = TOP_DB            # add to bring range to [0, TOP_DB]
SPEC_SCALE = TOP_DB            # divide to get [0, 1]


def make_mel_transform(sr: int = SR) -> nn.Sequential:
    return nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX,
        ),
        torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB),
    )


def audio_to_spec(audio_1d: np.ndarray, mel_transform: nn.Sequential) -> np.ndarray:
    """Convert a mono float32 array at SR to a (N_MELS, T) float16 spectrogram."""
    t = torch.tensor(audio_1d, dtype=torch.float32).unsqueeze(0)  # (1, samples)
    with torch.no_grad():
        s = mel_transform(t)                   # (1, N_MELS, T)
    s = (s.squeeze(0).numpy() + SPEC_SHIFT) / SPEC_SCALE  # → [0, 1]
    return s.astype(np.float16)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class OrcaDetectorCNN(nn.Module):
    """~1 M-param CNN binary classifier for orca call detection."""

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1,   32, stride=2),   # → (32, H/2, W/2)
            _conv_block(32,  64, stride=2),   # → (64, H/4, W/4)
            _conv_block(64,  128, stride=2),  # → (128, H/8, W/8)
            _conv_block(128, 256, stride=2),  # → (256, H/16, W/16)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, N_MELS, T)
        x = self.features(x)
        x = self.pool(x).flatten(1)    # (B, 256)
        return self.head(x).squeeze(1)  # (B,) logits


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class OrcaDetector:
    """Load a trained checkpoint and run inference on raw audio.

    Args:
        checkpoint_path: path to .pt checkpoint saved by train_orca_detector.py
        device: 'cuda' or 'cpu'
        threshold: probability threshold for has_orca()
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        threshold: float = 0.5,
    ):
        self.device = device
        self.threshold = threshold

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = OrcaDetectorCNN().to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self._mel = make_mel_transform(SR).to(device)

    @torch.no_grad()
    def _spec_tensor(self, audio_1d: np.ndarray) -> torch.Tensor:
        """Audio array → (1, 1, N_MELS, T) normalised tensor on device."""
        t = torch.tensor(audio_1d, dtype=torch.float32, device=self.device).unsqueeze(0)
        s = self._mel(t)                              # (1, N_MELS, T)
        s = (s + SPEC_SHIFT) / SPEC_SCALE
        return s.unsqueeze(0)                         # (1, 1, N_MELS, T)

    @torch.no_grad()
    def score(self, audio_1d: np.ndarray) -> float:
        """Orca probability for a ≤3 s mono clip at 44 100 Hz."""
        chunk = np.zeros(WIN_SAMPLES, dtype=np.float32)
        n = min(len(audio_1d), WIN_SAMPLES)
        chunk[:n] = audio_1d[:n]
        logit = self.model(self._spec_tensor(chunk))
        return float(torch.sigmoid(logit).item())

    @torch.no_grad()
    def scan(
        self,
        audio_1d: np.ndarray,
        hop_s: float = 1.5,
    ) -> list[tuple[float, float, float]]:
        """Slide 3 s windows over audio.

        Returns list of (t_start, t_end, probability).
        """
        hop = max(1, int(hop_s * SR))
        total = len(audio_1d)
        results: list[tuple[float, float, float]] = []
        for start in range(0, max(1, total - WIN_SAMPLES + 1), hop):
            chunk = audio_1d[start:start + WIN_SAMPLES]
            if len(chunk) < WIN_SAMPLES:
                padded = np.zeros(WIN_SAMPLES, dtype=np.float32)
                padded[:len(chunk)] = chunk
                chunk = padded
            prob = self.score(chunk)
            results.append((start / SR, start / SR + 3.0, prob))
        return results

    def has_orca(self, audio_1d: np.ndarray) -> bool:
        """True if any 3 s window scores ≥ threshold."""
        return any(p >= self.threshold for _, _, p in self.scan(audio_1d))
