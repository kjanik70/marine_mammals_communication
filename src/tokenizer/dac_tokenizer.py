"""Audio tokenizer wrapping the Descript Audio Codec (DAC).

Provides encode/decode interface for converting audio to discrete tokens
and back, matching the AudioTokenizer API for drop-in use.

DAC 44kHz model: 9 codebooks, hop_length=512, codebook_size=1024.
At 44100 Hz: ~86.1 tokens/sec per codebook.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class DACTokenizer:
    """Wraps the Descript Audio Codec for audio tokenization.

    Uses RVQ with 9 codebooks (44kHz model). Supports encoding to
    discrete codes and decoding back to audio. Interface matches
    AudioTokenizer for drop-in use.
    """

    def __init__(
        self,
        device: str = "cuda",
        n_codebooks: int = 9,
    ):
        import dac
        from dac.utils import download

        self.device = device
        self.n_codebooks = n_codebooks

        path = download(model_type="44khz")
        self.codec = dac.DAC.load(path)
        self.codec.eval()
        self.codec.to(device)

        self._sample_rate = self.codec.sample_rate    # 44100
        self._hop_length = self.codec.hop_length      # 512
        self._max_codebooks = self.codec.n_codebooks  # 9

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def hop_length(self) -> int:
        return self._hop_length

    @property
    def vocab_size(self) -> int:
        """Vocab size per codebook (1024) plus special tokens."""
        return 1024 + 2  # +2 for PAD(0) and offset

    @property
    def tokens_per_second(self) -> float:
        return self._sample_rate / self._hop_length

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode audio to discrete tokens.

        Args:
            audio: (batch, 1, samples) or (batch, samples) tensor at 44100 Hz

        Returns:
            codes: (batch, n_codebooks, time) discrete token IDs (1-1025, 0=PAD)
            z: (batch, latent_dim, time) continuous latent
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        audio = audio.to(self.device)
        z, codes, latents, _, _ = self.codec.encode(audio)

        codes = codes[:, :self.n_codebooks, :]

        # Offset codes by 1 to reserve 0 for PAD
        codes = codes + 1

        return codes, z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from continuous latent back to audio.

        Args:
            z: (batch, latent_dim, time) continuous latent

        Returns:
            audio: (batch, 1, samples) reconstructed audio
        """
        z = z.to(self.device)
        return self.codec.decode(z)

    def encode_file(self, path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an audio file to tokens."""
        import librosa
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != self._sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self._sample_rate)

        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return self.encode(audio_tensor)

    def flatten_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Flatten multi-codebook codes into a single interleaved sequence.

        Interleaves: [cb0_t1, cb1_t1, ..., cbN_t1, cb0_t2, ...]
        Adds codebook-specific offsets so tokens from different codebooks
        don't collide.

        Args:
            codes: (batch, n_codebooks, time)

        Returns:
            flat: (batch, n_codebooks * time) flattened tokens with offsets
        """
        B, C, T = codes.shape
        offsets = torch.arange(C, device=codes.device).unsqueeze(0).unsqueeze(2) * 1024
        offset_codes = codes + offsets
        flat = offset_codes.permute(0, 2, 1).reshape(B, T * C)
        return flat

    def unflatten_codes(self, flat: torch.Tensor, n_codebooks: int) -> torch.Tensor:
        """Reverse flatten_codes.

        Args:
            flat: (batch, T*n_codebooks) flattened tokens
            n_codebooks: number of codebooks

        Returns:
            codes: (batch, n_codebooks, T) with offsets removed
        """
        B, L = flat.shape
        T = L // n_codebooks
        codes = flat.view(B, T, n_codebooks).permute(0, 2, 1)
        offsets = torch.arange(n_codebooks, device=codes.device).unsqueeze(0).unsqueeze(2) * 1024
        codes = codes - offsets
        return codes

    def codes_to_sequence(self, codes: torch.Tensor) -> np.ndarray:
        """Convert codes tensor to 1D numpy array for dataset storage.

        For single codebook: just squeeze and return.
        For multi-codebook: flatten with offsets.
        """
        if codes.shape[1] == 1:
            return codes[0, 0].cpu().numpy().astype(np.int32)
        else:
            flat = self.flatten_codes(codes)
            return flat[0].cpu().numpy().astype(np.int32)

    def codes_to_2d(self, codes: torch.Tensor) -> np.ndarray:
        """Convert codes tensor to 2D numpy array (n_codebooks, T).

        Saves raw codes with +1 offset (PAD=0), no interleaving or
        codebook offsets. Used for hierarchical models where different
        codebooks are loaded separately.
        """
        return codes[0].cpu().numpy().astype(np.int32)

    @torch.no_grad()
    def decode_tokens_to_audio(
        self,
        tokens: np.ndarray,
        n_codebooks: int = 9,
        sep_token: int | None = None,
    ) -> np.ndarray:
        """Decode a 1D token sequence back to audio.

        Handles both single-codebook and multi-codebook (interleaved) tokens.
        Filters out PAD (0) and SEP tokens before decoding.

        Args:
            tokens: 1D numpy array of token IDs
            n_codebooks: Number of codebooks used during tokenization
            sep_token: SEP token ID to filter out (None = no filtering)

        Returns:
            audio: 1D numpy array of audio samples at 44100 Hz
        """
        # Filter out PAD and SEP tokens
        mask = tokens > 0
        if sep_token is not None:
            mask &= tokens != sep_token
        tokens = tokens[mask]

        if len(tokens) == 0:
            return np.zeros(0, dtype=np.float32)

        if n_codebooks == 1:
            codes = np.clip(tokens - 1, 0, 1023)
            codes_tensor = torch.tensor(codes, dtype=torch.long, device=self.device)
            codes_tensor = codes_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        else:
            trim_len = (len(tokens) // n_codebooks) * n_codebooks
            tokens = tokens[:trim_len]
            flat_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            codes_tensor = self.unflatten_codes(flat_tensor, n_codebooks)
            codes_tensor = torch.clamp(codes_tensor - 1, 0, 1023)

        # Pad to 9 codebooks (DAC expects all codebooks)
        T = codes_tensor.shape[-1]
        full_codes = torch.zeros(1, self._max_codebooks, T, dtype=torch.long, device=self.device)
        full_codes[:, :codes_tensor.shape[1], :] = codes_tensor

        z_q, _, _ = self.codec.quantizer.from_codes(full_codes)
        audio = self.codec.decode(z_q)
        return audio.squeeze().cpu().numpy()

    @torch.no_grad()
    def decode_2d_to_audio(
        self,
        codes_2d: np.ndarray,
        n_codebooks: int | None = None,
    ) -> np.ndarray:
        """Decode a 2D codes array (n_codebooks, T) back to audio.

        Args:
            codes_2d: 2D numpy array of shape (n_codebooks, T), values 1-1025
            n_codebooks: Use only first N codebooks (None = use all rows)

        Returns:
            audio: 1D numpy array of audio samples at 44100 Hz
        """
        if n_codebooks is not None:
            codes_2d = codes_2d[:n_codebooks, :]

        # Remove +1 offset, clip to valid range
        codes = np.clip(codes_2d - 1, 0, 1023)
        codes_tensor = torch.tensor(codes, dtype=torch.long, device=self.device).unsqueeze(0)

        # Pad to 9 codebooks
        C, T = codes_tensor.shape[1], codes_tensor.shape[2]
        full_codes = torch.zeros(1, self._max_codebooks, T, dtype=torch.long, device=self.device)
        full_codes[:, :C, :] = codes_tensor

        z_q, _, _ = self.codec.quantizer.from_codes(full_codes)
        audio = self.codec.decode(z_q)
        return audio.squeeze().cpu().numpy()
