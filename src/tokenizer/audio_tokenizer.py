"""Audio tokenizer wrapping the LAC (Learned Audio Codec) from WhAM.

Provides encode/decode interface for converting audio to discrete tokens
and back, for use with the autoregressive transformer.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np


class AudioTokenizer:
    """Wraps the LAC codec for audio tokenization.

    The LAC codec uses Residual Vector Quantization (RVQ) to produce
    multiple codebook indices per time step. This wrapper supports:
    - First codebook only (coarse tokens, ~57 tokens/sec with WhAM weights)
    - All codebooks flattened (interleaved, richer representation)
    """

    def __init__(
        self,
        codec_path: Optional[str | Path] = None,
        device: str = "cuda",
        n_codebooks: int = 1,  # 1 = first codebook only, 14 = all (WhAM)
    ):
        from lac.model.lac import LAC

        self.device = device
        self.n_codebooks = n_codebooks

        if codec_path and Path(codec_path).exists():
            self.codec = LAC.load(Path(codec_path))
        else:
            self.codec = LAC()

        self.codec.eval()
        self.codec.to(device)

        self._sample_rate = self.codec.sample_rate  # 44100
        self._hop_length = self.codec.hop_length    # 768 (WhAM), 512 (default LAC)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def vocab_size(self) -> int:
        """Vocab size per codebook (1024) plus special tokens."""
        return 1024 + 2  # +2 for PAD(0) and BOS/EOS if needed

    @property
    def tokens_per_second(self) -> float:
        return self._sample_rate / self._hop_length

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode audio to discrete tokens.

        Args:
            audio: (batch, 1, samples) or (batch, samples) tensor at codec sample rate

        Returns:
            codes: (batch, n_codebooks, time) discrete token IDs (0-1023)
            z: (batch, latent_dim, time) continuous latent for decoding
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        audio = audio.to(self.device)
        result = self.codec.encode(audio, self._sample_rate)

        codes = result["codes"][:, :self.n_codebooks, :]
        z = result["z"]

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
        result = self.codec.decode(z)
        return result["audio"]

    def encode_file(self, path: str | Path, target_sr: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an audio file to tokens."""
        import soundfile as sf
        import librosa

        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to codec sample rate
        if sr != self._sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self._sample_rate)

        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return self.encode(audio_tensor)

    def flatten_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Flatten multi-codebook codes into a single sequence.

        Interleaves codebooks: [cb1_t1, cb2_t1, ..., cbN_t1, cb1_t2, ...]

        Args:
            codes: (batch, n_codebooks, time)

        Returns:
            flat: (batch, n_codebooks * time) flattened tokens with codebook offsets
        """
        B, C, T = codes.shape
        # Add codebook-specific offsets so tokens from different codebooks don't collide
        offsets = torch.arange(C, device=codes.device).unsqueeze(0).unsqueeze(2) * 1024
        offset_codes = codes + offsets
        # Interleave: (B, C, T) -> (B, T, C) -> (B, T*C)
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
        # (B, T*C) -> (B, T, C) -> (B, C, T)
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

    @torch.no_grad()
    def decode_tokens_to_audio(
        self,
        tokens: np.ndarray,
        n_codebooks: int = 1,
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
            audio: 1D numpy array of audio samples at codec sample rate
        """
        # Filter out PAD and SEP tokens
        mask = tokens > 0
        if sep_token is not None:
            mask &= tokens != sep_token
        tokens = tokens[mask]

        if len(tokens) == 0:
            return np.zeros(0, dtype=np.float32)

        if n_codebooks == 1:
            # Single codebook: remove +1 offset, clip to [0, 1023]
            codes = np.clip(tokens - 1, 0, 1023)
            codes_tensor = torch.tensor(codes, dtype=torch.long, device=self.device)
            codes_tensor = codes_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        else:
            # Multi-codebook: trim to multiple of n_codebooks, then unflatten
            trim_len = (len(tokens) // n_codebooks) * n_codebooks
            tokens = tokens[:trim_len]
            flat_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            codes_tensor = self.unflatten_codes(flat_tensor, n_codebooks)
            # Remove +1 PAD offset and clip
            codes_tensor = torch.clamp(codes_tensor - 1, 0, 1023)

        # Pad to 14 codebooks (full codec expects all codebooks)
        T = codes_tensor.shape[-1]
        full_codes = torch.zeros(1, 14, T, dtype=torch.long, device=self.device)
        full_codes[:, :codes_tensor.shape[1], :] = codes_tensor

        z = self.codec.quantizer.from_codes(full_codes)[0]
        audio = self.codec.decode(z)["audio"]
        return audio.squeeze().cpu().numpy()
