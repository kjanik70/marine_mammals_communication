#!/usr/bin/env python3
"""Compare LAC (WhAM) vs DAC codec reconstruction quality.

Tests roundtrip quality at various codebook counts for both codecs.
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


def spectral_convergence(original, reconstructed, sr=44100, n_fft=2048):
    S_orig = np.abs(librosa.stft(original, n_fft=n_fft))
    S_recon = np.abs(librosa.stft(reconstructed, n_fft=n_fft))
    min_t = min(S_orig.shape[1], S_recon.shape[1])
    S_orig, S_recon = S_orig[:, :min_t], S_recon[:, :min_t]
    return float(np.linalg.norm(S_orig - S_recon) / (np.linalg.norm(S_orig) + 1e-10))


def signal_to_noise(original, reconstructed):
    min_len = min(len(original), len(reconstructed))
    orig, recon = original[:min_len], reconstructed[:min_len]
    noise = orig - recon
    signal_power = np.mean(orig ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return 100.0
    return float(10 * np.log10(signal_power / noise_power))


def mel_cepstral_distortion(original, reconstructed, sr=44100):
    mfcc_orig = librosa.feature.mfcc(y=original, sr=sr, n_mfcc=13, n_mels=80)
    mfcc_recon = librosa.feature.mfcc(y=reconstructed, sr=sr, n_mfcc=13, n_mels=80)
    min_t = min(mfcc_orig.shape[1], mfcc_recon.shape[1])
    return float(np.mean(np.sqrt(np.sum((mfcc_orig[:, :min_t] - mfcc_recon[:, :min_t]) ** 2, axis=0))))


def cross_correlation(original, reconstructed):
    min_len = min(len(original), len(reconstructed))
    orig = original[:min_len] - np.mean(original[:min_len])
    recon = reconstructed[:min_len] - np.mean(reconstructed[:min_len])
    denom = np.sqrt(np.sum(orig ** 2) * np.sum(recon ** 2))
    if denom < 1e-10:
        return 0.0
    return float(np.sum(orig * recon) / denom)


class LACCodec:
    """Wrapper for LAC (WhAM weights)."""

    def __init__(self, codec_path, device="cpu"):
        from lac.model.lac import LAC
        self.codec = LAC.load(Path(codec_path))
        self.codec.eval().to(device)
        self.device = device
        self.name = "LAC (WhAM)"
        self.sample_rate = self.codec.sample_rate  # 44100
        self.hop_length = self.codec.hop_length    # 768
        self.max_codebooks = 14
        self.tokens_per_sec = self.sample_rate / self.hop_length

    @torch.no_grad()
    def roundtrip(self, audio_tensor, n_cb):
        audio = audio_tensor.to(self.device)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        result = self.codec.encode(audio, self.sample_rate)
        codes = result["codes"]
        T = codes.shape[-1]
        full_codes = torch.zeros(1, 14, T, dtype=torch.long, device=self.device)
        full_codes[:, :n_cb, :] = codes[:, :n_cb, :]
        z = self.codec.quantizer.from_codes(full_codes)[0]
        recon = self.codec.decode(z)["audio"]
        return recon.squeeze().cpu().numpy()


class DACCodec:
    """Wrapper for Descript Audio Codec."""

    def __init__(self, device="cpu"):
        import dac
        from dac.utils import download
        path = download(model_type="44khz")
        self.model = dac.DAC.load(path).to(device)
        self.model.eval()
        self.device = device
        self.name = "DAC (44kHz)"
        self.sample_rate = self.model.sample_rate  # 44100
        self.hop_length = self.model.hop_length    # 512
        self.max_codebooks = self.model.n_codebooks  # 9
        self.tokens_per_sec = self.sample_rate / self.hop_length

    @torch.no_grad()
    def roundtrip(self, audio_tensor, n_cb):
        audio = audio_tensor.to(self.device)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        # Encode
        z, codes, latents, _, _ = self.model.encode(audio)
        # Zero out codebooks beyond n_cb
        # codes shape: (1, n_codebooks, T)
        codes_masked = codes.clone()
        codes_masked[:, n_cb:, :] = 0
        # Decode from codes
        z_q, _, _ = self.model.quantizer.from_codes(codes_masked)
        recon = self.model.decode(z_q)
        return recon.squeeze().cpu().numpy()


def bandpass_audio(audio, sr, low_hz=80, high_hz=20000):
    """Bandpass filter matching the SanctSound pipeline."""
    from scipy.signal import butter, sosfilt
    nyq = sr / 2
    low = min(low_hz / nyq, 0.95)
    high = min(high_hz / nyq, 0.95)
    if low < high:
        sos = butter(5, [low, high], btype='band', output='sos')
        audio = sosfilt(sos, audio).astype(np.float32)
    return audio


def load_audio(path, sr=44100, max_duration=10.0, preprocess=True):
    audio, file_sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    max_samples = int(max_duration * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    if preprocess:
        # Bandpass 80Hz-20kHz
        audio = bandpass_audio(audio, sr)
        # Peak normalize to 0.9
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio * (0.9 / peak)
    return audio


def main():
    parser = argparse.ArgumentParser(description="Compare LAC vs DAC codec quality")
    parser.add_argument("--lac-path", default="models/codec.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-files", type=int, default=10)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--output-dir", default="runs/codec_comparison")
    parser.add_argument("audio_dirs", nargs="*",
                        default=["data/raw/dswp", "data/raw/humpback_zenodo",
                                 "data/raw/orcasound", "data/raw/esp_orcas"])
    args = parser.parse_args()

    sr = 44100
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find files
    candidates = []
    for d in args.audio_dirs:
        p = Path(d)
        if p.exists():
            candidates.extend(sorted(p.glob("*.wav"))[:50])
            candidates.extend(sorted(p.glob("*.flac"))[:50])
    if not candidates:
        print("No audio files found!")
        return

    rng = np.random.default_rng(42)
    if len(candidates) > args.n_files:
        indices = rng.choice(len(candidates), size=args.n_files, replace=False)
        candidates = [candidates[i] for i in sorted(indices)]

    # Load codecs
    print("Loading LAC (WhAM)...")
    lac = LACCodec(args.lac_path, device=args.device)
    print(f"  SR={lac.sample_rate}, hop={lac.hop_length}, "
          f"tokens/sec={lac.tokens_per_sec:.1f}, max CB={lac.max_codebooks}")

    print("Loading DAC (44kHz)...")
    dac_codec = DACCodec(device=args.device)
    print(f"  SR={dac_codec.sample_rate}, hop={dac_codec.hop_length}, "
          f"tokens/sec={dac_codec.tokens_per_sec:.1f}, max CB={dac_codec.max_codebooks}")

    # Test configurations: (codec, n_codebooks)
    configs = [
        (lac, 1), (lac, 4), (lac, 8), (lac, 14),
        (dac_codec, 1), (dac_codec, 4), (dac_codec, 9),
    ]

    # Aggregate metrics
    all_metrics = {}
    for codec, n_cb in configs:
        key = f"{codec.name} {n_cb}CB"
        all_metrics[key] = {"sc": [], "snr": [], "mcd": [], "xcorr": []}

    print(f"\nTesting {len(candidates)} files\n")

    for fi, fpath in enumerate(candidates):
        audio = load_audio(fpath, sr, args.max_duration)
        dur = len(audio) / sr
        print(f"[{fi}] {fpath.name} ({dur:.1f}s)")

        # Save original
        sf.write(str(out_dir / f"file{fi}_original.wav"), audio, sr)

        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        for codec, n_cb in configs:
            key = f"{codec.name} {n_cb}CB"
            try:
                recon = codec.roundtrip(audio_tensor, n_cb)
                sc = spectral_convergence(audio, recon, sr)
                snr = signal_to_noise(audio, recon)
                mcd = mel_cepstral_distortion(audio, recon, sr)
                xcorr = cross_correlation(audio, recon)

                all_metrics[key]["sc"].append(sc)
                all_metrics[key]["snr"].append(snr)
                all_metrics[key]["mcd"].append(mcd)
                all_metrics[key]["xcorr"].append(xcorr)

                print(f"    {key:16s}: SC={sc:.4f}  SNR={snr:5.1f}dB  MCD={mcd:5.1f}  xcorr={xcorr:.4f}")

                # Save reconstruction
                safe_name = codec.name.replace(" ", "_").replace("(", "").replace(")", "")
                sf.write(str(out_dir / f"file{fi}_{safe_name}_{n_cb}cb.wav"), recon, sr)
            except Exception as e:
                print(f"    {key:16s}: ERROR - {e}")

        print()

    # Summary
    print("=" * 78)
    print(f"{'Codec':>18s} | {'SC (↓)':>8} | {'SNR dB (↑)':>10} | {'MCD (↓)':>8} | {'xcorr (↑)':>10} | {'tok/s':>6}")
    print("-" * 78)
    for codec, n_cb in configs:
        key = f"{codec.name} {n_cb}CB"
        m = all_metrics[key]
        if not m["sc"]:
            continue
        tps = codec.tokens_per_sec * n_cb
        print(f"{key:>18s} | {np.mean(m['sc']):8.4f} | {np.mean(m['snr']):10.1f} | "
              f"{np.mean(m['mcd']):8.1f} | {np.mean(m['xcorr']):10.4f} | {tps:6.0f}")
    print("=" * 78)
    print(f"\nAudio comparisons saved to {out_dir}/")


if __name__ == "__main__":
    main()
