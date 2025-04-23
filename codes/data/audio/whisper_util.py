from typing import Optional, Union

import numpy as np
import torch
from torch import nn

try:
    from whisper.audio import load_audio, mel_filters
except ImportError:
    # borrow from whipser
    import os
    import os.path as osp
    from functools import lru_cache
    from subprocess import CalledProcessError, run

    SAMPLE_RATE = 16000

    def load_audio(file: str, sr: int = SAMPLE_RATE):
        """
        Open an audio file and read as mono waveform, resampling as necessary

        Parameters
        ----------
        file: str
            The audio file to open

        sr: int
            The sample rate to resample the audio if necessary

        Returns
        -------
        A NumPy array containing the audio waveform, in float32 dtype.
        """

        # This launches a subprocess to decode audio while down-mixing
        # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
        # fmt: off
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-"
        ]
        # fmt: on
        try:
            out = run(cmd, capture_output=True, check=True).stdout
        except CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    @lru_cache(maxsize=None)
    def mel_filters(device, n_mels: int) -> torch.Tensor:
        """
        load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
        Allows decoupling librosa dependency; saved using:

            np.savez_compressed(
                "mel_filters.npz",
                mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
                mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
            )
        """
        assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

        filters_path = osp.join(osp.dirname(__file__), "assets", "mel_filters.npz")
        if not osp.exists(filters_path):
            dirname = osp.join(osp.dirname(__file__), "assets")
            os.makedirs(dirname, exist_ok=True)
            import librosa

            np.savez_compressed(
                filters_path,
                mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
                mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
            )
        with np.load(filters_path, allow_pickle=False) as f:
            return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram_batch(
    waveform: Union[str, np.ndarray, torch.Tensor],
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 128,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-mel spectrogram of the audio using PyTorch's GPU-accelerated STFT implementation with batching,
    yielding results similar to cpu computing with 1e-5 tolerance.
    """
    if not torch.is_tensor(waveform):
        if isinstance(waveform, str):
            waveform = load_audio(waveform)
        waveform = torch.from_numpy(waveform)

    window = torch.hann_window(n_fft)
    if device != "cpu":
        waveform = waveform.to(device)
        window = window.to(device)

    if padding > 0:
        waveform = nn.functional.pad(waveform, (0, padding))

    stft = torch.stft(waveform, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    mel_filter = mel_filters(device, n_mels).type(torch.float32)
    mel_spec = mel_filter.T @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    if waveform.dim() == 2:
        max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        log_spec = torch.maximum(log_spec, max_val - 8.0)
    else:
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    if device != "cpu":
        log_spec = log_spec.detach().cpu()

    return log_spec
