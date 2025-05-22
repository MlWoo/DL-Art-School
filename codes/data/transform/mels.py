import os
from typing import Any, Dict

import librosa
import torch

from data.builder import TRANSFORMS


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


@TRANSFORMS.register_module()
class CanonicalTorchMelSpectrogram:
    def __init__(
        self,
        input_key: str = "audio",
        output_key: str = "mel",
        sampling_rate: int = 24000,
        filter_length: int = 1024,
        hop_length: int = 240,
        win_length: int = 960,
        n_mel_channels: int = 80,
        mel_fmin: int = 0,
        mel_fmax: int = 12000,
        true_norm: bool = False,
        center: bool = False,
        rescaling_max: float = 0.999,
        mel_norm_file: str = None,
        **kwargs
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.true_norm = true_norm
        self.mel_norm_file = mel_norm_file
        self.center = center
        self.rescaling_max = rescaling_max
        self.pad_audio = int((self.filter_length - self.hop_length) / 2)

        self.mel_norms = None
        self.load_mel_norms(self.mel_norm_file)

    def load_mel_norms(self, mel_norm_file):
        if self.mel_norms is None and mel_norm_file is not None and os.path.exists(mel_norm_file):
            self.mel_norms = torch.load(mel_norm_file, weights_only=True)
        return self.mel_norms

    @torch.no_grad()
    def __call__(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        inp = batch_dict[self.input_key]
        if (
            len(inp.shape) == 3
        ):  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        mel = self.mel_spectrogram_torch(inp)
        mel_norms = self.load_mel_norms(self.mel_norm_file)
        if mel_norms is not None:
            if isinstance(mel_norms, torch.Tensor):
                mel_norms = mel_norms.to(mel.device)
                mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
            else:
                assert len(mel_norms) == 2
                mean = mel_norms["mean"].to(mel.device)[:, None]
                if "std_dev" in mel_norms:
                    stddev = mel_norms["std_dev"].to(mel.device)[:, None]
                else:
                    stddev = mel_norms["stddev"].to(mel.device)[:, None]
                mel = (mel - mean) / stddev
        batch_dict[self.output_key] = mel
        return batch_dict

    def mel_spectrogram_torch(self, x):
        norm_volume = False
        if torch.min(x) < -1.0:
            # print('min value is ', torch.min(x))
            norm_volume = True
        if torch.max(x) > 1.0:
            # print('max value is ', torch.max(x))
            norm_volume = True
        if norm_volume:
            x = x / torch.abs(x).max() * self.rescaling_max
        dtype = x.dtype
        device = x.device
        global mel_basis, hann_window
        dtype_device = str(dtype) + "_" + str(device)
        fmax_dtype_device = str(self.mel_fmax) + "_" + dtype_device
        wnsize_dtype_device = str(self.win_length) + "_" + dtype_device
        if fmax_dtype_device not in mel_basis:
            mel = librosa_mel_fn(
                sr=self.sampling_rate,
                n_fft=self.filter_length,
                n_mels=self.n_mel_channels,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,
            )
            mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=dtype, device=device)
        if wnsize_dtype_device not in hann_window:
            hann_window[wnsize_dtype_device] = torch.hann_window(self.win_length).to(dtype=dtype, device=device)

        x = torch.nn.functional.pad(x, (self.pad_audio, self.pad_audio), mode="reflect")

        spec = torch.stft(
            x.float(),
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=hann_window[wnsize_dtype_device].float(),
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = spec.abs().to(dtype=dtype)
        mels = torch.matmul(mel_basis[fmax_dtype_device], spec)
        mels = spectral_normalize_torch(mels)

        return mels
