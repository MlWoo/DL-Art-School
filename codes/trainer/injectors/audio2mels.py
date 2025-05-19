import os

import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from trainer.inject import Injector
from utils.options import opt_get

MEL_MIN = -11.512925148010254
TACOTRON_MEL_MAX = 2.3143386840820312
TORCH_MEL_MAX = 4.82  # FYI: this STILL isn't assertive enough...


def normalize_torch_mel(mel):
    return 2 * ((mel - MEL_MIN) / (TORCH_MEL_MAX - MEL_MIN)) - 1


def denormalize_torch_mel(norm_mel):
    return ((norm_mel + 1) / 2) * (TORCH_MEL_MAX - MEL_MIN) + MEL_MIN


def normalize_mel(mel):
    return 2 * ((mel - MEL_MIN) / (TACOTRON_MEL_MAX - MEL_MIN)) - 1


def denormalize_mel(norm_mel):
    return ((norm_mel + 1) / 2) * (TACOTRON_MEL_MAX - MEL_MIN) + MEL_MIN


class MelSpectrogramInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        from models.audio.tts.tacotron2 import TacotronSTFT

        # These are the default tacotron values for the MEL spectrogram.
        filter_length = opt_get(opt, ["filter_length"], 1024)
        hop_length = opt_get(opt, ["hop_length"], 300)
        win_length = opt_get(opt, ["win_length"], 1024)
        n_mel_channels = opt_get(opt, ["n_mel_channels"], 80)
        mel_fmin = opt_get(opt, ["mel_fmin"], 0)
        mel_fmax = opt_get(opt, ["mel_fmax"], 12000)
        sampling_rate = opt_get(opt, ["sampling_rate"], 24000)
        self.stft = TacotronSTFT(
            filter_length, hop_length, win_length, n_mel_channels, sampling_rate, mel_fmin, mel_fmax
        )
        self.do_normalization = opt_get(
            opt, ["do_normalization"], None
        )  # This is different from the TorchMelSpectrogramInjector. This just normalizes to the range [-1,1]

    def forward(self, state):
        inp = state[self.input]
        if (
            len(inp.shape) == 3
        ):  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        self.stft = self.stft.to(inp.device)
        mel = self.stft.mel_spectrogram(inp)
        if self.do_normalization:
            mel = normalize_mel(mel)
        return {self.output: mel}


class TorchMelSpectrogramInjector(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = opt_get(opt, ["filter_length"], 1024)
        self.hop_length = opt_get(opt, ["hop_length"], 240)
        self.win_length = opt_get(opt, ["win_length"], 960)
        self.n_mel_channels = opt_get(opt, ["n_mel_channels"], 80)
        self.mel_fmin = opt_get(opt, ["mel_fmin"], 0)
        self.mel_fmax = opt_get(opt, ["mel_fmax"], 12000)
        self.sampling_rate = opt_get(opt, ["sampling_rate"], 24000)
        norm = opt_get(opt, ["normalize"], False)
        self.true_norm = opt_get(opt, ["true_normalization"], False)
        self.spec_out = opt_get(opt, ["spec_out"], "spec")
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            power=2,
            normalized=norm,
            sample_rate=self.sampling_rate,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax,
            n_mels=self.n_mel_channels,
            norm="slaney",
        ).float()
        self.mel_norm_file = opt_get(opt, ["mel_norm_file"], None)
        if self.mel_norm_file is not None:
            self.mel_norms = torch.load(self.mel_norm_file)
        else:
            self.mel_norms = None

    def forward(self, state):
        with torch.no_grad():
            inp = state[self.input]
            if (
                len(inp.shape) == 3
            ):  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
                inp = inp.squeeze(1)
            assert len(inp.shape) == 2
            self.mel_stft = self.mel_stft.to(inp.device)
            specgram = self.mel_stft.spectrogram(inp)
            # mel = torch.matmul(specgram.to(torch.bfloat16).transpose(-1, -2),
            #                    self.mel_stft.mel_scale.fb.to(torch.bfloat16)).transpose(-1, -2)
            mel = self.mel_stft.mel_scale(specgram)
            # Perform dynamic range compression
            mel = torch.log(torch.clamp(mel, min=1e-5))
            if self.mel_norms is not None:
                self.mel_norms = self.mel_norms.to(mel.device)
                mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
            if self.true_norm:
                mel = normalize_torch_mel(mel)
            return {self.output: mel, "spec": specgram}


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


class CanonicalTorchMelSpectrogram(Injector):
    def __init__(self, opt, env):
        super().__init__(opt, env)
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = opt_get(opt, ["filter_length"], 1024)
        self.hop_length = opt_get(opt, ["hop_length"], 240)
        self.win_length = opt_get(opt, ["win_length"], 960)
        self.n_mel_channels = opt_get(opt, ["n_mel_channels"], 80)
        self.mel_fmin = opt_get(opt, ["mel_fmin"], 0)
        self.mel_fmax = opt_get(opt, ["mel_fmax"], 12000)
        self.sampling_rate = opt_get(opt, ["sampling_rate"], 24000)
        self.true_norm = opt_get(opt, ["true_normalization"], False)
        self.spec_out = opt_get(opt, ["spec_out"], "spec")
        self.mel_norm_file = opt_get(opt, ["mel_norm_file"], None)
        self.center = opt_get(opt, ["center"], False)
        self.rescaling_max = opt_get(opt, ["rescaling_max"], 0.999)
        self.mel_norms = None
        self.load_mel_norms(self.mel_norm_file)
        self.pad_audio = int((self.filter_length - self.hop_length) / 2)

    def load_mel_norms(self, mel_norm_file):
        if self.mel_norms is None and mel_norm_file is not None and os.path.exists(mel_norm_file):
            self.mel_norms = torch.load(mel_norm_file, weights_only=True)
        return self.mel_norms

    @torch.no_grad()
    def forward(self, state):
        inp = state.pop(self.input)
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

        return {"mel": mel, self.output: mel}

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
