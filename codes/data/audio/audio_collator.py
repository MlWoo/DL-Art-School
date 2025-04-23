import torch
from trainer.injectors.audio2mels import CanonicalTorchMelSpectrogram, TorchMelSpectrogramInjector
from utils.options import opt_get

from data.base.collator import Collator
from data.builder import COLLATIONS


@COLLATIONS.register_module()
class MelsInferCollator(Collator):
    SPEC_FUNCS = dict(original=TorchMelSpectrogramInjector, canonical=CanonicalTorchMelSpectrogram)

    def __init__(self, opt):
        super().__init__(opt)
        self.wav_key = "audio"
        self.len_key = "frame_lengths"
        self.frame_key = "mel"
        self.mel_inj_cfg = {
            "in": "in",
            "out": "out",
            "mel_fmin": 0,
            "mel_fmax": 8000,
            "sampling_rate": 16000,
            "n_mel_channels": 80,
            "filter_length": 1024,
            "hop_length": 160,
            "win_length": 640,
            "do_normalization": False,
            "mel_norm_file": None,
        }
        try:
            self.mel_inj_cfg.update(**opt["audio_process"]["mel"])
        except:  # noqa: E722
            pass
        spec_fn = opt_get(opt, ["spec_fn"], "canonical")
        self.spec_fn = self.SPEC_FUNCS[spec_fn](self.mel_inj_cfg, {})
        self.device = opt_get(opt, ["device"], -1)
        self.pad_mode = opt_get(opt, ["pad_mode"], "extra")
        self.pop_audio = opt_get(opt, ["pop_audio"], True)

    def _postprocess_tensor_(self, batch_dict, force_inter=False):
        if force_inter or self.inter_post:
            audios = batch_dict.pop(self.wav_key)
            if audios.dtype in [torch.int64, torch.int32, torch.int16]:
                audios = audios.to(torch.float32) / 32767.0
            if self.device >= 0:
                audios = audios.to(self.device)
            if self.pop_audio:
                audio_lengths = batch_dict.pop("audio_lengths")
            else:
                audio_lengths = batch_dict["audio_lengths"]
                batch_dict[self.wav_key] = audios

            outs = self.spec_fn({"in": audios})
            batch_dict[self.frame_key] = outs["mel"].to(torch.bfloat16)  # [..., :-1]
            extra_val = 1 if self.pad_mode == "extra" else 0
            batch_dict["mel_lengths"] = audio_lengths // self.mel_inj_cfg["hop_length"] + extra_val
            return batch_dict
        else:
            return batch_dict
