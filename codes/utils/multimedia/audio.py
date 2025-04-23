import os.path as osp

from utils.io import mfs, wav_save

from .base import MultiMedia


class LogAudios(MultiMedia):
    def __init__(self, log_path="./", sub_log_path="audio", persistent=["tensorboard", "file"], sample_rate=24000):
        super().__init__(log_path, sub_log_path)
        self.persistent = persistent
        self.sample_rate = sample_rate

    def process(self, iter, outputs, mode):
        log_path = osp.join(self.log_path, mode, self.sub_log_path)
        if not mfs.exists(log_path):
            mfs.makedirs(log_path)
        for key, value in outputs.items():
            if key.startswith("audio_"):
                name = key[6:]
                if "file" in self.persistent:
                    for i in range(value.size(0)):
                        file_name = f"{name}_{iter + 1}_step_{i}.wav"
                        file_path = osp.join(log_path, file_name)
                        if value[i].get_device() < 0:
                            val_np = value[i].numpy()
                        else:
                            val_np = value[i].cpu().numpy()
                        wav_save(path=file_path, wav=val_np, sr=self.sample_rate)
                if "tensorboard" not in self.persistent:
                    outputs.pop(key, None)
