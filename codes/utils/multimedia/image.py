import os.path as osp

from utils.io import image_save, mfs

from .base import MultiMedia


class LogImages(MultiMedia):
    def __init__(self, log_path="./", sub_log_path="images", persistent=["tensorboard", "file"]):
        super().__init__(log_path, sub_log_path)
        self.persistent = persistent

    def process(self, iter, outputs, mode):
        log_path = osp.join(self.log_path, mode, self.sub_log_path)
        if not mfs.exists(log_path):
            mfs.makedirs(log_path)
        for key, value in outputs.items():
            if key.startswith("image_"):
                name = key[6:]
                if "file" in self.persistent:
                    for i in range(value.size(0)):
                        file_name = f"{name}_{iter}_step_{i}.jpeg"
                        file_path = osp.join(log_path, file_name)
                        image_save(tensor=value[i], path=file_path)

                if "tensorboard" not in self.persistent:
                    outputs.pop("key", None)
