import os.path as osp

import torch
from utils.io import mfs, np_save

from .base import MultiMedia


class LogTensors(MultiMedia):
    def __init__(
        self,
        log_path="./",
        sub_log_path="audio",
        keywords=["audios"],
        split_on_batch=False,
        shapes_keywords=None,
        filtered_keywords=None,
    ):
        super().__init__(log_path, sub_log_path)
        self.keywords = keywords
        self.filtered_keywords = filtered_keywords
        self.shapes_keywords = shapes_keywords
        self.split_on_batch = split_on_batch

    def process(self, iter, outputs, mode):
        log_path = osp.join(self.log_path, mode, self.sub_log_path)
        if not mfs.exists(log_path):
            mfs.makedirs(log_path)
        labels = []
        tensors = []
        shapes_keywords = []
        for i, keyword in enumerate(self.keywords):
            if keyword in outputs.keys():
                val = outputs[keyword]
                labels.append(keyword)
                tensors.append(val)
                shapes_keywords.append(self.shapes_keywords[i])
                del outputs[keyword]

        if self.filtered_keywords is not None:
            for keyword in self.filtered_keywords:
                if keyword in outputs.keys():
                    del outputs[keyword]

        if len(tensors) > 0:
            if self.split_on_batch:
                batch_size = tensors[0].shape[0]
                for i, tensor in enumerate(tensors):
                    shapes = []
                    if isinstance(shapes_keywords[0], (list, tuple)):
                        shape_keywords = shapes_keywords[i]
                    else:
                        shape_keywords = shapes_keywords
                    for shape_keyword in shape_keywords:
                        if shape_keyword is None or shape_keyword not in outputs.keys():
                            shape_tensor = torch.LongTensor([-1] * batch_size)
                            shapes.append(shape_tensor)
                        else:
                            shapes.append(outputs[shape_keyword])

                    if tensor.get_device() < 0:
                        np_array = tensor.numpy()
                    else:
                        np_array = tensor.detach().cpu().numpy()
                    for j in range(batch_size):
                        arr = np_array[j]
                        for k, shape_cfg in enumerate(shapes):
                            if shape_cfg[j] >= 0:
                                arr = arr.take(indices=range(shape_cfg[j]), axis=k)
                        if "file_names" in outputs:
                            file_name = osp.basename(outputs["file_names"][j])
                            file_name = osp.splitext(file_name)[0]
                            file_name = f"{mode}_{labels[i]}_{file_name}.npy"
                        else:
                            file_name = f"{labels[i]}_batch_{iter + i}_{j}.npy"
                        file_path = osp.join(log_path, file_name)
                        np_save(file_path, arr)
                return None
            else:
                for i, tensor in enumerate(tensors):
                    file_name = f"{labels[i]}_{iter + 1}_step.npy"
                    file_path = osp.join(log_path, file_name)
                    if tensor.get_device() < 0:
                        np_array = tensor.numpy()
                    else:
                        np_array = tensor.detach().cpu().numpy()
                    np_save(file_path, np_array)
                return None
