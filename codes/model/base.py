from abc import ABC
from typing import List

import torch
import torch.nn as nn
from utils.multimedia import LogImages
from utils.plot import plot_images


class BaseModule(nn.Module, ABC):
    def __init__(
        self,
        track_inputs: bool = True,
        num_attn_visual_debug: int = 1,
        num_visual_debug: int = 1,
        debug_val_keys: List[str] = ["input_size", "input_length", "acc"],
    ):
        super().__init__()
        self.debug_info = {} if track_inputs else None
        self.num_attn_visual_debug = num_attn_visual_debug
        self.num_visual_debug = num_visual_debug
        self.debug_val_keys = debug_val_keys
        self.run_info = {}

    def feed_run_info(self, epoch, step, will_log, will_visual, **kwargs):
        self.run_info["epoch"] = epoch
        self.run_info["iter"] = step
        self.run_info["will_log"] = will_log
        self.run_info["will_visual"] = will_visual
        for k, v in kwargs.items():
            self.run_info[k] = v

    def get_debug_values(self, step, __):
        if self.debug_info is None:
            all_info = {}
        else:
            all_info = self.debug_info
        debug_info = {}
        for k in self.debug_val_keys:
            debug_info[k] = all_info[k]
        return debug_info

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_annotation(self, inputs_dict, indice):
        return None

    @torch.no_grad()
    def visual_dbg(self, it, model_vdbg_dir):
        num_split = 1
        logger = LogImages(model_vdbg_dir, "", persistent=["file"])
        indice = list(range(self.num_visual_debug))
        images_dict = dict()
        plot_cfg = self.visual_cfg()
        if plot_cfg is not None:
            annotation = None  # self.get_annotation(self.inputs_dict, indice)
            for cfg_k, cfg_v in plot_cfg.items():
                valid = True
                for k in cfg_v["tensor_keys"]:
                    if k not in self.debug_info or self.debug_info[k] is None:
                        valid = False
                        break
                if not valid:
                    continue
                images = plot_images(
                    self.debug_info,
                    tensor_keys=cfg_v["tensor_keys"],
                    visual_methods=cfg_v["visual_methods"],
                    color_info=cfg_v.get("color_info", None),
                    y_lim_info=cfg_v.get("y_lim_info", None),
                    y_pos_info=cfg_v.get("y_pos_info", None),
                    indice=indice,
                    titles=cfg_v.get("titles", None),
                    t_labels=cfg_v.get("t_labels", None),
                    l_labels=cfg_v.get("l_labels", None),
                    shapes_keys=cfg_v.get("shapes_keys", None),
                    texts=annotation,
                    split_text=True,
                    num_split=num_split,
                    width=cfg_v.get("width", 10),
                    height=cfg_v.get("height", 4),
                    align_direction=cfg_v.get("align_direction", "v"),
                    colorbar=True,
                )
                if images is not None:
                    images_dict["image_" + cfg_k] = images
        logger.process(it, images_dict, mode="")
