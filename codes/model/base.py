from abc import ABC
from typing import List

import torch.nn as nn


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
