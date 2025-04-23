import torch.nn.functional as F
from torch import Tensor, nn


class Dropout(nn.Module):
    def __init__(self, p: float, mode: str = "norm", inplace: bool = False):
        super().__init__()
        self.p = p
        self.mode = mode
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        if self.p > 0.0 and self.p < 1:
            if self.training or self.mode == "always_training":
                return F.dropout(input, p=self.p, training=True, inplace=self.inplace)
            else:
                return F.dropout(input, p=self.p, training=False, inplace=self.inplace)
        else:
            return input

    def extra_repr(self) -> str:
        return "p={}, mode={}, inplace={}".format(self.p, self.mode, self.inplace)
