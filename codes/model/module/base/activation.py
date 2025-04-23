import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NewGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        GELU activation
        https://arxiv.org/abs/1606.08415
        https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
        """
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        # return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(F.softplus(x))


class Tanh(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: ../scripts/activation_images/Tanh.png

    Examples::

        >>> m = nn.Tanh()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return torch.tanh(input)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else "inplace=False"
        return inplace_str


class Detach(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = x.detach()
        return x


ACT2FN = {
    "None": None,
    "relu": partial(nn.ReLU, inplace=True),
    "relu6": partial(nn.ReLU6, inplace=True),
    "leakey_relu": partial(nn.LeakyReLU, inplace=True),
    "tanh": partial(Tanh, inplace=True),
    "glu": nn.GLU,
    "swish": partial(nn.SiLU, inplace=True),
    "gelu": nn.GELU,
    "new_gelu": NewGELU,
    "mish": Mish,
    "softsign": nn.Softsign,
    "softplus": nn.Softplus,
    "detach": Detach,
}
