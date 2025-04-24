from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .activation import ACT2FN
from .dropout import Dropout
from .linear import Linear
from .normalization import LayerNorm


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal, stride=1, dilation=1, padding_mode="zeros", channel_last=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.causal = causal
        self.stride = stride
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.channel_last = channel_last

    def cal_length(self, input_length):
        return input_length // self.stride * self.stride + (self.kernel_size - 1) * self.dilation - input_length

    def cal_pad(self, x):
        ndim = x.ndim
        if ndim == 3:
            t = x.shape[-1]
            all_padding = self.cal_length(t)
            if self.causal:
                pad = (all_padding, 0)
            else:
                padding = all_padding // 2
                pad = (padding, padding)
        elif ndim == 4:
            t1, t2 = x.shape[-2], x.shape[-1]
            all_padding1 = self.cal_length(t1)
            all_padding2 = self.cal_length(t2)
            if self.causal:
                pad = (all_padding2, 0, all_padding1, 0)
            else:
                padding1 = all_padding1 // 2
                padding2 = all_padding2 // 2
                pad = (padding2, padding2, padding1, padding1)
        else:
            raise RuntimeError(f"input is {ndim} and not supported")
        return pad

    def forward(self, x):
        pad = self.cal_pad(x)
        if self.padding_mode == "zeros":
            x = F.pad(x, pad, mode="constant")
        else:
            x_type = x.dtype
            with torch.autocast(x.device.type, enabled=False):
                x = x.float()
                x = F.pad(x, pad, mode=self.padding_mode)
            x = x.to(x_type)
        return x

    def extra_repr(self) -> str:
        return "kernel_size={}, causal={}, stride={}, padding_mode={}".format(
            self.kernel_size, self.causal, self.stride, self.padding_mode
        )


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int, Tuple[int]] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        causal: bool = False,
    ) -> None:
        if causal:
            super().__init__(
                in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode, device, dtype
            )
            self.pad = SamePad(
                kernel_size=kernel_size,
                causal=causal,
                stride=stride,
                dilation=dilation,
                padding_mode=padding_mode,
                channel_last=False,
            )
        else:
            super().__init__(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                padding_mode,
                device,
                dtype,
            )
            self.pad = nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.pad(input)
        return super().forward(input)


class Conv1DBlock(nn.Module):
    ops_seq: torch.jit.Final[List[str]]

    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: int = 3,
        padding: Optional[int] = None,
        causal: bool = False,
        padding_mode: str = "zeros",
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        dropout_p: float = 0.0,
        dropout_mode: str = "norm",
        ops_seq: Tuple[str, ...] = ("conv", "bn", "relu", "dropout"),
        disable_afn: bool = False,
        bias: bool = True,
        res: bool = False,
        channel_last: bool = False,
        spectral_norm: bool = False,
        weight_norm: bool = False,
        bn_momentum: Optional[float] = None,
    ):
        super().__init__()
        if padding is None:
            if kernel_size % 2 == 0:
                padding = int((kernel_size * dilation) // 2)
            else:
                padding = int((kernel_size * dilation - dilation) // 2)
        self.op_list = nn.ModuleList()
        assert "conv" in ops_seq or "id" in ops_seq, f"Conv1DBlock should have conv or id ops but got {ops_seq}"
        self.channel_last = channel_last
        last_op_replace = False
        valid_ops_seq = []
        norm_features = n_in
        for op_name in ops_seq:
            if op_name == "conv" or op_name == "id":
                op = Conv1d(
                    n_in,
                    n_out,
                    kernel_size,
                    bias=bias,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                    groups=groups,
                    causal=causal,
                    padding_mode=padding_mode,
                )
                if spectral_norm:
                    op = nn.utils.parametrizations.spectral_norm(op)
                elif weight_norm:
                    op = nn.utils.weight_norm(op)
                norm_features = n_out
            elif op_name == "bn":
                if bn_momentum is None:
                    op = nn.BatchNorm1d(norm_features)
                else:
                    op = nn.BatchNorm1d(norm_features, momentum=bn_momentum)
            elif op_name == "ln":
                op = LayerNorm(norm_features, channel_last=self.channel_last)
            elif op_name == "dropout":
                if dropout_p == 0.0:
                    continue
                assert dropout_p > 0.0 and dropout_p < 1.0, f"Dropout probability: {dropout_p}"
                op = Dropout(p=dropout_p, mode=dropout_mode, inplace=True)
            elif op_name in ACT2FN.keys():
                if disable_afn:
                    continue
                if op_name == "glu":
                    op = Linear(n_in, 2 * n_in, channel_last=channel_last)
                    self.op_list.append(op)
                    valid_ops_seq.append("glu_linear")
                    if channel_last:
                        op = ACT2FN[op_name](dim=-1)
                    else:
                        op = ACT2FN[op_name](dim=1)
                else:
                    op = ACT2FN[op_name]()
            else:
                raise ValueError(f"{op_name} is not supported!")
            if hasattr(op, "inplace") and op.inplace:
                if last_op_replace:
                    op.inplace = False
                    last_op_replace = False
                else:
                    last_op_replace = True
            else:
                last_op_replace = False
            if op is not None:
                self.op_list.append(op)
                valid_ops_seq.append(op_name)
        self.res = res and (n_in == n_out)
        self.valid_ops_seq = tuple(valid_ops_seq)

    def reset_parameters(self, op):
        nn.init.xavier_normal_(op.weight)
        nn.init.zeros_(op.bias)

    def forward(self, x: Tensor, x_mask: Optional[Tensor] = None) -> Tensor:
        if self.res:
            residual = x
            for op, op_name in zip(self.op_list, self.valid_ops_seq):
                if op_name == "conv" or op_name == "id":
                    if x_mask is not None:
                        x = x * x_mask
                    if self.channel_last:
                        x = x.permute(0, 2, 1)
                        x = op(x)
                        x = x.permute(0, 2, 1)
                    else:
                        x = op(x)
                elif op_name == "bn" and self.channel_last:
                    x = x.permute(0, 2, 1)
                    x = op(x)
                    x = x.permute(0, 2, 1)
                else:
                    x = op(x)
            x = x + residual
        else:
            for op, op_name in zip(self.op_list, self.valid_ops_seq):
                if op_name == "conv" or op_name == "id":
                    if self.channel_last:
                        x = x.permute(0, 2, 1)
                        x = op(x)
                        x = x.permute(0, 2, 1)
                    else:
                        x = op(x)
                elif op_name == "bn" and self.channel_last:
                    x = x.permute(0, 2, 1)
                    x = op(x)
                    x = x.permute(0, 2, 1)
                else:
                    x = op(x)
        if x_mask is not None:
            x = x * x_mask
        return x


class Conv1DK1Block(Conv1DBlock):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        dropout_p: float = -1.0,
        dropout_mode: str = "norm",
        ops_seq: Tuple[str, ...] = ("conv",),
        bias: bool = True,
        spectral_norm: bool = False,
    ):
        super().__init__(
            n_in,
            n_out,
            kernel_size=1,
            dropout_p=dropout_p,
            dropout_mode=dropout_mode,
            ops_seq=ops_seq,
            bias=bias,
            spectral_norm=spectral_norm,
        )

    def forward(self, x: Tensor, x_mask: Optional[Tensor] = None) -> Tensor:
        x = super().forward(x, x_mask=x_mask)
        return x
