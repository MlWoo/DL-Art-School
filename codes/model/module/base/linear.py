from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import ModuleList

from .activation import ACT2FN
from .dropout import Dropout


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        channel_last: bool = True,
        spectral_norm: bool = False,
    ):
        super().__init__()
        if channel_last:
            if spectral_norm:
                self.op = nn.utils.parametrizations.spectral_norm(nn.Linear(in_features, out_features, bias))
            else:
                self.op = nn.Linear(in_features, out_features, bias)
        else:
            if spectral_norm:
                self.op = nn.utils.parametrizations.spectral_norm(
                    nn.Conv1d(in_features, out_features, kernel_size=1, bias=bias)
                )
            else:
                self.op = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1, bias=bias)
        self.spectral_norm = spectral_norm
        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.op.weight, 0.02)
        nn.init.constant_(self.op.bias, 0.0)

    def forward(self, input: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.op.forward(input)
        if mask is not None:
            x = x * mask
        return x


class LinearBlock(nn.Module):
    ops_seq: torch.jit.Final[List[str]]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ops_seq: Tuple[str, ...] = ("linear", "relu", "dropout"),
        dropout_p: float = -1.0,
        dropout_mode: str = "norm",
        disable_afn: bool = False,
        res: bool = False,
        bias: bool = True,
        channel_last: bool = True,
        spectral_norm: bool = False,
        norm_groups: int = 0,
    ):
        super().__init__()
        self.op_list = ModuleList()
        assert "linear" in ops_seq or "id" in ops_seq, f"LinearBlock should have linear ops but got {ops_seq}"
        self.channel_last = channel_last
        last_op_replace = False
        norm_features = in_features
        valid_ops_seq = []
        for op_name in ops_seq:
            if op_name == "linear" or op_name == "id":
                if channel_last:
                    op = nn.Linear(in_features, out_features, bias=bias)
                else:
                    op = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1, bias=bias)
                # op = Linear(in_features, out_features, bias, channel_last=channel_last, spectral_norm=spectral_norm)
                norm_features = out_features
                op_name = "id"
            elif op_name == "bn":
                op = nn.BatchNorm1d(norm_features)
            elif op_name == "gn":
                op = nn.GroupNorm(norm_groups, norm_features)
            elif op_name == "ln":
                op = nn.LayerNorm(norm_features)
            elif op_name == "dropout":
                if dropout_p == 0.0:
                    continue
                assert dropout_p > 0.0 and dropout_p < 1.0, f"dropout probability: {dropout_p}"
                op = Dropout(p=dropout_p, mode=dropout_mode, inplace=True)
            elif op_name in ACT2FN.keys():
                if disable_afn:
                    continue
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
        self.res = res and (in_features == out_features)
        self.valid_ops_seq = tuple(valid_ops_seq)
        # self.reset_parameters()

    def reset_parameters(self):
        for op, op_name in zip(self.op_list, self.valid_ops_seq):
            if op_name in ("id", "linear"):
                nn.init.normal_(op.weight, 0.02)
                nn.init.constant_(op.bias, 0.0)

    def forward(self, x: Tensor, x_mask: Optional[Tensor] = None) -> Tensor:
        if self.res:
            residual = x
            for op, op_name in zip(self.op_list, self.valid_ops_seq):
                if (op_name in ["bn", "gn"] and self.channel_last) or (op_name == "ln" and not self.channel_last):
                    x = x.permute(0, 2, 1)
                    x = op(x)
                    x = x.permute(0, 2, 1)
                else:
                    x = op(x)
            x = x + residual
        else:
            for op, op_name in zip(self.op_list, self.valid_ops_seq):
                if (op_name in ["bn", "gn"] and self.channel_last) or (op_name == "ln" and not self.channel_last):
                    x = x.permute(0, 2, 1)
                    x = op(x)
                    x = x.permute(0, 2, 1)
                else:
                    x = op(x)
        if x_mask is not None:
            x = x * x_mask
        return x
