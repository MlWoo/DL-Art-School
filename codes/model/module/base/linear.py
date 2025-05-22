from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .activation import ACT2FN
from .dropout import Dropout
from .util import extend2tuple


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
        self.op_list = nn.ModuleList()
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
                if op_name == "glu":
                    op = Linear(norm_features, 2 * norm_features, channel_last=channel_last)
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


class MultiLinearLayers(nn.Module):
    def __init__(
        self,
        layers: int,
        in_dim: int,
        inner_dim: int,
        ops_seq: Union[Tuple[str, ...], Tuple[Tuple[str, ...], ...]] = ("linear", "relu", "dropout"),
        dropout_p: float = 0.0,
        dropout_mode: str = "norm",
        dim_back: bool = False,
        per_residual: bool = False,
        residual: bool = False,
        last_activated: bool = False,
        channel_last: bool = True,
        spectral_norm: bool = False,
    ):
        super().__init__()
        if residual:
            dim_back = True
        self.residual = residual

        assert isinstance(inner_dim, (int, list, tuple))
        in_dim_tuple = (in_dim,) + extend2tuple(inner_dim, layers - 1)
        if dim_back:
            out_dim_tuple = extend2tuple(inner_dim, layers - 1) + (in_dim,)
        else:
            out_dim_tuple = extend2tuple(inner_dim, layers)

        assert isinstance(dropout_p, (float, list, tuple))
        dropout_p_tuple = extend2tuple(dropout_p, layers)
        assert isinstance(dropout_mode, (str, list, tuple))
        dropout_mode_tuple = extend2tuple(dropout_mode, layers)

        assert isinstance(per_residual, (bool, list, tuple))
        per_residual_tuple = extend2tuple(per_residual, layers)
        assert isinstance(channel_last, (bool, list, tuple))
        channel_last_tuple = extend2tuple(channel_last, layers)
        assert isinstance(spectral_norm, (bool, list, tuple))
        spectral_norm_tuple = extend2tuple(spectral_norm, layers)

        self.linear_modules = nn.ModuleList()
        for i in range(layers):
            disable_afn = (last_activated) and (i == layers - 1)
            if isinstance(ops_seq[0], tuple):
                assert (
                    len(ops_seq) == layers
                ), f"if ops_seq is double-nested list or tuple, the length \
                        should be layers_num, but got {len(ops_seq)} vs expected {layers}"
                block = LinearBlock(
                    in_features=in_dim_tuple[i],
                    out_features=out_dim_tuple[i],
                    ops_seq=ops_seq[i],
                    dropout_p=dropout_p_tuple[i],
                    dropout_mode=dropout_mode_tuple[i],
                    disable_afn=disable_afn,
                    res=per_residual_tuple[i],
                    channel_last=channel_last_tuple[i],
                    spectral_norm=spectral_norm_tuple[i],
                )
            else:
                block = LinearBlock(
                    in_features=in_dim_tuple[i],
                    out_features=out_dim_tuple[i],
                    ops_seq=ops_seq,
                    dropout_p=dropout_p_tuple[i],
                    dropout_mode=dropout_mode_tuple[i],
                    disable_afn=disable_afn,
                    res=per_residual_tuple[i],
                    channel_last=channel_last_tuple[i],
                    spectral_norm=spectral_norm_tuple[i],
                )
            self.linear_modules.append(block)

    def forward(self, x: Tensor, x_mask: Optional[Tensor] = None) -> Tensor:
        """
        :param x: [B x C x T]
        :return:
        """
        if self.residual:
            res = x
            for module in self.linear_modules:
                x = module(x)
            x = res + x
        else:
            for module in self.linear_modules:
                x = module(x)
        if x_mask is not None:
            x = x * x_mask
        return x  # B x C x T
