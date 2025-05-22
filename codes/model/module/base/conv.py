import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.conv import _pair, _single, _triple

from .activation import ACT2FN
from .dropout import Dropout
from .linear import Linear, MultiLinearLayers
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
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding: Optional[int] = None,
        padding_mode: str = "zeros",
        causal: bool = False,
        dropout_p: float = 0.0,
        dropout_mode: str = "norm",
        ops_seq: Tuple[str, ...] = ("conv", "bn", "relu", "dropout"),
        disable_afn: bool = False,
        res: bool = False,
        channel_last: bool = False,
        weight_norm: str = "none",
        bn_momentum: Optional[float] = None,
        norm_groups: int = 1,
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
                if weight_norm == "spectral":
                    op = nn.utils.spectral_norm(op)
                elif weight_norm == "weight":
                    op = nn.utils.weight_norm(op)
                norm_features = n_out
            elif op_name == "bn":
                if bn_momentum is None:
                    op = nn.BatchNorm1d(norm_features)
                else:
                    op = nn.BatchNorm1d(norm_features, momentum=bn_momentum)
            elif op_name == "ln":
                op = LayerNorm(norm_features, channel_last=self.channel_last)
            elif op_name == "gn":
                op = nn.GroupNorm(norm_groups, norm_features)
            elif op_name == "tgn":
                op = nn.GroupNorm(1, norm_features)
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


class LightWeightConv1d(nn.Module):
    """Lightweight Convolution assuming the input is BxCxT
    This is just an example that explains LightConv clearer than the TBC version.
    We don't use this module in the model.
    Args:
        n_in: # of channels of the input and output
        kernel_size: convolution channels
        padding_l: padding_l
        num_heads: number of heads used. The weight is of shape
            `(num_heads, 1, kernel_size)`
        weight_softmax: normalize the weight with softmax before the convolution
    Shape:
        Input: BxCxT, i.e. (batch_size, n_in, timesteps)
        Output: BxCxT, i.e. (batch_size, n_in, timesteps)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias: the learnable bias of the module of shape `(n_in)`
    """

    def __init__(
        self,
        n_in: int,
        kernel_size: int = 1,
        padding_l: Optional[int] = None,
        num_heads: int = 1,
        weight_softmax: bool = False,
        weight_dropout_p: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.n_in = n_in
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.padding_l = kernel_size // 2 if padding_l is None else padding_l
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_in))
        else:
            self.bias = None
        self.weight_dropout = Dropout(weight_dropout_p, mode="norm")
        self.reset_parameters()

    def reset_parameters(self):
        # default initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input: Tensor) -> Tensor:
        """
        input size: B x C x T
        output size: B x C x T
        """
        batch_size, channels, seqs_len = input.size()

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        weight = self.weight_dropout(weight)
        # Merge every C/H entries into the batch dimension (C = self.n_in)
        # B x C x T -> (B * C/H) x H x T
        # One can also expand the weight to C x 1 x K by a factor of C/H
        # and do not reshape the input instead, which is slow though
        input = input.view(-1, self.num_heads, seqs_len)
        output = F.conv1d(input, weight, padding=self.padding_l, groups=self.num_heads)
        output = output.view(batch_size, channels, seqs_len)

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        return output


def unfold1d(x: Tensor, kernel_size: int, padding_l: int, pad_value: float = 0.0) -> Tensor:
    """unfold B x T x C to B x T x C x K"""
    if kernel_size > 1:
        B, T, C = x.size()
        x = F.pad(x, (0, 0, padding_l, kernel_size - 1 - padding_l, 0, 0), value=pad_value)
        x = x.as_strided((B, T, C, kernel_size), ((T + kernel_size - 1) * C, C, 1, C))
    else:
        x = x.unsqueeze(3)
    return x


class LightweightConv1dBTC(nn.Module):
    """Lightweight Convolution assuming the input is BxTxC
    Args:
        n_in: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        bias: use bias
    Shape:
        Input: BxTxC, i.e. (batch_size, timesteps, n_in)
        Output: BxTxC, i.e. (batch_size, timesteps, n_in)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(n_in)`
    """

    def __init__(
        self,
        n_in: int,
        kernel_size: int = 1,
        padding_l: Optional[int] = None,
        num_heads: int = 1,
        weight_softmax: bool = False,
        weight_dropout_p: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.n_in = n_in
        self.kernel_size = kernel_size
        if padding_l is None:
            padding_l = kernel_size - 1
        else:
            padding_l = padding_l
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_softmax = weight_softmax
        if weight_softmax:
            weight_dropout_inplace = False
        else:
            weight_dropout_inplace = True
        self.weight_dropout = Dropout(weight_dropout_p, inplace=weight_dropout_inplace)
        self.weight_dropout_p = weight_dropout_p

        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_in))
        else:
            self.bias = None
        self.hk = num_heads * kernel_size  # 17 * 8 or 3 * 8  H * K
        self.reset_parameters()
        self.clear_buffer()

    def clear_buffer(self):
        self.input_buffer = None

    """
    def reset_parameters(self):
        # default initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
    """

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor, is_incremental: bool = False) -> Tensor:
        """Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, n_in)
            incremental_state: A dict to keep the state
            unfold: unfold the input or not. If not, we use the matrix trick instead
        """
        if is_incremental:
            output = self.incremental_forward(x)
        else:
            T = x.shape[1]
            if False and T > self.kernel_size:
                output = self._forward_unfolded(x)  # H * T * K
            elif False and torch.cuda.is_available() and x.is_cuda:
                output = self._forward_lconv(x)  # H * T * T
            else:
                output = self._forward_expanded(x)  # H * T * T

            if self.bias is not None:
                output = output + self.bias.view(1, 1, -1)
            # output_conv = self._forward_conv(x)

        return output

    def incremental_forward(self, input):
        # input: (B, T, C)
        if self.training:
            raise RuntimeError("incremental_forward only supports eval mode")

        # run forward pre hooks (e.g., weight norm)
        for hook in self._forward_pre_hooks.values():
            hook(self, input)

        # reshape weight
        B = input.size(0)  # input: bsz x len x dim
        K = self.kernel_size

        if K > 1:
            input = input.data
            if self.input_buffer is None:
                self.input_buffer = input.new(B, K, input.size(2))
                self.input_buffer.zero_()
            else:
                # shift buffer
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
            # append next input
            self.input_buffer[:, -1, :] = input[:, -1, :]
            input = self.input_buffer
        output = self._forward_unfolded(input)
        return output

    def _forward_lconv(self, x: Tensor):
        weight = self.weight.squeeze(dim=1)
        x = x.permute(0, 2, 1).contiguous()
        if self.weight_softmax:
            weight = F.softmax(weight, -1)
        weight = self.weight_dropout(weight)

        output = LightconvF.apply(x, weight, self.padding_l)
        output = output.permute(0, 2, 1)
        return output

    def _forward_unfolded(self, x: Tensor):
        """The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right."""
        B, T, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.n_in

        weight = self.weight.view(H, K)
        if self.weight_softmax:
            weight = F.softmax(weight, dim=1)

        # weight = weight.view(H, 1, K).expand(H, T, K).contiguous()
        weight = weight.view(H, K).expand(T, H, K).contiguous().view(T * H, K, 1)  # H * K * T vs H * T * T
        weight = self.weight_dropout(weight)
        # unfold the input: B x T x C --> B x T x C x K
        x_unfold = unfold1d(x, self.kernel_size, self.padding_l, 0)  # B * T, C, K
        x_unfold = x_unfold.view(B, T * H, R, K)

        output = torch.matmul(x_unfold, weight)  # B*T*H x R x 1
        output = output.view(B, T, C)
        return output

    def _forward_expanded(self, x: Tensor):
        """Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        """
        B, T, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.n_in

        weight = self.weight.view(H, K)

        if self.weight_softmax:
            x_dtype = x.dtype
            weight = weight.float()
            weight = F.softmax(weight, dim=1)
            weight = weight.to(x_dtype)

        weight = weight.view(H, 1, K).expand(H, T, K).contiguous()
        # B T C --> B T H R --> B H T R --> B * H, T, R
        x = x.view(B, T, H, R).permute(0, 2, 1, 3)  # B, H, T, R
        # B T C
        P = self.padding_l
        if K > T and P == K - 1:
            weight = weight.narrow(2, K - T, T)
            K, P = T, T - 1
        # turn the convolution filters into band matrices
        weight_expanded = weight.new_zeros(H, T, T + K - 1, requires_grad=False)
        weight_expanded.as_strided((H, T, K), (T * (T + K - 1), T + K, 1)).copy_(weight)
        weight_expanded = weight_expanded.narrow(2, P, T)  # H * T * T
        weight_expanded = self.weight_dropout(weight_expanded)
        output = torch.matmul(weight_expanded, x)  # B H T R
        # B H T R -> B T H R -> B T C
        output = output.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        import ipdb

        ipdb.set_trace()
        return output

    def _forward_conv(self, input: Tensor) -> Tensor:
        input = input.permute(0, 2, 1).contiguous()
        """
        input size: B x C x T
        output size: B x C x T
        """
        batch_size, channels, seqs_len = input.size()

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        weight = self.weight_dropout(weight)
        # Merge every C/H entries into the batch dimension (C = self.n_in)
        # B x C x T -> (B * C/H) x H x T
        # One can also expand the weight to C x 1 x K by a factor of C/H
        # and do not reshape the input instead, which is slow though
        input = input.view(-1, self.num_heads, seqs_len)
        output = F.conv1d(input, weight, padding=self.padding_l, groups=self.num_heads)  # B R H T
        output = output[:, :, :seqs_len].contiguous()
        output = output.view(batch_size, channels, seqs_len)

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        output = output.permute(0, 2, 1).contiguous()
        return output

    def extra_repr(self):
        s = "{}, kernel_size={}, padding_l={}, num_heads={}, weight_softmax={}, bias={}".format(
            self.n_in,
            self.kernel_size,
            self.padding_l,
            self.num_heads,
            self.weight_softmax,
            self.bias is not None,
        )
        if self.weight_dropout_p > 0.0:
            s += ", weight_dropout={}".format(self.weight_dropout_p)
        return s


class DynamicLightweightConv1dBTC(nn.Module):
    """Lightweight Convolution assuming the input is BxTxC
    Args:
        n_in: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        bias: use bias
    Shape:
        Input: BxTxC, i.e. (batch_size, timesteps, n_in)
        Output: BxTxC, i.e. (batch_size, timesteps, n_in)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(n_in)`
    """

    def __init__(
        self,
        n_in: int,
        kernel_size: int = 1,
        padding_l: Optional[int] = None,
        num_heads: int = 1,
        weight_softmax: bool = False,
        weight_dropout_p: float = 0.0,
        weight_bias: bool = False,
        conv_bias: bool = False,
        renorm_padding: bool = False,
    ):
        super().__init__()
        self.n_in = n_in
        self.kernel_size = kernel_size
        if padding_l is None:
            padding_l = kernel_size - 1
        else:
            padding_l = padding_l
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_softmax = weight_softmax
        if weight_softmax:
            weight_dropout_inplace = False
        else:
            weight_dropout_inplace = True
        self.weight_dropout = Dropout(weight_dropout_p, inplace=weight_dropout_inplace)
        self.weight_dropout_p = weight_dropout_p
        self.weight_linear = nn.Linear(n_in, num_heads * kernel_size, weight_bias)
        if conv_bias:
            self.bias = nn.Parameter(torch.Tensor(n_in))
        else:
            self.bias = None
        self.hk = num_heads * kernel_size  # 17 * 8 or 3 * 8  H * K
        self.renorm_padding = renorm_padding

        self.causal = padding_l == kernel_size - 1
        self.input_buffer = {}
        self.reset_parameters()

    def reset_parameters(self):
        # default initialization
        # nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    """

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    """

    def forward(self, x) -> Tensor:
        """Assuming the input, x, of the shape B x T x C and producing an output in the shape B x T x C
        args:
            x: Input of shape B x T x C, i.e. (batch_size, timesteps, n_in)
            incremental_state: A dict to keep the state
            unfold: unfold the input or not. If not, we use the matrix trick instead
        """
        T = x.shape[1]
        if False and T > self.kernel_size:
            output = self._forward_unfolded(x)  # H * T * K
        elif False and torch.cuda.is_available() and x.is_cuda:
            output = self._forward_lconv(x)  # H * T * T
        else:
            output = self._forward_expanded(x)  # H * T * T
        if self.bias is not None:
            output = output + self.bias.view(1, 1, -1).to(output.dtype)
        # output_conv = self._forward_conv(x)
        return output

    def incremental_forward(self, input: Tensor):
        # input: (B, T, C)
        if self.training:
            raise RuntimeError("incremental_forward only supports eval mode")

        # run forward pre hooks (e.g., weight norm)
        for hook in self._forward_pre_hooks.values():
            hook(self, input)

        # reshape weight
        B = input.size(0)  # input: bsz x len x dim
        K = self.kernel_size

        if K > 1:
            input = input.data
            if self.input_buffer is None:
                self.input_buffer = input.new(B, K, input.size(2))
                self.input_buffer.zero_()
            else:
                # shift buffer
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
            # append next input
            self.input_buffer[:, -1, :] = input[:, -1, :]
            input = self.input_buffer
        output = self._forward_unfolded(input)
        return output

    def _forward_lconv(self, x: Tensor):
        weight = self.weight.squeeze(dim=1)
        x = x.permute(0, 2, 1).contiguous()
        if self.weight_softmax:
            weight = F.softmax(weight, -1)
        weight = self.weight_dropout(weight)

        output = LightconvF.apply(x, weight, self.padding_l)
        output = output.permute(0, 2, 1)
        return output

    def _forward_unfolded(self, x: Tensor):
        """The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right."""
        B, T, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.n_in

        weight = self.weight_linear(x).view(B * T * H, -1)  # B*T*H x R
        if not self.renorm_padding:
            if self.weight_softmax:
                weight = F.softmax(weight, dim=1)
            weight = self.weight_dropout(weight)
        weight = weight.view(B, T, H, K).permute(0, 2, 1, 3).contiguous()  # B H T C

        # weight = weight.view(H, 1, K).expand(H, T, K).contiguous()
        weight = weight.view(H, K).expand(T, H, K).contiguous().view(T * H, K, 1)  # H * K * T vs H * T * T
        weight = self.weight_dropout(weight)
        # unfold the input: B x T x C --> B x T x C x K
        x_unfold = unfold1d(x, self.kernel_size, self.padding_l, 0)  # B * T, C, K
        x_unfold = x_unfold.view(B, T * H, R, K)

        output = torch.matmul(x_unfold, weight)  # B*T*H x R x 1
        output = output.view(B, T, C)
        return output

    def _forward_expanded(self, x: Tensor):
        """Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        """
        B, T, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.n_in

        weight = self.weight_linear(x).view(B * T * H, -1)  # B*T*H x K
        if not self.renorm_padding:
            if self.weight_softmax:
                x_dtype = x.dtype
                weight = weight.float()
                weight = F.softmax(weight, dim=1)
                weight = weight.to(x_dtype)
            weight = self.weight_dropout(weight)
        weight = weight.view(B, T, H, K).permute(0, 2, 1, 3).contiguous()  # B H T K

        if self.weight_softmax and self.renorm_padding:
            # turn the convolution filters into band matrices
            weight_expanded = weight.new(B, H, T, T + K - 1).fill_(float("-inf"))
            weight_expanded.as_strided((B, H, T, K), (H * T * (T + K - 1), T * (T + K - 1), T + K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(3, self.padding_l, T)
            # normalize the weight over valid positions like self-attention
            weight_expanded = F.softmax(weight_expanded, dim=2)
            weight_expanded = self.weight_dropout(weight_expanded, inplace=False)
        else:
            P = self.padding_l
            if K > T and P == K - 1:
                weight = weight.narrow(2, K - T, T)
                K, P = T, T - 1
            # turn the convolution filters into band matrices
            weight_expanded = weight.new_zeros(B, H, T, T + K - 1, requires_grad=False)
            weight_expanded.as_strided((B, H, T, K), (H * T * (T + K - 1), T * (T + K - 1), T + K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(3, P, T)  # B x H x T x T

        x = x.view(B, T, H, R).permute(0, 2, 1, 3)  # B H T R

        output = torch.matmul(weight_expanded, x)  # B H T R
        # B H T R -> B T H R -> B T C
        output = output.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        return output

    def _forward_conv(self, input: Tensor) -> Tensor:
        input = input.permute(0, 2, 1).contiguous()
        """
        input size: B x C x T
        output size: B x C x T
        """
        batch_size, channels, seqs_len = input.size()

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        weight = self.weight_dropout(weight)
        # Merge every C/H entries into the batch dimension (C = self.n_in)
        # B x C x T -> (B * C/H) x H x T
        # One can also expand the weight to C x 1 x K by a factor of C/H
        # and do not reshape the input instead, which is slow though
        input = input.view(-1, self.num_heads, seqs_len)
        output = F.conv1d(input, weight, padding=self.padding_l, groups=self.num_heads)  # B R H T
        output = output[:, :, :seqs_len].contiguous()
        output = output.view(batch_size, channels, seqs_len)

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        output = output.permute(0, 2, 1).contiguous()
        return output

    def extra_repr(self):
        s = "{}, kernel_size={}, padding_l={}, num_heads={}, weight_softmax={}, bias={}".format(
            self.n_in,
            self.kernel_size,
            self.padding_l,
            self.num_heads,
            self.weight_softmax,
            self.bias is not None,
        )
        if self.weight_dropout_p > 0.0:
            s += ", weight_dropout={}".format(self.weight_dropout_p)
        return s


class AddCoords(nn.Module):
    def __init__(self, rank: int, with_radius: bool = False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_radius = with_radius

    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: shape (N, C_in, H, W)
        :return:
        """
        device = input.get_device()
        if self.rank == 1:
            batch_size, _, dim_x = input.shape
            grid_x = torch.arange(dim_x, dtype=torch.int32)

            grid_x = grid_x.float() / (dim_x - 1)
            grid_x = grid_x * 2 - 1
            grid_x = grid_x[:, :, 1].repeat(batch_size, 1, 1)

            grid_x = grid_x.to(device)
            out = torch.cat([input, grid_x], dim=1)

            if self.with_radius:
                rr = torch.sqrt(torch.pow(grid_x - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size, _, dim_y, dim_x = input.shape
            grid_x_vy, grid_y_vx = torch.meshgrid(torch.arange(dim_y), torch.arange(dim_x))

            grid_x_vy = grid_x_vy.float() / (dim_x - 1)
            grid_y_vx = grid_y_vx.float() / (dim_y - 1)

            grid_x_vy = grid_x_vy * 2 - 1
            grid_y_vx = grid_y_vx * 2 - 1

            grid_x_vy = grid_x_vy.repeat(batch_size, 1, 1, 1)
            grid_y_vx = grid_y_vx.repeat(batch_size, 1, 1, 1)

            grid_x_vy = grid_x_vy.to(device)
            grid_y_vx = grid_y_vx.to(device)
            out = torch.cat([input, grid_x_vy, grid_y_vx], dim=1)

            if self.with_radius:
                rr = torch.sqrt(torch.pow(grid_x_vy - 0.5, 2) + torch.pow(grid_y_vx - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size, _, dim_z, dim_y, dim_x = input.shape
            grid_xy_vz, grid_xz_vy, grid_yz_vx = torch.meshgrid(
                torch.arange(dim_z), torch.arange(dim_y), torch.arange(dim_x)
            )
            grid_fx = grid_xy_vz + grid_xz_vy
            grid_fy = grid_xy_vz + grid_yz_vx
            grid_fz = grid_xz_vy + grid_yz_vx

            grid_fx = grid_fx.float() / (dim_x - 1)
            grid_fy = grid_fy.float() / (dim_y - 1)
            grid_fz = grid_fz.float() / (dim_z - 1)

            grid_fx = grid_fx * 2 - 1
            grid_fy = grid_fy * 2 - 1
            grid_fz = grid_fz * 2 - 1

            grid_fx = grid_fx.repeat(batch_size, 1, 1, 1, 1)
            grid_fy = grid_fy.repeat(batch_size, 1, 1, 1, 1)
            grid_fz = grid_fz.repeat(batch_size, 1, 1, 1, 1)

            grid_fx = grid_fx.to(device)
            grid_fy = grid_fy.to(device)
            grid_fz = grid_fz.to(device)
            out = torch.cat([input, grid_fx, grid_fy, grid_fz], dim=1)

            if self.with_radius:
                rr = torch.sqrt(torch.pow(grid_fx - 0.5, 2) + torch.pow(grid_fy - 0.5, 2) + torch.pow(grid_fz - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out

    def extra_repr(self) -> str:
        return "rank={}, with_radius={}".format(self.rank, self.with_radius)


class _CoordConv(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: Union[int, Tuple[int, int], Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int], Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int], Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int], Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        with_radius: bool = False,
        rank=1,
    ):
        super().__init__()
        if rank == 1:
            self.conv = nn.Conv1d(
                n_in + rank + int(with_radius), n_out, kernel_size, stride, padding, dilation, groups, bias
            )
        elif rank == 2:
            self.conv = nn.Conv2d(
                n_in + rank + int(with_radius), n_out, kernel_size, stride, padding, dilation, groups, bias
            )
        elif rank == 3:
            self.conv = nn.Conv3d(
                n_in + rank + int(with_radius), n_out, kernel_size, stride, padding, dilation, groups, bias
            )
        self.addcoords = AddCoords(rank, with_radius)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        input_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out
        :return: CoordConv2d Result
        """
        out = self.addcoords(inputs)
        out = self.conv(out)
        return out


class CoordConv1d(_CoordConv):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        with_radius: bool = False,
    ):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super().__init__(
            n_in,
            n_out,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups,
            bias=bias,
            with_radius=with_radius,
            rank=1,
        )


class CoordConv2d(_CoordConv):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        with_radius: bool = False,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            n_in,
            n_out,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups,
            bias=bias,
            with_radius=with_radius,
            rank=2,
        )


class CoordConv3d(_CoordConv):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: _size_3_t = 3,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        with_radius: bool = False,
    ):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super().__init__(
            n_in,
            n_out,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups,
            bias=bias,
            with_radius=with_radius,
            rank=3,
        )


class Conv1dIncremental(nn.Conv1d):
    """Extended nn.Conv1d for incremental dilated convolutions"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_buffer()
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input, is_incremental=False):
        if is_incremental:
            return self.incremental_forward(input)
        else:
            return super().forward(input)

    def incremental_forward(self, input):
        # input: (B, T, C)
        if self.training:
            raise RuntimeError("incremental_forward only supports eval mode")

        # run forward pre hooks (e.g., weight norm)
        for hook in self._forward_pre_hooks.values():
            hook(self, input)

        # reshape weight
        weight = self._get_linearized_weight()
        kw = self.kernel_size[0]
        dilation = self.dilation[0]

        bsz = input.size(0)  # input: bsz x len x dim
        if kw > 1:
            input = input.data
            if self.input_buffer is None:
                self.input_buffer = input.new(bsz, kw + (kw - 1) * (dilation - 1), input.size(2))
                self.input_buffer.zero_()
            else:
                # shift buffer
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
            # append next input
            self.input_buffer[:, -1, :] = input[:, -1, :]
            input = self.input_buffer
            if dilation > 1:
                input = input[:, 0::dilation, :].contiguous()
        output = F.linear(input.view(bsz, -1), weight, self.bias)
        return output.view(bsz, 1, -1)

    def clear_buffer(self):
        self.input_buffer = None

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            # nn.Conv1d
            if self.weight.size() == (self.out_channels, self.in_channels, kw):
                weight = self.weight.transpose(1, 2).contiguous()
            else:
                # fairseq.modules.conv_tbc.ConvTBC
                weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None


class LightWeightConv1DBlock(nn.Module):
    res: torch.jit.Final[bool]

    def __init__(
        self,
        n_in: int,
        kernel_size: int = 3,
        padding_l: Optional[int] = None,
        num_heads: int = 1,
        ops_seq: Tuple[str, ...] = ("glu", "conv"),
        weight_softmax: bool = False,
        weight_dropout_p: float = 0.0,
        weight_bias: bool = True,
        conv_bias: bool = True,
        dynamic: bool = False,
        output_dropout_p: float = 0.0,
        output_dropout_mode: str = "norm",
        disable_afn: bool = False,
        res: bool = False,
        channel_last: bool = False,
        spectral_norm: bool = False,
        weight_norm: str = "none",
        norm_groups: int = 1,
    ):
        super().__init__()
        if padding_l is None:
            padding_l = kernel_size - 1
        else:
            padding_l = padding_l
        self.op_list = nn.ModuleList()
        assert "conv" in ops_seq or "id" in ops_seq, f"LConv1DBlock should have conv or id ops but got {ops_seq}"
        self.channel_last = channel_last
        last_op_replace = False
        valid_ops_seq = []
        norm_features = n_in
        for op_name in ops_seq:
            if op_name == "conv" or op_name == "id":
                if dynamic:
                    op = DynamicLightweightConv1dBTC(
                        n_in=n_in,
                        kernel_size=kernel_size,
                        padding_l=padding_l,
                        num_heads=num_heads,
                        weight_softmax=weight_softmax,
                        weight_dropout_p=weight_dropout_p,
                        weight_bias=weight_bias,
                        conv_bias=conv_bias,
                    )
                else:
                    op = LightweightConv1dBTC(
                        n_in=n_in,
                        kernel_size=kernel_size,
                        padding_l=padding_l,
                        num_heads=num_heads,
                        weight_softmax=weight_softmax,
                        weight_dropout_p=weight_dropout_p,
                        bias=conv_bias,
                    )
                if spectral_norm:
                    op = nn.utils.parametrizations.spectral_norm(op)
            elif op_name == "bn":
                op = nn.BatchNorm1d(norm_features)
            elif op_name == "ln":
                op = LayerNorm(norm_features, channel_last=self.channel_last)
            elif op_name == "weight_norm":
                op = weight_norm(op, weight_norm, groups=norm_groups)
            elif op_name == "dropout":
                assert output_dropout_p > 0.0 and output_dropout_p < 1.0
                op = Dropout(p=output_dropout_p, mode=output_dropout_mode)
            elif op_name in ACT2FN.keys():
                if disable_afn:
                    continue
                if op_name == "glu":
                    op = Linear(n_in, 2 * n_in, channel_last=channel_last)
                    self.op_list.append(op)
                    valid_ops_seq.append(op_name)
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
        self.res = res
        self.valid_ops_seq = tuple(valid_ops_seq)

    def forward(self, x: Tensor, x_mask: Optional[Tensor] = None, incremental_state=None) -> Tensor:
        if self.res:
            residual = x
            for op, op_name in zip(self.op_list, self.valid_ops_seq):
                if op_name == "conv" or op_name == "id":
                    if x_mask is not None:
                        x = x * x_mask
                    if self.channel_last:
                        x = op(x)
                    else:
                        x = x.permute(0, 2, 1)
                        x = op(x)
                        x = x.permute(0, 2, 1)
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
                    if x_mask is not None:
                        x = x * x_mask
                    if self.channel_last:
                        x = op(x)
                    else:
                        x = x.permute(0, 2, 1)
                        x = op(x)
                        x = x.permute(0, 2, 1)
                elif op_name == "bn" and self.channel_last:
                    x = x.permute(0, 2, 1)
                    x = op(x)
                    x = x.permute(0, 2, 1)
                else:
                    x = op(x)
        if x_mask is not None:
            x = x * x_mask
        return x


class Conv2DBlock(nn.Module):
    valid_ops_seq: torch.jit.Final[Tuple[str, ...]]

    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        padding: Optional[int] = None,
        causal: bool = False,
        padding_mode: str = "zeros",
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        dropout_p: float = -1.0,
        dropout_mode: str = "norm",
        ops_seq: Tuple[str, ...] = ("conv", "bn", "relu", "dropout"),
        disable_afn: bool = False,
        bias: bool = True,
        res: bool = False,
        channel_last: bool = False,
        spectral_norm: bool = False,
        weight_norm: str = "none",
        norm_groups: int = 1,
    ):
        super().__init__()
        if padding is None:
            if isinstance(kernel_size, tuple):
                _padding = tuple([_kernel_size // 2 for _kernel_size in kernel_size])
            else:
                _padding = kernel_size // 2
        else:
            _padding = padding
        self.op_list = nn.ModuleList()
        assert "conv" in ops_seq or "id" in ops_seq, f"Conv2DBlock should have conv or id ops but got {ops_seq}"
        self.channel_last = channel_last
        last_op_replace = False
        valid_ops_seq = []
        norm_features = n_in
        for op_name in ops_seq:
            if op_name == "conv" or op_name == "id":
                if causal:
                    op = SamePad(
                        kernel_size=kernel_size,
                        causal=causal,
                        stride=stride,
                        dilation=dilation,
                        padding_mode=padding_mode,
                        channel_last=channel_last,
                        simple=False,
                    )
                    valid_ops_seq.append("conv_pad")
                    self.op_list.append(op)
                    op = nn.Conv2d(
                        n_in,
                        n_out,
                        kernel_size,
                        bias=bias,
                        padding=_padding,
                        stride=stride,
                        dilation=dilation,
                        groups=groups,
                    )
                else:
                    op = nn.Conv2d(n_in, n_out, kernel_size, bias=bias, padding=_padding, stride=stride, groups=groups)
                if spectral_norm:
                    op = nn.utils.parametrizations.spectral_norm(op)
                norm_features = n_out
            elif op_name == "bn":
                op = nn.BatchNorm2d(norm_features)
            elif op_name == "ln":
                op = LayerNorm(norm_features, channel_last=self.channel_last)
            elif op_name == "weight_norm":
                op = weight_norm(op, weight_norm, groups=norm_groups)
            elif op_name == "dropout":
                assert dropout_p > 0.0 and dropout_p < 1.0
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
        self.res = res and (n_in == n_out)
        self.valid_ops_seq = tuple(valid_ops_seq)

    def forward(self, x: Tensor, x_mask: Optional[Tensor] = None) -> Tensor:
        if x_mask is not None:
            x = x * x_mask

        if self.res:
            residual = x
            for op, op_name in zip(self.op_list, self.valid_ops_seq):
                if (op_name == "conv" or op_name == "id") and self.channel_last:
                    x = x.permute(0, 3, 1, 2)
                    x = op(x)
                    x = x.permute(0, 2, 3, 1)
                elif op_name == "bn" and self.channel_last:
                    x = x.permute(0, 3, 1, 2)
                    x = op(x)
                    x = x.permute(0, 2, 3, 1)
                else:
                    x = op(x)
            x = x + residual
        else:
            for op, op_name in zip(self.op_list, self.valid_ops_seq):
                if (op_name == "conv" or op_name == "id") and self.channel_last:
                    x = x.permute(0, 3, 1, 2)
                    x = op(x)
                    x = x.permute(0, 2, 3, 1)
                elif op_name == "bn" and self.channel_last:
                    x = x.permute(0, 3, 1, 2)
                    x = op(x)
                    x = x.permute(0, 2, 3, 1)
                else:
                    x = op(x)
        if x_mask is not None:
            x = x * x_mask
        return x


class CoordConv1DBlock(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        with_radius: bool = False,
        dropout_p: float = -1.0,
        dropout_mode: str = "norm",
        ops_seq: Tuple[str, ...] = ("conv", "bn", "relu", "dropout"),
        disable_afn: bool = False,
        res: bool = False,
        channel_last: bool = False,
        weight_norm: str = "none",
        norm_groups: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.op_list = nn.ModuleList()
        assert not channel_last, "only `channel_last` is supportted now"
        assert "conv" in ops_seq, f"Conv1DBlock should have conv ops but got {ops_seq}"
        self.channel_last = channel_last
        self.dropout_mode = dropout_mode
        valid_ops_seq = []
        norm_features = n_in
        for op_name in ops_seq:
            if op_name == "conv":
                op = CoordConv1d(
                    n_in,
                    n_out,
                    kernel_size,
                    bias=bias,
                    padding=padding,
                    dilation=dilation,
                    stride=stride,
                    groups=groups,
                    with_radius=with_radius,
                )
                norm_features = n_out
            elif op_name == "bn":
                op = nn.BatchNorm1d(norm_features)
            elif op_name == "ln":
                op = LayerNorm(norm_features, channel_last=self.channel_last)
            elif op_name == "weight_norm":
                op = weight_norm(op, weight_norm, groups=norm_groups)
            elif op_name == "dropout":
                assert dropout_p > 0.0 and dropout_p < 1.0
                op = Dropout(p=dropout_p, mode=dropout_mode, inplace=True)
            elif op_name in ACT2FN.keys():
                if disable_afn:
                    continue
                op = ACT2FN[op_name]()
            else:
                raise ValueError(f"{op_name} is not supported!")
            if op is not None:
                self.op_list.append(op)
                valid_ops_seq.append(op_name)
        self.res = res and (n_in == n_out)
        self.valid_ops_seq = tuple(valid_ops_seq)

    def forward(self, x: Tensor, x_mask: Optional[Tensor] = None) -> Tensor:
        if x_mask is not None:
            x = x * x_mask

        if self.res:
            residual = x
            for op, op_name in zip(self.op_list, self.valid_ops_seq):
                if (op_name == "conv" or op_name == "id") and self.channel_last:
                    x = x.permute(0, 2, 1)
                    x = op(x)
                    x = x.permute(0, 2, 1)
                else:
                    x = op(x)
            x = x + residual
        else:
            for op, op_name in zip(self.op_list, self.valid_ops_seq):
                if (op_name == "conv" or op_name == "id") and self.channel_last:
                    x = x.permute(0, 2, 1)
                    x = op(x)
                    x = x.permute(0, 2, 1)
                else:
                    x = op(x)
        if x_mask is not None:
            x = x * x_mask
        return x


class CoordConv2DBlock(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        ops_seq: Tuple[str, ...] = ("conv", "bn", "relu", "dropout"),
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: int = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        with_radius: bool = False,
        dropout_p: float = -1.0,
        dropout_mode: str = "norm",
        disable_afn: bool = False,
        res: bool = False,
        channel_last: bool = False,
        weight_norm: str = "none",
        norm_groups: int = 1,
    ):
        super().__init__()
        if padding is None:
            if isinstance(kernel_size, (list, tuple)):
                _padding = tuple([_kernel_size // 2 for _kernel_size in kernel_size])
            else:
                _padding = kernel_size // 2
        else:
            _padding = padding
        self.op_list = nn.ModuleList()
        assert "conv" in ops_seq, f"Conv2DBlock should have conv ops but got {ops_seq}"
        self.channel_last = channel_last
        valid_ops_seq = []
        norm_features = n_in
        for op_name in ops_seq:
            if op_name == "conv":
                op = CoordConv2d(
                    n_in,
                    n_out,
                    kernel_size,
                    bias=bias,
                    padding=_padding,
                    stride=stride,
                    dilation=dilation,
                    groups=groups,
                    with_radius=with_radius,
                )
                norm_features = n_out
            elif op_name == "bn":
                op = nn.BatchNorm2d(norm_features)
            elif op_name == "ln":
                op = LayerNorm(norm_features, channel_last=self.channel_last)
            elif op_name == "weight_norm":
                op = weight_norm(op, weight_norm, groups=norm_groups)
            elif op_name == "dropout":
                assert dropout_p > 0.0 and dropout_p < 1.0
                op = Dropout(p=dropout_p, mode=dropout_mode, inplace=True)
            elif op_name in ACT2FN.keys():
                if disable_afn:
                    continue
                op = ACT2FN[op_name]()
            else:
                raise ValueError(f"{op_name} is not supported!")
            if op is not None:
                self.op_list.append(op)
                valid_ops_seq.append(op_name)
        self.res = res and (n_in == n_out)
        self.valid_ops_seq = tuple(valid_ops_seq)

    def forward(self, x: Tensor, x_mask: Optional[Tensor] = None) -> Tensor:
        if x_mask is not None:
            x = x * x_mask
        if self.res:
            residual = x
            for op, op_name in zip(self.op_list, self.valid_ops_seq):
                if (op_name == "conv" or op_name == "id") and self.channel_last:
                    x = x.permute(0, 3, 1, 2)
                    x = op(x)
                    x = x.permute(0, 2, 3, 1)
                else:
                    x = op(x)
            x = x + residual
        else:
            for op, op_name in zip(self.op_list, self.valid_ops_seq):
                if (op_name == "conv" or op_name == "id") and self.channel_last:
                    x = x.permute(0, 3, 1, 2)
                    x = op(x)
                    x = x.permute(0, 2, 3, 1)
                else:
                    x = op(x)
        if x_mask is not None:
            x = x * x_mask
        return x


class CoordConv3DBlock(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: int = 1,
        padding: Optional[Union[int, Tuple[int, int, int]]] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        with_radius: bool = False,
        dropout_p: float = -1.0,
        dropout_mode: str = "norm",
        ops_seq: Tuple[str, ...] = ("conv", "bn", "relu", "dropout"),
        disable_afn: bool = False,
        res: bool = False,
        channel_last: bool = False,
        weight_norm: str = "none",
        norm_groups: int = 1,
    ):
        super().__init__()
        if padding is None:
            if isinstance(kernel_size, (list, tuple)):
                _padding = tuple([_kernel_size // 2 for _kernel_size in kernel_size])
            else:
                _padding = kernel_size // 2
        else:
            _padding = padding
        self.op_list = nn.ModuleList()
        assert "conv" in ops_seq, f"Conv2DBlock should have conv ops but got {ops_seq}"
        self.channel_last = channel_last
        valid_ops_seq = []
        norm_features = n_in
        for op_name in ops_seq:
            if op_name == "conv":
                op = CoordConv3d(
                    n_in,
                    n_out,
                    kernel_size,
                    bias=bias,
                    padding=_padding,
                    stride=stride,
                    dilation=dilation,
                    groups=groups,
                    with_radius=with_radius,
                )
                norm_features = n_in
            elif op_name == "bn":
                op = nn.BatchNorm3d(norm_features)
            elif op_name == "ln":
                op = LayerNorm(norm_features, channel_last=self.channel_last)
            elif op_name == "weight_norm":
                op = weight_norm(op, weight_norm, groups=norm_groups)
            elif op_name == "dropout":
                assert dropout_p > 0.0 and dropout_p < 1.0
                op = Dropout(p=dropout_p, mode=dropout_mode, inplace=True)
            elif op_name in ACT2FN.keys():
                if disable_afn:
                    continue
                op = ACT2FN[op_name]()
            else:
                raise ValueError(f"{op_name} is not supported!")
            if op is not None:
                self.op_list.append(op)
                valid_ops_seq.append(op_name)
        self.res = res and (n_in == n_out)
        self.valid_ops_seq = tuple(valid_ops_seq)

    def forward(self, x: Tensor, x_mask: Optional[Tensor] = None) -> Tensor:
        if x_mask is not None:
            x = x * x_mask

        if self.res:
            residual = x
            for op, op_name in zip(self.op_list, self.valid_ops_seq):
                if (op_name == "conv" or op_name == "id") and self.channel_last:
                    x = x.permute(0, 4, 1, 2, 3)
                    x = op(x)
                    x = x.permute(0, 2, 3, 4, 1)
                else:
                    x = op(x)
            x = x + residual
        else:
            for op, op_name in zip(self.op_list, self.valid_ops_seq):
                if (op_name == "conv" or op_name == "id") and self.channel_last:
                    x = x.permute(0, 4, 1, 2, 3)
                    x = op(x)
                    x = x.permute(0, 2, 3, 4, 1)
                else:
                    x = op(x)
        if x_mask is not None:
            x = x * x_mask
        return x


class LConv1DBlock(nn.Module):
    def __init__(
        self,
        conv_in_dim: int,
        conv_kernel_size: int = 3,
        conv_padding_l: Optional[int] = None,
        conv_bias: bool = True,
        conv_weight_bias: bool = False,
        conv_heads: int = 8,
        conv_dynamic: bool = False,
        conv_ops_seq: Tuple[str, ...] = ("glu", "conv"),
        weight_softmax: bool = False,
        weight_dropout_p: float = 0.0,
        ff_ops_seq: Tuple[str, ...] = ("linear", "relu", "dropout"),
        ff_dropout_p: float = 0.1,
        norm_before: bool = False,
        norm_after: bool = False,
        weight_norm: str = "none",
        norm_groups: int = 1,
    ):
        super().__init__()
        self.conv_layer = LightWeightConv1DBlock(
            n_in=conv_in_dim,
            kernel_size=conv_kernel_size,
            padding_l=conv_padding_l,
            weight_bias=conv_weight_bias,
            conv_bias=conv_bias,
            num_heads=conv_heads,
            dynamic=conv_dynamic,
            ops_seq=conv_ops_seq,
            weight_softmax=weight_softmax,
            weight_dropout_p=weight_dropout_p,
            res=False,
            channel_last=True,
            weight_norm=weight_norm,
            norm_groups=norm_groups,
        )

        self.ff = MultiLinearLayers(
            layers=2,
            in_dim=conv_in_dim,
            inner_dim=4 * conv_in_dim,
            ops_seq=ff_ops_seq,
            dropout_p=ff_dropout_p,
            dim_back=True,
            last_activated=False,
            residual=False,
            channel_last=True,
        )
        assert not (
            norm_before and norm_after
        ), f"LayerNorms only exist before {norm_before} or after {norm_after} main body"
        if norm_before or norm_after:
            self.ln_layers = nn.ModuleList([nn.LayerNorm(conv_in_dim) for _ in range(2)])
        self.norm_before = norm_before
        self.norm_after = norm_after

    def layer_norm(self, idx, x):
        x = self.ln_layers[idx](x)
        return x

    def forward(self, x, x_mask=None, y=None, incremental_state=None):
        residual = x
        if self.norm_before:
            x = self.layer_norm(0, x)
        x = self.conv_layer(x, x_mask)
        x = residual + x
        if self.norm_after:
            x = self.layer_norm(0, x)

        residual = x
        if self.norm_before:
            x = self.layer_norm(1, x)
        x = self.ff(x)
        if y is not None:
            x = x + y
        x = residual + x
        if self.norm_after:
            x = self.layer_norm(1, x)
        return x

    def clear_buffer(self):
        self.conv_layer.clear_buffer()


class LConv1DCrossAttnBlock(LConv1DBlock):
    def __init__(
        self,
        conv_in_dim: int,
        conv_kernel_size: int = 3,
        conv_padding_l: Optional[int] = None,
        conv_bias: bool = True,
        conv_weight_bias: bool = False,
        conv_heads: int = 8,
        conv_dynamic: bool = False,
        conv_ops_seq: Tuple[str, ...] = ("glu", "conv"),
        weight_softmax: bool = False,
        weight_dropout_p: float = 0.0,
        ff_ops_seq: Tuple[str, ...] = ("linear", "relu", "dropout"),
        ff_dropout_p: float = 0.1,
        norm_before: bool = False,
        norm_after: bool = False,
        num_attn_heads: int = 2,
        kv_channels: int = 64,
        weight_norm: str = "none",
        norm_groups: int = 1,
    ):
        super().__init__(
            conv_in_dim,
            conv_kernel_size,
            conv_padding_l,
            conv_bias,
            conv_weight_bias,
            conv_heads,
            conv_dynamic,
            conv_ops_seq,
            weight_softmax,
            weight_dropout_p,
            ff_ops_seq,
            ff_dropout_p,
            norm_before,
            norm_after,
            weight_norm=weight_norm,
            norm_groups=norm_groups,
        )
        head_channels = conv_in_dim // num_attn_heads
        self.attn = MultiHeadAttn(
            in_channels=conv_in_dim,
            head_channels=head_channels,
            kv_channels=kv_channels,
            num_heads=num_attn_heads,
            attn_func_type=3,
        )

    def layer_norm(self, idx, x):
        x = self.ln_layers[idx](x)
        return x

    def forward(self, x, enc_kv, x_mask=None, kv_mask=None, return_attn=False):
        attn_mask = get_outter_attn_mask(None, x_mask, kv_mask)
        residual = x
        if self.norm_before:
            x = self.layer_norm(0, x)
        x = self.conv_layer(x, x_mask)
        x = residual + x
        if self.norm_after:
            x = self.layer_norm(0, x)

        residual = x
        context, _ = self.attn(
            x, enc_kv=enc_kv, attn_mask=attn_mask, sample=False, return_attn=return_attn, pos_info=None
        )
        x = residual + context

        residual = x
        if self.norm_before:
            x = self.layer_norm(1, x)
        x = self.ff(x)
        x = residual + x
        if self.norm_after:
            x = self.layer_norm(1, x)
        return x

    def clear_buffer(self):
        self.conv_layer.clear_buffer()


class MultiLConv1DLayers(nn.Module):
    """ """

    def __init__(
        self,
        num_layers: int,
        num_lconv_in_dim: int,
        lconv_ops_seq: Tuple[str, ...] = ("glu", "conv"),
        lconv_kernel_size: int = 3,
        lconv_padding_l: Optional[int] = None,
        lconv_weight_bias: bool = False,
        lconv_bias: bool = True,
        lconv_dynamic: bool = False,
        num_lconv_heads: int = 8,
        lconv_weight_softmax: bool = False,
        lconv_weight_dropout_p: float = 0.0,
        ff_ops_seq: Tuple[str, ...] = ("linear", "relu", "dropout"),
        ff_dropout_p: float = 0.1,
        norm_before: bool = False,
        norm_after: bool = False,
        weight_norm: str = "none",
        norm_groups: int = 1,
    ):
        super().__init__()
        self.lconv_blocks = nn.ModuleList()
        for _ in range(num_layers):
            conv_layer = LConv1DBlock(
                conv_in_dim=num_lconv_in_dim,
                conv_kernel_size=lconv_kernel_size,
                conv_padding_l=lconv_padding_l,
                conv_weight_bias=lconv_weight_bias,
                conv_bias=lconv_bias,
                conv_dynamic=lconv_dynamic,
                conv_heads=num_lconv_heads,
                conv_ops_seq=lconv_ops_seq,
                weight_softmax=lconv_weight_softmax,
                weight_dropout_p=lconv_weight_dropout_p,
                ff_ops_seq=ff_ops_seq,
                ff_dropout_p=ff_dropout_p,
                norm_before=norm_before,
                norm_after=norm_after,
                weight_norm=weight_norm,
                norm_groups=norm_groups,
            )
            self.lconv_blocks.append(conv_layer)

    def forward(
        self,
        x: Tensor,
        x_mask: Optional[Tensor] = None,
        y_list: Optional[List[Tensor]] = None,
        return_all: bool = False,
    ) -> Union[Tensor, List[Tensor]]:
        """
        Shapes:
            x: [B, T, C]
            x_mask: [B, T, C]
        """
        if x_mask is not None:
            x = x * x_mask
        if y_list is None:
            if return_all:
                x_list = []
                for block in self.lconv_blocks:
                    x = block(x, x_mask=x_mask)
                    x_list.append(x)
            else:
                for block in self.lconv_blocks:
                    x = block(x, x_mask=x_mask)
                x_list = x
        else:
            if isinstance(y_list, (list, tuple)):
                if return_all:
                    x_list = []
                    for block, y in zip(self.lconv_blocks, y_list):
                        x = block(x, x_mask=x_mask, y=y)
                        x_list.append(x)
                else:
                    for block, y in zip(self.lconv_blocks, y_list):
                        x = block(x, x_mask=x_mask, y=y)
                    x_list = x
            else:
                if return_all:
                    x_list = []
                    for block in self.lconv_blocks:
                        x = block(x, x_mask=x_mask, y=y_list)
                        x_list.append(x)
                else:
                    for block in self.lconv_blocks:
                        x = block(x, x_mask=x_mask, y=y_list)
                    x_list = x
        return x_list

    def clear_buffer(self):
        for block in self.lconv_blocks:
            block.clear_buffer()


class MultiLConv1DCrossAttnLayers(nn.Module):
    """ """

    def __init__(
        self,
        num_layers: int,
        num_lconv_in_dim: int,
        lconv_ops_seq: Tuple[str, ...] = ("glu", "conv"),
        lconv_kernel_size: int = 3,
        lconv_padding_l: Optional[int] = None,
        lconv_weight_bias: bool = False,
        lconv_bias: bool = True,
        lconv_dynamic: bool = False,
        num_lconv_heads: int = 8,
        lconv_weight_softmax: bool = False,
        lconv_weight_dropout_p: float = 0.0,
        ff_ops_seq: Tuple[str, ...] = ("linear", "relu", "dropout"),
        ff_dropout_p: float = 0.1,
        norm_before: bool = False,
        norm_after: bool = False,
        num_attn_heads: int = 4,
        kv_channels: int = 64,
        weight_norm: str = "none",
        norm_groups: int = 1,
    ):
        super().__init__()
        self.lconv_blocks = nn.ModuleList()
        for _ in range(num_layers):
            conv_layer = LConv1DCrossAttnBlock(
                conv_in_dim=num_lconv_in_dim,
                conv_kernel_size=lconv_kernel_size,
                conv_padding_l=lconv_padding_l,
                conv_weight_bias=lconv_weight_bias,
                conv_bias=lconv_bias,
                conv_dynamic=lconv_dynamic,
                conv_heads=num_lconv_heads,
                conv_ops_seq=lconv_ops_seq,
                weight_softmax=lconv_weight_softmax,
                weight_dropout_p=lconv_weight_dropout_p,
                ff_ops_seq=ff_ops_seq,
                ff_dropout_p=ff_dropout_p,
                norm_before=norm_before,
                norm_after=norm_after,
                num_attn_heads=num_attn_heads,
                kv_channels=kv_channels,
                weight_norm=weight_norm,
                norm_groups=norm_groups,
            )

            self.lconv_blocks.append(conv_layer)

    def forward(
        self,
        x: Tensor,
        enc_kv: Tensor,
        x_mask: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
        return_all: bool = False,
    ) -> Union[Tensor, List[Tensor]]:
        """
        Shapes:
            x: [B, T, C]
            x_mask: [B, T, C]
        """
        if x_mask is not None:
            x = x * x_mask
        if return_all:
            x_list = []
            for block in self.lconv_blocks:
                x = block(x, enc_kv=enc_kv, x_mask=x_mask, kv_mask=kv_mask)
                x_list.append(x)
        else:
            for block in self.lconv_blocks:
                x = block(x, enc_kv=enc_kv, x_mask=x_mask, kv_mask=kv_mask)
            x_list = x
        return x_list

    def clear_buffer(self):
        for block in self.lconv_blocks:
            block.clear_buffer()
