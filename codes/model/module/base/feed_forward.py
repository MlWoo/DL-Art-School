from typing import Optional, Tuple, Union

from torch import Tensor, nn

from .conv import Conv1DBlock
from .linear import LinearBlock


class PositionwiseFeedForwardNetwork(nn.Module):
    """Feed Forward Inner layers for Transformer.
    Args:
        in_channels (int): input tensor channels.
        out_channels (int): output tensor channels.
        hidden_channels (int): inner layers hidden channels.
        kernel_size (int): conv1d filter kernel size.
        dropout_p (float, optional): dropout rate. Defaults to 0.
    """

    def __init__(
        self, in_channels: int, out_channels: int, hidden_channels: int, act_func: str = "mish", dropout_p: float = 0.0
    ):

        super().__init__()
        self.linear_1 = LinearBlock(in_channels, hidden_channels, ops_seq=("linear", act_func))
        self.linear_2 = LinearBlock(hidden_channels, out_channels, ops_seq=("linear", "dropout"), dropout_p=dropout_p)

    def forward(self, x: Tensor, x_mask: Optional[Tensor] = None) -> Tensor:
        # B T C ---> B C T
        x = self.linear_1(x)
        x = self.linear_2(x)
        if x_mask is not None:
            x = x * x_mask
        return x


class ConformerFeedForwardNetwork(nn.Module):
    """
    Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep model
    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        act_func: str = "swish",
        dropout_p: float = 0.1,
        causal: bool = False,
        padding_mode: str = "zeros",
        norm_groups: int = 0,
    ):
        super().__init__()
        if causal:
            assert kernel_size % 2 == 0, "kernel_size should be a even number for 'SAME' padding in causal mode"
        else:
            assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        self.conv_block_1 = Conv1DBlock(
            in_channels,
            in_channels,
            kernel_size,
            causal=causal,
            padding_mode=padding_mode,
            ops_seq=("ln", "glu", "conv"),
            groups=in_channels,
            channel_last=True,
        )
        if norm_groups > 0:
            ops_seq = ("gn", act_func, "linear", "dropout")
        else:
            ops_seq = ("bn", act_func, "linear", "dropout")
        self.conv_block_2 = LinearBlock(
            in_channels, in_channels, ops_seq=ops_seq, dropout_p=dropout_p, channel_last=True, norm_groups=norm_groups
        )

    def forward(self, x: Tensor, x_mask: Optional[Tensor] = None) -> Tensor:
        # x: B T C
        # x_mask: B T C
        x = self.conv_block_1(x, x_mask=x_mask)
        x = self.conv_block_2(x, x_mask=x_mask)
        return x


class ConvFeedForwardNetwork(nn.Module):
    """Feed Forward Inner layers for Transformer.
    Args:
        in_channels (int): input tensor channels.
        out_channels (int): output tensor channels.
        hidden_channels (int): inner layers hidden channels.
        kernel_size (int): conv1d filter kernel size.
        dropout_p (float, optional): dropout rate. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        act_func: str = "mish",
        dropout_p: float = 0.0,
        channel_last: bool = False,
        causal: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        if isinstance(kernel_size, (tuple, list)):
            self.conv_1 = Conv1DBlock(
                in_channels,
                hidden_channels,
                kernel_size[0],
                ops_seq=("conv", act_func),
                causal=causal,
                padding_mode=padding_mode,
            )
            self.conv_2 = Conv1DBlock(
                hidden_channels,
                out_channels,
                kernel_size[1],
                ops_seq=("conv", "dropout"),
                causal=causal,
                padding_mode=padding_mode,
                dropout_p=dropout_p,
            )
        else:
            self.conv_1 = Conv1DBlock(
                in_channels,
                hidden_channels,
                kernel_size,
                ops_seq=("conv", act_func),
                causal=causal,
                padding_mode=padding_mode,
            )
            self.conv_2 = Conv1DBlock(
                hidden_channels,
                out_channels,
                kernel_size,
                ops_seq=("conv", "dropout"),
                causal=causal,
                padding_mode=padding_mode,
                dropout_p=dropout_p,
            )
        self.channel_last = channel_last

    def forward(self, x: Tensor, x_mask: Optional[Tensor] = None) -> Tensor:
        # B T C ---> B C T
        if x_mask is None:
            conv_x_mask = None
        else:
            conv_x_mask = x_mask.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        x = self.conv_1(x, x_mask=conv_x_mask)
        x = self.conv_2(x, x_mask=conv_x_mask)
        x = x.permute(0, 2, 1)
        if x_mask is not None:
            x = x * x_mask
        return x
