from model.audio.module.bestrq_conformer import ConformerEncoder  # noqa F401
from model.audio.module.bestrq_flex import Conv1dSubsampling, Conv2dSubsampling  # noqa F401
from model.audio.module.bestrq_flex import Transformer as Conformer
from torch import Tensor, nn


class USMSpeechEncoder(nn.Module):
    """Universal Speech Model (USMEncoder) is a model that can be used for any speech task.


    Args:
        dim (int): Dimension of the input features.
        heads (int): Number of heads in the multi-head attention.
        ff_dim (int): Dimension of the feed-forward
        depth (int): Number of layers in the model.
        depthwise_conv_kernel_size (int): Kernel size of the depthwise convolution.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        use_group_norm (bool, optional): Whether to use group normalization. Defaults to False.
        conv_first (bool, optional): Whether to use convolution first. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.


    Examples:
    >>> from usm import USMEncoder
    >>> model = USMEncoder(
    ...     dim=80,
    ...     heads=4,
    ...     ffn_dim=256,
    ...     depth=4,
    ...     depthwise_conv_kernel_size=32,
    ...     dropout=0.1,
    ...     use_group_norm=True,
    ...     conv_first=True
    ... )
    >>> input = torch.randn(10, 80, 100)
    >>> output = model(input)


    """

    def __init__(
        self,
        in_dim,
        heads,
        ffn_dim,
        depth,
        depthwise_conv_kernel_size,
        hidden_dim: int = 512,
        sub_layers: int = 2,
        sub_stride: int = 2,
        sub_num_resnet_blocks: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        use_group_norm: bool = False,
        conv_first: bool = False,
        *args,
        **kwargs,
    ):
        super(USMSpeechEncoder, self).__init__()
        self.subsampler = Conv2dSubsampling(
            in_channels=in_dim,
            num_layers=sub_layers,
            num_resnet_blocks=sub_num_resnet_blocks,
            hidden_dim=hidden_dim,
            stride=sub_stride,
            activation=activation,
            positional_dims=1,
        )
        channels_up = sub_stride ** (sub_layers - 1)
        self.conformer_layer = Conformer(
            input_dim=hidden_dim * channels_up,
            num_heads=heads,
            ffn_dim=ffn_dim,
            num_layers=depth,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            use_group_norm=use_group_norm,
            convolution_first=conv_first,
        )
        self.conformer1 = Conformer(
            input_dim=hidden_dim * channels_up,
            heads=heads,
            ff_dim=ffn_dim,
            depth=depth,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=use_group_norm,
            conv_first=conv_first,
            *args,
            **kwargs,
        )
        self.conformer2 = Conformer(
            input_dim=hidden_dim * channels_up,
            heads=heads,
            ff_dim=ffn_dim,
            depth=depth,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=use_group_norm,
            conv_first=conv_first,
            *args,
            **kwargs,
        )
        self.reduction_steps = self.subsampler.reduction_steps
        self.output_dim = hidden_dim * channels_up

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """Forward pass of the model.
        Args:
            input_values (Tensor): with shape `(B, T, D)`
            input_lengths (Tensor): with shape `(B)`

        Returns:
        """
        folded_lengths = lengths // self.reduction_steps
        transformed_input_lengths = folded_lengths * self.reduction_steps
        folded_lengths = transformed_input_lengths // self.reduction_steps
        x = x.permute(0, 2, 1)
        x = self.subsampler(x)
        x = x.permute(0, 2, 1)
        x, folded_lengths = self.conformer_layer(x, folded_lengths)
        enc1, folded_lengths1 = self.conformer1(x, folded_lengths)
        enc2, folded_lengths2 = self.conformer2(enc1, folded_lengths1)
        return enc1, enc2, folded_lengths1, folded_lengths2
