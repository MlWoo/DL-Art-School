import math
from typing import List, Optional, Tuple, Union

import torch
from model.audio.module.util import apply_compile_transformer
from model.module.base.activation import ACT2FN
from model.module.base.conv import SamePad
from model.module.base.mask import sequence_mask
from model.module.base.positional_encoding import PositionalEncoding, RelativePositionalEncoding
from model.module.base.transformer import Transformer as TransformerBase
from model.module.base.util import extend2tuple
from torch import Tensor, nn
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput
from utils import logging
from utils.util import default, print_network

logger = logging.getLogger("base")


class Transformer(nn.Module):
    """Base class for FS2Encoder."""

    def __init__(
        self,
        attn_type: str = "base",
        num_hidden_layers: int = 4,
        num_attention_heads: int = 2,
        num_attn_in_dim: int = 512,
        num_attn_head_dim: int = 256,
        num_attn_out_dim: int = 512,
        ffn_dim: int = 1024,
        attn_func_type: int = 0,
        ffn_kernel_size: Union[int, Tuple[int, int]] = (9, 1),
        ffn_act_func: str = "mish",
        attn_dropout_p: float = 0.0,
        cxt_dropout_p: float = 0.0,
        ffn_dropout_p: float = 0.0,
        pre_LN: bool = False,
        final_LN: bool = True,
        use_macaron: bool = False,
        conv_module_type: Optional[str] = None,
        conv_module_kernel_size: int = 31,
        conformer_conv_dropout_p: float = 0.0,
        pre_conv_module: bool = True,
        ffn_cat_after: bool = False,
        causal: bool = False,
        padding_mode: str = "zeros",
        chunk_len: Optional[int] = None,
        norm_groups: int = 0,
        pos_type: str = "base",
        pos_dropout_p: float = 0.0,
        window_size: Optional[int] = None,
        max_len: int = 1000,
    ):
        super().__init__()
        if attn_type == "base":
            x_scale = None
            scaled = False
            if pos_type == "scaled":
                scaled = True
            elif pos_type == "nlp":
                x_scale = None
            else:
                x_scale = 1.0
            self.pos_enc = PositionalEncoding(
                num_attn_in_dim, dropout_p=pos_dropout_p, scale=x_scale, scaled=scaled, max_len=max_len
            )
        elif attn_type == "rpr":
            if pos_type == "nlp":
                x_scale = None
            else:
                x_scale = 1.0
            self.pos_enc = RelativePositionalEncoding(
                num_attn_head_dim * num_attention_heads,
                dropout_p=pos_dropout_p,
                scale=x_scale,
                max_len=max_len,
                win_len=window_size,
                chunk_len=chunk_len,
            )
        elif attn_type == "rpr-native":
            self.x_scale = math.sqrt(num_attn_in_dim)
        elif attn_type == "wo-pos":
            self.x_scale = 1.0
        else:
            raise ValueError(f"Transformer: {attn_type} is not supported for the moment")

        self.transformer = TransformerBase(
            in_attn_size=num_attn_in_dim,
            head_attn_size=num_attn_head_dim,
            out_attn_size=num_attn_out_dim,
            hidden_ffn_size=ffn_dim,
            num_hidden_layers=num_hidden_layers,
            num_heads=num_attention_heads,
            out_ffn_size=num_attn_out_dim,
            attn_type=attn_type,
            attn_func_type=attn_func_type,
            ffn_kernel_size=ffn_kernel_size,
            ffn_act_func=ffn_act_func,
            attn_dropout_p=attn_dropout_p,
            cxt_dropout_p=cxt_dropout_p,
            ffn_dropout_p=ffn_dropout_p,
            pre_LN=pre_LN,
            use_macaron=use_macaron,
            conv_module_type=conv_module_type,
            conv_module_kernel_size=conv_module_kernel_size,
            conformer_conv_dropout_p=conformer_conv_dropout_p,
            pre_conv_module=pre_conv_module,
            ffn_cat_after=ffn_cat_after,
            causal=causal,
            padding_mode=padding_mode,
            chunk_len=chunk_len,
            norm_groups=norm_groups,
            final_LN=final_LN,
            window_size=window_size,
            max_len=max_len,
        )
        self.attn_type = attn_type

    def forward(
        self,
        x: Tensor,
        extra: Optional[Tensor] = None,
        x_mask: Optional[Tensor] = None,
        fertilities: Optional[Tensor] = None,
        indices: Optional[Tensor] = None,
        enc_kv: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        sample: bool = False,
        return_all: bool = False,
        return_attn: bool = False,
        return_attn_num: int = -1,
        layer_idx: int = -1,
    ):
        """
        text - (batch, maxseqlen)
        """
        if self.attn_type == "base":
            assert (
                fertilities is None or indices is None
            ), "The encoding is confused if both fertilities and indices are provided."
            x, pos_encoding = self.pos_enc(x, fertilities, indices)
            x, attn_list = self.transformer(
                x,
                x_mask=x_mask,
                enc_kv=enc_kv,
                kv_mask=kv_mask,
                attn_mask=attn_mask,
                sample=sample,
                return_all=return_all,
                return_attn=return_attn,
                return_attn_num=return_attn_num,
                layer_idx=layer_idx,
            )
        else:
            if self.attn_type == "rpr":
                x, pos_encoding = self.pos_enc(x)
            else:
                x = x * self.x_scale
                pos_encoding = None
            x, attn_list = self.transformer(
                x,
                pos_info=pos_encoding,
                x_mask=x_mask,
                enc_kv=enc_kv,
                kv_mask=kv_mask,
                attn_mask=attn_mask,
                sample=sample,
                return_all=return_all,
                return_attn=return_attn,
                return_attn_num=return_attn_num,
                layer_idx=layer_idx,
            )
        return x, attn_list


class Conv1dSubsampling(nn.Module):
    """
    2d Convolutional subsampling.
    Subsamples time and freq domains of input spectrograms by a factor of 4, d_model times.

    Parameters:
    d_model (int): Dimension of the model

    Inputs:
    x (Tensor): Input spectrogram (batch_size, time, d_input)

    Outputs:
    Tensor (batch_size, time, d_model * (d_input // 4)): Output tensor from the conlutional subsampling module

    """

    def __init__(
        self,
        sub_layers: int = 2,
        input_channels: int = 1,  # unused
        input_dim: int = 128,
        conv_hidden_size: int = 144,
        hidden_size: int = 144,
        padding_mode: str = "zero",
        causal: bool = True,
        activation: str = "relu",
    ):
        super(Conv1dSubsampling, self).__init__()
        if isinstance(conv_hidden_size, (list, tuple)):
            conv_hidden_size = conv_hidden_size[0]
        if sub_layers == 2:
            self.layers = nn.Sequential(
                SamePad(kernel_size=3, causal=causal, stride=2, padding_mode=padding_mode),
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=conv_hidden_size,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                ACT2FN[activation](),
                SamePad(kernel_size=3, causal=causal, stride=2, padding_mode=padding_mode),
                nn.Conv1d(
                    in_channels=conv_hidden_size,
                    out_channels=hidden_size,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                ACT2FN[activation](),
            )
        else:
            self.layers = nn.Sequential(
                SamePad(kernel_size=3, causal=causal, stride=2, padding_mode=padding_mode),
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=hidden_size,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                ACT2FN[activation](),
            )

    def forward(self, x):
        # x (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.layers(x)  # (B, T, C)
        x = x.permute(0, 2, 1)
        return x


class Conv2dSubsampling(nn.Module):
    """
    2d Convolutional subsampling.
    Subsamples time and freq domains of input spectrograms by a factor of 4, d_model times.

    Parameters:
    d_model (int): Dimension of the model

    Inputs:
    x (Tensor): Input spectrogram (batch_size, time, d_input)

    Outputs:
    Tensor (batch_size, time, d_model * (d_input // 4)): Output tensor from the conlutional subsampling module

    """

    def __init__(
        self,
        sub_layers: int = 2,
        input_channels: int = 1,
        input_dim: int = 128,
        conv_hidden_size: List[int] = [144, 144],
        hidden_size: int = 1024,
        padding_mode: str = "zero",
        proj_dropout_p: float = 0.1,
        causal: bool = True,
        activation: str = "relu",
    ):
        super(Conv2dSubsampling, self).__init__()
        if not isinstance(conv_hidden_size, (list, tuple)):
            conv_hidden_size = [
                conv_hidden_size,
            ] * sub_layers
        if sub_layers == 2:
            self.layers = nn.Sequential(
                SamePad(kernel_size=3, causal=causal, stride=2, padding_mode=padding_mode),
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=conv_hidden_size[0],
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                ACT2FN[activation](),
                SamePad(kernel_size=3, causal=causal, stride=2, padding_mode=padding_mode),
                nn.Conv2d(
                    in_channels=conv_hidden_size[0],
                    out_channels=conv_hidden_size[1],
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                ACT2FN[activation](),
            )
        else:
            self.layers = nn.Sequential(
                SamePad(kernel_size=3, causal=causal, stride=2, padding_mode=padding_mode),
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=conv_hidden_size[-1],
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    padding_mode=padding_mode,
                ),
                ACT2FN[activation](),
            )

        if sub_layers == 2:
            self.linear_proj = nn.Linear(
                conv_hidden_size[-1] * (((input_dim) // 2) // 2), hidden_size
            )  # project subsamples to d_model
            self.reduction_factors = 4
        else:
            self.linear_proj = nn.Linear(
                conv_hidden_size[-1] * ((input_dim) // 2), hidden_size
            )  # project subsamples to d_model
            self.reduction_factors = 2
        self.proj_dropout = nn.Dropout(p=proj_dropout_p, inplace=True)

    def forward(self, x):
        output = self.layers(x.unsqueeze(1))  # (batch_size, 1, time, d_input)
        batch_size, d_model, subsampled_time, subsampled_freq = output.size()
        x = output.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size, subsampled_time, d_model * subsampled_freq)
        x = self.linear_proj(x)
        x = self.proj_dropout(x)
        return x


class BestRqConformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        input_channels: int = 1,
        num_attention_heads: int = 8,
        hidden_size: int = 1024,
        hidden_act: str = "swish",
        num_hidden_layers: int = 10,
        conv_depthwise_kernel_size: int = 5,
        feat_proj_dropout: float = 0.0,  # unused
        activation_dropout: float = 0.0,
        hidden_dropout: float = 0.0,  # unused
        conformer_conv_dropout: float = 0.0,  # unused
        final_dropout: float = 0.0,  # unused
        attention_dropout: float = 0.0,
        max_source_positions: int = 4096,
        num_preconformer_layers: int = 3,
        num_preconformer_heads: int = 4,
        preconformer_hidden_size: int = 384,
        preconformer_ffn_dim: int = 1536,
        sub_type: str = "2d",
        sub_layers: int = 2,
        conv_hidden_size: List[int] = [8, 32],
        layerdrop: float = 0.0,  # unused
        ffn_dim: Optional[int] = None,
        proj_dropout_p: float = 0.1,
        cxt_dropout_p: float = 0.1,
        ffn_dropout_p: float = 0.1,
        pos_dropout_p: float = 0.0,
        causal: bool = True,
        window_size: int = 64,
        in_attn_size: Optional[int] = None,
        padding_mode: str = "zeros",
        chunk_len: Optional[int] = None,
        norm_groups: int = 8,
        codebook_dim: int = -1,
        channel_last: bool = True,
        conv_hidden_act: str = "gelu",
        preconformer_input_feature_projection: bool = False,  # unused
        no_scale_embedding: bool = False,  # unused
        rotary_embedding_base: int = 10000,  # unused
        position_embeddings_type: str = "relative",
        compile: bool = False,
    ) -> None:
        super().__init__()
        # compatible
        if position_embeddings_type == "relative":
            attn_type = "rpr"
        else:
            attn_type = position_embeddings_type

        conv_module_kernel_size = (
            conv_depthwise_kernel_size // 2 * 2 if causal else conv_depthwise_kernel_size // 2 * 2 + 1
        )
        conformer_conv_dropout_p = activation_dropout

        if in_attn_size is not None:
            input_dim = in_attn_size

        attn_func_type = 1 if causal else 0
        attn_func_type = attn_func_type if window_size is None else attn_func_type + 1

        conv_hidden_size = extend2tuple(conv_hidden_size, sub_layers)
        if num_preconformer_layers > 0:
            if conv_hidden_size[0] == input_dim:
                self.proj_linear = None
            else:
                self.proj_linear = nn.Linear(input_dim, preconformer_hidden_size)
            num_attn_in_dim = num_attn_out_dim = preconformer_hidden_size
            num_attn_head_dim = num_attn_in_dim // num_preconformer_heads
            preconformer_ffn_dim = default(preconformer_ffn_dim, num_attn_in_dim * 4)
            if attn_type == "wo-pos":
                pre_conformer_attn_type = "rpr"
                logger.warning("Pre Conformer should be with relative positional encoding")
            else:
                pre_conformer_attn_type = attn_type

            self.pre_conformer = Transformer(
                attn_type=pre_conformer_attn_type,
                num_hidden_layers=num_preconformer_layers,
                num_attention_heads=num_preconformer_heads,
                num_attn_in_dim=num_attn_in_dim,
                num_attn_out_dim=num_attn_out_dim,
                num_attn_head_dim=num_attn_head_dim,
                ffn_dim=preconformer_ffn_dim,
                attn_func_type=attn_func_type,
                ffn_kernel_size=1,
                ffn_act_func=hidden_act,
                attn_dropout_p=attention_dropout,
                cxt_dropout_p=cxt_dropout_p,
                ffn_dropout_p=ffn_dropout_p,
                pos_dropout_p=pos_dropout_p,
                pre_LN=True,
                final_LN=False,
                use_macaron=True,
                conv_module_type="original",
                conv_module_kernel_size=conv_module_kernel_size,
                causal=causal,
                padding_mode=padding_mode,
                window_size=window_size,
                max_len=max_source_positions,
                norm_groups=norm_groups,
            )
            input_dim = num_attn_in_dim
        else:
            self.proj_linear = None
            self.pre_conformer = None

        if sub_type == "2d":
            self.conv_subsample = Conv2dSubsampling(
                sub_layers=sub_layers,
                input_dim=input_dim,
                input_channels=input_channels,
                conv_hidden_size=conv_hidden_size,
                hidden_size=hidden_size,
                padding_mode=padding_mode,
                proj_dropout_p=proj_dropout_p,
                causal=causal,
                activation=conv_hidden_act,
            )
        elif sub_type == "1d":
            self.conv_subsample = Conv1dSubsampling(
                sub_layers=sub_layers,
                input_channels=input_channels,
                input_dim=input_dim,
                conv_hidden_size=conv_hidden_size,
                hidden_size=hidden_size,
                padding_mode=padding_mode,
                causal=causal,
                activation=conv_hidden_act,
            )
        else:
            raise ImportError(f"{sub_type} Subsampling Module")
        if sub_layers == 2:
            self.reduction_factors = 4
        else:
            self.reduction_factors = 2
        num_attn_in_dim = num_attn_out_dim = hidden_size
        num_attn_head_dim = num_attn_in_dim // num_attention_heads
        ffn_dim = default(ffn_dim, num_attn_in_dim * 4)

        if chunk_len is not None:
            chunk_len = chunk_len // self.reduction_factors
        self.chunk_len = chunk_len

        self.conformer = Transformer(
            attn_type=attn_type,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_attn_in_dim=num_attn_in_dim,
            num_attn_out_dim=num_attn_out_dim,
            num_attn_head_dim=num_attn_head_dim,
            ffn_dim=ffn_dim,
            attn_func_type=attn_func_type,
            ffn_kernel_size=1,
            ffn_act_func=hidden_act,
            attn_dropout_p=attention_dropout,
            cxt_dropout_p=cxt_dropout_p,
            ffn_dropout_p=ffn_dropout_p,
            pre_LN=True,
            final_LN=True,
            use_macaron=True,
            conv_module_type="original",
            conv_module_kernel_size=conv_module_kernel_size,
            conformer_conv_dropout_p=conformer_conv_dropout_p,
            causal=causal,
            padding_mode=padding_mode,
            chunk_len=chunk_len,
            window_size=window_size,
            max_len=max_source_positions // self.reduction_factors,
            norm_groups=norm_groups,
        )
        if codebook_dim > 0:
            self.output_proj = nn.Linear(hidden_size, codebook_dim)
            self.output_dim = codebook_dim
        else:
            self.output_proj = nn.Identity()
            self.output_dim = hidden_size
        self.channel_last = channel_last
        self.hidden_size = hidden_size
        self.compiled = False

    def apply_compile(self):
        if not self.compiled:
            apply_compile_transformer(self.conv_subsample)
            apply_compile_transformer(self.conformer)
            if self.pre_conformer is not None:
                apply_compile_transformer(self.pre_conformer)
            self.compiled = True

    def forward(self, input_values, input_lengths=None, layer_idx: int = -1, return_attn_num: int = -1):
        if not self.channel_last:
            input_values = input_values.permute(0, 2, 1)

        if input_lengths is None:
            mask = None
        else:
            mask = sequence_mask(input_lengths, max_len=input_values.shape[1]).unsqueeze(-1)
            assert mask.shape[1] == input_values.shape[1], f"mask {mask.shape}, input {input_values.shape}"

        if self.proj_linear is not None:
            input_values = self.proj_linear(input_values)
        if self.pre_conformer is not None:
            input_values, _ = self.pre_conformer(input_values, x_mask=mask)

        input_features = self.conv_subsample(input_values)

        if input_lengths is None:
            features_mask = None
        else:
            features_lengths = input_lengths // self.reduction_factors
            features_mask = sequence_mask(features_lengths, max_len=input_features.shape[1]).unsqueeze(-1)
            assert (
                features_mask.shape[1] == input_features.shape[1]
            ), f"features mask {features_mask.shape}, features input {input_features.shape}"

        output_values, attn_list = self.conformer(
            x=input_features, x_mask=features_mask, layer_idx=layer_idx, return_attn_num=return_attn_num
        )
        if layer_idx == -1:
            output_values = self.output_proj(output_values)
        if not self.channel_last:
            output_values = output_values.permute(0, 2, 1)

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=output_values,
            extract_features=input_features,
            hidden_states=None,
            attentions=None,
        )


if __name__ == "__main__":
    bs = 2
    sub_layers = 2
    sub_type = "1d"
    max_length = 1600
    input_dim = 80
    window_size = 8
    num_preconformer_layers = 3
    conv_hidden_size = 512
    num_attention_heads = 8
    num_hidden_layers = 24
    hidden_size = 1024
    attn_type = "wo-pos"
    chunk_len = 800
    causal = True
    norm_groups = 8

    for sub_type in ["1d", "2d"]:
        for causal in [False, True]:
            encoder = BestRqConformerEncoder(
                attn_type=attn_type,
                num_preconformer_layers=num_preconformer_layers,
                sub_type=sub_type,
                sub_layers=sub_layers,
                conv_hidden_size=conv_hidden_size,
                input_dim=input_dim,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                window_size=window_size,
                chunk_len=chunk_len,
                causal=causal,
                norm_groups=norm_groups,
            ).cuda()
            print_network(encoder)
            for max_length_ in range(max_length, max_length + 4):
                print(f"sub_type {sub_type}, causal: {causal}, max_length {max_length_}")
                input_lengths = torch.randint(max_length_ // 2, max_length_, (bs,)).cuda()
                input_lengths[-1] = max_length_
                inputs = torch.randn(bs, max_length_, input_dim).cuda()

                outputs, _ = encoder(x=inputs, x_lengths=input_lengths)

                print(outputs.shape)
