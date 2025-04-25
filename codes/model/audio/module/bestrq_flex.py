from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from model.audio.module.util import apply_compile_transformer
from model.module.base.activation import ACT2FN
from model.module.base.conv import SamePad
from model.module.base.mask import sequence_mask
from model.module.base.transformer import Transformer, TransformerCompile  # noqa F401
from model.module.base.util import extend2tuple
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput
from utils.logging_utils import get_root_logger
from utils.util import default, print_network

logger = get_root_logger()


@dataclass
class BestRqConformerEncoderOutput(Wav2Vec2BaseModelOutput):
    """
    Base class for models that have been trained with the Wav2Vec2 loss objective.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim[-1])`):
            Sequence of extracted feature vectors of the last convolutional layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`
            is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of
            each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    extract_features: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    score_mask: Optional[Tuple[torch.BoolTensor, ...]] = None


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
        stream_chunk_size: Optional[int] = None,
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
        attn_func_type = attn_func_type if window_size is None else attn_func_type + 2

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
            input_dim = num_attn_in_dim
        else:
            self.proj_linear = None

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

        if num_preconformer_layers > 0:
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
                stream_chunk_size=stream_chunk_size * self.reduction_factors if stream_chunk_size is not None else None,
                window_size=window_size * self.reduction_factors if window_size is not None else None,
                max_len=max_source_positions,
                norm_groups=norm_groups,
            )
        else:
            self.pre_conformer = None

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
            chunkwise_size=chunk_len,
            stream_chunk_size=stream_chunk_size,
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
        self.compile = compile
        self.compiled = False

    def apply_compile(self):
        if self.compile and not self.compiled:
            apply_compile_transformer(self.conv_subsample)
            apply_compile_transformer(self.conformer)
            if self.pre_conformer is not None:
                apply_compile_transformer(self.pre_conformer)
            self.compiled = True

    def forward(self, input_values, input_lengths=None, layer_idx: int = -1, num_attn: int = -1):
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

        output_values, attn_tuple, score_mask_tuple = self.conformer(
            x=input_features, x_mask=features_mask, layer_idx=layer_idx, num_attn=num_attn
        )
        last_hidden_state = output_values[-1]
        if layer_idx == -1:
            last_hidden_state = self.output_proj(last_hidden_state)
        if not self.channel_last:
            last_hidden_state = last_hidden_state.permute(0, 2, 1)

        return BestRqConformerEncoderOutput(
            last_hidden_state=last_hidden_state,
            extract_features=input_features,
            hidden_states=output_values,
            attentions=attn_tuple,
            score_mask=score_mask_tuple,
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
