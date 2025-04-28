import inspect
import math
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .conv import Conv1DBlock
from .feed_forward import ConformerFeedForwardNetwork, ConvFeedForwardNetwork, PositionwiseFeedForwardNetwork
from .linear import Linear
from .mask import get_chunks_ceil_score_mask, get_inner_score_mask, get_outter_score_mask
from .multi_heads_attn import MultiHeadAttn, NativeRelativePositionMultiHeadAttn, RelativePositionMultiHeadAttn
from .normalization import ForgottenLayerNorm
from .positional_encoding import PositionalEncoding, RelativePositionalEncoding, RotaryPositionalEncoding
from .util import extend2tuple


class AbcResAttnBlock(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        attn,
        attn_size: int,
        hidden_ffn_size: int,
        out_ffn_size: Optional[int] = None,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        ffn_act_func: str = "mish",
        ffn_dropout_p: float = 0,
        ffn_scale: float = 1.0,
        norm_groups: int = 0,
        use_macaron: bool = False,
        conv_module_type: Optional[str] = None,
        conv_module_kernel_size: int = 31,
        conformer_conv_dropout_p: float = 0.0,
        causal: bool = False,
        padding_mode: str = "zeros",
        pre_conv_module: bool = True,
        pre_LN: bool = False,
        concat_after: bool = False,
        LN_learnable: bool = True,
        channel_last: bool = True,
    ):
        assert attn_size == attn.in_size
        super().__init__()
        self.attn = attn

        if concat_after:
            self.concat_linear = Linear(attn_size + attn_size, attn_size, channel_last=channel_last)

        if out_ffn_size is None:
            out_ffn_size = attn_size

        # check pffn or cffn
        pffn = True
        if isinstance(kernel_size, (list, tuple)):
            for ks in kernel_size:
                if ks > 1:
                    pffn = False
                    break
        else:
            if kernel_size > 1:
                pffn = False

        if pffn:
            self.ffn = PositionwiseFeedForwardNetwork(
                attn_size, out_ffn_size, hidden_ffn_size, act_func=ffn_act_func, dropout_p=ffn_dropout_p
            )
        else:
            self.ffn = ConvFeedForwardNetwork(
                attn_size,
                out_ffn_size,
                hidden_ffn_size,
                kernel_size,
                act_func=ffn_act_func,
                causal=causal,
                padding_mode=padding_mode,
                dropout_p=ffn_dropout_p,
            )

        if LN_learnable:
            ln_0 = nn.LayerNorm(attn_size)
            ln_1 = nn.LayerNorm(attn_size)
        else:
            ln_0 = ForgottenLayerNorm(attn_size)
            ln_1 = ForgottenLayerNorm(attn_size)
        self.layer_norm_list = nn.ModuleList([ln_0, ln_1])

        if use_macaron:
            self.ffn_scale = 0.5
            if pffn:
                self.macaron_ffn = PositionwiseFeedForwardNetwork(
                    attn_size, out_ffn_size, hidden_ffn_size, act_func=ffn_act_func, dropout_p=ffn_dropout_p
                )
            else:
                self.macaron_ffn = ConvFeedForwardNetwork(
                    attn_size,
                    out_ffn_size,
                    hidden_ffn_size,
                    kernel_size,
                    act_func=ffn_act_func,
                    causal=causal,
                    padding_mode=padding_mode,
                    dropout_p=ffn_dropout_p,
                    norm_groups=norm_groups,
                )
            if LN_learnable:
                ln_2 = nn.LayerNorm(attn_size)
            else:
                ln_2 = ForgottenLayerNorm(attn_size)
            self.layer_norm_list.append(ln_2)
        else:
            self.ffn_scale = ffn_scale
            self.macaron_ffn = None

        if conv_module_type == "original":
            self.conv_module = ConformerFeedForwardNetwork(
                attn_size,
                causal=causal,
                padding_mode=padding_mode,
                kernel_size=conv_module_kernel_size,
                norm_groups=norm_groups,
                dropout_p=conformer_conv_dropout_p,
            )
            ln_3 = nn.LayerNorm(attn_size)
            self.layer_norm_list.append(ln_3)
        elif conv_module_type == "tts":
            self.conv_module = Conv1DBlock(
                attn_size,
                attn_size,
                ops_seq=("id",),
                causal=causal,
                padding_mode=padding_mode,
                kernel_size=conv_module_kernel_size,
                groups=attn_size,
            )
            ln_3 = nn.LayerNorm(attn_size)
            self.layer_norm_list.append(ln_3)
        else:
            self.conv_module = None

        self.attn_size = attn_size
        self.pre_LN = pre_LN
        self.concat_after = concat_after
        self.LN_learnable = LN_learnable
        self.pre_conv_module = pre_conv_module
        self.channel_last = channel_last

    def get_kwargs(self, init_func, local_vars):
        module_kwargs = inspect.getfullargspec(init_func).args
        discarded_kw = ["self", "version"]
        module_kwargs = dict((name, local_vars[name]) for name in module_kwargs if name not in discarded_kw)
        return module_kwargs

    @abstractmethod
    def layernorm(self, x: Tensor, idx: int = 0):
        raise NotImplementedError

    def forward(
        self,
        x: Tensor,
        enc_kv: Optional[Tensor] = None,
        x_mask: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
        outter_score_mask: Optional[Tensor] = None,
        score_mask: Optional[Tensor] = None,
        sample: bool = False,
        num_attn: int = 0,
        pos_info: Optional[Tensor] = None,
    ):
        # B T C
        outter_score_mask = get_outter_score_mask(outter_score_mask, x_mask, kv_mask)

        # macaron_ffn module
        if self.macaron_ffn is not None:
            residual = x
            if self.pre_LN:
                x = self.layernorm(x, idx=2)
            if self.ffn_scale == 1.0:
                x = residual + self.macaron_ffn(x, x_mask)
            else:
                x = residual + self.ffn_scale * self.macaron_ffn(x, x_mask)
            if not self.pre_LN:
                x = self.layernorm(x, idx=2)

        # pre conv module
        if self.pre_conv_module and self.conv_module is not None:
            residual = x
            if self.pre_LN:
                x = self.layernorm(x, idx=3)
            x = self.conv_module(x, x_mask)
            if self.ffn_scale == 1.0:
                x = residual + x
            else:
                x = residual + self.ffn_scale * x
            if not self.pre_LN:
                x = self.layernorm(x, idx=3)

        # attention module
        residual = x
        if self.pre_LN:
            x = self.layernorm(x, idx=0)

        context, attn, score_mask = self.attn(
            x,
            enc_kv=enc_kv,
            score_mask=score_mask,
            outter_score_mask=outter_score_mask,
            sample=sample,
            num_attn=num_attn,
            pos_info=pos_info,
        )

        if x_mask is not None:
            context = context * x_mask
        # concat or not
        if self.concat_after:
            x_concat = torch.cat((x, context), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + context

        if not self.pre_LN:
            x = self.layernorm(x, idx=0)

        # post conv module
        if self.pre_conv_module and self.conv_module is not None:
            residual = x
            if self.pre_LN:
                x = self.layernorm(x, idx=3)
            x = self.conv_module(x, x_mask)
            if self.ffn_scale == 1.0:
                x = residual + x
            else:
                x = residual + self.ffn_scale * x
            if not self.pre_LN:
                x = self.layernorm(x, idx=3)

        # ffn module
        residual = x
        if self.pre_LN:
            x = self.layernorm(x, idx=1)

        if self.ffn_scale == 1.0:
            x = residual + self.ffn(x, x_mask)
        else:
            x = residual + self.ffn_scale * self.ffn(x, x_mask)

        if not self.pre_LN:
            x = self.layernorm(x, idx=1)

        return x, attn, score_mask


class ResAttnBlock(AbcResAttnBlock):
    def __init__(
        self,
        attn,
        attn_size: int,
        hidden_ffn_size: int,
        out_ffn_size: Optional[int] = None,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        ffn_act_func: str = "mish",
        ffn_dropout_p: float = 0,
        ffn_scale: float = 1.0,
        use_macaron: bool = False,
        conv_module_type: Optional[str] = None,
        conv_module_kernel_size: int = 31,
        conformer_conv_dropout_p: float = 0.0,
        norm_groups: int = 0,
        pre_conv_module: bool = True,
        pre_LN: bool = False,
        concat_after: bool = False,
        causal: bool = False,
        padding_mode: str = "zeros",
    ):
        module_kwargs = super().get_kwargs(self.__init__, locals())
        super().__init__(**module_kwargs)

    def layernorm(self, x: Tensor, idx: int = 0):
        return self.layer_norm_list[idx](x)


class AdaptLNResAttnBlock(AbcResAttnBlock):
    def __init__(
        self,
        attn,
        attn_size: int,
        hidden_ffn_size: int,
        out_ffn_size: Optional[int] = None,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        ffn_act_func: str = "mish",
        ffn_dropout_p: float = 0,
        ffn_scale: float = 1.0,
        use_macaron: bool = False,
        conv_module_type: Optional[str] = None,
        conv_module_kernel_size: int = 31,
        conformer_conv_dropout_p: float = 0.0,
        norm_groups: int = 0,
        pre_conv_module: bool = True,
        pre_LN: bool = False,
        concat_after: bool = False,
        causal: bool = False,
        padding_mode: str = "zeros",
    ):
        module_kwargs = super().get_kwargs(self.__init__, locals())
        module_kwargs["LN_learnable"] = True
        super().__init__(**module_kwargs)
        self.ln_params = {}

    def layernorm(self, x: Tensor, idx: int = 0, **kwargs):
        x = self.layer_norm_list[idx](x, weight=kwargs[f"weight_{idx}"], bias=kwargs[f"bias_{idx}"])
        return x

    def forward(
        self,
        x: Tensor,
        enc_kv: Optional[Tensor] = None,
        x_mask: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
        score_mask: Optional[Tensor] = None,
        sample: bool = False,
        num_attn: int = 0,
        pos_info: Optional[Tensor] = None,
        **kwargs,
    ):
        self.ln_params.update(**kwargs)
        return super().forward(
            x,
            enc_kv=enc_kv,
            x_mask=x_mask,
            kv_mask=kv_mask,
            score_mask=score_mask,
            sample=sample,
            num_attn=num_attn,
            pos_info=pos_info,
        )


class AbcTransformerBlocks(nn.Module, metaclass=ABCMeta):
    ATTN_CLS_TYPE_MAP = {
        "base": MultiHeadAttn,
        "wo-pos": MultiHeadAttn,
        "rpr": RelativePositionMultiHeadAttn,
        "rpr-native": NativeRelativePositionMultiHeadAttn,
    }
    """Transformer.
        https://arxiv.org/pdf/2002.04745v1.pdf
        Args:
            in_attn_size (int): number of channels of the input tensor.
            out_chanels (int): number of channels of the output tensor.
            hidden_size (int): model hidden channels.
            hidden_size_ffn (int): hidden channels of FeedForwardNetwork.
            num_heads (int): number of attention heads.
            num_hidden_layers (int): number of transformer layers.
            kernel_size (int, optional): kernel size of feed-forward inner layers. Defaults to 1.
            dropout_p (float, optional): dropout rate for self-attention and feed-forward inner layers_per_stack.
                Defaults to 0.
            rel_attn_window_size (int, optional): relation attention window size.
                If 4, for each time step next and previous 4 time steps are attended.
                If default, relative encoding is disabled and it is a regular transformer.
                Defaults to None.
            input_length (int, optional): input lenght to limit position encoding. Defaults to None.
    """

    def __init__(
        self,
        in_attn_size: int,
        head_attn_size: int,
        out_attn_size: int,
        hidden_ffn_size: int,
        out_ffn_size: int,
        num_heads: int,
        num_hidden_layers: int,
        attn_type: str = "base",
        attn_func_type: Union[int, str] = 0,
        ffn_kernel_size: Union[int, Tuple[int, int]] = 1,
        ffn_act_func: str = "mish",
        attn_dropout_p: float = 0.0,
        cxt_dropout_p: float = 0.0,
        ffn_dropout_p: float = 0.0,
        ffn_rescale: float = 1.0,
        use_macaron: bool = False,
        conv_module_type: Optional[str] = None,
        conv_module_kernel_size: int = 31,
        conformer_conv_dropout_p: float = 0.0,
        norm_groups: int = 0,
        pre_conv_module: bool = True,
        ffn_cat_after: bool = False,
        pre_LN: bool = True,
        final_LN: bool = True,
        causal: bool = False,
        padding_mode: str = "zeros",
        chunkwise_size: Optional[int] = None,
        stream_chunk_size: Optional[int] = None,
        channel_last: bool = True,
        spectral_norm: bool = False,
        window_size: int = 5,
        zero_triu: bool = False,
        max_len: int = 1024,
    ):
        """Transformer.
        https://arxiv.org/pdf/2002.04745v1.pdf

        Args:
            - in_attn_size: int, required
                Number of channels of the input tensor.
            - head_attn_size: int, required
                Number of attention hidden channels.
            - out_attn_size: int, required
                Number of channels of the attention output tensor.
            - hidden_ffn_size: int, required
                Number of hidden channels of FeedForwardNetwork.
            - out_ffn_size: int, required
                Number of channels of the output tensor of FeedForwardNetwork.
            - num_heads: int, required
                Number of attention heads.
            - num_hidden_layers: int, required
                Number of transformer layers.
            - attn_func_type: int, default 0
                Attention function type idx.
            - ffn_kernel_size: Union[int, Tuple[int, int]], default 1
                Kernel size of feed-forward inner layers.
            - ffn_act_func: str, defalut 'mish'
                Activation function of feed-forward layers.
            - attn_dropout_p: float, default 0.
                Dropout rate for self-attention inner layers_per_stack.
            - cxt_dropout_p: float, default 0.
                Dropout rate for attention output tensor.
            - ffn_dropout_p: float, default 0.
                Dropout rate for feed-forward inner layers_per_stack.
            - ffn_rescale: float, default 1.0.
                Feed-forward output tensor scale before adding to the residual.
            - use_macaron: bool, default False
                Whether to use `ConvFeedForwardNetwork` instead of `PositionwiseFeedForwardNetwork`.
            - ffn_cat_after: bool, default False
                Whether to concat feed-forward output tensor with the residual instead of adding.
            - pre_LN: bool, default True
                Whether to do LN before combine feed-forward output tensor and the residual
            - final_LN: bool, defalut True
                Whether to do LN for transformer final output tensor
        """
        super().__init__()
        assert (
            stream_chunk_size is None or stream_chunk_size > 0 and causal
        ), "stream_chunk_size must be positive and causal must be True"
        in_attn_size_tuple = (in_attn_size,) + extend2tuple(out_attn_size, num_hidden_layers - 1)
        head_attn_size_tuple = extend2tuple(head_attn_size, num_hidden_layers)
        out_attn_size_tuple = extend2tuple(out_attn_size, num_hidden_layers)
        hidden_ffn_size_tuple = extend2tuple(hidden_ffn_size, num_hidden_layers)
        out_block_size_tuple = extend2tuple(out_attn_size_tuple[:-1], num_hidden_layers - 1) + (out_ffn_size,)
        self.layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            attn_kwargs = dict(
                in_size=in_attn_size_tuple[i],
                head_size=head_attn_size_tuple[i],
                out_size=out_attn_size_tuple[i],
                num_heads=num_heads,
                attn_dropout_p=attn_dropout_p,
                cxt_dropout_p=cxt_dropout_p,
                attn_func_type=attn_func_type,
                channel_last=channel_last,
                spectral_norm=spectral_norm,
                window_size=window_size,
                zero_triu=zero_triu,
                max_len=max_len,
            )
            attn_layer = self.ATTN_CLS_TYPE_MAP[attn_type](**attn_kwargs)
            cls_res_attn_block = self.choose_res_block()
            res_attn_block = cls_res_attn_block(
                attn_layer,
                attn_size=out_attn_size_tuple[i],
                hidden_ffn_size=hidden_ffn_size_tuple[i],
                out_ffn_size=out_block_size_tuple[i],
                kernel_size=ffn_kernel_size,
                ffn_act_func=ffn_act_func,
                ffn_dropout_p=ffn_dropout_p,
                ffn_scale=ffn_rescale,
                use_macaron=use_macaron,
                conv_module_type=conv_module_type,
                conv_module_kernel_size=conv_module_kernel_size,
                conformer_conv_dropout_p=conformer_conv_dropout_p,
                norm_groups=norm_groups,
                pre_conv_module=pre_conv_module,
                pre_LN=pre_LN,
                concat_after=ffn_cat_after,
                causal=causal,
                padding_mode=padding_mode,
            )
            self.layers.append(res_attn_block)
        if pre_LN and final_LN:
            self.final_ln = nn.LayerNorm(out_attn_size_tuple[-1])
        else:
            self.final_ln = None
        self.num_hidden_layers = num_hidden_layers
        if use_macaron:
            self.num_ln_layers = 3
        else:
            self.num_ln_layers = 2
        self.channel_last = channel_last
        self.chunkwise_size = chunkwise_size
        self.stream_chunk_size = stream_chunk_size

    def get_kwargs(self, init_func, local_vars):
        module_kwargs = inspect.getfullargspec(init_func).args
        discarded_kw = ["self", "version"]
        module_kwargs = dict((name, local_vars[name]) for name in module_kwargs if name not in discarded_kw)
        return module_kwargs

    @abstractmethod
    def choose_res_block(self):
        raise NotImplementedError

    def get_score_mask(self, x, enc_kv, x_mask, kv_mask, outter_score_mask, sample):
        score_mask = None
        with torch.no_grad():
            outter_score_mask = get_outter_score_mask(outter_score_mask, x_mask, kv_mask)
            if x_mask is not None and outter_score_mask is not None:
                outter_score_mask = outter_score_mask.masked_fill(~x_mask[:, None], True)
            scores_mask = self.layers[0].attn.scores_mask
            window_size = self.layers[0].attn.window_size
            sample_t = self.layers[0].attn.sample_t
            seq_l = x.shape[1]
            device = x.device

            score_mask = None
            if self.chunkwise_size is not None:
                if self.channel_last:
                    # (-1, chunk_size, chunk_size)
                    folded_chunk_score_mask, chunks = get_chunks_ceil_score_mask(
                        seq_l, self.chunkwise_size, x.device, return_folded=True
                    )
                    folded_chunk_inner_score_mask = get_inner_score_mask(
                        scores_mask,
                        self.chunkwise_size,
                        self.chunkwise_size,
                        device,
                        sample,
                        sample_t,
                        window_size,
                    )

                    if folded_chunk_inner_score_mask is not None:
                        folded_chunk_score_mask = torch.logical_or(
                            folded_chunk_score_mask, folded_chunk_inner_score_mask
                        )

                    chunk_score_mask = folded_chunk_score_mask.view(
                        chunks, chunks, self.chunkwise_size, self.chunkwise_size
                    )
                    chunk_score_mask = chunk_score_mask.permute(0, 2, 1, 3).reshape(seq_l, seq_l)
                    score_mask = chunk_score_mask[:seq_l, :seq_l].reshape(1, 1, seq_l, seq_l)
                else:
                    raise NotImplementedError("Attention with channels last is not supported.")

            else:
                inner_score_mask = get_inner_score_mask(
                    scores_mask,
                    seq_l,
                    seq_l if enc_kv is None else enc_kv.shape[-1],
                    device,
                    sample,
                    sample_t,
                    window_size,
                )
                if self.stream_chunk_size is not None:
                    if self.channel_last:
                        chunk_score_mask, _ = get_chunks_ceil_score_mask(seq_l, self.stream_chunk_size, device)
                        chunk_score_mask = chunk_score_mask[:seq_l, :seq_l].reshape(1, 1, seq_l, seq_l)

                        if inner_score_mask is not None:
                            score_mask = torch.logical_or(chunk_score_mask, inner_score_mask)
                        else:
                            score_mask = chunk_score_mask
                    else:
                        raise NotImplementedError("Attention with channels last is not supported.")
                else:
                    score_mask = inner_score_mask

            if outter_score_mask is not None and score_mask is not None:
                score_mask = torch.logical_and(score_mask, outter_score_mask)
            else:
                score_mask = outter_score_mask
        return score_mask


class TransformerBlocks(AbcTransformerBlocks):
    """Transformer.
    https://arxiv.org/pdf/2002.04745v1.pdf
    Args:
        in_attn_size (int): number of channels of the input tensor.
        out_chanels (int): number of channels of the output tensor.
        hidden_size (int): model hidden channels.
        hidden_size_ffn (int): hidden channels of FeedForwardNetwork.
        num_heads (int): number of attention heads.
        num_hidden_layers (int): number of transformer layers.
        kernel_size (int, optional): kernel size of feed-forward inner layers. Defaults to 1.
        dropout_p (float, optional): dropout rate for self-attention and feed-forward inner layers_per_stack.
            Defaults to 0.
        rel_attn_window_size (int, optional): relation attention window size.
            If 4, for each time step next and previous 4 time steps are attended.
            If default, relative encoding is disabled and it is a regular transformer.
            Defaults to None.
        input_length (int, optional): input lenght to limit position encoding. Defaults to None.
    """

    def __init__(
        self,
        in_attn_size: int,
        head_attn_size: int,
        out_attn_size: int,
        hidden_ffn_size: int,
        out_ffn_size: int,
        num_heads: int,
        num_hidden_layers: int,
        attn_type: str = "base",
        attn_func_type: Union[int, str] = 0,
        ffn_kernel_size: Union[int, Tuple[int, int]] = 1,
        ffn_act_func: str = "mish",
        attn_dropout_p: float = 0.0,
        cxt_dropout_p: float = 0.0,
        ffn_dropout_p: float = 0.0,
        ffn_rescale: float = 1.0,
        use_macaron: bool = False,
        conv_module_type: Optional[str] = None,
        conv_module_kernel_size: int = 31,
        conformer_conv_dropout_p: float = 0.0,
        norm_groups: int = 0,
        pre_conv_module: bool = True,
        ffn_cat_after: bool = False,
        pre_LN: bool = True,
        final_LN: bool = True,
        causal: bool = False,
        padding_mode: str = "zeros",
        chunkwise_size: Optional[int] = None,
        stream_chunk_size: Optional[int] = None,
        channel_last: bool = True,
        spectral_norm: bool = False,
        window_size: int = 5,
        zero_triu: bool = False,
        max_len: int = 1024,
    ):
        module_kwargs = super().get_kwargs(self.__init__, locals())
        super().__init__(**module_kwargs)

    def choose_res_block(self):
        return ResAttnBlock

    def forward(
        self,
        x: Tensor,
        enc_kv: Optional[Tensor] = None,
        x_mask: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
        outter_score_mask: Optional[Tensor] = None,
        sample: bool = False,
        out_hidden_states: bool = False,
        out_score_mask: bool = False,
        num_attn: int = 0,
        pos_info: Optional[Tensor] = None,
        layer_idx: int = -1,
    ):
        """
        Shapes:
            x: [B, T, C]
            x_mask: [B, T, C] usually B, T, 1
        """
        # B 1 T_q T_kv
        score_mask = self.get_score_mask(x, enc_kv, x_mask, kv_mask, outter_score_mask, sample)

        attn_tuple = []
        hidden_state_tuple = []
        score_mask_tuple = []
        for i, block in enumerate(self.layers):
            x, attn, score_mask_out = block(
                x,
                enc_kv=enc_kv,
                x_mask=x_mask,
                kv_mask=kv_mask,
                score_mask=score_mask,
                outter_score_mask=outter_score_mask,
                sample=sample,
                num_attn=num_attn,
                pos_info=pos_info,
            )
            if out_score_mask:
                score_mask_tuple.append(score_mask_out if score_mask_out is not None else None)
            else:
                score_mask_tuple.append(None)
            attn_tuple.append(attn)
            if out_hidden_states:
                hidden_state_tuple.append(x)
            else:
                hidden_state_tuple.append(None)
            if i == layer_idx:
                hidden_state_tuple[-1] = x
                return hidden_state_tuple, attn_tuple, score_mask_tuple

        if self.final_ln is not None:
            x = self.final_ln(x)
        if x_mask is not None:
            x = x * x_mask
        hidden_state_tuple.append(x)

        return hidden_state_tuple, attn_tuple, score_mask_tuple


class AdaptLNTransformerBlocks(AbcTransformerBlocks):
    """Transformer.
    https://arxiv.org/pdf/2002.04745v1.pdf
    Args:
        in_attn_size (int): number of channels of the input tensor.
        out_chanels (int): number of channels of the output tensor.
        hidden_size (int): model hidden channels.
        hidden_size_ffn (int): hidden channels of FeedForwardNetwork.
        num_heads (int): number of attention heads.
        num_hidden_layers (int): number of transformer layers.
        kernel_size (int, optional): kernel size of feed-forward inner layers. Defaults to 1.
        dropout_p (float, optional): dropout rate for self-attention and feed-forward inner layers_per_stack.
            Defaults to 0.
        rel_attn_window_size (int, optional): relation attention window size.
            If 4, for each time step next and previous 4 time steps are attended.
            If default, relative encoding is disabled and it is a regular transformer.
            Defaults to None.
        input_length (int, optional): input lenght to limit position encoding. Defaults to None.
    """

    def __init__(
        self,
        in_attn_size: int,
        head_attn_size: int,
        out_attn_size: int,
        hidden_ffn_size: int,
        out_ffn_size: int,
        num_heads: int,
        num_hidden_layers: int,
        attn_type: str = "base",
        attn_func_type: Union[int, str] = 0,
        ffn_kernel_size: Union[int, Tuple[int, int]] = 1,
        ffn_act_func: str = "mish",
        attn_dropout_p: float = 0.0,
        cxt_dropout_p: float = 0.0,
        ffn_dropout_p: float = 0.0,
        ffn_rescale: float = 1.0,
        use_macaron: bool = False,
        conv_module_type: Optional[str] = None,
        conv_module_kernel_size: int = 31,
        conformer_conv_dropout_p: float = 0.0,
        norm_groups: int = 0,
        pre_conv_module: bool = True,
        ffn_cat_after: bool = False,
        pre_LN: bool = True,
        final_LN: bool = True,
        channel_last: bool = True,
        causal: bool = False,
        padding_mode: str = "zeros",
        chunkwise_size: Optional[int] = None,
        spectral_norm: bool = False,
        window_size: int = 5,
        zero_triu: bool = False,
        max_len: int = 1024,
    ):
        module_kwargs = super().get_kwargs(self.__init__, locals())
        super().__init__(**module_kwargs)

    def choose_res_block(self):
        return AdaptLNResAttnBlock

    def forward(
        self,
        x: Tensor,
        enc_kv: Optional[Tensor] = None,
        x_mask: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
        outter_score_mask: Optional[Tensor] = None,
        sample: bool = False,
        out_hidden_states: bool = False,
        out_score_mask: bool = False,
        num_attn: int = -1,
        pos_info: Optional[Tensor] = None,
        ln_mean_scale: Optional[Tensor] = None,
    ):
        """
        Shapes:
            x: [B, T, C]
            x_mask: [B, T, C] usually B, T, 1
        """
        # B 1 T_q T_kv
        score_mask = self.get_score_mask(x, enc_kv, x_mask, kv_mask, outter_score_mask, sample)
        attn_tuple = []
        hidden_state_tuple = []
        score_mask_tuple = []

        assert (
            ln_mean_scale is not None
        ), "ForgottenLayerNorm should take external mean and scale vectors whose name is `ln_mean_scale`"
        if not isinstance(ln_mean_scale, (list, tuple)):
            mean_scale_vectors = torch.chunk(ln_mean_scale, self.num_ln_layers * 2 * self.num_hidden_layers, dim=-1)
        for i, block in enumerate(self.layers):
            if self.num_ln_layers == 2:
                control_ln_dict = dict(
                    weight_0=mean_scale_vectors[4 * i],
                    bias_0=mean_scale_vectors[4 * i + 1],
                    weight_1=mean_scale_vectors[4 * i + 2],
                    bias_1=mean_scale_vectors[4 * i + 3],
                )
            else:
                control_ln_dict = dict(
                    weight_0=mean_scale_vectors[6 * i],
                    bias_0=mean_scale_vectors[6 * i + 1],
                    weight_1=mean_scale_vectors[6 * i + 2],
                    bias_1=mean_scale_vectors[6 * i + 3],
                    weight_2=mean_scale_vectors[6 * i + 4],
                    bias_2=mean_scale_vectors[6 * i + 5],
                )
            x, attn, score_mask_out = block(
                x,
                enc_kv=enc_kv,
                x_mask=x_mask,
                kv_mask=kv_mask,
                score_mask=score_mask,
                sample=sample,
                num_attn=num_attn,
                pos_info=pos_info,
                **control_ln_dict,
            )
            attn_tuple.append(attn)
            if out_score_mask:
                score_mask_tuple.append(score_mask_out.cpu() if score_mask_out is not None else None)
            else:
                score_mask_tuple.append(None)

            if out_hidden_states:
                hidden_state_tuple.append(x)
            else:
                hidden_state_tuple.append(None)

        if self.final_ln is not None:
            x = self.final_ln(x)
        if x_mask is not None:
            x = x * x_mask
        hidden_state_tuple.append(x)

        return hidden_state_tuple, attn_tuple, score_mask_tuple


class ResAdaptTransformerBlocks(AbcTransformerBlocks):
    """Transformer.
    https://arxiv.org/pdf/2002.04745v1.pdf
    Args:
        in_attn_size (int): number of channels of the input tensor.
        out_chanels (int): number of channels of the output tensor.
        hidden_size (int): model hidden channels.
        hidden_size_ffn (int): hidden channels of FeedForwardNetwork.
        num_heads (int): number of attention heads.
        num_hidden_layers (int): number of transformer layers.
        kernel_size (int, optional): kernel size of feed-forward inner layers. Defaults to 1.
        dropout_p (float, optional): dropout rate for self-attention and feed-forward inner layers_per_stack.
            Defaults to 0.
        rel_attn_window_size (int, optional): relation attention window size.
            If 4, for each time step next and previous 4 time steps are attended.
            If default, relative encoding is disabled and it is a regular transformer.
            Defaults to None.
        input_length (int, optional): input lenght to limit position encoding. Defaults to None.
    """

    def __init__(
        self,
        in_attn_size: int,
        head_attn_size: int,
        out_attn_size: int,
        hidden_ffn_size: int,
        out_ffn_size: int,
        num_heads: int,
        num_hidden_layers: int,
        attn_type: str = "base",
        attn_func_type: Union[int, str] = 0,
        ffn_kernel_size: Union[int, Tuple[int, int]] = 1,
        ffn_act_func: str = "mish",
        attn_dropout_p: float = 0.0,
        cxt_dropout_p: float = 0.0,
        ffn_dropout_p: float = 0.0,
        ffn_rescale: float = 1.0,
        use_macaron: bool = False,
        conv_module_type: Optional[str] = None,
        conv_module_kernel_size: int = 31,
        conformer_conv_dropout_p: float = 0.0,
        norm_groups: int = 0,
        pre_conv_module: bool = True,
        ffn_cat_after: bool = False,
        pre_LN: bool = True,
        final_LN: bool = True,
        channel_last: bool = True,
        causal: bool = False,
        padding_mode: str = "zeros",
        chunkwise_size: Optional[int] = None,
        spectral_norm: bool = False,
        window_size: int = 5,
        zero_triu: bool = False,
        max_len: int = 1024,
    ):
        module_kwargs = super().get_kwargs(self.__init__, locals())
        super().__init__(**module_kwargs)

    def choose_res_block(self):
        return ResAttnBlock

    def forward(
        self,
        x: Tensor,
        enc_kv: Optional[Tensor] = None,
        x_mask: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
        outter_score_mask: Optional[Tensor] = None,
        sample: bool = False,
        out_hidden_states: bool = False,
        out_score_mask: bool = False,
        num_attn: int = 0,
        pos_info: Optional[Tensor] = None,
        residual_adapter=None,
    ):
        """
        Shapes:
            x: [B, T, C]
            x_mask: [B, T, C] usually B, T, 1
        """
        # B 1 T_q T_kv
        score_mask = self.get_score_mask(x, enc_kv, x_mask, kv_mask, outter_score_mask, sample)

        attn_tuple = []
        hidden_state_tuple = []
        score_mask_tuple = []
        for i, block in enumerate(self.layers):
            x, attn, score_mask_out = block(
                x,
                enc_kv=enc_kv,
                x_mask=x_mask,
                kv_mask=kv_mask,
                score_mask=score_mask,
                sample=sample,
                num_attn=num_attn,
                pos_info=pos_info,
            )
            if residual_adapter is not None:
                x = x + residual_adapter(x, i)
            attn_tuple.append(attn)
            if out_score_mask:
                score_mask_tuple.append(score_mask_out.cpu() if score_mask_out is not None else None)
            else:
                score_mask_tuple.append(None)
            if out_hidden_states:
                hidden_state_tuple.append(x)
            else:
                hidden_state_tuple.append(None)

        if self.final_ln is not None:
            x = self.final_ln(x)
        if x_mask is not None:
            x = x * x_mask
        hidden_state_tuple.append(x)

        return hidden_state_tuple, attn_tuple, score_mask_tuple


class Transformer(nn.Module):
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
        chunkwise_size: Optional[int] = None,
        stream_chunk_size: Optional[int] = None,
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
                chunkwise_len=chunkwise_size,
            )
        elif attn_type == "rope":
            self.pos_enc = RotaryPositionalEncoding(num_attn_head_dim * num_attention_heads, base=10000)
        elif attn_type == "rpr-native":
            self.x_scale = math.sqrt(num_attn_in_dim)
        elif attn_type == "wo-pos":
            self.x_scale = 1.0
        else:
            raise ValueError(f"Transformer: {attn_type} is not supported for the moment")

        self.layers = TransformerBlocks(
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
            chunkwise_size=chunkwise_size,
            stream_chunk_size=stream_chunk_size,
            norm_groups=norm_groups,
            final_LN=final_LN,
            window_size=window_size,
            max_len=max_len,
        )
        self.attn_type = attn_type

    def forward(
        self,
        x: Tensor,
        x_mask: Optional[Tensor] = None,
        extra: Optional[Tensor] = None,
        fertilities: Optional[Tensor] = None,
        indices: Optional[Tensor] = None,
        enc_kv: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
        outter_score_mask: Optional[Tensor] = None,
        sample: bool = False,
        out_hidden_states: bool = False,
        out_score_mask: bool = False,
        num_attn: int = -1,
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
            hidden_state_tuple, attn_tuple, score_mask_tuple = self.layers(
                x,
                x_mask=x_mask,
                enc_kv=enc_kv,
                kv_mask=kv_mask,
                outter_score_mask=outter_score_mask,
                sample=sample,
                out_hidden_states=out_hidden_states,
                out_score_mask=out_score_mask,
                num_attn=num_attn,
                layer_idx=layer_idx,
            )
        else:
            if self.attn_type in ["rpr", "rope"]:
                x, pos_encoding = self.pos_enc(x)
            else:
                x = x * self.x_scale
                pos_encoding = None
            hidden_state_tuple, attn_tuple, score_mask_tuple = self.layers(
                x,
                pos_info=pos_encoding,
                x_mask=x_mask,
                enc_kv=enc_kv,
                kv_mask=kv_mask,
                outter_score_mask=outter_score_mask,
                sample=sample,
                out_hidden_states=out_hidden_states,
                out_score_mask=out_score_mask,
                num_attn=num_attn,
                layer_idx=layer_idx,
            )

        return hidden_state_tuple, attn_tuple, score_mask_tuple


class TransformerCompile(nn.Module):
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
        chunkwise_size: Optional[int] = None,
        stream_chunk_size: Optional[int] = None,
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
                chunkwise_len=chunkwise_size,
            )
        elif attn_type == "rope":
            self.pos_enc = RotaryPositionalEncoding(num_attn_head_dim * num_attention_heads, base=10000)
        elif attn_type == "rpr-native":
            self.x_scale = math.sqrt(num_attn_in_dim)
        elif attn_type == "wo-pos":
            self.x_scale = 1.0
        else:
            raise ValueError(f"Transformer: {attn_type} is not supported for the moment")
        self.attn_type = attn_type

        self.blocks = TransformerBlocks(
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
            chunkwise_size=chunkwise_size,
            stream_chunk_size=stream_chunk_size,
            norm_groups=norm_groups,
            final_LN=final_LN,
            window_size=window_size,
            max_len=max_len,
        )
        self.layers = self.blocks.layers

    def blocks_forward(
        self,
        x: Tensor,
        x_mask: Optional[Tensor] = None,
        extra: Optional[Tensor] = None,
        fertilities: Optional[Tensor] = None,
        indices: Optional[Tensor] = None,
        enc_kv: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
        outter_score_mask: Optional[Tensor] = None,
        sample: bool = False,
        out_hidden_states: bool = False,
        out_score_mask: bool = False,
        num_attn: int = -1,
        layer_idx: int = -1,
        pos_info: Optional[Tensor] = None,
    ):
        score_mask = self.blocks.get_score_mask(x, enc_kv, x_mask, kv_mask, outter_score_mask, sample)

        attn_tuple = []
        hidden_state_tuple = []
        score_mask_tuple = []
        for i, block in enumerate(self.layers):
            x, attn, score_mask_out = block(
                x,
                enc_kv=enc_kv,
                x_mask=x_mask,
                kv_mask=kv_mask,
                score_mask=score_mask,
                sample=sample,
                num_attn=num_attn,
                pos_info=pos_info,
            )
            score_mask_tuple.append(score_mask_out)
            attn_tuple.append(attn)
            if out_hidden_states:
                hidden_state_tuple.append(x)
            else:
                hidden_state_tuple.append(None)
            if i == layer_idx:
                if out_hidden_states:
                    hidden_state_tuple[-1] = x
                else:
                    hidden_state_tuple = x
                return hidden_state_tuple, attn_tuple, score_mask_tuple

        if self.blocks.final_ln is not None:
            x = self.blocks.final_ln(x)
        if x_mask is not None:
            x = x * x_mask
        hidden_state_tuple.append(x)

        return hidden_state_tuple, attn_tuple, score_mask_tuple

    def forward(
        self,
        x: Tensor,
        x_mask: Optional[Tensor] = None,
        extra: Optional[Tensor] = None,
        fertilities: Optional[Tensor] = None,
        indices: Optional[Tensor] = None,
        enc_kv: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
        outter_score_mask: Optional[Tensor] = None,
        out_score_mask: bool = False,
        sample: bool = False,
        out_hidden_states: bool = False,
        num_attn: int = -1,
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
            hidden_state_tuple, attn_tuple, score_mask_tuple = self.blocks_forward(
                x,
                x_mask=x_mask,
                enc_kv=enc_kv,
                kv_mask=kv_mask,
                outter_score_mask=outter_score_mask,
                out_score_mask=out_score_mask,
                sample=sample,
                out_hidden_states=out_hidden_states,
                num_attn=num_attn,
                layer_idx=layer_idx,
            )
        else:
            if self.attn_type in ["rpr", "rope"]:
                x, pos_encoding = self.pos_enc(x)
            else:
                x = x * self.x_scale
                pos_encoding = None
            hidden_state_tuple, attn_tuple, score_mask_tuple = self.blocks_forward(
                x,
                pos_info=pos_encoding,
                x_mask=x_mask,
                enc_kv=enc_kv,
                kv_mask=kv_mask,
                outter_score_mask=outter_score_mask,
                sample=sample,
                out_hidden_states=out_hidden_states,
                out_score_mask=out_score_mask,
                num_attn=num_attn,
                layer_idx=layer_idx,
            )

        return hidden_state_tuple, attn_tuple, score_mask_tuple
