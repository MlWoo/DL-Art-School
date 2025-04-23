import inspect
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .conv import Conv1DBlock
from .feed_forward import ConformerFeedForwardNetwork, ConvFeedForwardNetwork, PositionwiseFeedForwardNetwork
from .linear import Linear
from .mask import get_outter_attn_mask
from .multi_heads_attn import MultiHeadAttn, NativeRelativePositionMultiHeadAttn, RelativePositionMultiHeadAttn
from .normalization import ForgottenLayerNorm
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
        chunk_len: Optional[int] = None,
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
        self.chunk_len = chunk_len
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
        attn_mask: Optional[Tensor] = None,
        sample: bool = False,
        return_attn: bool = False,
        pos_info: Optional[Tensor] = None,
    ):
        # B T C
        attn_mask = get_outter_attn_mask(attn_mask, x_mask, kv_mask)

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

        if self.chunk_len is not None:
            bs, time, channel = x.shape
            if self.channel_last:
                x = x.reshape(-1, self.chunk_len, channel)
            else:
                raise NotImplementedError("Attention with channels last is not supported.")

        context, attn = self.attn(
            x, enc_kv=enc_kv, attn_mask=attn_mask, sample=sample, return_attn=return_attn, pos_info=pos_info
        )

        if self.chunk_len is not None:
            context = context.reshape(bs, time, channel)

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

        return x, attn


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
        chunk_len: Optional[int] = None,
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
        chunk_len: Optional[int] = None,
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
        attn_mask: Optional[Tensor] = None,
        sample: bool = False,
        return_attn: bool = False,
        pos_info: Optional[Tensor] = None,
        **kwargs,
    ):
        self.ln_params.update(**kwargs)
        return super().forward(
            x,
            enc_kv=enc_kv,
            x_mask=x_mask,
            kv_mask=kv_mask,
            attn_mask=attn_mask,
            sample=sample,
            return_attn=return_attn,
            pos_info=pos_info,
        )


class AbcTransformer(nn.Module, metaclass=ABCMeta):
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
        attn_func_type: int = 0,
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
        chunk_len: Optional[int] = None,
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
                chunk_len=chunk_len,
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
        self.chunk_len = chunk_len

    def get_kwargs(self, init_func, local_vars):
        module_kwargs = inspect.getfullargspec(init_func).args
        discarded_kw = ["self", "version"]
        module_kwargs = dict((name, local_vars[name]) for name in module_kwargs if name not in discarded_kw)
        return module_kwargs

    @abstractmethod
    def choose_res_block(self):
        raise NotImplementedError


class Transformer(AbcTransformer):
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
        attn_func_type: int = 0,
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
        chunk_len: Optional[int] = None,
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
        attn_mask: Optional[Tensor] = None,
        sample: bool = False,
        return_all: bool = False,
        return_attn: bool = False,
        return_attn_num: int = -1,
        pos_info: Optional[Tensor] = None,
        layer_idx: int = -1,
    ):
        """
        Shapes:
            x: [B, T, C]
            x_mask: [B, T, C] usually B, T, 1
        """
        # B 1 T_q T_kv
        with torch.no_grad():
            attn_mask = get_outter_attn_mask(attn_mask, x_mask, kv_mask)
            if x_mask is not None:
                attn_mask = attn_mask.masked_fill(~x_mask[:, None], True)

            if self.chunk_len is not None:
                bs, time, channel = x.shape
                if self.channel_last:
                    if attn_mask is not None:
                        chunks = time // self.chunk_len
                        attn_mask_o = attn_mask
                        attn_mask1 = attn_mask_o.reshape(bs, chunks, self.chunk_len, chunks, self.chunk_len)
                        attn_mask2 = attn_mask1.permute(0, 1, 3, 2, 4).reshape(bs, -1, self.chunk_len, self.chunk_len)
                        attn_mask3 = attn_mask2[:, 0 :: (chunks + 1)].reshape(-1, 1, self.chunk_len, self.chunk_len)
                        attn_mask = attn_mask3
                else:
                    raise NotImplementedError("Attention with channels last is not supported.")

        attn_tuple = []
        result_tuple = []
        for i, block in enumerate(self.layers):
            x, attn = block(
                x,
                enc_kv=enc_kv,
                x_mask=x_mask,
                kv_mask=kv_mask,
                attn_mask=attn_mask,
                sample=sample,
                return_attn=return_attn,
                pos_info=pos_info,
            )
            if return_attn:
                if return_attn_num == -1 or return_attn_num >= attn.shape[0]:
                    attn_tuple.append(attn)
                else:
                    attn_tuple.append(attn[:return_attn_num])
            if return_all:
                result_tuple.append(x)
            if i == layer_idx:
                if return_all:
                    result_tuple[-1] = x
                else:
                    result_tuple = x
                return result_tuple, attn_tuple
        if self.final_ln is not None:
            x = self.final_ln(x)
        if x_mask is not None:
            x = x * x_mask
        if return_all:
            result_tuple[-1] = x
        else:
            result_tuple = x
        return result_tuple, attn_tuple


class AdaptLNTransformer(AbcTransformer):
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
        attn_func_type: int = 0,
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
        chunk_len: Optional[int] = None,
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
        attn_mask: Optional[Tensor] = None,
        sample: bool = False,
        return_all: bool = False,
        return_attn: bool = False,
        return_attn_num: int = -1,
        pos_info: Optional[Tensor] = None,
        ln_mean_scale: Optional[Tensor] = None,
    ):
        """
        Shapes:
            x: [B, T, C]
            x_mask: [B, T, C] usually B, T, 1
        """
        # B 1 T_q T_kv
        attn_mask = get_outter_attn_mask(attn_mask, x_mask, kv_mask)
        attn_tuple = []
        result_tuple = []

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
            x, attn = block(
                x,
                enc_kv=enc_kv,
                x_mask=x_mask,
                kv_mask=kv_mask,
                attn_mask=attn_mask,
                sample=sample,
                return_attn=return_attn,
                pos_info=pos_info,
                **control_ln_dict,
            )
            if return_attn:
                if return_attn_num == -1 or return_attn_num >= attn.shape[0]:
                    attn_tuple.append(attn)
                else:
                    attn_tuple.append(attn[:return_attn_num])
            if return_all:
                result_tuple.append(x)
        if self.final_ln is not None:
            x = self.final_ln(x)
        if x_mask is not None:
            x = x * x_mask
        if return_all:
            result_tuple[-1] = x
        else:
            result_tuple = x
        return result_tuple, attn_tuple


class ResAdaptTransformer(AbcTransformer):
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
        attn_func_type: int = 0,
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
        chunk_len: Optional[int] = None,
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
        attn_mask: Optional[Tensor] = None,
        sample: bool = False,
        return_all: bool = False,
        return_attn: bool = False,
        return_attn_num: int = -1,
        pos_info: Optional[Tensor] = None,
        residual_adapter=None,
    ):
        """
        Shapes:
            x: [B, T, C]
            x_mask: [B, T, C] usually B, T, 1
        """
        # B 1 T_q T_kv
        attn_mask = get_outter_attn_mask(attn_mask, x_mask, kv_mask)
        attn_tuple = []
        result_tuple = []
        for i, block in enumerate(self.layers):
            x, attn = block(
                x,
                enc_kv=enc_kv,
                x_mask=x_mask,
                kv_mask=kv_mask,
                attn_mask=attn_mask,
                sample=sample,
                return_attn=return_attn,
                pos_info=pos_info,
            )
            if residual_adapter is not None:
                x = x + residual_adapter(x, i)
            if return_attn:
                if return_attn_num == -1 or return_attn_num >= attn.shape[0]:
                    attn_tuple.append(attn)
                else:
                    attn_tuple.append(attn[:return_attn_num])
            if return_all:
                result_tuple.append(x)
        if self.final_ln is not None:
            x = self.final_ln(x)
        if x_mask is not None:
            x = x * x_mask
        if return_all:
            result_tuple[-1] = x
        else:
            result_tuple = x
        return result_tuple, attn_tuple
