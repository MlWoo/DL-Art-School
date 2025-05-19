import math
from abc import ABCMeta
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from utils.logging_utils import get_root_logger

from .dropout import Dropout
from .linear import Linear
from .mask import get_inner_score_mask
from .positional_embedding import RelativePositionalEmbedding
from .positional_encoding import RelativePositionalEncoding

logger = get_root_logger()

EPSILON = 1e-6


class ABCScaledDotProductAttention(nn.Module, metaclass=ABCMeta):
    """Scaled Dot-Product Attention"""

    def __init__(
        self, sqrt_scale_inv: float = 1.0, attn_dropout_p: float = 0.0, stripe_bandwdith: Optional[int] = None
    ):
        super().__init__()
        self.sqrt_scale_inv = sqrt_scale_inv
        self.attn_dropout = Dropout(attn_dropout_p, inplace=False) if attn_dropout_p > 0.0 else lambda x: x
        self.half_bandwdith = 0 if stripe_bandwdith is None else int(stripe_bandwdith) // 2
        self.score_mask_value = float("-inf")

    def _scaled_dot_scores(self, query: Tensor, key: Tensor):
        if self.training:
            scores = torch.matmul(query * self.sqrt_scale_inv, key * self.sqrt_scale_inv)
        else:
            scores = torch.matmul(query, key)
            scores.mul_(self.sqrt_scale_inv * self.sqrt_scale_inv)
        return scores

    def _get_attn(
        self,
        score: Tensor,
        padding_mask: Optional[Tensor] = None,
        stripe_mask: Optional[Tensor] = None,
        return_score: bool = False,
    ) -> Tensor:
        score_type = score.dtype
        score = score.float()
        if padding_mask is not None:
            if score.dim() > padding_mask.dim():
                padding_mask = padding_mask.unsqueeze(dim=1).float()
            mask = padding_mask.expand_as(score)
        else:
            mask = None
        if stripe_mask is not None:
            if mask is not None:
                mask = torch.logical_and(mask, stripe_mask)
            else:
                mask = stripe_mask
        if mask is not None:
            score = score.masked_fill(mask == 0, self.score_mask_value)
        weight = F.softmax(score, dim=-1).type(score_type)
        # apply dropout to attention weights
        weight = self.attn_dropout(weight)
        if return_score:
            return weight, score
        else:
            return weight

    def _get_stripe_mask(self, key_lengths: Tensor, query_lengths: Tensor) -> Tensor:
        with torch.no_grad():
            batch_size = len(key_lengths)
            max_query_length = query_lengths.max()
            max_key_length = key_lengths.max()
            grid_x, grid_y = torch.meshgrid(torch.arange(max_query_length), torch.arange(max_key_length))
            grid_x = grid_x.to(key_lengths.device)
            grid_y = grid_y.to(key_lengths.device)
            slope = key_lengths / query_lengths
            slope = slope.view(-1, 1, 1)
            grid_x = grid_x.unsqueeze(dim=0).expand(batch_size, -1, -1)  # B, T
            grid_y = grid_y.unsqueeze(dim=0).expand(batch_size, -1, -1)  # B, T
            grid_y_upper = grid_x * slope + self.half_bandwdith  # B Tt
            grid_y_lower = grid_x * slope - self.half_bandwdith  # B Tt
            triu = grid_y >= grid_y_lower
            tril = grid_y <= grid_y_upper
            mask = torch.logical_and(triu, tril)
            return mask

    def get_inner_query(self, query):
        return query

    def get_inner_key(self, key):
        return key

    def get_inner_value(self, val):
        return val

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        query_lengths: Optional[Tensor] = None,
        key_lengths: Optional[Tensor] = None,
        return_score: bool = False,
    ):
        query = self.get_inner_query(query)
        key = self.get_inner_key(key)
        scores = self._scaled_dot_scores(query, key.transpose(-2, -1))
        if query_lengths is not None and key_lengths is not None and self.half_bandwdith > 0:
            stripe_mask = self._get_stripe_mask(query_lengths=query_lengths, key_lengths=key_lengths)
        else:
            stripe_mask = None
        attn_weight = self._get_attn(scores, padding_mask=mask, stripe_mask=stripe_mask, return_score=return_score)
        value = self.get_inner_value(value)
        return attn_weight, value


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(
        self,
        num_query_dim: int,
        num_key_dim: int,
        inner_dim: int,
        scale: Optional[float] = None,
        attn_dropout_p: float = 0.0,
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.key_linear = Linear(num_key_dim, inner_dim, bias=False, spectral_norm=spectral_norm)
        self.query_linear = Linear(num_query_dim, inner_dim, bias=False, spectral_norm=spectral_norm)
        self.sqrt_scale_inv = math.sqrt(1.0 / math.sqrt(num_key_dim)) if scale is None else scale
        self.attn_dropout = Dropout(attn_dropout_p, inplace=False) if attn_dropout_p > 0.0 else lambda x: x
        self.score_mask_value = float("-inf")

    def _scaled_dot_scores(self, query: Tensor, key: Tensor):
        if self.training:
            scores = torch.matmul(query * self.sqrt_scale_inv, key * self.sqrt_scale_inv)
        else:
            scores = torch.matmul(query, key)
            scores.mul_(self.sqrt_scale_inv * self.sqrt_scale_inv)
        return scores

    def _get_attn(self, scores: Tensor, mask: Optional[Tensor] = None):
        score_type = scores.dtype
        scores = scores.float()
        if mask is not None:
            mask = mask.expand_as(scores)
            scores = scores.masked_fill(mask == 0, self.score_mask_value)
        weight = F.softmax(scores, dim=-1).type(score_type)
        # apply dropout to attention weights
        attn = self.attn_dropout(weight)
        return attn

    def forward(self, query: Tensor, key: Tensor, value: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        query = self.query_linear(query)
        key = self.key_linear(key)
        scores = self._scaled_dot_scores(query, key.transpose(-2, -1))
        attn_weight = self._get_attn(scores, mask)
        if value is None:
            attn_cxt = None
        else:
            attn_cxt = torch.bmm(attn_weight, value)  # [N, seq_len, prosody_embedding_dim]

        return attn_cxt, attn_weight


class MultiHeadAttn(nn.Module):
    """Multi-Head Attention module"""

    def __init__(
        self,
        in_size: int,
        head_size: int,
        out_size: Optional[int] = None,
        kv_size: Optional[int] = None,
        num_heads: int = 4,
        scale: Optional[float] = None,
        attn_dropout_p: float = 0.1,
        cxt_dropout_p: float = 0.1,
        attn_func_type: Union[int, str] = 0,
        channel_last: bool = True,
        spectral_norm: bool = False,
        window_size: int = 5,
        lookforward_size: Optional[int] = None,
        *args,
        **kawrgs,
    ):
        """
        attn_func_type
        0 : self attention, production attention, no mask
        1 : self attention, production attention, triangle mask
        2 : self attention, production attention, window mask
        3 : self attention, production attention, window mask
        4 : self attention with rotary embedding, production attention, no mask
        5 : self attention with rotary embedding, production attention, triangle mask
        6 : self attention with rotary embedding, production attention, window mask
        7 : self attention with rotary embedding, production attention, window mask
        8 : cross attention, production attention, no mask
        9 : cross attention, production attention, triangle mask
        10 : cross attention, production attention, window mask
        11 : cross attention, production attention, window mask
        """
        attn_func_map_by_id = {
            0: (self._individual_qkv, self._scaled_dot_scores, None, "self_attn_base"),
            1: (self._individual_qkv, self._scaled_dot_scores, "autoregressive", "self_attn_ar"),
            2: (self._individual_qkv, self._scaled_dot_scores, "window", "self_attn_window"),
            3: (self._individual_qkv, self._scaled_dot_scores, "window_ar", "self_attn_window_ar"),
            4: (self._individual_rope_qkv, self._scaled_dot_scores, None, "self_attn_rope_base"),
            5: (self._individual_rope_qkv, self._scaled_dot_scores, "autoregressive", "self_attn_rope_ar"),
            6: (self._individual_rope_qkv, self._scaled_dot_scores, "window", "self_attn_rope_window"),
            7: (self._individual_rope_qkv, self._scaled_dot_scores, "window_ar", "self_attn_rope_window_ar"),
            8: (self._mutual_qkv, self._scaled_dot_scores, None, "cross_attn_base"),
            9: (self._mutual_qkv, self._scaled_dot_scores, "autoregressive", "cross_attn_ar"),
            10: (self._mutual_qkv, self._scaled_dot_scores, "window", "cross_attn_window"),
            11: (self._mutual_qkv, self._scaled_dot_scores, "window_ar", "cross_attn_window_ar"),
        }
        attn_func_map_by_name = {}
        for k, v in attn_func_map_by_id.items():
            qkv_func, scores_func, scores_mask_type, attn_type = v
            attn_func_map_by_name[attn_type] = (qkv_func, scores_func, scores_mask_type, k)
        if isinstance(attn_func_type, str):
            attn_func_type = attn_func_map_by_name[attn_func_type][3]
        super().__init__()
        assert in_size % num_heads == 0, " [!] channels should be divisible by num_heads."
        # class attributes
        self.in_size = in_size
        if out_size is None:
            self.out_size = in_size
        else:
            self.out_size = out_size
        self.head_size = head_size
        self.num_heads = num_heads
        n_state = num_heads * head_size
        self.n_state = n_state
        if attn_func_type > 7:
            if kv_size is None:
                kv_size = in_size
            self.inner_proj_q = Linear(in_size, n_state, channel_last=channel_last, spectral_norm=spectral_norm)
            self.inner_proj_kv = Linear(kv_size, n_state * 2, channel_last=channel_last, spectral_norm=spectral_norm)
        elif attn_func_type > 3:
            self.inner_proj_qk = Linear(in_size, n_state * 2, channel_last=channel_last, spectral_norm=spectral_norm)
            self.inner_proj_v = Linear(in_size, n_state, channel_last=channel_last, spectral_norm=spectral_norm)
        else:
            self.inner_proj_qkv = Linear(in_size, n_state * 3, channel_last=channel_last, spectral_norm=spectral_norm)
        self.attn_func_type = attn_func_type
        self.proj = Linear(n_state, self.out_size, channel_last=channel_last, spectral_norm=spectral_norm)

        if scale is None or scale < EPSILON:
            self.scale = math.sqrt(head_size)
            self.sqrt_scale_inv = math.sqrt(1.0 / self.scale)
        else:
            self.scale = scale
            self.sqrt_scale_inv = math.sqrt(1.0 / scale)

        self.attn_dropout = Dropout(attn_dropout_p, inplace=False) if attn_dropout_p > 0.0 else lambda x: x
        self.cxt_dropout = Dropout(cxt_dropout_p, inplace=True) if cxt_dropout_p > 0.0 else lambda x: x
        self.attn_dropout_p = attn_dropout_p
        self.qkv, self.scores, self.scores_mask, self.attn_type = attn_func_map_by_id[attn_func_type]
        self.mask_score = self.scores_mask is not None
        self.window_size = window_size

        self.sample_t = 1
        self.lookforward_size = lookforward_size
        self.channel_last = channel_last
        self.reset_parameters()
        self.score_mask_value = float("-inf")
        self.sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.n_state)
        if self.attn_func_type > 7:
            nn.init.kaiming_uniform_(self.inner_proj_kv.op.weight[: self.n_state, :], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.inner_proj_kv.op.weight[self.n_state :, :], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.inner_proj_q.op.weight, a=math.sqrt(5))
            nn.init.uniform_(self.inner_proj_kv.op.bias, -bound, bound)
            nn.init.uniform_(self.inner_proj_q.op.bias, -bound, bound)
        elif self.attn_func_type > 3:
            nn.init.kaiming_uniform_(self.inner_proj_qk.op.weight[: self.n_state, :], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.inner_proj_qk.op.weight[self.n_state :, :], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.inner_proj_v.op.weight, a=math.sqrt(5))
            nn.init.uniform_(self.inner_proj_qk.op.bias, -bound, bound)
            nn.init.uniform_(self.inner_proj_v.op.bias, -bound, bound)
        else:
            nn.init.kaiming_uniform_(self.inner_proj_qkv.op.weight[: self.n_state, :], a=math.sqrt(5))
            nn.init.kaiming_uniform_(
                self.inner_proj_qkv.op.weight[self.n_state : (self.n_state * 2), :], a=math.sqrt(5)
            )
            nn.init.kaiming_uniform_(self.inner_proj_qkv.op.weight[(self.n_state * 2) :, :], a=math.sqrt(5))
            nn.init.uniform_(self.inner_proj_qkv.op.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.proj.op.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(self.out_size)
        nn.init.uniform_(self.proj.op.bias, -bound, bound)

    def _merge_heads(self, x: Tensor) -> Tensor:
        if self.channel_last:
            # x : B C T H
            # return : B T C
            x = x.permute(0, 2, 1, 3).contiguous()
            new_x_shape = (x.size(0), -1, x.size(-2) * x.size(-1))
            return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states
        else:
            # x : B H T C
            # return : B C T
            x = x.permute(0, 1, 3, 2).contiguous()
            new_x_shape = (x.size(0), -1, x.size(-1))
            return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def _split_heads(self, x: Tensor, k: bool = False) -> Tensor:
        if self.channel_last:
            # x: B T C
            # return B H T C
            # B T H C
            new_x_shape = (x.size(0), -1, self.num_heads, x.size(-1) // self.num_heads)
            x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
            if k:
                # B H C T
                return x.permute(0, 2, 3, 1)
            else:
                # B H T C
                return x.permute(0, 2, 1, 3)
        else:
            # x: B C T
            # B H C T
            new_x_shape = (x.size(0), self.num_heads, x.size(1) // self.num_heads, x.size(-1))
            x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
            if k:
                # B H C T
                return x
            else:
                # B H T C
                return x.permute(0, 1, 3, 2)

    def _scaled_dot_scores(self, query: Tensor, key: Tensor) -> Tensor:
        if False and self.training:
            scores = torch.matmul(query * self.sqrt_scale_inv, key * self.sqrt_scale_inv)
        else:
            scores = torch.matmul(query, key)
            scores.mul_(self.sqrt_scale_inv * self.sqrt_scale_inv)
        return scores

    def _individual_qkv(self, x: Tensor, enc_kv: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        assert enc_kv is None
        qkv = self.inner_proj_qkv(x)
        if self.channel_last:
            query, key, value = torch.split(qkv, self.n_state, dim=-1)
        else:
            query, key, value = torch.split(qkv, self.n_state, dim=1)
        return query, key, value

    def _individual_rope_qkv(self, x: Tensor, enc_kv: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # enc_kv is the query applied with rotary embedding
        qk = self.inner_proj_qk(enc_kv)
        if self.channel_last:
            query, key = torch.split(qk, self.n_state, dim=-1)
        else:
            query, key = torch.split(qk, self.n_state, dim=1)
        value = self.inner_proj_v(x)
        return query, key, value

    def _mutual_qkv(self, x: Tensor, enc_kv: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        assert enc_kv is not None
        query = self.inner_proj_q(x)
        kv = self.inner_proj_kv(enc_kv)
        if self.channel_last:
            key, value = torch.split(kv, self.n_state, dim=-1)
        else:
            key, value = torch.split(kv, self.n_state, dim=1)
        return query, key, value

    def get_scores(self, x, enc_kv=None, sample=False, pos_info=None):
        if pos_info is not None:
            qk = self._apply_rotary_embedding(x, pos_info)
            query, key, value = self.qkv(x, enc_kv=qk)
        else:
            query, key, value = self.qkv(x, enc_kv=enc_kv)
        query = self._split_heads(query)
        key = self._split_heads(key, k=True)
        value = self._split_heads(value)
        scores = self.scores(query, key)
        return scores, query, key, value, sample

    def get_qkv(self, x, enc_kv=None, sample=False, pos_info=None):
        if pos_info is not None:
            qk = self._apply_rotary_embedding(x, pos_info)
            query, key, value = self.qkv(x, enc_kv=qk)
        else:
            query, key, value = self.qkv(x, enc_kv=enc_kv)
        query = self._split_heads(query)
        key = self._split_heads(key, k=False)
        value = self._split_heads(value)

        return None, query, key, value, sample

    def get_attn(self, scores, mask=None):
        if mask is not None:
            # mask = mask.expand_as(scores)
            scores = scores.masked_fill(~mask, self.score_mask_value)
        # score_type = scores.dtype
        # scores = scores.float()
        weight = F.softmax(scores, dim=-1)  # .type(score_type)
        # apply dropout to attention weights
        attn = self.attn_dropout(weight)
        return attn

    def get_context(self, attn, value):
        context = torch.matmul(attn, value)
        context = self._merge_heads(context)
        return context

    def get_mask(
        self,
        device,
        q_l,
        kv_l=None,
        score_mask=None,
        outter_score_mask=None,
        sample=False,
    ):
        if score_mask is None:
            with torch.no_grad():
                inner_score_mask = get_inner_score_mask(
                    self.scores_mask,
                    q_l,
                    kv_l,
                    device,
                    sample,
                    self.sample_t,
                    self.window_size,
                    self.lookforward_size,
                )
                if outter_score_mask is not None and inner_score_mask is not None:
                    score_mask = torch.logical_and(inner_score_mask, outter_score_mask)
                elif inner_score_mask is not None:
                    score_mask = inner_score_mask
                elif outter_score_mask is not None:
                    score_mask = outter_score_mask
                else:
                    score_mask = None

        return score_mask

    def get_attn_context(self, score, value, score_mask=None):
        attn = self.get_attn(score, mask=score_mask)
        context = self.get_context(attn, value)
        return context, attn

    def post_attn_context(self, context, attn, num_attn: int = 0):
        context = self.proj(context)
        if attn is None or num_attn == 0 or num_attn < -1:
            return self.cxt_dropout(context), None
        elif num_attn == -1 or num_attn >= attn.shape[0]:
            logger.info(
                f"It's very slow to use num_attn: {num_attn} to record all attention weights when formal training."
            )
            return self.cxt_dropout(context), attn.detach().cpu()
        else:
            return self.cxt_dropout(context), attn.detach()[:num_attn].cpu()

    def _apply_rotary_embedding(self, hidden_states, relative_position_embeddings):
        batch_size, sequence_length, hidden_size = hidden_states.size()
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_size)
        cos = relative_position_embeddings[0, :sequence_length, ...]
        sin = relative_position_embeddings[1, :sequence_length, ...]

        # rotate hidden_states with rotary embeddings
        hidden_states = hidden_states.transpose(0, 1)
        rotated_states_begin = hidden_states[..., : self.head_size // 2]
        rotated_states_end = hidden_states[..., self.head_size // 2 :]
        rotated_states = torch.cat((-rotated_states_end, rotated_states_begin), dim=rotated_states_begin.ndim - 1)
        hidden_states = (hidden_states * cos) + (rotated_states * sin)
        hidden_states = hidden_states.transpose(0, 1)

        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads * self.head_size)

        return hidden_states

    def forward(
        self, x, enc_kv=None, score_mask=None, outter_score_mask=None, sample=False, num_attn: int = 0, pos_info=None
    ):
        if self.sdpa:
            _, query, key, value, sample = self.get_qkv(x, enc_kv=enc_kv, sample=sample, pos_info=pos_info)
            score_mask = self.get_mask(
                x.device,
                query.size(-2),
                key.size(-2),
                score_mask=score_mask,
                outter_score_mask=outter_score_mask,
                sample=sample,
            )
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=score_mask,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=False,
            )
            context = self._merge_heads(context)
            attn = None
        else:
            score, query, key, value, sample = self.get_scores(x, enc_kv=enc_kv)
            score_mask = self.get_mask(
                x.device,
                query.size(-2),
                key.size(-1),
                score_mask=score_mask,
                outter_score_mask=outter_score_mask,
                sample=sample,
            )
            context, attn = self.get_attn_context(score, value, score_mask=score_mask)

        context, attn = self.post_attn_context(context, attn, num_attn=num_attn)
        return context, attn, score_mask


class NativeRelativePositionMultiHeadAttn(MultiHeadAttn):
    """Multi-Head Attention layer with relative position encoding (new implementation).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
    """

    def __init__(
        self,
        in_size: int,
        head_size: int,
        out_size: Optional[int] = None,
        kv_size: Optional[int] = None,
        num_heads: int = 4,
        scale: Optional[float] = None,
        attn_dropout_p: float = 0.1,
        cxt_dropout_p: float = 0.1,
        attn_func_type: Union[int, str] = 0,
        channel_last: bool = True,
        spectral_norm: bool = False,
        window_size: int = 5,
        *args,
        **kawrgs,
    ):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(
            in_size,
            head_size,
            out_size,
            kv_size,
            num_heads,
            scale,
            attn_dropout_p,
            cxt_dropout_p,
            attn_func_type=attn_func_type,
            spectral_norm=spectral_norm,
            window_size=window_size,
            channel_last=channel_last,
        )
        if self.window_size is not None:
            self.relative_position_key = RelativePositionalEmbedding(self.head_size, window_size)
            self.relative_position_val = RelativePositionalEmbedding(self.head_size, window_size)
        else:
            raise ValueError(" Relative window size must be specified for NativeRelativePositionMultiHeadAttn.")

    def get_relative_position_scores(self, x, enc_kv=None, sample=False):
        query, key, value = self.qkv(x, enc_kv=enc_kv)
        len_q = query.size(1)
        len_k = key.size(1)
        # B H T1 C
        query = self._split_heads(query)
        # B H C T2
        key = self._split_heads(key, k=True)
        # B H T2 C
        value = self._split_heads(value)
        # B H T1 T2
        score_qk = self.scores(query, key)

        # B T C
        r_p_key_embedding = self.relative_position_key(len_q, len_k)  # T1 x T2 x C
        r_p_key_embedding = r_p_key_embedding.permute(0, 2, 1)  # T1 x C x T2
        # B H T1 1 T2 -> B H T1 T2
        score_qp = self.scores(query.unsqueeze(3), r_p_key_embedding).squeeze(3)
        scores = score_qk + score_qp
        return scores, query, key, value, sample

    def relative_position_context_val_bias(self, context, attn, len_q, len_v):
        # attn B x H x T1 x T2
        r_p_key_embedding = self.relative_position_val(len_q, len_v)  # T1 x T2 x C
        # B H T1 1 T2 * T1 T2 C -> B H T1 1 C -> B H T1 C -> B T1 H C
        context_val = torch.matmul(attn.unsqueeze(3), r_p_key_embedding).squeeze(3)
        # B H T1 C -> B T1 H C -> B T1 CC
        context_val = context_val.permute(0, 2, 1, 3).contiguous().reshape(-1, len_q, self.n_state)
        context = context + context_val
        return context

    def forward(
        self, x, enc_kv=None, score_mask=None, outter_score_mask=None, sample=False, num_attn: int = 0, pos_info=None
    ):
        if self.window_size is None:
            context, attn = super().forward(
                x,
                enc_kv=enc_kv,
                score_mask=score_mask,
                outter_score_mask=outter_score_mask,
                sample=sample,
                num_attn=num_attn,
                pos_info=pos_info,
            )
        else:
            scores, query, key, value, sample = self.get_relative_position_scores(x, enc_kv=enc_kv)
            score_mask = self.get_mask(
                x.device,
                query.size(-2),
                key.size(-1),
                score_mask=score_mask,
                outter_score_mask=outter_score_mask,
                sample=sample,
            )
            context, attn = self.get_attn_context(scores, value, score_mask=score_mask)
            context = self.relative_position_context_val_bias(context, attn, query.size(-2), value.size(-2))
            context, attn = self.post_attn_context(context, attn, num_attn=num_attn)

        return context, attn, score_mask


class RelativePositionMultiHeadAttn(MultiHeadAttn):
    """Multi-Head Attention layer with relative position encoding (new implementation).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
    """

    def __init__(
        self,
        in_size: int,
        head_size: int,
        out_size: Optional[int] = None,
        kv_size: Optional[int] = None,
        num_heads: int = 4,
        scale: Optional[float] = None,
        attn_dropout_p: float = 0.1,
        cxt_dropout_p: float = 0.1,
        attn_func_type: Union[int, str] = 0,
        channel_last: bool = True,
        spectral_norm: bool = False,
        window_size: int = 5,
        lookforward_size: Optional[int] = None,
        zero_triu: bool = False,
        max_len: int = 5000,
    ):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(
            in_size,
            head_size,
            out_size,
            kv_size,
            num_heads,
            scale,
            attn_dropout_p,
            cxt_dropout_p,
            attn_func_type=attn_func_type,
            spectral_norm=spectral_norm,
            window_size=window_size,
            lookforward_size=lookforward_size,
            channel_last=channel_last,
        )
        self.zero_triu = zero_triu
        self.hidden_size = head_size * num_heads

        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(head_size * num_heads, head_size * num_heads, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # https://arxiv.org/pdf/1803.02155.pdf
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.num_heads, self.head_size))  # H C
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.num_heads, self.head_size))  # H C
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)
        self.pe = None
        self.max_len = max_len

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        # only keep the positions from 0 to time2
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def get_scores(self, x, pos_encoding, enc_kv=None, sample=False):
        query, key, value = self.qkv(x, enc_kv=enc_kv)
        # B H T C
        query = self._split_heads(query)
        # B H C T
        key = self._split_heads(key, k=True)
        value = self._split_heads(value)
        # (batch, head, time1, d_k)
        query_with_bias_key = query + self.pos_bias_u.unsqueeze(dim=1)
        # (batch, head, time1, d_k)
        query_with_bias_pos = query + self.pos_bias_v.unsqueeze(dim=1)

        pos = self.linear_pos(pos_encoding).view(-1, self.num_heads, self.head_size)  # T H C
        pos = pos.permute(1, 2, 0)  # (head, d_k, 2*time1-1) H C T

        score_ac = self.scores(query_with_bias_key, key)
        score_bd = self.scores(query_with_bias_pos, pos)  # B H time1 * 2*time1-1
        score_bd = self.rel_shift(score_bd)
        scores = score_ac + score_bd  # (batch, head, time1, time2)
        return scores, query, key, value, sample

    def forward(
        self, x, enc_kv=None, score_mask=None, outter_score_mask=None, sample=False, num_attn: int = 0, pos_info=None
    ):
        if pos_info is None:
            logger.warn("pos encoding could be shared across layers. It will save memory if passed from params.")
            if self.pe is None:
                self.pe = RelativePositionalEncoding(
                    self.hidden_size * self.num_heads, 0, scale=1.0, max_len=self.max_len, win_len=self.window_size
                )
            x, pos_encoding = self.pe(x)
        else:
            pos_encoding = pos_info

        scores, query, key, value, sample = self.get_scores(x, pos_encoding, enc_kv=enc_kv)
        score_mask = self.get_mask(
            x.device,
            query.size(-2),
            key.size(-1),
            score_mask=score_mask,
            outter_score_mask=outter_score_mask,
            sample=sample,
        )
        context, attn = self.get_attn_context(scores, value, score_mask=score_mask)
        context, attn = self.post_attn_context(context, attn, num_attn=num_attn)

        return context, attn, score_mask
