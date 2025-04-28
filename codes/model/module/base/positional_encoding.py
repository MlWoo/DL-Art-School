#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positional Encoding Module."""
import math
from typing import Optional

import torch
from torch import Tensor, nn

from .dropout import Dropout
from .mask import pad_list, sequence_mask


class PositionalEncoding(nn.Module):
    """Positional encoding.
    Args:
        hidden_channels (int): encoding dimension.
        dropout_p (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position.
    """

    def __init__(
        self,
        hidden_channels,
        dropout_p,
        max_len=5000,
        scale=None,
        scaled=False,
        reverse=False,
        mode: Optional[str] = "add",
    ):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.hidden_channels = hidden_channels
        self.reverse = reverse
        if scaled:
            self.x_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.x_scale = math.sqrt(self.hidden_channels) if scale is None else scale
        self.dropout = Dropout(p=dropout_p, inplace=True)  # if dropout_p > 0.0 else lambda x: x
        self.pe = None
        if mode is None:
            self.mode = None
        else:
            self.mode = mode.lower()
        self.extend_pe(max_len, torch.float32, -1)

    def reset_parameters(self):
        """Reset parameters."""
        assert isinstance(self.x_scale, Tensor), "Only scale is a parameter could be reset."
        self.x_scale.data = torch.tensor(1.0)

    def fertilize_pe(self, pe, fertilities):
        fertilities = fertilities.squeeze(dim=-1)
        batch_size = fertilities.shape[0]
        duration_cumsum = fertilities.cumsum(dim=1)
        max_len = duration_cumsum.max()
        pos_indices = torch.arange(0, max_len)[None, :].expand(batch_size, -1)
        if fertilities.is_cuda:
            pos_indices = pos_indices.to(fertilities.get_device())
        start_cumsum = torch.cat([pos_indices[:, :1], duration_cumsum[:, :-1]], dim=-1)
        x_temp = [torch.repeat_interleave(x, d, dim=0) for x, d in zip(start_cumsum, fertilities)]
        start_cumsum_x = pad_list(x_temp, 0)
        relative_pos = pos_indices - start_cumsum_x + 1

        mask = sequence_mask(duration_cumsum[:, -1], valid=False)
        relative_pos = relative_pos.masked_fill_(mask, 0)[:, :, None].expand(-1, -1, self.hidden_channels)
        relative_pe = torch.gather(pe.expand(batch_size, -1, -1), 1, relative_pos)

        return relative_pe

    '''
    def fertilize_pe(self, pe, durations):
        """
            :param durations: [B, N]
            :return positional_encoding_outputs: [B, T, positional_hidden]
        """

        B = durations.size(0)
        N = durations.size(1)

        pos_durations = durations.long()
        sum_len = pos_durations.sum(dim=1)
        max_len = sum_len.max()
        diff_len = max_len - sum_len
        pos_durations[:, -1] = pos_durations[:, -1] + diff_len

        ids = torch.arange(max_len, device=durations.device).expand(B, N, max_len)
        pos_mask = (ids < pos_durations.view(B, N, 1))
        pos_ids = ids[pos_mask].view(-1, max_len)  # [B, T]
        positional_encoding_outputs = pe(pos_ids)

        return positional_encoding_outputs
    '''

    def extend_pe(self, max_len, dtype, device):
        """Reset the positional encodings."""
        if self.pe is None or self.pe.size(1) < max_len:
            pe = torch.zeros(max_len, self.hidden_channels)
            if self.reverse:
                position = torch.arange(max_len - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
            else:
                position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.hidden_channels, 2, dtype=torch.float32)
                * -(math.log(10000.0) / self.hidden_channels)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.to(dtype=dtype)
            if device >= 0:
                self.pe = pe.to(device)
            else:
                self.pe = pe
        else:
            if self.pe.dtype != dtype or self.pe.get_device() != device:
                if device >= 0:
                    self.pe = self.pe.to(dtype=dtype, device=torch.device("cuda:" + str(device)))
                else:
                    self.pe = self.pe.to(dtype=dtype, device=torch.device("cpu"))

    def forward(
        self, x: Tensor, fertilities: Optional[torch.LongTensor] = None, indices: Optional[torch.LongTensor] = None
    ):
        """Add positional encoding.
        Args:
            x (Tensor): Input tensor (batch, time, `*`).
        Returns:
            Tensor: Encoded tensor (batch, time, `*`).
        """
        assert (
            fertilities is None or indices is None
        ), "The encoding is confused if both fertilities and indices are provided."

        dtype = x.dtype
        device = x.get_device()
        if fertilities is not None:
            max_len = torch.max(fertilities.sum(dim=-1)).item() + 1
            self.extend_pe(max_len, dtype, device)
            pe = self.fertilize_pe(self.pe, fertilities)
        elif indices is not None:
            max_len = torch.max(indices).item() + 1
            self.extend_pe(max_len, dtype, device)
            batch_size = x.size(0)
            t_length = x.size(1)
            pe = torch.index_select(self.pe, 0, indices.view(-1)).view(batch_size, t_length, -1)
        else:
            max_len = x.size(1)
            self.extend_pe(max_len, dtype, device)
            pe = self.pe.unsqueeze(0).expand(x.size(0), -1, -1)

        if self.mode in ["add", "addition"]:
            if self.x_scale == 1.0:
                x = x + pe[:, : x.size(1)]
            else:
                x = x * self.x_scale + pe[:, : x.size(1)]
            return self.dropout(x), None
        elif self.mode in ["cat", "concat"]:
            if self.x_scale == 1.0:
                x = torch.cat([x, pe[:, : x.size(1)]], dim=-1)
            else:
                x = torch.cat([x * self.x_scale, pe[:, : x.size(1)]], dim=-1)
            return self.dropout(x), None
        elif self.mode is None:
            if self.x_scale == 1.0:
                return x, pe[:, : x.size(1)]
            else:
                return x, pe[:, : x.size(1)] / self.x_scale


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding module (new implementation).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        hidden_channels (int): Encoding dimension.
        dropout_p (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(
        self, hidden_channels, dropout_p, scale=None, scaled=False, max_len=5000, win_len=None, chunkwise_len=None
    ):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.hidden_channels = hidden_channels
        if scaled:
            self.x_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.x_scale = math.sqrt(self.hidden_channels) if scale is None else scale
        self.dropout = Dropout(p=dropout_p, inplace=False)  # if dropout_p > 0.0 else lambda x: x
        self.pe = None
        if chunkwise_len is not None:
            self.extend_pe(torch.tensor(0.0).expand(1, chunkwise_len))
        elif win_len is not None:
            self.extend_pe(torch.tensor(0.0).expand(1, win_len))
        else:
            self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self.win_len = win_len
        self.chunkwise_len = chunkwise_len

    @torch.no_grad()
    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.chunkwise_len is not None:
                pe_len = 2 * self.chunkwise_len - 1
            elif self.win_len is not None:
                pe_len = 2 * self.win_len - 1
            else:
                pe_len = x.size(1) * 2 - 1
            if self.pe.size(1) >= pe_len:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.hidden_channels)
        pe_negative = torch.zeros(x.size(1), self.hidden_channels)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_channels, 2, dtype=torch.float32) * -(math.log(10000.0) / self.hidden_channels)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)  # 1 x (2 * T + 1) x C
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: Tensor):
        """Add positional encoding.
        Args:
            x (Tensor): Input tensor (batch, time, `*`).
        Returns:
            Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        with torch.no_grad():
            if self.chunkwise_len is not None:
                pos_emb = self.pe
            elif self.win_len is not None:
                diff_len = x.size(1) - self.win_len
                if diff_len > 0:
                    pad_negative = self.pe[:, 0].unsqueeze(1).expand(-1, diff_len, -1)
                    pad_positive = self.pe[:, -1].unsqueeze(1).expand(-1, diff_len, -1)
                    pos_emb = torch.cat([pad_negative, self.pe, pad_positive], dim=1)
                else:
                    pos_emb = self.pe[:, self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1)]
            else:
                pos_emb = self.pe[:, self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1)]

        x = x * self.x_scale
        return self.dropout(x), self.dropout(pos_emb)


class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding
    Reference : https://blog.eleuther.ai/rotary-embeddings/ Paper: https://arxiv.org/pdf/2104.09864.pdf
    """

    def __init__(self, hidden_size, base: int = 10000):
        super().__init__()
        dim = hidden_size

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_sequence_length = None
        self.cached_rotary_positional_encoding = None

    def forward(self, hidden_states):
        sequence_length = hidden_states.shape[1]

        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_encoding is not None:
            return hidden_states, self.cached_rotary_positional_encoding

        self.cached_sequence_length = sequence_length
        # encodings are computed in the dtype of the inv_freq constant
        time_stamps = torch.arange(sequence_length).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
        encodings = torch.cat((freqs, freqs), dim=-1)

        cos_encodings = encodings.cos()[:, None, None, :]
        sin_encodings = encodings.sin()[:, None, None, :]
        # Computed encodings are cast to the dtype of the hidden state inputs
        self.cached_rotary_positional_encoding = torch.stack([cos_encodings, sin_encodings]).type_as(hidden_states)
        return hidden_states, self.cached_rotary_positional_encoding
