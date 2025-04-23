# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Subsampling layer definition."""

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelaudio.module.wenet.mask import make_pad_mask


class BaseSubsampling(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class EmbedinigNoSubsampling(BaseSubsampling):
    """Embedding input without subsampling"""

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        super().__init__()
        self.embed = torch.nn.Embedding(idim, odim)
        self.pos_enc = pos_enc_class

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.embed(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class Conv1dSubsampling2(BaseSubsampling):
    """Convolutional 1D subsampling (to 1/2 length).
       It is designed for Whisper, ref:
       https://github.com/openai/whisper/blob/main/whisper/model.py

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        """Construct an Conv1dSubsampling2 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
        )
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 2
        # 4 = (3 - 1) * 1 + (3 - 1) * 1
        self.right_context = 4

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.
            torch.Tensor: positional encoding

        """
        time = x.size(1)
        x = x.transpose(1, 2)  # (b, f, t)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (b, t, f)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, (time + 1) % 2 :: 2]


class Conv1dSubsampling4(BaseSubsampling):
    """Convolutional 1D subsampling (to 1/2 length).
       It is designed for Whisper, ref:
       https://github.com/openai/whisper/blob/main/whisper/model.py

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        """Construct an Conv1dSubsampling2 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(odim, odim, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
        )
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 2
        # 4 = (3 - 1) * 1 + (3 - 1) * 1
        self.right_context = 4

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.
            torch.Tensor: positional encoding

        """
        x = x.transpose(1, 2)  # (b, f, t)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (b, t, f)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 0::2][:, :, 0::2]


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling6 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim)
        self.pos_enc = pos_enc_class
        # 10 = (3 - 1) * 1 + (5 - 1) * 2
        self.subsampling_rate = 6
        self.right_context = 10

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 4::3]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2][:, :, 2::2]


class StackNFramesSubsampling(BaseSubsampling):

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module, stride: int = 2):

        super().__init__()
        del dropout_rate
        self.pos_enc_class = pos_enc_class
        self.stride = stride
        self.idim = idim

        self.norm = torch.nn.LayerNorm(idim * stride, eps=1e-5)
        self.out = torch.nn.Linear(idim * stride, odim)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // stride.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // stride.
            torch.Tensor: positional encoding
        """
        with torch.no_grad():
            b, s, _ = x.size()

            seq_len = x_mask.sum(-1).view(b)
            r = s % self.stride
            s -= r
            x = x[:, :s, :]
            seq_len = torch.where(seq_len > s, s, seq_len)
            seq_len = seq_len // self.stride
            new_mask = ~make_pad_mask(seq_len, max_len=s // self.stride)
            x = x.view(b, s // self.stride, self.idim * self.stride)
            _, pos_emb = self.pos_enc_class(x, offset)
        x = self.norm(x)
        x = self.out(x)
        return x, pos_emb, new_mask.unsqueeze(1)

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        return self.pos_enc_class.position_encoding(offset, size)


class SamePad1d(nn.Module):
    def __init__(
        self, kernel_size, causal, stride=1, dilation=1, padding_mode="replicate", channel_last=False, simple=True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.causal = causal
        self.stride = stride
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.channel_last = channel_last
        self.simple = simple

    def cal_length(self, input_length):
        if self.simple:
            all_padding = (self.kernel_size - 1) * self.dilation
        else:
            out_length = input_length // self.stride
            if self.causal:
                all_padding = out_length * self.stride + (self.kernel_size - 1) * self.dilation - input_length
            else:
                all_padding = out_length * self.stride + (self.kernel_size - 1) * self.dilation - input_length
        return all_padding

    def cal_pad(self, x):
        ndim = x.ndim
        if ndim == 3:
            t = x.shape[-2]
            all_padding = self.cal_length(t)
            if self.causal:
                pad = (0, 0, all_padding, 0)
            else:
                padding = all_padding // 2
                pad = (0, 0, padding, padding)
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
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            x = F.pad(x, pad, mode=self.padding_mode)
        return x

    def extra_repr(self) -> str:
        return "kernel_size={}, causal={}, stride={}".format(self.kernel_size, self.causal, self.stride)


class StreamConv2dSubsampling(BaseSubsampling):
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
        idim: int,
        odim: int,
        dropout_rate: float,
        pos_enc_class: torch.nn.Module,
        sub_layers=2,
        causal=True,
        hidden_dim=256,
    ):
        super().__init__()
        if sub_layers == 2:
            self.module = nn.Sequential(
                SamePad1d(kernel_size=3, causal=causal, stride=2, simple=False),
                nn.Conv2d(
                    in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=2, padding=0, padding_mode="replicate"
                ),
                nn.ReLU(inplace=True),
                SamePad1d(kernel_size=3, causal=causal, stride=2, simple=False),
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
            )
        else:
            self.module = nn.Sequential(
                SamePad1d(kernel_size=3, causal=causal, stride=2, simple=False),
                nn.Conv2d(
                    in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=2, padding=0, padding_mode="replicate"
                ),
                nn.ReLU(inplace=True),
            )
        self.linear_layer = nn.Linear(hidden_dim * idim // 4, odim)
        self.pos_enc = pos_enc_class

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, offset: Union[int, torch.Tensor] = 0):
        # [32, 1, 1536, 128]
        output = self.module(x.unsqueeze(1))  # (batch_size, 1, time, d_input)
        # [32, hidden_size, 384, 32]
        batch_size, d_model, subsampled_time, subsampled_freq = output.size()
        output = output.permute(0, 2, 1, 3)
        output = output.contiguous().view(batch_size, subsampled_time, d_model * subsampled_freq)
        output = self.linear_layer(output)
        x, pos_emb = self.pos_enc(output, offset)
        return x, pos_emb, x_mask[:, :, 1::2][:, :, 1::2]
