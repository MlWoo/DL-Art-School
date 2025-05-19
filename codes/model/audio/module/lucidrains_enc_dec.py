from functools import partial

import torch
import torch.nn as nn
from model.module.base.activation import ACT2FN


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class ResBlock(nn.Module):
    def __init__(self, chan, conv, activation, dropout_p=0.0, ln=False):
        super().__init__()

        self.net = nn.Sequential(
            conv(chan, chan, 3, padding=1),
            Snake1d(chan) if activation == "snake" else ACT2FN[activation](),
            conv(chan, chan, 3, padding=1),
            Snake1d(chan) if activation == "snake" else ACT2FN[activation](),
            conv(chan, chan, 1),
        )
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 and dropout_p < 1.0 else nn.Identity()
        if ln:
            self.in_ln = nn.LayerNorm(chan)
        else:
            self.in_ln = None

    def forward(self, x):
        res = x
        if self.in_ln is not None:
            x = x.permute(0, 2, 1)
            x = self.in_ln(x)
            x = x.permute(0, 2, 1)
        x = self.dropout(self.net(x))
        return res + x


class StackedResBlock(nn.Module):
    def __init__(self, num_blocks, in_channels, activation, dropout_p, positional_dims=1, ln=True):
        super().__init__()
        if positional_dims == 2:
            conv = nn.Conv2d
        else:
            conv = nn.Conv1d
        layers = []
        for _ in range(num_blocks):
            layers.append(ResBlock(in_channels, conv, activation, dropout_p=dropout_p, ln=ln))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LucidrainsEncoder(nn.Module):
    def __init__(
        self,
        in_channels=128,
        num_layers=3,
        num_resnet_blocks=0,
        hidden_dim=64,
        stride=2,
        kernel_size=3,
        encoder_norm: bool = False,
        activation="relu",
        positional_dims=2,
        causal: bool = False,
        codebook_dim: int = -1,
    ):
        super().__init__()
        self.num_channels = in_channels

        if positional_dims == 2:
            conv = nn.Conv2d
        else:
            conv = nn.Conv1d

        enc_layers = []
        if num_layers > 0:
            enc_chans = [hidden_dim * 2**i for i in range(num_layers)]
            enc_chans = [in_channels, *enc_chans]
            enc_chans_io = list(zip(enc_chans[:-1], enc_chans[1:]))
            pad = (kernel_size, 0) if causal else (kernel_size - 1) // 2
            for enc_in, enc_out in enc_chans_io:
                enc_layers.append(
                    nn.Sequential(
                        conv(enc_in, enc_out, kernel_size, stride=stride, padding=pad),
                        Snake1d(enc_out) if activation == "snake" else ACT2FN[activation](),
                    )
                )
                if encoder_norm:
                    enc_layers.append(nn.GroupNorm(8, enc_out))
            innermost_dim = enc_chans[-1]
        else:
            enc_layers.append(
                nn.Sequential(
                    conv(in_channels, hidden_dim, 1),
                    Snake1d(hidden_dim) if activation == "snake" else ACT2FN[activation](),
                )
            )
            innermost_dim = hidden_dim

        for _ in range(num_resnet_blocks):
            enc_layers.append(ResBlock(innermost_dim, conv, activation))

        if codebook_dim > 0:
            enc_layers.append(conv(innermost_dim, codebook_dim, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.reduction_steps = stride**num_layers

    def forward(self, x, x_lengths=None):
        return self.encoder(x)


class UpsampledConv(nn.Module):
    def __init__(self, conv, *args, **kwargs):
        super().__init__()
        assert "stride" in kwargs.keys()
        self.stride = kwargs["stride"]
        del kwargs["stride"]
        self.conv = conv(*args, **kwargs)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            up = nn.functional.interpolate(x, scale_factor=self.stride, mode="nearest")
        y = self.conv(up)
        return y


class LucidrainsDecoder(nn.Module):
    def __init__(
        self,
        in_channels=512,
        num_layers=3,
        num_resnet_blocks=0,
        hidden_dim=64,
        out_channels=80,
        stride=2,
        kernel_size=3,
        activation="relu",
        use_transposed_convs: bool = False,
        positional_dims=2,
        g_clean_dim: int = 0,
        g_noise_dim: int = 0,
    ):
        super().__init__()
        has_resblocks = num_resnet_blocks > 0

        if positional_dims == 2:
            conv = nn.Conv2d
            conv_transpose = nn.ConvTranspose2d
        else:
            conv = nn.Conv1d
            conv_transpose = nn.ConvTranspose1d
        if not use_transposed_convs:
            conv_transpose = partial(UpsampledConv, conv)

        dec_layers = []
        if num_layers > 0:
            enc_chans = [hidden_dim * 2**i for i in range(num_layers)]
            dec_chans = list(reversed(enc_chans))

            dec_init_chan = in_channels if not has_resblocks else dec_chans[0]
            dec_chans = [dec_init_chan, *dec_chans]

            dec_chans_io = list(zip(dec_chans[:-1], dec_chans[1:]))
            pad = (kernel_size - 1) // 2
            for dec_in, dec_out in dec_chans_io:
                dec_layers.append(
                    nn.Sequential(
                        conv_transpose(dec_in, dec_out, kernel_size, stride=stride, padding=pad),
                        Snake1d(dec_out) if activation == "snake" else ACT2FN[activation](),
                    )
                )
            dec_out_chans = dec_chans[-1]
            innermost_dim = dec_chans[0]
        else:
            dec_out_chans = hidden_dim
            innermost_dim = hidden_dim

        if num_layers > 0:
            dec_layers.insert(0, conv(in_channels, innermost_dim, 1))

        for _ in range(num_resnet_blocks):
            dec_layers.append(ResBlock(dec_out_chans, conv, activation))

        dec_layers.append(conv(dec_out_chans, out_channels, 1))

        self.decoder = nn.Sequential(*dec_layers)
        if g_noise_dim > 0:
            self.g_noise_proj = nn.Linear(g_noise_dim, in_channels)
        else:
            self.g_noise_proj = None

        if g_clean_dim > 0:
            self.g_clean_proj = nn.Linear(g_clean_dim, in_channels)
        else:
            self.g_clean_proj = None

    def forward(self, x, g_c=None, g_n=None):
        if g_c is not None and self.g_clean_proj is not None:
            x = x + self.g_clean_proj(g_c).unsqueeze(-1)
        if g_n is not None and self.g_noise_proj is not None:
            x = x + self.g_noise_proj(g_n).unsqueeze(-1)

        return self.decoder(x)
