import torch
from model.audio.module.wenet.attention import (
    MultiHeadedAttention,
    MultiHeadedCrossAttention,
    RelPositionMultiHeadedAttention,
    RopeMultiHeadedAttention,
    ShawRelPositionMultiHeadedAttention,
)
from model.audio.module.wenet.embedding import (
    LearnablePositionalEncoding,
    NoPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    RopePositionalEncoding,
    WhisperPositionalEncoding,
)
from model.audio.module.wenet.positionwise_feed_forward import GatedVariantsMLP, MoEFFNLayer, PositionwiseFeedForward
from modelaudio.module.wenet.subsampling import Conv1dSubsampling4, StreamConv2dSubsampling
from torch.nn import BatchNorm1d, LayerNorm


class RMSNorm(torch.nn.Module):
    """https://arxiv.org/pdf/1910.07467.pdf"""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        return x * self.weight


WENET_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    # "swish": getattr(torch.nn, "SiLU", Swish),
    "swish": torch.nn.SiLU,
    "gelu": torch.nn.GELU,
}

WENET_RNN_CLASSES = {
    "rnn": torch.nn.RNN,
    "lstm": torch.nn.LSTM,
    "gru": torch.nn.GRU,
}

WENET_SUBSAMPLE_CLASSES = {
    "conv1d4": Conv1dSubsampling4,
    "stream_conv2d4": StreamConv2dSubsampling,
}

WENET_EMB_CLASSES = {
    "rel_pos": RelPositionalEncoding,
    "rope_pos": RopePositionalEncoding,
    "abs_pos": PositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
    "no_pos": NoPositionalEncoding,
}

WENET_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
    "crossattn": MultiHeadedCrossAttention,
    "shaw_rel_selfattn": ShawRelPositionMultiHeadedAttention,
    "rope_abs_selfattn": RopeMultiHeadedAttention,
}

WENET_MLP_CLASSES = {
    "position_wise_feed_forward": PositionwiseFeedForward,
    "moe": MoEFFNLayer,
    "gated": GatedVariantsMLP,
}

WENET_NORM_CLASSES = {"layer_norm": LayerNorm, "batch_norm": BatchNorm1d, "rms_norm": RMSNorm}
