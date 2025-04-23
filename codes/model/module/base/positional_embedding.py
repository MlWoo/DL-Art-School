# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positional Embedding Module."""
import torch
import torch.nn as nn


class RelativePositionalEmbedding(nn.Module):

    def __init__(self, hidden_channels, max_relative_position):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.max_relative_position = max_relative_position - 1
        rel_stddev = hidden_channels**-0.5
        self.weight = nn.Parameter(rel_stddev * torch.randn(max_relative_position * 2 + 1, hidden_channels))
        self.embeddings_table = self.weight
        # nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q).to(device=device)
        range_vec_k = torch.arange(length_k).to(device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class PositionalEmbedding(nn.Module):
    """Positional encoding.
    Args:
        hidden_channels (int): Embedding dimension.
        dropout_p (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position.
    """

    def __init__(self, hidden_channels, max_len=5000):
        """Construct an PositionalEmbedding object."""
        super().__init__()
        self.hidden_channels = hidden_channels
        self.pe = None
        self.weight = nn.Parameter(torch.Tensor(max_len, hidden_channels))
        self.embeddings_table = self.weight

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        batch_size, t_length = x.shape
        return torch.index_select(self.embeddings_table, 0, x.view(-1)).view(batch_size, t_length, -1)
