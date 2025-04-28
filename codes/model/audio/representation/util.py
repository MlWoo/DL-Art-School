import os

import torch
from model.builder import STAT
from model.module.base.mask import sequence_mask


@STAT.register_module()
class MeanVarianceOnlineStats:
    def __init__(self, n_dim=128, feature_dim=1, data_key=None, data_length_key=None):
        self.n = 0
        self.mean = torch.zeros(n_dim)
        self.M2 = torch.zeros(n_dim)
        self.feature_dim = feature_dim
        self.data_key = data_key
        self.data_length_key = data_length_key

    def update(self, data_dict):
        """
        Args:
            data (Tensor): with shape `(B, D, T)`
            data_length (Tensor): with shape `(B)`
        """
        data = data_dict[self.data_key]
        data_length = data_dict[self.data_length_key]

        stat_dims = tuple([i for i in range(data.ndim) if i != self.feature_dim])
        mask = sequence_mask(data_length).unsqueeze(self.feature_dim)
        data = data * mask
        data_sum = torch.sum(data, dim=stat_dims)
        m = data_length.sum()
        if m == 0:
            return

        y_mean = data_sum / m
        y_mean_expanded = y_mean
        for stat_dim in stat_dims:
            y_mean_expanded = y_mean_expanded.unsqueeze(stat_dim)
        y_M2 = torch.sum((data - y_mean_expanded) ** 2 * mask, dim=stat_dims)

        new_n = self.n + m
        delta_mean = y_mean - self.mean
        new_mean = self.mean + delta_mean * m / new_n
        new_M2 = self.M2 + y_M2 + (self.n * m / new_n) * delta_mean**2

        self.n = new_n
        self.mean = new_mean
        self.M2 = new_M2

    def get_mean(self):
        return self.mean

    def get_variance(self):
        return self.M2 / self.n if self.n > 0 else torch.zeros_like(self.M2)

    def get_stddev(self):
        return self.get_variance().sqrt()

    def reset(self):
        self.n = 0
        self.mean.zero_()
        self.M2.zero_()

    def save(self, cnt, save_dir):
        mean = self.get_mean().cpu()
        std_dev = self.get_stddev().cpu()
        data = dict(
            mean=mean,
            std_dev=std_dev,
        )
        torch.save(data, os.path.join(save_dir, f"{self.data_key}_stat_{cnt}.pt"))
