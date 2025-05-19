# -*- coding: utf8 -*-
#   Copyright 2019 JSALT2019 Distant Supervision Team
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.util.ops import safe_squeeze
from torch import distributed
from torch.distributions.laplace import Laplace
from torch.distributions.normal import Normal


# Quantizations with losses
class BaseDataQuantizer(torch.nn.Module):
    def __init__(self, **kwargs):
        super(BaseDataQuantizer, self).__init__(**kwargs)
        self.num_levels = 1

    def quantize(self, x):
        return self(x)

    def forward(self, x):
        """Encode the values, e.g. by quantizeing.

        Args:
            x: the data to quantize
        Returns:
            tuple of:
                - tensor with encoded values
                - tensor with
        """
        return x, x

    def dequantize(self, x):
        return x

    def mean_field(self, logits):
        raise NotImplementedError

    def sample(self, logits):
        raise NotImplementedError

    def loss(self, x, targets):
        raise NotImplementedError


class SoftmaxUniformQuantizer(BaseDataQuantizer):
    def __init__(self, num_levels, min=0.0, max=1.0, **kwargs):
        assert min == 0.0, "Not implemented"
        assert max == 1.0, "Not implemented"
        super(SoftmaxUniformQuantizer, self).__init__(**kwargs)
        self.num_levels = num_levels

    def forward(self, x):
        assert x.min() >= 0.0
        assert x.max() <= 1.0
        targets = (x * self.num_levels).clamp(0, self.num_levels - 1).long()
        assert targets.min() >= 0
        assert targets.max() < self.num_levels
        inputs = self.dequantize(targets)
        return inputs, targets

    def dequantize(self, q):
        return q.float() / (self.num_levels - 1)

    def mean_field(self, logits):
        dim = -1
        probs = F.softmax(logits, dim)
        ndim = [1] * probs.dim()
        ndim[dim] = self.num_levels
        probs *= torch.arange(self.num_levels, dtype=torch.float32, device=probs.device).view(*ndim)
        return probs.sum(dim)

    def sample(self, logits):
        *lead_dims, softmax_dim = logits.shape
        probs = torch.softmax(logits, -1).view(-1, softmax_dim)
        samples = torch.multinomial(probs, 1)
        samples = samples.view(*lead_dims)
        return self.dequantize(samples)

    def loss(self, logits, targets):
        assert logits.size()[:4] == targets.size()
        logits = logits.permute(0, 4, 1, 2, 3)
        loss = F.cross_entropy(logits, targets.long(), reduction="none")
        return loss


class SoftmaxQuantizer(BaseDataQuantizer):
    def __init__(self, levels=[0.0, 0.25, 0.5, 0.75], **kwargs):
        super(SoftmaxQuantizer, self).__init__(**kwargs)
        assert levels == sorted(levels), "Levels should be sorted"
        self.register_buffer("levels", torch.tensor(levels))
        self.num_levels = len(levels)

    def forward(self, x):
        _, targets = torch.min((x.unsqueeze(-1) - self.levels) ** 2, -1)
        return self.dequantize(targets), targets

    def dequantize(self, q):
        return self.levels[q]

    def mean_field(self, logits):
        dim = -1
        probs = F.softmax(logits, dim)
        ndim = [1] * probs.dim()
        ndim[dim] = self.num_levels
        probs *= self.levels.view(*ndim)
        return probs.sum(dim)

    def sample(self, logits):
        *lead_dims, softmax_dim = logits.shape
        probs = torch.softmax(logits, -1).view(-1, softmax_dim)
        samples = torch.multinomial(probs, 1)
        samples = samples.view(*lead_dims)
        return self.dequantize(samples)

    def loss(self, logits, targets):
        assert logits.size()[:4] == targets.size()
        logits = logits.permute(0, 4, 1, 2, 3)
        loss = F.cross_entropy(logits, targets.long(), reduction="none")
        return loss


class BinaryXEntropy(BaseDataQuantizer):
    def __init__(self, **kwargs):
        super(BinaryXEntropy, self).__init__(**kwargs)

    def forward(self, x):
        assert x.min() >= 0.0
        assert x.max() <= 1.0
        return x, x

    def dequantize(self, x):
        return x

    def mean_field(self, logits):
        logits = safe_squeeze(logits, -1)
        return torch.sigmoid(logits)

    def sample(self, logits):
        logits = safe_squeeze(logits, -1)
        probs = torch.sigmoid(logits)
        return (torch.rand_like(probs) < probs).float()

    def loss(self, logits, targets):
        logits = safe_squeeze(logits, -1)
        assert logits.size() == targets.size()
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")


class L1Loss(BaseDataQuantizer):
    def __init__(self, **kwargs):
        super(L1Loss, self).__init__(**kwargs)

    def forward(self, x):
        return x, x

    def dequantize(self, x):
        return x

    def mean_field(self, logits):
        logits = safe_squeeze(logits, -1)
        return logits

    def sample(self, logits):
        logits = safe_squeeze(logits, -1)
        return Laplace(logits, 1.0).sample()

    def loss(self, logits, targets):
        logits = safe_squeeze(logits, -1)
        assert logits.size() == targets.size(), f"{logits.size()} != {targets.size()}"
        return F.l1_loss(logits, targets, reduction="none")


class L2Loss(BaseDataQuantizer):
    def __init__(self, **kwargs):
        super(L2Loss, self).__init__(**kwargs)

    def forward(self, x):
        return x, x

    def dequantize(self, x):
        return x

    def mean_field(self, logits):
        logits = safe_squeeze(logits, -1)
        return logits

    def sample(self, logits):
        logits = safe_squeeze(logits, -1)
        return Normal(logits, 1.0).sample()

    def loss(self, logits, targets):
        logits = safe_squeeze(logits, -1)
        assert logits.size() == targets.size()
        return F.mse_loss(logits, targets, reduction="none")


class NormalMeanScaleLoss(BaseDataQuantizer):
    def __init__(self, **kwargs):
        super(NormalMeanScaleLoss, self).__init__(**kwargs)
        self.num_levels = 2

    def forward(self, x):
        return x, x

    def dequantize(self, x):
        return x

    def mean_field(self, logits):
        return logits[:, :, :, :, 0]

    def _get_normal(self, logits):
        loc, scale = logits.chunk(2, dim=-1)
        loc = safe_squeeze(loc, -1)
        scale = torch.exp(safe_squeeze(scale, -1))
        return Normal(loc, scale)

    def sample(self, logits):
        return self._get_normal(logits).sample()

    def loss(self, logits, targets):
        assert logits.size()[:-1] == targets.size()
        return -self._get_normal(logits).log_prob(targets)


class PerceptualLoss(BaseDataQuantizer):
    def __init__(self, layers=6):
        super(PerceptualLoss, self).__init__()
        import torchvision.models as tvmodels

        self.vgg = tvmodels.vgg16(pretrained=True).features[:layers]
        self.vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return x, x

    def dequantize(self, x):
        return x

    def mean_field(self, logits):
        logits = safe_squeeze(logits, -1)
        return logits

    def sample(self, logits):
        return safe_squeeze(logits, -1)

    def loss(self, logits, targets):
        logits = safe_squeeze(logits, -1)
        logits = logits.permute(0, 3, 2, 1)
        B, C, H, W = logits.shape
        logits = logits.expand(B, 3, H, W)
        targets = targets.permute(0, 3, 2, 1)
        targets = targets.expand(B, 3, H, W)
        return F.l1_loss(self.vgg(logits * 2 - 1), self.vgg(targets * 2 - 1), reduction="none")


class RandomProjectionQuantizer(nn.Module):
    def __init__(
        self,
        input_feature_size,
        reduction_factors,
        codebook_size=8192,
        codebook_dim=16,
    ):
        super().__init__()
        # should normalize input feature first
        # print("Input feature should be normalized firstly before using RandomProjectionQuantizer")

        P_init = torch.empty((input_feature_size * reduction_factors, codebook_dim))
        self.register_buffer("project_mat", nn.init.xavier_uniform_(P_init))

        codebook_weight_norm = F.normalize(torch.randn(codebook_size, codebook_dim), p=2.0, dim=-1)
        self.register_buffer("codebook_norm", codebook_weight_norm)

    @torch.no_grad()
    def forward(self, masked_target_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_values (torch.Tensor): with shape `(B, L, D)`
            mask_time_indices (torch.Tensor): with shape `(B, L)`

        Returns:
            torch.Tensor with shape `(N)`
        """
        targets = torch.matmul(masked_target_values, self.project_mat)
        targets_norm = F.normalize(targets, p=2.0, dim=-1)
        vector_distances = torch.cdist(targets_norm, self.codebook_norm)
        labels = torch.argmin(vector_distances, dim=-1)

        return labels


class NoBottleneck(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        norm_type = kwargs.get("norm_type", None)
        codebook_dim = kwargs.get("codebook_dim")
        if norm_type is None:
            self.norm = nn.Identity()
        elif norm_type == "batch_norm":
            if distributed.is_initialized() and distributed.get_world_size() > 0:
                self.norm = nn.SyncBatchNorm(codebook_dim, affine=False)
            else:
                self.norm = nn.BatchNorm1d(codebook_dim, affine=False)
        else:
            raise ValueError(f"Only Batch norm is support for the moment, but got {norm_type}")

    def forward(self, xs):
        zero = torch.zeros(()).cuda(xs.device)
        commit_losses = zero
        metrics = dict(entropy=zero, usage=zero, used_curr=zero, pn=zero, dk=zero)
        return xs, xs, commit_losses, None, metrics


class JukeboxQuantize(nn.Module):
    def __init__(
        self,
        in_dim,
        codebook_dim,
        codebook_size,
        decay=0.99,
        eps=1e-5,
        debug=False,
        max_threshold=-1.0,
        min_threshold=1.0,
        new_return_order=False,
        norm_type=None,
        dist_type="euclidean",
        dead_dist_update="single",
        dead_update_type="rand",
        ignore_extra_process=True,
        dynamic_ema_decay: bool = False,
        store_run_info=False,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps
        self.reset_k()
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.debug = debug
        self.ignore_extra_process = ignore_extra_process

        if in_dim == codebook_dim:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(in_features=in_dim, out_features=codebook_dim)

        if norm_type is None or dist_type == "cosine":
            self.norm = nn.Identity()
        elif norm_type == "batch_norm":
            if distributed.is_initialized() and distributed.get_world_size() > 0:
                self.norm = nn.SyncBatchNorm(codebook_dim, affine=False)
            else:
                self.norm = nn.BatchNorm1d(codebook_dim, affine=False)
        else:
            raise ValueError(f"Only Batch norm is support for the moment, but got {norm_type}")

        self.dist_type = dist_type
        self.dead_dist_update = dead_dist_update
        self.dead_update_type = dead_update_type
        self.diversity_enabled = True

        self.dynamic_ema_decay = dynamic_ema_decay
        if dynamic_ema_decay:
            store_run_info = True
        if store_run_info:
            self.run_info = dict(
                epoch=0,
                iter=0,
                will_log=False,
                lr=0,
            )
        else:
            self.run_info = None

    def store_run_info(self, epoch, step, will_log, **kwargs):
        if self.run_info is not None:
            self.run_info["epoch"] = epoch
            self.run_info["iter"] = step
            self.run_info["will_log"] = will_log
            self.run_info["lr"] = kwargs.get("lr", 0)

    def reset_k(self):
        codebook = torch.zeros(self.codebook_size, self.codebook_dim)
        self.register_buffer("codebook", codebook)
        self.register_buffer("codebook_cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("codebook_sum", torch.ones(self.codebook_size, self.codebook_dim))
        ###
        self.register_buffer("init", torch.BoolTensor([False]))
        self.register_buffer("dk", torch.ones(1))

    def _tile(self, x):
        d, ew = x.shape
        if d < self.codebook_size:
            n_repeats = (self.codebook_size + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def init_codebook(self, x):
        codebook_dim, codebook_size = self.codebook_dim, self.codebook_size
        # init k_w using random vectors from x
        y = self._tile(x)
        codebook_rand = y[torch.randperm(y.shape[0])][:codebook_size]
        if distributed.is_initialized() and distributed.get_world_size() > 1:
            distributed.broadcast(codebook_rand, 0)
        self.codebook = codebook_rand
        assert self.codebook.shape == (
            codebook_size,
            codebook_dim,
        ), f"codebook shape: {self.codebook.shape} vs codebook_size: {codebook_size} codebook_dim: {codebook_dim}"
        self.codebook_sum = self.codebook
        self.codebook_cluster_size = torch.ones(codebook_size, device=self.codebook.device)
        self.init = torch.BoolTensor([True]).to(self.init.device)

    def restore_codebook(self, num_tokens=None, min_threshold=1.0):
        codebook_dim, codebook_size = self.codebook_dim, self.codebook_size
        self.init = torch.BoolTensor([True]).to(self.init.device)
        assert self.codebook.shape == (codebook_size, codebook_dim)
        self.codebook_sum = self.codebook.clone()
        self.codebook_cluster_size = torch.ones(codebook_size, device=self.codebook.device)
        if num_tokens is not None:
            expected_usage = num_tokens / codebook_size
            self.codebook_cluster_size.data.mul_(expected_usage)
            self.codebook_sum.data.mul_(expected_usage)
        self.min_threshold = min_threshold

    def update_codebook(self, x, x_q, distances=None):
        codebook_size = self.codebook_size
        if self.dynamic_ema_decay:
            decay = 1.0 - 2.0 * self.run_info["lr"]
            decay = min(max(0.0, (1.0 - self.dk[0])), self.decay)
        else:
            decay = self.decay

        with torch.no_grad():
            # Calculate new centres
            x_q_onehot = F.one_hot(x_q, self.codebook_size).type(x.dtype)
            x_q_onehot = x_q_onehot.t()

            _codebook_sum = torch.matmul(x_q_onehot, x)  # codebook_size, w
            _codebook_cluster_size = x_q_onehot.sum(dim=-1)  # codebook_size
            y = self._tile(x)

            if distributed.is_initialized() and distributed.get_world_size() > 1:
                distributed.all_reduce(_codebook_sum)
                distributed.all_reduce(_codebook_cluster_size)
                if self.dead_dist_update == "single":
                    if self.dead_update_type == "rand":
                        codebook_rand = y[torch.randperm(y.shape[0])][:codebook_size]
                        distributed.broadcast(codebook_rand, 0)
                    elif self.dead_update_type == "confidence":
                        pass
                    else:
                        raise RuntimeError(f"{self.dead_update_type} is not implemented now.")

                else:
                    world_size = distributed.get_world_size()
                    if self.dead_update_type == "rand":
                        codebook_size_local = codebook_size // world_size + 1
                        codebook_rand_local = y[torch.randperm(y.shape[0])][:codebook_size_local]
                        codebook_rand_list = [torch.empty_like(codebook_rand_local) for _ in range(world_size)]
                        distributed.all_gather(codebook_rand_list, codebook_rand_local)
                        codebook_rand = torch.cat(codebook_rand_list, dim=0)[:codebook_size]
                    elif self.dead_update_type == "confidence":
                        raise RuntimeError(f"{self.dead_update_type} is not implemented now.")
                    else:
                        raise RuntimeError(f"{self.dead_update_type} is not implemented now.")
            else:
                if self.dead_update_type == "rand":
                    codebook_rand = y[torch.randperm(y.shape[0])][:codebook_size]
                elif self.dead_update_type == "confidence":
                    pass
                else:
                    raise RuntimeError(f"{self.dead_update_type} is not implemented now.")

            # Update centres
            old_k = self.codebook
            self.codebook_sum = decay * self.codebook_sum + (1.0 - decay) * _codebook_sum  # w, codebook_size
            self.codebook_cluster_size = (
                decay * self.codebook_cluster_size + (1.0 - decay) * _codebook_cluster_size
            )  # codebook_size
            usage_mask = torch.zeros_like(self.codebook_cluster_size).bool()
            if self.min_threshold > 0:
                usage_mask_min = self.codebook_cluster_size >= self.min_threshold
                usage_mask = torch.logical_or(usage_mask, usage_mask_min)
            if self.max_threshold > 0 and self.max_threshold > self.min_threshold:
                usage_mask_max = self.codebook_cluster_size <= self.max_threshold
                usage_mask = torch.logical_or(usage_mask, usage_mask_max)

            usage = usage_mask.float().view(self.codebook_size, 1)

            if self.dead_update_type == "confidence":
                codebook_rand = torch.zeros_like(self.codebook).to(y.dtype)
                num_dead_codes = (self.codebook_cluster_size < self.min_threshold).sum()
                dead_index = torch.where(usage_mask == False)[0]
                if distributed.is_initialized() and distributed.get_world_size() > 1:
                    _, selected_indice = torch.topk(distances, num_dead_codes)
                    selected_codebook = torch.index_select(y, 0, selected_indice)
                    codebook_rand.index_copy_(0, dead_index, selected_codebook)
                    distributed.broadcast(codebook_rand, 0)
                else:
                    _, selected_indice = torch.topk(distances, num_dead_codes)
                    selected_codebook = torch.index_select(y, 0, selected_indice)
                    codebook_rand.index_copy_(0, dead_index, selected_codebook)

            self.codebook = (
                usage * (self.codebook_sum / self.codebook_cluster_size.view(self.codebook_size, 1))
                + (1 - usage) * codebook_rand
            )

            _k_prob = _codebook_cluster_size / torch.sum(
                _codebook_cluster_size
            )  # x_q_onehotorch.mean(dim=-1)  # prob of each bin
            entropy = -torch.sum(_k_prob * torch.log(_k_prob + 1e-8))  # entropy ie how diverse
            used_curr = (_codebook_cluster_size >= self.min_threshold).sum()
            usage = torch.sum(usage)
            dk = torch.norm(self.codebook - old_k) / np.sqrt(np.prod(old_k.shape))
            self.dk[0] = dk

        return dict(entropy=entropy, used_curr=used_curr, usage=usage, dk=dk, ema_decay=decay)

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  # x_en = (N * L, w), k_j = (w, codebook_size)

        if x.shape[-1] == self.codebook_dim:
            prenorm = torch.norm(x - torch.mean(x)) / np.sqrt(np.prod(x.shape))
        elif x.shape[-1] == 2 * self.codebook_dim:
            x1, x2 = x[..., : self.codebook_dim], x[..., self.codebook_dim :]
            prenorm = (torch.norm(x1 - torch.mean(x1)) / np.sqrt(np.prod(x1.shape))) + (
                torch.norm(x2 - torch.mean(x2)) / np.sqrt(np.prod(x2.shape))
            )
            # Normalise
            x = x1 + x2
        else:
            assert False, f"Expected {x.shape[-1]} to be (1 or 2) * {self.codebook_dim}"
        return x, prenorm

    def postprocess(self, x_q, x_d, x_shape):
        # [NT, C] -> NTC -> NCT
        N, T = x_shape
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        if self.debug:
            return x_d
        else:
            x_q = x_q.view(N, T)
            return x_q, x_d

    def quantise(self, x):
        # Calculate latent code x_q
        # x: (N * L, b)
        embed = self.codebook.detach()
        if self.dist_type == "cosine":
            embed = F.normalize(embed)
            x = F.normalize(x)
        # (N * L, b)
        if self.diversity_enabled:
            distance = torch.cdist(x, embed)
        else:
            with torch.no_grad():
                distance = torch.cdist(x, embed)
        # last dim is
        with torch.no_grad():
            min_distance, x_q = torch.min(distance, dim=-1)
            fit = torch.mean(min_distance)
        return x_q, fit, min_distance, distance

    def dequantise(self, x_q):
        with torch.no_grad():
            x = F.embedding(x_q, self.codebook)
        return x

    def encode(self, x):
        N, width, T = x.shape

        # Preprocess.
        x, prenorm = self.preprocess(x)

        # Quantise
        x_q, fit = self.quantise(x)

        # Postprocess.
        x_q = x_q.view(N, T)
        return x_q

    def decode(self, x_q):
        N, T = x_q.shape
        width = self.codebook_dim

        # Dequantise
        x_d = self.dequantise(x_q)

        # Postprocess
        x_d = x_d.view(N, T, width).permute(0, 2, 1).contiguous()
        return x_d

    def forward(self, x, input_lengths=None):
        x = self.projection(x)
        if isinstance(self.norm, nn.BatchNorm1d) or isinstance(self.norm, nn.SyncBatchNorm):
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.norm(x)
        # Preprocess
        if self.ignore_extra_process:
            N, T, _ = x.shape
            prenorm = None
            x = x.reshape(-1, x.shape[-1])
        else:
            N, _, T = x.shape
            x, prenorm = self.preprocess(x)

        update_codebook = self.training
        # Init k if not inited
        if update_codebook and not self.init.item():
            self.init_codebook(x)

        # Quantise and dequantise through bottleneck
        x_q, fit, _, distances = self.quantise(x)
        x_d = self.dequantise(x_q)

        # Update embeddings
        if update_codebook:
            update_metrics = self.update_codebook(x, x_q, None)
        else:
            update_metrics = {}

        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        if not self.ignore_extra_process:
            if self.debug:
                x_d = self.postprocess(x_q, x_d, (N, T))
            else:
                x_q, x_d = self.postprocess(x_q, x_d, (N, T))
        else:
            x_d = x_d.reshape(N, T, -1)
        x_q = x_q.reshape(N, -1)
        confidence = -distances
        # quant value; quant idx, loss
        return x_d, x_q, commit_loss, confidence, dict(fit=fit, pn=prenorm, **update_metrics)
