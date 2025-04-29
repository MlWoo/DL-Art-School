import warnings

import torch
import torch.nn.functional as F
import vqtorch
from einops import repeat
from stringcolor import cs


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeanspp(samples, num_clusters, use_cosine_sim=False, num_iters=10, tol=1e-4):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device  # noqa F841

    means = sample_vectors(samples, num_clusters)
    i = 0
    for i in range(num_iters):
        means_pre = means.clone()
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            # (N * L, b)
            dists = torch.cdist(samples, means)
        buckets = dists.min(dim=-1).indices

        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

        means_shift = torch.sum((means - means_pre) ** 2)
        if means_shift < tol:
            break
    return means, bins, i, means_shift


@torch.no_grad()
def data_dependent_init_forward_hook(self, inputs, outputs, use_kmeans=True, verbose=False):
    """initializes codebook from data"""

    if (not self.training) or (self.data_initialized.item() == 1):
        return

    if verbose:
        print(cs("initializing codebook with k-means++", "y"))

    def sample_centroids(z_e, num_codes):
        """replaces the data of the codebook with z_e randomly."""

        z_e = z_e.reshape(-1, z_e.size(-1))

        if num_codes >= z_e.size(0):
            e_msg = (
                f"\ncodebook size > warmup samples: {num_codes} vs {z_e.size(0)}. "
                + "recommended to decrease the codebook size or increase batch size."
            )

            warnings.warn(str(cs(e_msg, "yellow")))

            # repeat until it fits and add noise
            repeat = num_codes // z_e.shape[0]
            new_codes = z_e.data.tile([repeat, 1])[:num_codes]
            new_codes += 1e-3 * torch.randn_like(new_codes.data)

        else:
            # you have more warmup samples than codebook. subsample data
            if use_kmeans:
                # from torchpq.clustering import KMeans
                # kmeans = KMeans(n_clusters=num_codes, distance="euclidean", init_mode="kmeans++")
                # kmeans.fit(z_e.data.T.contiguous())
                # new_codes = kmeans.centroids.T
                new_codes, bins, i, means_shift = kmeanspp(
                    z_e, num_clusters=num_codes, num_iters=100, use_cosine_sim=False, tol=1e-4
                )
                return new_codes.float()
            else:
                indices = torch.randint(low=0, high=num_codes, size=(num_codes,))
                indices = indices.to(z_e.device)
                new_codes = torch.index_select(z_e, 0, indices).to(z_e.device).data

        return new_codes

    _, misc = outputs
    z_e, z_q = misc["z"], misc["z_q"]  # noqa F841

    if type(self) is vqtorch.nn.VectorQuant:
        num_codes = self.codebook.weight.shape[0]
        new_codebook = sample_centroids(z_e, num_codes)
        self.codebook.weight.data = new_codebook

    elif type(self) is vqtorch.nn.GroupVectorQuant:
        if self.share:
            print(self.codebook.weight.shape)
            new_codebook = sample_centroids(z_e, self.group_size)
            self.codebook.weight.data = new_codebook
        else:
            for i in range(self.groups):
                offset = i * self.group_size
                new_codebook = sample_centroids(z_e[..., i, :], self.group_size)
                self.codebook.weight.data[offset : offset + self.group_size] = new_codebook

    elif type(self) is vqtorch.nn.ResidualVectorQuant:
        z_e = misc["z_res"]

        if self.share:
            new_codebook = sample_centroids(z_e, self.group_size)
            self.codebook.weight.data = new_codebook
        else:
            for i in range(self.groups):
                offset = i * self.group_size
                new_codebook = sample_centroids(z_e[..., i, :], self.group_size)
                self.codebook.weight.data[offset : offset + self.group_size] = new_codebook

    self.data_initialized.fill_(1)
    return
