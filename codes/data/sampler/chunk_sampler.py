import math
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from utils.function import mp_executor
from utils.logging_utils import get_root_logger

from data.sampler.util import chunk, split_list, split_with_n


def get_buckets(sizes, num_buckets, bucket_method="percent"):
    if bucket_method == "percent":
        buckets = np.unique(
            np.percentile(
                sizes,
                np.linspace(0, 100, num_buckets + 1),
                method="lower",
            )[1:]
        )
    elif bucket_method == "size":
        buckets = np.ceil(np.linspace(sizes.min(), sizes.max(), num_buckets + 1)[1:]).astype(int)
    else:
        raise ValueError(f"Invalid bucket method: {bucket_method}")
    return buckets


def get_bucketed_indices(sorted_sizes: torch.Tensor, bucket_boundaries):
    assert sorted_sizes.min() >= 0
    start_val = -1
    bucket_volumes = []
    for i, end_val in enumerate(bucket_boundaries):
        if end_val < 0:
            assert i == len(bucket_boundaries) - 1, "Only the last bucket boundaries is negative"
            mask = sorted_sizes > start_val
        else:
            mask = torch.logical_and((sorted_sizes > start_val), (sorted_sizes <= end_val))
        start_val = end_val
        bucket_volumes.append(mask.sum())
    return bucket_volumes


def acc_data(data, coeffs=((1.2, 1),)):
    ret = 0
    for pow, coeff in coeffs:
        ret += coeff * math.pow(data, pow)
    return ret


def batch_by_size(indices, num_tokens_vec, max_tokens, max_samples, bsz_mult=1, acc_coeffs=((1.2, 1),), drop_last=True):

    if indices.shape[0] == 0:
        return None

    assert (
        max_tokens <= 0 or torch.max(num_tokens_vec) <= max_tokens
    ), f"Sentences lengths should not exceed max_tokens={max_tokens}"

    indices_len = indices.shape[0]
    batches_ends = torch.zeros_like(indices, dtype=torch.int32)

    batches_count = 0
    batch_start = 0
    tail_max_tokens = 0
    batch_max_tokens = 0

    for pos in range(indices_len):
        # At every pos we keep stats about the last complete batch [batch_start:batch_end),
        #      and tail [batch_end:pos].
        # 1) Every time when (batch + tail) forms a valid batch
        #      (according to max_tokens, max_sentences) we append tail to batch.
        # 2) When (batch+tail) violates max_tokens or max_sentences constraints
        #      we finalize running batch, and tail becomes a new batch.
        # 3) There is a corner case when tail also violates constraints.
        #      In that situation [batch_end:pos-1] (tail without the current pos)
        #      gets added to the finalized batches, while [pos:pos] becomes a new tail.
        #
        # Important: For the sake of performance try to avoid using function calls within this loop.

        tail_max_tokens = tail_max_tokens if tail_max_tokens > num_tokens_vec[pos] else num_tokens_vec[pos]
        tail_max_tokens_acc = acc_data(tail_max_tokens, acc_coeffs)
        new_batch_end = pos + 1
        new_batch_max_tokens = batch_max_tokens if batch_max_tokens > tail_max_tokens_acc else tail_max_tokens_acc
        new_batch_sentences = new_batch_end - batch_start
        new_batch_num_tokens = new_batch_sentences * new_batch_max_tokens

        overflow = new_batch_sentences > max_samples > 0 or new_batch_num_tokens > max_tokens > 0

        size_matches_with_bsz_mult = new_batch_sentences < bsz_mult or new_batch_sentences % bsz_mult == 0
        if overflow:
            tail_num_tokens = tail_max_tokens_acc * (new_batch_end - batches_ends[batches_count])
            tail_overflow = tail_num_tokens > max_tokens > 0
            # In case of a tail overflow finalize two batches
            if tail_overflow:
                batches_count += 1
                batches_ends[batches_count] = pos
                tail_max_tokens = num_tokens_vec[pos]
            batch_start = batches_ends[batches_count]
            batches_count += 1
            new_batch_max_tokens = acc_data(tail_max_tokens, acc_coeffs)

        if overflow or size_matches_with_bsz_mult:
            batches_ends[batches_count] = new_batch_end
            batch_max_tokens = new_batch_max_tokens
            tail_max_tokens = 0
    if drop_last:
        indices = indices[: batches_ends[batches_count]]
        batches_ends = batches_ends[:batches_count]
    else:
        if batches_ends[batches_count] != indices_len:
            batches_count += 1

    # Memory and time-efficient split
    return list(torch.tensor_split(indices, batches_ends[:batches_count].tolist()))


def batch_by_size_wrapper(options: Tuple[List, List], cfg: Dict[str, Any]):
    indices, num_tokens_vec = options
    max_tokens = cfg.get("max_tokens")
    max_samples = cfg.get("max_samples")
    bsz_mult = cfg.get("bsz_mult", 1)
    acc_coeffs = cfg.get("acc_coeffs", 1.0)
    drop_last = cfg.get("drop_last", True)
    batch_list = batch_by_size(
        indices, num_tokens_vec, max_tokens, max_samples, bsz_mult, acc_coeffs=acc_coeffs, drop_last=drop_last
    )
    return None, batch_list


class LengthChunkSampler:
    def __init__(
        self,
        batch_size: Optional[int] = None,
        buffer_batch_group: int = 8,
        shuffle: bool = True,
        last_samples: str = "pad",
        copies: int = 1,
        seed: int = 42,
        num_replicas: int = 1,
        rank: int = 0,
        limited_type: Optional[str] = None,
        similar_type: Optional[str] = None,
        length_range: Tuple[int, int] = (0, -1),
        phase: str = "unknown",
        bucket_batch_volume: Optional[int] = 32,
        bucket_boundaries: Optional[Tuple[int, ...]] = None,
        num_buckets: Optional[int] = None,
        bucket_method: str = "percent",
        bucket_padding_noise: float = 0.0,
        batch_mode: str = "dynamical",
        acc_coeffs: Tuple[Tuple[float, float], ...] = ((1.0, 1.0),),
        max_tokens: int = 1024,
        max_samples: int = 64,
        bucket_max_batch_tokens: int = 4096,
        bucket_min_samples: int = 1,
        bucket_max_samples: int = 64,
        descending: bool = False,
        partition_record_path: Optional[str] = None,
        bucketed_batch_size_map: Optional[Dict[int, int]] = None,
    ):
        """The sampler generate dataset indices in term of sequencial data which has length info.

        Args:
            batch_size (Optional[int]): the batch size, it is determined automatically by
                `max_tokens`, `max_samples` and `num_replicas` if `dynamical` mode provided.
                Defaults to None.
            buffer_batch_group (int, optional): KVDataset buffer size to increase loading
                efficiency. Defaults to 8.
            shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to True.
            last_samples (str, optional): How to handle with the tailfing samples in the last batch
                of the epoch or the bucket. Defaults to 'pad'.
            copies (int, optional): How many epoches to release dataloader resources. The option
                will increase dataloader efficiency. Defaults to 1.
            seed (int, optional): The random seed of dataloaders in  distributed training.
                Defaults to 42.
            num_replicas (int, optional): The numbers of GPUs or other devices . Defaults to 1.
            rank (int, optional): The rank of GPUs or other devices. Defaults to 0.
            limited_type (Optional[str], optional): The samplers discard the samples according to
                 the length type. Defaults to None.
            similar_type (Optional[str], optional): The samplers bucket the samples according to
                 the length type. Defaults to 'lab'.
            length_range (Tuple[int, int], optional): The samplers discard the samples whose
                lengths exceed the range. Limited the number of per class in this dataloader.
                The dataset of one class will be ignored if the number is lower than left boundary.
                The redundant samples will be discarded if the number exceeds the right boundary.
                Defaults to (0,-1).
            phase (str, optional): training phase, validation phase or test phase.
                Defaults to 'unknown'.
            bucket_batch_volume (Optional[int], optional): The bucket size is the product of
                `bucket_batch_volume` and `batch_size`. Defaults to 32.
            bucket_boundaries (Optional[Tuple[int, ...]], optional): The bucket boundaries.
                Defaults to None.
            num_buckets (Optional[int], optional): The numbers of buckets. Defaults to None.
            bucket_padding_noise (float, optional): The nosie to make the samples could be
                interchanged among differnet buckets. Defaults to 0.1.
            dynamical (bool, optional): The batch size is determined automatically each batch by
                `max_tokens`, `max_samples` and `num_replicas` if `dynamical` mode provided.
                Defaults to False.
            max_tokens (int, optional): The max tokens in each batch if `dynamical` mode provided.
                Defaults to 1024.
            max_samples (int, optional): The max samples in each batch if `dynamical` mode
                provided. Defaults to 64.
            descending (bool, optional): It is used to debug the code and could be ignored.
                Defaults to False.

        Raises:
            RuntimeError: _description_
            RuntimeError: _description_
            RuntimeError:
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if batch_size is None and not batch_mode == "dynamical":
            raise RuntimeError("batch size should be provided if dynamical batching is disabled")
        self.batch_size = batch_size
        self.last_samples = last_samples
        self.buffer_batch_group = buffer_batch_group
        self.num_replicas = num_replicas
        self.rank = rank
        self.copies = copies
        self.epoch = 0
        self.copy_idx = 0
        self._sample_idx = 0
        self.iter_is_null = True
        self.truncate_mode = False
        self.ready = False
        self.dataset_len = None
        self.dataset_indices = None
        self.shuffle = shuffle
        self.seed = seed
        self.phase = phase
        self.logger = get_root_logger()
        self.length_range = (0, -1) if length_range is None else length_range
        self.limited_type = limited_type
        self.descending = descending
        if bucket_batch_volume is not None:
            if bucket_batch_volume % num_replicas == 0:
                self.bucket_batch_volume = bucket_batch_volume
            else:
                self.logger.info(
                    " ".join(
                        [
                            "It is recommanded that the batch volume of a bucket is multiplier of number of GPUs.",
                            "The batch volume is setted to the product of original setting and the number of GPUs.",
                        ]
                    )
                )
        self.similar_type = similar_type
        self.dataset = None
        self.bucket_padding_noise = bucket_padding_noise
        self.bucket_boundaries = bucket_boundaries
        self.max_tokens = max_tokens
        self.max_samples = max_samples
        self.batch_mode = batch_mode
        self.acc_coeffs = acc_coeffs
        self.num_buckets = num_buckets
        self.partition_record_path = partition_record_path
        self.protected = False
        self.bucket_min_samples = bucket_min_samples
        self.bucket_max_samples = bucket_max_samples
        self.bucket_max_batch_tokens = bucket_max_batch_tokens
        self.bucket_boundaries_batch_size_map = bucketed_batch_size_map
        self.bucket_method = bucket_method
        torch.manual_seed(seed)

    def finalize_dataset(self, dataset, indices: Union[torch.LongTensor, Tuple[int]], verbose: bool = True):
        self.dataset = dataset
        self.dataset_len = len(indices)
        if isinstance(indices, torch.LongTensor):
            self.dataset_indices = indices
        else:
            self.dataset_indices = torch.LongTensor(indices)

        sorted_type = self.limited_type if self.similar_type is None else self.similar_type
        if sorted_type is not None and self.bucket_boundaries is None and self.num_buckets is not None:
            sizes = torch.LongTensor(self.dataset.info_dict[sorted_type])[indices].numpy()
            self.bucket_boundaries = torch.from_numpy(get_buckets(sizes, self.num_buckets, self.bucket_method))
        if self.bucket_boundaries is not None and self.batch_mode == "bucketed":
            if self.bucket_boundaries_batch_size_map is None:
                self.bucket_boundaries_batch_size_map = {}
                for bucket_boundary in self.bucket_boundaries.tolist():
                    batch_size = self.bucket_max_batch_tokens // bucket_boundary
                    if batch_size > self.bucket_max_samples:
                        batch_size = self.bucket_max_samples
                    elif batch_size < self.bucket_min_samples:
                        batch_size = self.bucket_min_samples
                    self.bucket_boundaries_batch_size_map[bucket_boundary] = batch_size
            else:
                for bucket_boundary in self.bucket_boundaries.tolist():
                    if bucket_boundary not in self.bucket_boundaries_batch_size_map:
                        raise ValueError(f"bucket_boundary {bucket_boundary} not in bucket_boundaries_batch_size_map")

        if self.bucket_boundaries is not None and self.rank <= 0:
            self.logger.info(f"sampler bucket boundaries: {self.bucket_boundaries.tolist()}")

        if self.batch_mode == "dynamical":
            self.total_size = self.dataset_len
            self.num_samples = self.total_size

            if verbose and self.rank <= 0:
                self.logger.info(
                    f"Phase {self.phase}, Batch mode: {self.batch_mode}, coeffs: {self.acc_coeffs}, last samples: {self.last_samples}"  # noqa: E501
                )
                self.logger.info(
                    "dynamical Batch, Now calculating the dynamical info to aprroximate the batch num, waiting a moment...."  # noqa: E501
                )
                self.set_epoch(0)
                self.protected = True
                self.logger.info("Chunk sampler info: ")
                self.logger.info(f"    Total size: {self.total_size}, Sample size: {self.num_samples}")
                self.logger.info(f"    Batch_group: {self.buffer_batch_group}, Batch Num: {self.num_batch}")
            else:
                self.set_epoch(0)
                self.protected = True
                self._sample_idx = 0
        else:
            self.buffer_batch_group_size = self.batch_size * self.buffer_batch_group

            if self.last_samples == "drop":
                self.num_batch = int(math.floor(self.dataset_len * 1.0 / (self.num_replicas * self.batch_size)))
            else:
                self.num_batch = int(math.ceil(self.dataset_len * 1.0 / (self.num_replicas * self.batch_size)))

            if self.last_samples == "keep":
                self.total_size = self.dataset_len
                self.num_samples = int(math.ceil(self.dataset_len * 1.0 / self.num_replicas))
            else:
                self.total_size = self.num_batch * self.batch_size * self.num_replicas
                self.num_samples = self.num_batch * self.batch_size
            self._num_samples_ = self.num_samples
            if verbose and self.rank <= 0:
                self.logger.info(f"Phase {self.phase}, Chunk sampler info: ")
                self.logger.info(
                    f"    Total size: {self.total_size}, Sample size: {self.num_samples}, Dataset length: {self.dataset_len}"  # noqa: E501
                )
                self.logger.info(
                    f"    Batch_group: {self.buffer_batch_group}, Batch size: {self.batch_size}, Batch Num: {self.num_batch}"  # noqa: E501
                )

    def argsort_by_noise_padding(self, g):
        """ """
        if self.similar_type is None:
            raise RuntimeError("Similar type should be not None in bucketing mode")
        else:
            indices = self.dataset_indices.clone()
            lengths = torch.LongTensor(self.dataset.info_dict[self.similar_type])[indices]

        if self.bucket_boundaries is None:
            if self.total_size > len(indices):
                if self.shuffle:
                    perm = torch.randperm(self.dataset_len, generator=g)
                    lengths = torch.cat([lengths, lengths[perm[: (self.total_size - len(indices))]]])
                    indices = torch.cat([indices, indices[perm[: (self.total_size - len(indices))]]])
                else:
                    lengths = torch.cat([lengths, lengths[: (self.total_size - len(indices))]])
                    indices = torch.cat([indices, indices[: (self.total_size - len(indices))]])
            elif self.total_size < len(indices):
                lengths = lengths[: self.total_size]
                indices = indices[: self.total_size]
        max_length = lengths.max()
        min_length = lengths.min()
        torch.manual_seed(self.seed)
        noise = (torch.randn(*lengths.size()) - 0.5) * self.bucket_padding_noise
        noisy_length = (lengths * noise + lengths).long()
        noisy_length = torch.clip(noisy_length, min=min_length, max=max_length)
        # if self.length_range is not None and self.length_range[-1] > 0:
        #    max_length = self.length_range[-1]
        #    noisy_length = torch.clip(noisy_length, max=max_length)
        sorted_noisy_lengths, sorted_indices = torch.sort(noisy_length)
        new_dataset_indices = indices[sorted_indices]
        new_dataset_lengths = lengths[sorted_indices]
        return new_dataset_indices, new_dataset_lengths, sorted_noisy_lengths

    def bucket_with_sorted_dataset(
        self, sorted_dataset_indices, sorted_dataset_lengths, sorted_noisy_lengths, g, copy_idx=0
    ):
        shuffled_batches_indices_list = []

        if self.partition_record_path is not None and self.batch_mode == "dynamical":
            if osp.exists(osp.join(self.partition_record_path, self.phase + "_bucket_partition.pt")):
                ddp_bucket_batches_indices_list = torch.load(
                    osp.join(self.partition_record_path, self.phase + "_bucket_partition.pt")
                )
                if self.shuffle:
                    perm = torch.randperm(len(ddp_bucket_batches_indices_list), generator=g)
                    shuffled_ddp_bucket_batches_indices_list = [ddp_bucket_batches_indices_list[i] for i in perm]
                    ddp_bucket_batches_indices_list = shuffled_ddp_bucket_batches_indices_list

                return ddp_bucket_batches_indices_list

        if self.bucket_boundaries is not None:
            bucket_volumes = get_bucketed_indices(sorted_noisy_lengths, self.bucket_boundaries)
            accum_start = 0
            bucket_indices_pool = []
            bucket_lengths_pool = []
            ddp_bucket_batches_indices_list = []
            for i, bucket_volume in enumerate(bucket_volumes):
                accum_end = accum_start + bucket_volume
                bucket_indices = sorted_dataset_indices[accum_start:accum_end].contiguous()
                bucket_lengths = sorted_dataset_lengths[accum_start:accum_end].contiguous()
                # butcket internal shuffle
                if self.shuffle:
                    perm = torch.randperm(bucket_volume, generator=g)
                    bucket_indices = bucket_indices[perm]
                    bucket_lengths = bucket_lengths[perm]
                else:
                    bucket_indices = bucket_indices
                    bucket_lengths = bucket_lengths

                if self.batch_mode == "dynamical":
                    # add bucket_lengths to bucket_lengths_pool to partition
                    bucket_lengths_pool.append(bucket_lengths)
                    bucket_indices_pool.append(bucket_indices)
                else:
                    if self.batch_mode == "bucketed":
                        ddp_batch_size = (
                            self.bucket_boundaries_batch_size_map[self.bucket_boundaries[i].item()] * self.num_replicas
                        )
                    elif self.batch_mode == "fixed":
                        ddp_batch_size = self.batch_size * self.num_replicas
                    else:
                        raise ValueError(f"Invalid batch mode: {self.batch_mode}")

                    if self.last_samples == "drop":
                        ddp_bucket_batches_indices = split_list(bucket_indices, ddp_batch_size, keep_tail=False)
                    else:
                        total_batch_num = len(bucket_indices)
                        extra_batch_num = total_batch_num - total_batch_num % self.num_replicas
                        extra_indices = torch.randperm(total_batch_num)[:extra_batch_num]  # 随机排列后取前n个
                        bucket_indices = torch.cat([bucket_indices, bucket_indices[extra_indices]], dim=0)

                        ddp_bucket_batches_indices = split_list(bucket_indices, ddp_batch_size, keep_tail=False)

                    for ddp_bucket_batch_indices in ddp_bucket_batches_indices:
                        ddp_bucket_batches_indices_list.append(torch.chunk(ddp_bucket_batch_indices, self.num_replicas))

                accum_start = accum_end

            if self.batch_mode == "dynamical":
                indices_sum_ref = sorted_dataset_indices.sum()
                indices_sum_shuffle = sum([bucket_indices.sum() for bucket_indices in bucket_indices_pool])
                assert (
                    indices_sum_ref == indices_sum_shuffle
                ), "Any shuffle operation should not change the selected data indices"

                drop_last = False
                cfg = dict(
                    max_tokens=self.max_tokens,
                    max_samples=self.max_samples,
                    bsz_mult=1,
                    acc_coeffs=self.acc_coeffs,
                    drop_last=drop_last,
                )
                pool = None
                bucket_batches_indices_list = mp_executor(
                    pool, batch_by_size_wrapper, (bucket_indices_pool, bucket_lengths_pool), cfg, True, -1, -1
                )
                if False:
                    for bucket_batches_indices in bucket_batches_indices_list:
                        for batch_indices in bucket_batches_indices:
                            seq_lengths = self.dataset.dummy_lengths[batch_indices]
                            batch_tokens = seq_lengths.max() * len(batch_indices)
                            if batch_tokens > self.max_tokens:
                                print(batch_tokens)
                bucket_batches_indices = sum(bucket_batches_indices_list, [])

                if not drop_last:
                    indices_sum_partion = sum(
                        [indices.sum() for batches_indicest in bucket_batches_indices for indices in batches_indicest]
                    )
                    assert (
                        indices_sum_ref == indices_sum_partion
                    ), "Any Partition operation should not change the selected data indices"

                if self.last_samples == "drop":
                    total_batch_num = (len(bucket_batches_indices) // self.num_replicas) * self.num_replicas
                    bucket_batches_indices = bucket_batches_indices[:total_batch_num]
                else:
                    total_batch_num = len(bucket_batches_indices)
                    extra_batch_num = total_batch_num - total_batch_num % self.num_replicas
                    extra_indices = torch.randperm(total_batch_num)[:extra_batch_num]  # 随机排列后取前n个
                    bucket_batches_indices = bucket_batches_indices + [bucket_batches_indices[i] for i in extra_indices]

                # chunk the list with num_replicas
                for i in range(0, len(bucket_batches_indices), self.num_replicas):
                    ddp_bucket_batches_indices_list.append(bucket_batches_indices[i : i + self.num_replicas])

            # butcket external shuffle
            if self.shuffle:
                perm = torch.randperm(len(ddp_bucket_batches_indices_list), generator=g)
                shuffled_ddp_bucket_batches_indices_list = [ddp_bucket_batches_indices_list[i] for i in perm]
                ddp_bucket_batches_indices_list = shuffled_ddp_bucket_batches_indices_list

            num_batch = len(ddp_bucket_batches_indices_list)
            self.logger.info(
                f"Batch Mode: {self.batch_mode}, Last samples: {self.last_samples}, Copy idx: {copy_idx}, Chunk sampler info: "  # noqa: E501
            )
            self.logger.info(
                f"    Batch size: {self.batch_size}, Batch Num: {num_batch}, Batch_group: {self.buffer_batch_group}"
            )

        elif self.bucket_batch_volume is not None:
            bucket_volume = self.batch_size * self.bucket_batch_volume * self.num_replicas
            buckets_indices = torch.split(sorted_dataset_indices, bucket_volume)
            for bucket_indices in buckets_indices:
                if self.shuffle:
                    bucket_volume_ = len(bucket_indices)
                    perm = torch.randperm(bucket_volume_, generator=g)
                    shuffled_bucket_indices = bucket_indices[perm]
                else:
                    shuffled_bucket_indices = bucket_indices
                indices_list = torch.split(shuffled_bucket_indices, self.batch_size * self.num_replicas)
                shuffled_batches_indices_list.extend(indices_list)
            if self.shuffle:
                perm = torch.randperm(len(shuffled_batches_indices_list), generator=g)
                shuffled_batches_indices_list = [shuffled_batches_indices_list[i] for i in perm]
        else:
            raise ValueError("The sampler are confused with bucket strategy")

        if self.rank == 0 and self.partition_record_path is not None and self.batch_mode == "dynamical":
            os.makedirs(self.partition_record_path, exist_ok=True)
            torch.save(
                ddp_bucket_batches_indices_list,
                osp.join(self.partition_record_path, self.phase + "_bucket_partition.pt"),
            )
        return ddp_bucket_batches_indices_list

    def generate_dataset_indices(self):
        if self.dataset_len is None:
            raise RuntimeError("The dataset shouldn't have no samples, but the length of dataset is 0")
        if self.similar_type is None:
            if self.batch_size is None:
                raise RuntimeError("Batch size should be provided without bucked strategy")
            all_indices_copies = []
            indices = self.dataset_indices.clone().to(torch.int32)
            for _copy in range(self.copy_idx, self.copies):
                if self.shuffle:
                    assert self.last_samples in [
                        "pad",
                        "drop",
                    ], "shuffle mode is only applied to training mode, the last samples should be dropped or padded to make a whole batch"  # noqa: E501
                    g = torch.Generator()
                    g.manual_seed(self.seed + self.epoch + _copy)
                    if self.total_size > len(indices):
                        perm = torch.randperm(self.dataset_len, generator=g)
                        indices = torch.cat([indices, indices[perm[: (self.total_size - len(indices))]]])
                    elif self.total_size < len(indices):
                        indices = indices[: (self.total_size - len(indices))]
                    perm = torch.randperm(self.total_size, generator=g)
                    indices = indices[perm]
                else:
                    indices = self.dataset_indices.clone().to(torch.int32)
                    if self.last_samples == "pad":
                        # add extra samples to make it evenly divisible
                        indices = torch.cat((indices, indices[: (self.total_size - self.dataset_len)]))
                    elif self.last_samples == "drop":
                        indices = indices[: self.total_size]
                ddp_batch_indices_list_ = torch.split(indices, self.batch_size * self.num_replicas)
                batch_indices_list = []
                for ddp_batch_indices in ddp_batch_indices_list_:
                    batch_indices_list.append(torch.chunk(ddp_batch_indices, self.num_replicas))
                all_indices_copies += batch_indices_list

            indices_copies = all_indices_copies
            self.num_batch = len(indices_copies)
        else:
            g = torch.Generator()
            shuffled_batches_indices_list = []
            for _copy in range(self.copy_idx, self.copies):
                g.manual_seed(self.seed + self.epoch + _copy)
                # deterministically shuffle based on epoch
                new_dataset_indices, new_dataset_lengths, sorted_noisy_lengths = self.argsort_by_noise_padding(g)

                shuffled_batches_indices_list_ = self.bucket_with_sorted_dataset(
                    new_dataset_indices, new_dataset_lengths, sorted_noisy_lengths, g, _copy
                )
                shuffled_batches_indices_list += shuffled_batches_indices_list_

            copies = self.copies - self.copy_idx
            all_len = sum([len(sample_indices) for sample_indices in shuffled_batches_indices_list])

            if self.bucket_boundaries is not None:
                if self.batch_mode == "dynamical" or self.last_samples == "keep":
                    pass
            else:
                assert (
                    all_len == self.num_samples * self.copies
                ), f"dist sampler phase {self.phase} {all_len} vs {self.num_samples} x {copies}"

            indices_copies = shuffled_batches_indices_list
            self.num_batch = len(indices_copies)

        return indices_copies

    def __real_iter__(self):
        assert not self.truncate_mode
        indices_copies = self.generate_dataset_indices()
        self.iterable = indices_copies  # iter(indices_copies)
        self.iter_is_null = False

    def __iter__(self):
        if self.iter_is_null:
            if self.phase == "train":
                raise RuntimeError("please call `set_epoch` first before iterating the dataloader")
            else:
                indices = self.dataset_indices.clone()
                batch_indices_list_ = torch.split(indices, self.batch_size * self.num_replicas)
                total_size = len(batch_indices_list_)
                batch_indices_list = [indice.tolist() for indice in batch_indices_list_]
                indices_copies = batch_indices_list[self.rank : total_size : self.num_replicas]
                self.iterable = indices_copies
                self.iter_is_null = False
        data = chunk(self.iterable, self.buffer_batch_group, self.rank)
        return data

    def __len__(self):
        return self.num_batch

    def set_epoch(self, epoch, copy_idx=0, sample_idx=0):
        if not self.protected:
            self.reset_iter()
            self.epoch = epoch
            self.copy_idx = copy_idx
            self.__real_iter__()
        self.protected = False
        self._sample_idx = sample_idx
        if sample_idx > 0:
            remained = len(self.iterable) - sample_idx
            piece = split_with_n(self.iterable, sample_idx)
            self.iterable = piece
            self.num_batch = remained

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.copy_idx = state_dict["copy_idx"]
        self._sample_idx = state_dict["sample_idx"]
        self.set_epoch(self.epoch, self.copy_idx, self._sample_idx)
        self.protected = False

    def state_dict(self):
        return {"epoch": self.epoch, "copy_idx": self.copy_idx, "sample_idx": self._sample_idx}

    # dump iterable, deprecated
    def dump_iter(self):
        return self.iterable

    # load external iterable, deprecated
    def load_iter(self, external_iter):
        self.iter_is_null = False
        self.iterable = external_iter

    # truncate iterable, deprecated
    def truncate_iter(self, n):
        if self.iter_is_null:
            self.num_samples = self._num_samples_
            total_iterable = self.__real_iter__()
        else:
            total_iterable = self.iterable
        self.iter_is_null = False
        if self.batch_mode != "dynamical":
            n = self.batch_size * n
        piece = split_with_n(total_iterable, n)
        self.iterable = piece
        self.num_samples = n
        self.truncate_mode = True
        return total_iterable

    # reset iterable, deprecated
    def reset_iter(self):
        self.iter_is_null = True
        self.truncate_mode = False
        if self.batch_mode != "dynamical":
            self.num_samples = self._num_samples_


if __name__ == "__main__":
    from data.base.dataset import MyDataset

    max_tokens = 4096
    dataset = MyDataset(1000, min_length=100, max_length=max_tokens)

    similar_type = "dummy_length"
    similar_type = None
    batch_mode = "dynamical"
    batch_mode = "fixed"
    sampler = LengthChunkSampler(
        batch_size=8,
        buffer_batch_group=2,
        shuffle=True,
        last_samples="pad",
        copies=1,
        seed=42,
        num_replicas=2,
        rank=0,
        limited_type=None,
        similar_type=similar_type,
        length_range=(0, -1),
        phase="train",
        bucket_batch_volume=32,
        bucket_boundaries=None,
        num_buckets=6,
        bucket_padding_noise=0.0,
        batch_mode=batch_mode,
        acc_coeffs=((1.0, 1.0),),
        max_tokens=max_tokens,
        max_samples=64,
        descending=False,
        partition_record_path=None,
    )
    sampler.finalize_dataset(dataset, dataset.indices)
    sampler.set_epoch(0)
    for sample in sampler:
        print("==" * 10)
        batch_tokens = dataset.dummy_lengths[sample[0]].max() * len(sample[0])
        print(f"Correct {batch_tokens <= max_tokens},  batch_tokens: {batch_tokens} vs max_tokens: {max_tokens}")
        print(sample[0])
        if len(sample) > 1:
            batch_tokens = dataset.dummy_lengths[sample[1]].max() * len(sample[1])
            print(f"Correct {batch_tokens <= max_tokens},  batch_tokens: {batch_tokens} vs max_tokens: {max_tokens}")
            print(sample[1])
        # print(sample)
