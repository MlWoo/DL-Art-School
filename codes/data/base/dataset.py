import bisect
import copy
from typing import List

import torch
from torch.utils.data import Dataset
from utils.logging_utils import get_root_logger


def sanity_check_merge_dict_item(dict1, dict2):
    for k2, v2 in dict2.items():
        if k2 in dict1:
            assert isinstance(dict1[k2], list) and isinstance(v2, list)
            dict1[k2] = dict1[k2] + v2
        else:
            raise ValueError(f"The item {k2} does not occur in different datasets")


class DynamicDataset(Dataset):
    def __init__(self, size, min_length=100, max_length=1023):

        self.size = size
        self.dummy_lengths = torch.randint(min_length, max_length, (size,))
        self.info_dict = {"dummy_length": self.dummy_lengths}
        self.indices = torch.arange(size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.randint(0, 4096, (self.dummy_lengths[index],))

    def create_dummy_input(self, batch_size, bucket_boundary, device):
        dummy_input = torch.randint(0, 10000, (batch_size, bucket_boundary))
        dummy_length = torch.randint(bucket_boundary // 2, bucket_boundary, (batch_size,))
        dummy_input = dummy_input.to(device)
        return dummy_input, dummy_length


class ConcatDataset(Dataset):
    """A wrapper of concatenated dataset.
    samples as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.
    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            length = len(e)
            r.append(length + s)
            s += length
        return r

    def __init__(self, datasets):
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        for d in self.datasets:
            assert isinstance(d, Dataset), "ConcatDataset does only support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.logger = get_root_logger()
        self.dataset_len = 0
        self.cumulative_sizes = []
        self.info_dict = dict()
        for i, dataset in enumerate(datasets):
            if i == 0:
                self.info_dict = copy.deepcopy(dataset.info_dict)
            else:
                sanity_check_merge_dict_item(self.info_dict, dataset.info_dict)
            self.dataset_len += dataset.dataset_len
            self.cumulative_sizes.append(self.dataset_len)

        self.num_datasets = len(datasets)

    def __collect__(self, dataset_indices, sample_indices_indices):
        dataset_bucket = [[] for _ in range(self.num_datasets)]
        global_indices = []
        idx = 0
        for dataset_idx, sample_indices_idx in zip(dataset_indices, sample_indices_indices):
            global_indices.append((dataset_idx, len(dataset_bucket[dataset_idx])))
            dataset_bucket[dataset_idx].append(sample_indices_idx)
            idx += 1
        data_bucket = []
        for dataset_idx, dataset_sample_indices in enumerate(dataset_bucket):
            if len(dataset_sample_indices) > 0:
                data_bucket.append(self.datasets[dataset_idx][dataset_sample_indices])
            else:
                data_bucket.append([])
        data_dict = dict()
        for dataset_idx, data_bucket_idx in global_indices:
            for k, v in data_bucket[dataset_idx].items():
                if k in data_dict:
                    data_dict[k].append(v[data_bucket_idx])
                else:
                    data_dict[k] = [v[data_bucket_idx]]
        return data_dict

    def __getitem__(self, index):
        if isinstance(index[0], (list, tuple)):
            total_index = []
            num_parts = [len(sub_indice) for sub_indice in index]
            for sub_index in index:
                total_index.extend(sub_index)
            dataset_indices = []
            sample_indices_indices = []
            for idx in total_index:
                if idx < 0:
                    if -idx > len(self):
                        raise ValueError("absolute value of index should not exceed dataset length")
                    idx = len(self) + idx
                dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
                dataset_indices.append(dataset_idx)
                if dataset_idx == 0:
                    sample_indices_idx = idx
                else:
                    sample_indices_idx = idx - self.cumulative_sizes[dataset_idx - 1]
                sample_indices_indices.append(sample_indices_idx)
            values_dict = self.__collect__(dataset_indices, sample_indices_indices)
            values_dict_list = list()
            accum_num = 0
            for i, num_part in enumerate(num_parts):
                sub_value_dict = dict()
                for k, v in values_dict.items():
                    sub_value_dict[k] = v[accum_num : accum_num + num_part]
                values_dict_list.append(sub_value_dict)
                accum_num += num_part
            return values_dict_list
        else:
            dataset_indices = []
            sample_indices_indices = []
            for idx in index:
                if idx < 0:
                    if -idx > len(self):
                        raise ValueError("absolute value of index should not exceed dataset length")
                    idx = len(self) + idx
                dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
                dataset_indices.append(dataset_idx)
                if dataset_idx == 0:
                    sample_indices_idx = idx
                else:
                    sample_indices_idx = idx - self.cumulative_sizes[dataset_idx - 1]
                if sample_indices_idx.item() >= self.datasets[dataset_idx].dataset_len:
                    print(idx, dataset_idx, sample_indices_idx)
                sample_indices_indices.append(sample_indices_idx)
            data_dict = self.__collect__(dataset_indices, sample_indices_indices)

            return data_dict
