import numpy as np
import torch
import torch.nn.functional as F
from utils.distributed import apply_to_sample

from data.builder import COLLATIONS
from data.sampler.util import find_surrounding_elements

Tensor = torch.Tensor


@COLLATIONS.register_module()
class Collator(object):
    def __init__(self, opt):
        data_cfg = opt.get("data_cfg", None)
        batch_size = opt.get("batch_size", 16)
        buffer_batch_group = opt.get("buffer_batch_group", 8)
        apply_half = opt.get("apply_half", False)
        half_type = opt.get("half_type", "fp16")
        half_list = opt.get("half_list", None)
        self.batch_size = batch_size
        self.buffer_batch_group = buffer_batch_group
        self.data_cfg = data_cfg
        self.apply_half = apply_half
        self.half_type = half_type
        self.half_list = half_list
        self.inter_post = True
        self.ignored_keys = []
        self.bucket_boundaries_batch_size_map = {}

    def _postprocess_items_(self, batch_dict):
        return batch_dict

    def _postprocess_tensor_(self, batch_dict, force_inter=False):
        return batch_dict

    def _extra_info_(self, batch_dict):
        return batch_dict

    def _decode_(self, key, raw_data):
        if key in self.data_cfg:
            return raw_data
        else:
            if key not in self.ignored_keys:
                print(f"ignore the key {key}")
                self.ignored_keys.append(key)
            return None

    def _fp_convert_sample(self, sample):
        def apply_half(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.half)
            return t

        def apply_bfloat16(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.bfloat16)
            return t

        if self.half_type == "fp16":
            sample = apply_to_sample(apply_half, sample)
        elif self.half_type == "bf16":
            sample = apply_to_sample(apply_bfloat16, sample)

        return sample

    def _prepare_sample_(self, batch_dict):
        new_batch_dict = dict()
        if self.apply_half:
            if self.half_list is None:
                for k, v in batch_dict.items():
                    sample = self._fp_convert_sample(v)
                    new_batch_dict[k] = sample
            else:
                for k, v in batch_dict.items():
                    if k in self.half_list:
                        sample = self._fp_convert_sample(v)
                    else:
                        sample = v
                    new_batch_dict[k] = sample
        else:
            new_batch_dict = batch_dict
        return new_batch_dict

    def _collate_tensors_(self, data, key):
        result = []
        largest_dims = [0 for _ in range(data[0].dim())]
        for elem in data:
            result.append(elem)
            largest_dims = [
                max(current_largest, new_consideration)
                for current_largest, new_consideration in zip(largest_dims, elem.shape)
            ]
        # Now pad each tensor by the largest dimension.
        batch_size = len(result)
        for i in range(batch_size):
            padding_tuple = ()
            for d in range(len(largest_dims)):
                padding_needed = largest_dims[d] - result[i].shape[d]
                assert padding_needed >= 0
                if key in self.bucket_boundaries_batch_size_map:
                    assert i == 1, "padding_needed in second dimension should be 1"
                    original_size = result[i].shape[1]
                    _, right = find_surrounding_elements(
                        list(self.bucket_boundaries_batch_size_map[key].keys()), original_size
                    )
                    assert right is not None, "padding_needed is too large"
                    assert (
                        batch_size == self.bucket_boundaries_batch_size_map[key][right]
                    ), "batch_size is not equal to the bucket_boundaries_batch_size_map"
                    padding_tuple = right - result[i].shape[1]
                padding_tuple = (0, padding_needed) + padding_tuple
            try:
                constant_val = self.data_cfg[key]["padding_val"]
            except:  # noqa: E722
                constant_val = 0
            result[i] = F.pad(result[i], padding_tuple, value=constant_val)
        return torch.stack(result, dim=0)

    def set_bucketed_batch_size(self, key, bucket_boundaries_batch_size_map):
        self.bucket_boundaries_batch_size_map[key] = bucket_boundaries_batch_size_map

    def _collate_one_batch_(self, batch):
        collated = {}
        for key, data in batch.items():
            if data is None:
                continue
            if isinstance(data[0], int):
                collated[key] = torch.LongTensor(data)
            elif isinstance(data[0], float):
                collated[key] = torch.FloatTensor(data)
            elif isinstance(data[0], str):
                collated[key] = data
            else:
                if isinstance(data[0], (np.ndarray)) or isinstance(data[0], (torch.Tensor)):
                    try:
                        tensors = []
                        for d in data:
                            if isinstance(d, np.ndarray):
                                tensors.append(torch.from_numpy(d.copy()))
                            else:
                                tensors.append(d)
                    except:  # noqa: E722
                        import pdb

                        pdb.set_trace()
                else:
                    tensors = data
                if len(tensors[0].shape) > 0:
                    collated[key] = self._collate_tensors_(tensors, key)
                else:
                    try:
                        collated[key] = torch.stack(tensors)
                    except:  # noqa: E722
                        raise RuntimeError(f"{key} type is not compatitable.")
        return collated

    def __call__(self, compound_data):
        if isinstance(compound_data, (list,)):
            # It's a legacy that you should not pay much attention on it.
            # it handles with the old kv dataset which partitions a group of batches into a list of dicts.
            for data_dict in compound_data:
                for d_k, d_v in data_dict.items():
                    if d_v is None:
                        continue
                    if isinstance(d_v, (list, tuple)):
                        batch_group = True
                    else:
                        batch_group = False
                    break
                break
            # assert batch_group == self.buffer_batch_group > 0, f"{d_k} data are not orginized by list"
            if batch_group:
                value_batch_group = []
                for data_dict in compound_data:
                    value_batch = dict()
                    for d_k, d_v in data_dict.items():
                        batch_val = self._decode_(d_k, d_v)
                        if batch_val is not None:
                            value_batch[d_k] = batch_val
                    value_batch = self._postprocess_items_(value_batch)
                    collated_batch = self._collate_one_batch_(value_batch)
                    collated_batch = self._postprocess_tensor_(collated_batch)
                    collated_batch = self._prepare_sample_(collated_batch)
                    value_batch_group.append(collated_batch)
                return value_batch_group
            else:
                value_batch = dict()
                for data_dict in compound_data:
                    for d_k, d_v in data_dict.items():
                        val = self._decode_(d_k, [d_v])
                        if val is not None:
                            if d_k in value_batch:
                                value_batch[d_k] += val
                            else:
                                value_batch[d_k] = val
                value_batch = self._postprocess_items_(value_batch)
                collated_batch = self._collate_one_batch_(value_batch)
                collated_batch = self._postprocess_tensor_(collated_batch)
                collated_batch = self._prepare_sample_(collated_batch)
                return collated_batch
        else:
            # The current practice, the dataset does not need to partition a group batches.
            # The input is a dict with the key which represents the feature name and the value which is a ordered
            # list of raw value.
            assert isinstance(compound_data, dict), "data from dataset to collate should be a dict or a list of dict"
            # get the length of a group of batches.
            data_size = 1
            for _, val in compound_data.items():
                data_size = len(val)
                break

            # patition the group
            enc_value_batch_group = []
            for i in range(self.buffer_batch_group):
                if (i + 1) * self.batch_size > data_size:
                    if i == 0:
                        print("batch size is too large to the dataset")
                if i * self.batch_size >= data_size:
                    if i == 0:
                        raise RuntimeError("could not get any samplers in the dataset")
                    break

                value_batch = dict()
                # partition the value of the dict into batches respected to each feature
                for d_k, d_v in compound_data.items():
                    data = d_v[i * self.batch_size : (i + 1) * self.batch_size]
                    value_batch[d_k] = data
                enc_value_batch_group.append(value_batch)

            value_batch_group = []
            for enc_value_batch in enc_value_batch_group:
                value_batch = dict()
                for d_k, d_v in enc_value_batch.items():
                    batch_val = self._decode_(d_k, d_v)
                    if batch_val is not None:
                        value_batch[d_k] = batch_val
                    value_batch[d_k] = batch_val
                value_batch = self._postprocess_items_(value_batch)
                collated_batch = self._collate_one_batch_(value_batch)
                collated_batch = self._postprocess_tensor_(collated_batch)
                collated_batch = self._prepare_sample_(collated_batch)
                value_batch_group.append(collated_batch)
            return value_batch_group
