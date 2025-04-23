from abc import ABCMeta
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
from utils.xtype import concat_list, index_list, is_scalar

T_co = TypeVar("T_co", covariant=True)
Tensor = torch.Tensor


class AudioABCDataset(torch.utils.data.Dataset[T_co], metaclass=ABCMeta):

    def __len__(self):
        return self.dataset_len

    def _get_supplement_data_(self, values_dict, indice: List[int]):
        values_dict = dict()
        if self.carry_filename:
            values_dict["file_names"] = [
                f"{self.prefix_path}" + file_path
                for file_path in self._get_data_(indice, [self.file_path_key])[self.file_path_key]
            ]

        return values_dict

    def get_supplement_data(self, values_dict, indice: Union[int, List[int]]):
        if isinstance(indice, (list, tuple)):
            if self.sorted:
                indice = sorted(indice)
            origin_supplement = self._get_supplement_data_(values_dict, indice)
        else:
            supplement = self._get_supplement_data_(values_dict, [indice])
            origin_supplement = dict()
            for k, v in supplement.items():
                origin_supplement[k] = supplement[k][0]

        return origin_supplement

    def postprocess_items(self, values_dict: Dict[str, List[Any]], **kwargs) -> Dict[str, Any]:
        return values_dict

    def complete_data(self, values_dict: Dict[str, Any], indice: Union[int, List[int]]):
        values_dict.update(self.get_supplement_data(values_dict, indice))
        whole_values_dict = self.postprocess_items(values_dict)
        return whole_values_dict

    def __getitem__(self, indice: Union[int, List[int], List[List[int]]]):
        values_dict = self._get_data_(indice, self.read_keys)
        if self.stand_dict is not None:
            stand_value_dict = dict()
            for k, v in values_dict.items():
                if k in self.stand_dict:
                    stand_value_dict[self.stand_dict[k]] = v
                else:
                    stand_value_dict[k] = v
            values_dict = stand_value_dict
        if is_scalar(indice) or is_scalar(indice[0]):
            whole_values_dict = self.complete_data(values_dict, indice)
        else:
            whole_values_dict = []
            for values_dict_, indice_ in zip(values_dict, indice):
                whole_values_dict.append(self.complete_data(values_dict_, indice_))
        return whole_values_dict

    def _get_data_(self, indice, read_keys) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """In order to improve IO efficiency of HuggingfaceAudioDataset on HDFS, the kvreader
        fetches multiple batches (batch group) onces.
        Return is list of batches. IterableDataloader should pop back only one.
        There are two kinds of implemenets. If IterableDataloader is not modified,
        In order to avoid modifying dataloader code of pytorch, the kv Sampler pops
        multiple batches indices at once to help reader read data of multiple batches.
        but the trainer or fetcher of dataloader only fetches data of single batch at once.
        So the kv Sampler provides None to disable reader read consequential data in the
        following code.
        Args:
            index ([type]): [description]

        Returns:
            [type]: [description]
        """
        if is_scalar(indice) or is_scalar(indice[0]):
            if self.sorted:
                indice = sorted(indice)
            all_values_dict = self.get_item(indice)
            values_dict = {key: all_values_dict[key] for key in read_keys}
            return values_dict
        else:
            num_parts = [len(sub_indice) for sub_indice in indice]
            all_indice = concat_list(indice)
            if self.sorted:
                all_indice = sorted(all_indice)

            all_values_dict = self.get_item(all_indice)
            values_dict = {key: all_values_dict[key] for key in read_keys}
            values_dict_list = list()
            accum_num = 0
            for i, num_part in enumerate(num_parts):
                sub_value_dict = dict()
                for k, v in values_dict.items():
                    sub_value_dict[k] = v[accum_num : accum_num + num_part]
                values_dict_list.append(sub_value_dict)
                accum_num += num_part
            return values_dict_list

    def create_dummy_input(self, batch_size, bucket_boundary, device):
        dummy_input = torch.randint(0, 10000, (batch_size, bucket_boundary))
        dummy_length = torch.randint(bucket_boundary // 2, bucket_boundary, (batch_size,))
        dummy_input = dummy_input.to(device)
        return dummy_input, dummy_length

    def filter_indices_by_size(
        self,
        dataset_indices,
        length_type: Optional[str] = None,
        length_range: Tuple[int, int] = (0, -1),
        descending: bool = False,
    ):
        if length_type is not None and length_type in self.info_dict:
            lengths = index_list(self.info_dict[length_type], dataset_indices)
            sorted_lengths, sorted_indices = torch.sort(torch.LongTensor(lengths))
            min_length, max_length = length_range[0], length_range[1]
            if max_length > 0:
                assert max_length > min_length, " ".join(
                    "The right boundary should be greater than the left one,",
                    f"but here max_length {max_length} <= min_length {min_length}",
                )
            if min_length > 0:
                start_idx = (sorted_lengths < min_length).sum()
            else:
                start_idx = 0
            if max_length > 0:
                end_idx = (sorted_lengths <= max_length).sum()
            else:
                end_idx = len(sorted_lengths) + 1

            tensor_dataset_indices = torch.LongTensor(dataset_indices)
            tensor_dataset_lengths = torch.LongTensor(lengths)
            if descending:
                sorted_dataset_lengths = torch.flip(tensor_dataset_lengths[sorted_indices[start_idx:end_idx]], dims=[0])
                sorted_dataset_indices = torch.flip(tensor_dataset_indices[sorted_indices[start_idx:end_idx]], dims=[0])
            else:
                sorted_dataset_lengths = tensor_dataset_lengths[sorted_indices[start_idx:end_idx]]
                sorted_dataset_indices = tensor_dataset_indices[sorted_indices[start_idx:end_idx]]
            self.enable_similar_length = False
        else:
            self.enable_similar_length = False
            sorted_dataset_lengths = None
            sorted_dataset_indices = torch.LongTensor(dataset_indices)
        return sorted_dataset_lengths, sorted_dataset_indices

    def filter_and_assign_dataset(
        self,
        phases: Union[str, Tuple[str, ...]],
        phases_ratios: Tuple[float, ...],
        num_classes_limit: int = -1,
        range_per_class: Tuple[int, int] = (0, -1),
        length_type: Optional[str] = None,
        length_range: Tuple[int, int] = (0, -1),
        descending: bool = False,
    ):
        if self.assigned:
            return
        if phases is None:
            phase = "default"
            self.phases_indice_dict[phase] = torch.LongTensor(list(range(self.dataset_len)))
        else:
            if isinstance(phases, str):
                phases = [phases]
                phases_ratios = []
            assert isinstance(phases, (tuple, list)) and isinstance(
                phases_ratios, (tuple, list)
            ), "both of `phases` and `phases_ratios` should be `Iterators`"
            assert (
                len(phases) == len(phases_ratios) + 1
            ), f"the number of `phases_ratios` {phases_ratios} should 1 less than that of `phases`: {phases}"
            assert len(phases_ratios) == 0 or (
                sum(phases_ratios) <= 1.0 and min(phases_ratios) >= 0.0
            ), "`phases_ratios` should be all positive and its sum should be not greater than 1."
            num_phase = len(phases)

            if num_phase == 1 and range_per_class == (0, -1) and num_classes_limit == -1:
                phase = phases[0]
                dataset_indices = torch.LongTensor(list(range(self.dataset_len)))
                _, filter_dataset_indices = self.filter_indices_by_size(
                    dataset_indices=dataset_indices,
                    length_type=length_type,
                    length_range=length_range,
                    descending=descending,
                )
                sorted_indices, _ = torch.sort(filter_dataset_indices)
                self.phases_indice_dict[phase] = sorted_indices
            else:
                classes_len_dict = self.dataset_len_dict
                accum_phase_len = [
                    0,
                ] * len(classes_len_dict)
                train_valid_speakers_names = []
                train_valid_speakers_lengths = []
                limited_num = 0
                for i, phase in enumerate(phases):
                    accum_dataset_len = 0
                    map_dataset_indice = torch.zeros(0).long()
                    valid_classess_names = []
                    for j, (class_name, class_num) in enumerate(classes_len_dict.items()):
                        if (
                            num_classes_limit is not None
                            and num_classes_limit > 0  # noqa E503
                            and len(valid_classess_names) >= num_classes_limit  # noqa E503
                        ):
                            break
                        range_per_class = (0, -1) if range_per_class is None else range_per_class
                        min_num, max_num = range_per_class[0], range_per_class[1]
                        if max_num > 0:
                            assert max_num > min_num, " ".join(
                                "The right boundary should be greater than the left one,",
                                f"but here max_length {max_num} <= min_length {min_num}",
                            )

                        if class_num >= min_num:
                            dataset_start = accum_dataset_len + accum_phase_len[j]
                            if max_num > 0:
                                limited_num = min(class_num, max_num)
                                if i < num_phase - 1:
                                    dataset_len_upper = int(round(phases_ratios[i] * limited_num))
                                else:
                                    dataset_len_upper = int(class_num - accum_phase_len[j])
                            else:
                                if "train" in phases:
                                    dataset_len_upper = 0
                                else:
                                    dataset_len_upper = class_num

                            if phase == "train":
                                assert i == 0, "Train phase should be created firstly"
                                map_data_indice = list(range(dataset_start, accum_dataset_len + class_num))
                                # sampler will selected the dataset indices
                                _, filter_dataset_indices = self.filter_indices_by_size(
                                    map_data_indice,
                                    length_type=length_type,
                                    length_range=length_range,
                                    descending=descending,
                                )
                                filtered_dataset_len = len(filter_dataset_indices)
                                aux_dataset_len_max = int(round(phases_ratios[i] * filtered_dataset_len))
                                aux_dataset_len_min = int(round(phases_ratios[i] * min_num))
                                if aux_dataset_len_max > aux_dataset_len_min:
                                    if (aux_dataset_len_max < dataset_len_upper) or (dataset_len_upper == 0):
                                        dataset_len = aux_dataset_len_max
                                    else:
                                        dataset_len = dataset_len_upper
                                else:
                                    dataset_len = dataset_len_upper
                                if filtered_dataset_len > dataset_len:
                                    sorted_indices, _ = torch.sort(filter_dataset_indices)
                                    map_dataset_indice = torch.cat(
                                        (map_dataset_indice, sorted_indices[:dataset_len]), dim=0
                                    )
                                    dataset_offset = sorted_indices[dataset_len - 1] - dataset_start
                                    valid_classess_names.append(class_name)
                                elif filtered_dataset_len == dataset_len and dataset_len > 0:
                                    sorted_indices, _ = torch.sort(filter_dataset_indices)
                                    map_dataset_indice = torch.cat((map_dataset_indice, sorted_indices), dim=0)
                                    dataset_offset = sorted_indices[-1] - dataset_start
                                    valid_classess_names.append(class_name)
                                else:
                                    sorted_indices = None
                                    dataset_offset = 0
                                train_valid_speakers_lengths.append(filtered_dataset_len)
                            else:
                                if (class_name in train_valid_speakers_names) or ("train" not in phases):
                                    if dataset_len_upper == 0:
                                        if i < num_phase - 1:
                                            dataset_len = int(round(phases_ratios[i] * train_valid_speakers_lengths[j]))
                                        else:
                                            dataset_len = int(class_num - accum_phase_len[j])
                                    else:
                                        dataset_len = dataset_len_upper
                                    map_data_indice = list(range(dataset_start, dataset_start + dataset_len))
                                    if len(map_data_indice) > 0:
                                        map_dataset_indice = torch.cat((map_dataset_indice, map_data_indice), dim=0)
                                        dataset_offset = len(map_data_indice)
                                        valid_classess_names.append(class_name)
                                    else:
                                        dataset_offset = 0
                                else:
                                    dataset_offset = 0
                        else:
                            dataset_len = 0
                            dataset_offset = 0
                        accum_dataset_len += class_num
                        accum_phase_len[j] += dataset_offset
                    self.logger.info(
                        " ".join(
                            f"{len(valid_classess_names)} valid classess will be used at {phase},",
                            f"classess details: {valid_classess_names}.",
                        ),
                        flush_right_now=False,
                    )
                    if phase == "train":
                        train_valid_speakers_names = valid_classess_names
                    self.phases_indice_dict[phase] = map_dataset_indice

    def sort_dataset_indice(self, dataset_indice, length_type, descending: bool = False):
        lengths = index_list(self.dataset_len_dict[length_type], dataset_indice)
        _, sorted_indices = torch.sort(torch.LongTensor(lengths))
        tensor_dataset_indices = torch.LongTensor(dataset_indice)
        dataset_indices = tensor_dataset_indices[sorted_indices]
        if descending:
            dataset_indices = torch.flip(dataset_indices, dims=[0])
        return dataset_indices

    def sort_dataset(self, subset: str, length_type: Optional[str] = None, descending: bool = False):
        if len(self.phases_indice_dict) == 0:
            raise RuntimeError(
                "HuggingfaceAudioDataset is not assigned to subset, call `filter_and_assign_indices` first"
            )
        if subset not in self.phases_indice_dict:
            raise KeyError(f"Dataset not loaded: {subset}")
        if length_type is not None and length_type not in self.dataset_len_dict:
            self.logger.info("The dataset is not buckted by the length.", flush_right_now=False)
        else:
            dataset_indice = self.phases_indice_dict[subset]
            sort_dataset_indice = self.sort_dataset_indice(dataset_indice, length_type, descending)
            self.phases_indice_dict[subset] = sort_dataset_indice

    def filter_assign_and_sort_dataset(
        self,
        phases: Tuple[str, ...],
        phases_ratios: Tuple[float, ...],
        num_classes_limit: int = -1,
        range_per_class: Tuple[int, int] = (0, -1),
        limited_length_type: Optional[str] = None,
        length_range: Tuple[int, int] = (0, -1),
        sorted_subsets: Optional[Tuple[str]] = None,
        sorted_length_type: Optional[str] = None,
        descending: bool = False,
    ):
        self.filter_and_assign_dataset(
            phases, phases_ratios, num_classes_limit, range_per_class, limited_length_type, length_range, descending
        )
        if sorted_subsets is not None:
            for subset in sorted_subsets:
                self.sort_dataset(subset, sorted_length_type, descending)
