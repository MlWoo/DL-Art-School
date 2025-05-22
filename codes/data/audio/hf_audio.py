import math
import os  # noqa: F401
import os.path as osp
import time

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from utils.io import mfs
from utils.logging_utils import get_root_logger
from utils.options import opt_get

from data.audio.io import load_audio
from data.audio.xdata_abc import AudioABCDataset
from data.builder import DATASETS

datasets.config.DEFAULT_MAX_BATCH_SIZE = 1000000000


def get_data_length(data):
    data_length = []
    if isinstance(data[0], (list, tuple)):
        for _data in data:
            data_length.append([x.shape[-1] for x in _data])
    else:
        data_length = [x.shape[-1] for x in data]
    return data_length


@DATASETS.register_module()
class HuggingfaceAudioDataset(AudioABCDataset):
    def __init__(self, opt):
        super().__init__()
        self.sr = opt_get(opt, ["sr"], 16000)
        self.channels = opt_get(opt, ["channels"], 1)
        self.min_duration = opt_get(opt, ["min_duration"], None)
        self.max_duration = opt_get(opt, ["max_duration"], None)
        self.dataset_prefix = opt_get(opt, ["dataset_prefix"], None)
        store_type = opt_get(opt, ["store_type"], "json")
        cache_dir = opt_get(opt, ["cache_dir"], "/mnt/user/wumenglin/cache/huggingface")
        read_keys = opt_get(opt, ["read_keys"], ["audio"])

        phase = opt_get(opt, ["phase"], None)
        self.phase = phase

        start_time = time.time()
        if store_type == "freeze":
            data_file = opt_get(opt, ["data_file"])
            dataset = load_from_disk(dataset_path=data_file)
        elif store_type == "arrow":
            data_file = opt_get(opt, ["data_file"])
            dataset_dict = load_dataset(data_file, num_proc=64)
            dataset = dataset_dict["train"]
        else:
            data_file = opt_get(opt, ["data_file"])
            if data_file:
                data_files = {f"{phase}": data_file}
            else:
                path = opt_get(opt, ["path"])
                data_files = {f"{phase}": mfs.glob(f"{path}/*{phase}*{store_type}*")}

            dataset_dict = load_dataset(store_type, data_files=data_files, cache_dir=cache_dir, num_proc=64)
            dataset = dataset_dict[phase]

        self.name = opt_get(opt, ["name"], None)
        end_time = time.time()
        self.logger = get_root_logger()
        self.logger.info(f"load {self.name} {phase} dataset time: {end_time - start_time}")

        dataset.set_format("np")
        # filtered min duration
        durations = dataset["duration"] if "duration" in dataset.features else dataset["duration_ms"] / 1000
        min_mask = durations > self.min_duration + (1.0 / self.sr)

        if self.max_duration is None or self.max_duration < 0.0:
            mask = min_mask
        else:
            max_mask = durations < self.max_duration - (1.0 / self.sr)
            mask = np.logical_and(min_mask, max_mask)

        self.files = np.where(mask)[0]
        self.durations = durations[mask]

        frame_lengths = []
        hop_length = opt_get(opt, ["audio_process", "hop_length"], 160)

        frame_lengths = np.ceil(self.durations * self.sr / hop_length).astype(np.int32)
        self.feature_dim = opt_get(opt, ["audio_process", "mel_bins"], 128)

        self.sample_duration = opt_get(opt, ["sample_duration"], None)

        sample_frame_length = (
            None
            if self.sample_duration is None
            else np.ceil(self.sample_duration * self.sr / hop_length).astype(np.int32)
        )

        self.info_dict = dict(
            frame_length=frame_lengths,
            sample_frame_length=np.clip(frame_lengths, 0, sample_frame_length),
        )
        import pdb

        pdb.set_trace()

        self.dataset = dataset
        self.dataset_len = len(self.files)
        self.phases_indice_dict = self.create_datasets(
            phase,
            self.dataset_len,
            phases_ratios=opt_get(opt, ["phases_ratios"], [0.985]),
            seed=opt_get(opt, ["seed"], 1234),
        )

        self.read_keys = read_keys
        self.sorted = False
        self.stand_dict = None
        self.carry_filename = False

    def get_audio_chunk(self, indice, item=None, offsets=None):
        meta_list = self.dataset[indice]

        audio_ready = False
        if "path" in meta_list:
            paths = meta_list["path"]
        elif "audio_path" in meta_list:
            paths = meta_list["audio_path"]
        elif "audio_filepath" in meta_list:
            paths = meta_list["audio_filepath"]
        elif "audio" in meta_list:
            audio_ready = True
        else:
            raise ValueError(f"No path found in meta_list: {meta_list}")

        if audio_ready:
            data_group = []
            path_group = []
            audio_lengths = []
            begin_time_group = []
            duration_group = []

            if offsets is None:
                offsets = [0] * len(meta_list["audio"])

            for i, (audio_meta, offset) in enumerate(zip(meta_list["audio"], offsets)):
                path = audio_meta["path"]
                audio = audio_meta["array"]
                if audio.ndim == 1:
                    audio = audio.reshape(1, -1)
                sr = audio_meta["sampling_rate"]

                audio_length = audio.shape[-1]
                offset = min(offset, 0)

                if audio_length > self.sample_duration * sr + offset:
                    offset = offset + np.random.rand() * (audio_length - self.sample_duration * sr - offset)
                    duration = self.sample_duration
                    sample_length = int(self.sample_duration * sr)
                else:
                    sample_length = audio_length - offset
                    duration = round(sample_length / sr, 2)

                try:
                    data = audio[:, int(offset) : int(offset + sample_length)].reshape(1, -1)
                    assert data.shape == (
                        self.channels,
                        sample_length,
                    ), f"Expected {(self.channels, sample_length)}, got {data.shape}"
                except:  # noqa: E722
                    continue

                data_group.append(data)
                path_group.append(path)
                audio_lengths.append(sample_length)
                begin_time_group.append(round(offset / sr, 2))
                duration_group.append(duration)
            return dict(
                audio=data_group,
                path=path_group,
                audio_lengths=audio_lengths,
                begin_time=begin_time_group,
                duration=duration_group,
            )
        else:
            begin_times = meta_list["begin_time"]
            durations = meta_list["duration"]

            data_group = []
            path_group = []
            audio_lengths = []
            begin_time_group = []
            duration_group = []
            for path, begin_time, duration in zip(paths, begin_times, durations):
                if duration > self.sample_duration:
                    begin_time = begin_time + np.random.rand() * (duration - self.sample_duration)
                    duration = self.sample_duration
                audio_offset = int(begin_time * self.sr)
                duration_sample = int(duration * self.sr)
                if self.dataset_prefix is not None:
                    path = osp.join(self.dataset_prefix, path)
                data, sr = load_audio(
                    path, sr=self.sr, offset=audio_offset, resample=True, duration=duration_sample, check_duration=False
                )
                if data is None:
                    continue
                data = data[:, : int((self.sample_duration * self.sr))]
                data = (data / max(0.1, np.abs(data).max())) * 0.9
                data_group.append(data)
                path_group.append(path)
                audio_lengths.append(data.shape[1])
                begin_time_group.append(begin_time)
                duration_group.append(duration)
            return dict(
                audio=data_group,
                path=path_group,
                audio_lengths=audio_lengths,
                begin_time=begin_time_group,
                duration=duration_group,
            )

    def __len__(self):
        return self.dataset_len

    def get_index_offset(self, indice):
        # For a given dataset item and shift, return audio index and offset within audio
        file_indice = []
        for index in indice:
            file_indice.append(self.files[index])
        return file_indice

    def get_item(self, item):
        index = self.get_index_offset(item)
        return self.get_audio_chunk(index, item)


@DATASETS.register_module()
class HuggingfaceAudioClipDataset(AudioABCDataset):
    def __init__(self, opt):
        super().__init__()
        self.sr = opt_get(opt, ["sr"], 16000)
        self.channels = opt_get(opt, ["channels"], 1)
        self.sample_length = opt_get(opt, ["sample_length"], int(16000 * 2))
        self.min_duration = opt_get(opt, ["min_duration"], None) or math.ceil(self.sample_length / self.sr)
        assert self.sample_length / self.sr <= self.min_duration, (
            f"Sample length {self.sample_length} per sr {self.sr} ({self.sample_length / self.sr:.2f})"
            + f" should be shorter than min duration {self.min_duration}"
        )
        self.max_duration = opt_get(opt, ["max_duration"], None)
        self.aug_shift = opt_get(opt, ["aug_shift"], False)
        self.dataset_prefix = opt_get(opt, ["dataset_prefix"], None)

        self.name = opt_get(opt, ["name"], None)
        store_type = opt_get(opt, ["store_type"], "json")
        cache_dir = opt_get(opt, ["cache_dir"], "/home/wumenglin/.cache/cache/huggingface")

        phase = opt_get(opt, ["phase"], None)
        self.phase = phase

        data_file = opt_get(opt, ["data_file"])
        start_time = time.time()
        if store_type == "freeze":
            dataset = load_from_disk(dataset_path=data_file)
        elif store_type == "arrow":
            dataset_dict = load_dataset(data_file, num_proc=64)
            dataset = dataset_dict["train"]
        else:
            if data_file:
                data_files = {f"{phase}": data_file}
            else:
                path = opt_get(opt, ["path"])
                data_files = {f"{phase}": mfs.glob(f"{path}/*{phase}*{store_type}*")}
            dataset_dict = load_dataset(store_type, data_files=data_files, cache_dir=cache_dir, num_proc=128)
            dataset = dataset_dict[phase]

        end_time = time.time()
        self.logger = get_root_logger()
        self.logger.info(f"load {self.name} {phase} dataset time: {end_time - start_time}")

        dataset.set_format("np")
        # filtered min duration
        durations = dataset["duration"] if "duration" in dataset.features else dataset["duration_ms"] / 1000
        min_mask = durations > self.min_duration + (1.0 / self.sr)

        if self.max_duration is None or self.max_duration < 0 or self.aug_shift:
            mask = min_mask
        else:
            max_mask = durations < self.max_duration - (1.0 / self.sr)
            mask = np.logical_and(min_mask, max_mask)

        self.files = np.where(mask)[0]
        self.durations = durations[mask]
        self.dataset = dataset

        durations_sample = (self.durations * self.sr).astype(np.int32)
        self.cumsum = np.cumsum(durations_sample)

        self.dataset_len = int(np.floor(self.cumsum[-1] / self.sample_length))  # len(dataset)

        self.phases_indice_dict = self.create_datasets(
            phase,
            self.dataset_len,
            phases_ratios=opt_get(opt, ["phases_ratios"], [0.985]),
            seed=opt_get(opt, ["seed"], 1234),
        )

        self.read_keys = ["audio", "audio_lengths"]
        self.sorted = True
        self.stand_dict = None
        self.carry_filename = False
        self.info_dict = dict()

    def get_audio_chunk(self, indice, offsets=None):
        meta_list = self.dataset[indice]

        audio_ready = False
        if "path" in meta_list:
            paths = meta_list["path"]
        elif "audio_path" in meta_list:
            paths = meta_list["audio_path"]
        elif "audio_filepath" in meta_list:
            paths = meta_list["audio_filepath"]
        elif "audio" in meta_list:
            audio_ready = True
        else:
            raise ValueError(f"No path found in meta_list: {meta_list}")
        data_group = []
        path_group = []
        audio_lengths = []
        begin_time_group = []
        duration_group = []
        if audio_ready:
            if offsets is None:
                offsets = [0] * len(paths)
            for i, (audio_meta, offset) in enumerate(zip(meta_list["audio"], offsets)):
                path = audio_meta["path"]
                audio = audio_meta["array"]
                sr = audio_meta["sampling_rate"]

                offset = min(offset, 0)
                end_offset = offset + self.sample_length
                if end_offset > len(audio):
                    offset = len(audio) - self.sample_length
                end_offset = offset + self.sample_length

                try:
                    data = audio[offset:end_offset].reshape(1, -1)
                    assert data.shape == (
                        self.channels,
                        self.sample_length,
                    ), f"Expected {(self.channels, self.sample_length)}, got {data.shape}"
                except:  # noqa: E722
                    continue

                # audio_length = audio.shape[1]
                # if audio_length > self.max_duration * sr:
                #     offset = np.random.rand() * (audio_length - self.max_duration * sr)
                #     duration = self.max_duration
                # else:
                #     offset = 0
                #     duration = audio_length / sr

                # data = audio[:, int(offset): int(offset + duration * sr)]

                data_group.append(data)
                path_group.append(path)
                audio_lengths.append(data.shape[1])
        else:
            begin_times = meta_list["begin_time"]
            durations = meta_list["duration"]
            for path, begin_time, duration in zip(paths, begin_times, durations):
                if duration > self.max_duration:
                    begin_time = begin_time + np.random.rand() * (duration - self.max_duration)
                    duration = self.max_duration
                audio_offset = int(begin_time * self.sr)
                duration_sample = int(duration * self.sr)
                if self.dataset_prefix is not None:
                    path = osp.join(self.dataset_prefix, path)
                data, sr = load_audio(
                    path, sr=self.sr, offset=audio_offset, resample=True, duration=duration_sample, check_duration=False
                )
                if data is None:
                    continue
                data = data[:, : int((self.max_duration * self.sr))]
                data = (data / max(0.1, np.abs(data).max())) * 0.9
                data_group.append(data)
                path_group.append(path)
                audio_lengths.append(data.shape[1])
                begin_time_group.append(begin_time)
                duration_group.append(duration)

        return dict(
            audio=data_group,
            path=path_group,
            audio_lengths=audio_lengths,
            begin_time=begin_time_group,
            duration=duration_group,
        )

    def get_index_offset(self, indice):
        # For a given dataset item and shift, return audio index and offset within audio
        half_interval = self.sample_length // 2
        shift = np.random.randint(-half_interval, half_interval) if self.aug_shift else 0
        file_indice = []
        offsets = []
        for item in indice:
            offset = item * self.sample_length + shift  # Note we centred shifts, so adding now
            midpoint = offset + half_interval
            assert 0 <= midpoint < self.cumsum[-1], f"Midpoint {midpoint} of item beyond total length {self.cumsum[-1]}"
            index = np.searchsorted(self.cumsum, midpoint)  # index <-> midpoint of interval lies in this audio
            start, end = self.cumsum[index - 1] if index > 0 else 0.0, self.cumsum[
                index
            ]  # start and end of current audio
            assert (
                start <= midpoint <= end
            ), f"Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}"
            if offset > end - self.sample_length:  # Going over audio
                offset = max(start, offset - half_interval)  # Now should fit
            elif offset < start:  # Going under audio
                offset = min(end - self.sample_length, offset + half_interval)  # Now should fit
            assert (
                start <= offset <= end - self.sample_length
            ), f"Offset {offset} not in [{start}, {end - self.sample_length}]. End: {end}, SL: {self.sample_length}, Index: {index}"  # noqa: E501
            offset = offset - start
            file_indice.append(self.files[index])
            offsets.append(offset)

        return file_indice, offsets


if __name__ == "__main__":
    opt = dict(audio_dataset="/mnt/shared-storage/tenant/user/wml/hf_cache/audio-half-2sec/", spec_fn="canonical")
    """
    dataset = FilesAudioHfDataset(opt)
    dataset_len = len(dataset)
    indices = torch.randint(0, dataset_len, (16, )).tolist()
    start_time = time.time()
    data = dataset[indices]
    print(time.time() - start_time)
    """
    mode = "ja_hf_dataset"
    dave_params = {
        "name": "librittsR_train",
        "n_workers": 25,
        "batch_size": 1796,
        "mode": f"{mode}",
        "phase": "train",
        "audio_dataset": "/mnt/shared-storage/tenant/user/wml/hf_cache/audio-half-2sec/",
        "cache_dir": "./cache/huggingface2",
        "buffer_batch_group": 2,
        "bucket_batch_volume": 32,
        "similar_type": None,
        "last_samples": "pad",
        "shuffle": True,
        "limited_length_type": "frame_length",
        "length_range": [0, -1],
        "copies": 1,
        "persistent_workers": True,
        "data_cfg": {"audio": {"padding_val": -0.0}},
        "apply_half": True,
        "half_type": "bf16",
        "pool_threads": 0,
        "spec_fn": "canonical",
        "buffer_background": False,
        "buffer_background_size": 4,
        "process_background": False,
        "process_device": -1,
        "device": -1,
    }
    # from multiprocess import set_start_method
    # set_start_method("spawn")
    from data import create_dataloader, create_dataset

    ds, collate_fn = create_dataset(dave_params, return_collate=True)
    dl = create_dataloader(ds, dave_params, collate_fn=collate_fn)
    dl.set_epoch(1)
    # print(dl.generator.sampler)
    i = 0
    last_time = time.time()
    mel_max = -999999999
    mel_min = 999999999
    mel_means = []
    mel_stds = []
    mel_vars = []
    end_cnt = 560
    end_in_adv = True
    total_time = 0
    for b in tqdm(dl):
        cur_time = time.time()
        if i % 25 == 0 and i > 0:
            print(i, total_time / (i))
        interval = cur_time - last_time
        total_time += interval

        mel = b["mel"]
        mel_means.append(mel.mean((0, 2)).cpu())
        mel_stds.append(mel.std((0, 2)).cpu())
        mel_vars.append(mel.var((0, 2)).cpu())
        mel_max = max(mel.max().item(), mel_max)
        mel_min = min(mel.min().item(), mel_min)

        # time.sleep(1.85)
        i += 1
        if end_in_adv and i > end_cnt:
            break
        last_time = cur_time

    mel_means = torch.stack(mel_means).mean(0)
    mel_stds = torch.stack(mel_stds).mean(0)
    mel_vars = torch.stack(mel_vars).mean(0)
    torch.save((mel_means, mel_max, mel_min, mel_stds, mel_vars), "mel_norms-new-1m-128.pth")
