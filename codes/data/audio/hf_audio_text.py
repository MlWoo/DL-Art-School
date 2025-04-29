import os  # noqa: F401
import os.path as osp
import re
import time
from functools import partial
from typing import Any, Dict, List, TypeVar, Union

import numpy as np
import torch
import torchaudio
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.io import mfs
from utils.logging_utils import get_root_logger
from utils.options import opt_get

from data.audio.io import load_audio  # noqa F401
from data.audio.xdata_abc import AudioABCDataset
from data.builder import DATASETS

T_co = TypeVar("T_co", covariant=True)
Tensor = torch.Tensor


def rm_punctuation(eg, dataset_name):
    text = eg["text"]
    data_dict = {}
    if dataset_name == "GigaSpeech":
        data_dict["text"] = re.sub(r"<.*?>", "", text)
    else:
        data_dict["text"] = text
    return data_dict


# caching all derived data
def cache_text_processor(eg, audio_token, tokenizers_dict=None):
    def apply_chat_template(text, tokenizer):
        chat = [
            {
                "role": "user",
                "content": f"Transcribe the speech.<|startofspeech|>{audio_token * eg['speech_token_length']}<|endofspeech|>",  # noqa E501
            }
        ]
        chat.append({"role": "assistant", "content": text})
        tokens = tokenizer.apply_chat_template(chat)
        return tokens

    data_dict = {}
    if tokenizers_dict is not None:
        for key, tokenizer in tokenizers_dict.items():
            text = eg["text"]
            token = apply_chat_template(text, tokenizer)
            token_name = "text_token"
            token_dict = {f"{token_name}": token, f"{token_name}_length": len(token)}
            data_dict.update(**token_dict)

    return data_dict


def apply_chat_template(eg, tokenizer, audio_token):

    text = eg["text"]
    speech_token_length = len(eg["speech_token"])
    chat = [
        {
            "role": "user",
            "content": f"Transcribe the speech.<|startofspeech|>{audio_token * speech_token_length}<|endofspeech|>",
        }
    ]
    chat.append({"role": "assistant", "content": text})
    tokens = tokenizer.apply_chat_template(chat)
    return tokens


def add_duration_and_speech_token_length(eg, token_frame_rate):
    duration = round(eg["end_time"] - eg["begin_time"], 3)
    speech_token_length = int(round(duration * token_frame_rate, 3))
    return {"duration": duration, "speech_token_length": speech_token_length}


def process_speech_func(
    eg,
    token_frame_rate,
    dataset_name,
    rm_punc=False,
):
    if "duration" in eg:
        duration = round(eg["duration"], 3)
    elif "begin_time" in eg and "end_time" in eg:
        duration = round(eg["end_time"] - eg["begin_time"], 3)
    else:
        raise ValueError("No duration or begin_time and end_time in dataset")

    if "speech_token" in eg:
        speech_token_length = len(eg["speech_token"])
    else:
        speech_token_length = int(round(duration * token_frame_rate, 3))

    text = eg["text"]
    if rm_punc and dataset_name == "GigaSpeech":
        text = re.sub(r"<.*?>", "", text)

    return {"duration": duration, "speech_token_length": speech_token_length, "text": text}


@DATASETS.register_module()
class HuggingfaceMinmoASRDataset(AudioABCDataset):

    classes_dict: Union[Dict[str, int], None]

    def __init__(
        self,
        opt,
    ):
        cache_dir = opt_get(opt, ["cache_dir"], None)
        data_file = opt_get(opt, ["data_file"])
        self.tokenizers = opt_get(opt, ["tokenizers"], None)
        self.cache_text = opt_get(opt, ["cache_text"], True)
        self.read_keys = opt_get(opt, ["read_keys"], None)
        self.text_key = opt_get(opt, ["text_key"], "text")
        self.text_token_key = opt_get(opt, ["text_token_key"], "text_token")
        self.min_duration = opt_get(opt, ["min_duration"], None)
        self.max_duration = opt_get(opt, ["max_duration"], None)
        self.sample_duration = opt_get(opt, ["sample_duration"], None)
        self.text_type = opt_get(opt, ["text_type"], 0)
        self.speech_type = opt_get(opt, ["speech_type"], 1)
        self.asr_text_type = opt_get(opt, ["asr_text_type"], 2)
        self.token_frame_rate = opt_get(opt, ["token_frame_rate"], 25)
        self.name = opt_get(opt, ["name"], None)
        store_type = opt_get(opt, ["store_type"], "json")

        assert self.text_key in self.read_keys or self.text_token_key in self.read_keys

        phase = opt_get(opt, ["phase"], None)
        self.phase = phase

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

            dataset_dict = load_dataset(store_type, data_files=data_files, cache_dir=cache_dir, num_proc=64)
            dataset = dataset_dict[phase]

        end_time = time.time()
        self.logger = get_root_logger()
        self.logger.info(f"load {self.name} {phase} dataset time: {end_time - start_time}")

        dataset.set_format("np")

        try:
            from multiprocess import set_start_method

            set_start_method("spawn")
        except Exception as e:  # noqa F841
            pass

        tokenizers_dict = opt_get(opt, ["tokenizers"], None)
        self.audio_token = opt_get(opt, ["audio_id"], "<|speech|>")
        self.start_of_speech_token = opt_get(opt, ["start_of_speech_id"], "<|startofspeech|>")
        self.end_of_speech_token = opt_get(opt, ["end_of_speech_id"], "<|endofspeech|>")
        tokenizer_dict_instances = {}
        for k, v in tokenizers_dict.items():
            tokenizer = AutoTokenizer.from_pretrained(v)
            tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.audio_token, self.start_of_speech_token, self.end_of_speech_token]}
            )
            self.audio_token_id = tokenizer.convert_tokens_to_ids([self.audio_token])[0]
            tokenizer_dict_instances[k] = tokenizer

        self.tokenizers_dict = tokenizer_dict_instances
        text_tokenizer = self.tokenizers_dict["text"]
        self.ignore_id = opt_get(opt, ["ignore_id"], None)
        self.pad_token_id = text_tokenizer.pad_token_id
        self.text_tokenizer = text_tokenizer

        self.sr = opt_get(opt, ["sample_rate"], 16000)
        dataset = self.normalize_dataset(dataset, sr=self.sr, rm_punc=opt_get(opt, ["rm_punc"], False))
        self.cache_text = opt_get(opt, ["cache_text"], False)

        if self.cache_text and self.tokenizers_dict is not None:
            dataset = dataset.map(
                partial(cache_text_processor, tokenizers_dict=self.tokenizers_dict, audio_token=self.audio_token),
                num_proc=128,
                batch_size=256,
                writer_batch_size=1024,
                desc="cache text",
            )

        self.dataset = dataset

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

        self.dataset_len = len(dataset)

        self.phases_indice_dict = self.create_datasets(
            phase,
            self.dataset_len,
            phases_ratios=opt_get(opt, ["phases_ratios"], [0.985]),
            seed=opt_get(opt, ["seed"], 1234),
        )

        self.info_dict["text_token_length"] = dataset["text_token_length"]
        self.info_dict["speech_token_length"] = dataset["speech_token_length"]

        self.carry_filename = opt_get(opt, ["carry_filename"], False)
        self.selected_names = opt_get(opt, ["selected_names"], None)
        self.meta_path = opt_get(opt, ["meta_path"], None)
        self.stand_dict = opt_get(opt, ["stand_dict"], None)
        if opt_get(opt, ["shrink"], False):
            dataset = dataset.select_columns(self.read_keys)
        self.sorted = opt_get(opt, ["sorted"], False)
        if self.sorted:
            self.logger.info("It's dangerouts to sort sampler idx when using buffer_batch_group > 1.")
        self.dataset = dataset
        self.file_path_key = opt_get(opt, ["file_path_key"], "id")
        self.channels = opt_get(opt, ["channels"], 1)

    def normalize_dataset(self, dataset, **kwargs):
        dataset = dataset.map(
            partial(
                process_speech_func,
                token_frame_rate=self.token_frame_rate,
                dataset_name=self.name,
                rm_punc=kwargs.get("rm_punc", False),
            ),
            num_proc=128,
            batch_size=256,
            writer_batch_size=1024,
            desc="add duration and speech token length",
        )
        return dataset

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
        elif "source" in meta_list:
            paths = meta_list.pop("source")
        else:
            raise ValueError(f"No path found in meta_list: {meta_list}")

        data_group = []
        path_group = []
        audio_lengths = []
        begin_time_group = []
        duration_group = []

        if audio_ready:
            if offsets is None:
                offsets = [0] * len(meta_list["audio"])

            audio_metas = meta_list.pop("audio")
            for i, (audio_meta, offset) in enumerate(zip(audio_metas, offsets)):
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
        else:
            begin_times = meta_list["begin_time"] if "begin_time" in meta_list else meta_list["start"]
            durations = meta_list["duration"]

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
        data = dict(
            audio=data_group,
            path=path_group,
            audio_lengths=audio_lengths,
            begin_time=begin_time_group,
            duration=duration_group,
        )

        data.update(**meta_list)

        return data

    def layout_data(self, asr_input_ids, text_token):
        input_type = []
        text_token_len = len(text_token)
        i = 0
        while i < text_token_len:
            if text_token[i] == self.audio_token_id:
                input_type.append(self.speech_type)
            else:
                input_type.append(self.text_type)
            i += 1
        assert (text_token[-(len(asr_input_ids) + 2) : -2] == asr_input_ids).all()
        input_type[-(len(asr_input_ids) + 2) : -1] = [self.asr_text_type] * (len(asr_input_ids) + 1)

        input_tokens = torch.Tensor(text_token).long()
        input_type = torch.Tensor(input_type).long()
        asr_input_mask = input_type == self.asr_text_type
        asr_output_mask = torch.cat([asr_input_mask, torch.zeros(1, dtype=torch.bool)])[1:]

        output_tokens = torch.zeros_like(input_tokens).long() + self.ignore_id
        output_tokens[asr_output_mask] = input_tokens[asr_input_mask]
        return input_tokens, output_tokens, input_type

    def postprocess_items(self, values_dict: Dict[str, List[Any]], **kwargs) -> Dict[str, Any]:
        if "audio" not in values_dict and "source" in values_dict:
            source = values_dict.pop("source")
            audios = []
            for _source in source:
                audio, sr = torchaudio.load(_source)
                audios.append(audio)
        elif "audio" in values_dict:
            audios = values_dict.pop("audio")
        else:
            raise ValueError("No audio found in values_dict")
        values_dict["audio"] = audios
        values_dict["audio_lengths"] = [audio.shape[-1] for audio in audios]
        # file_names = values_dict.pop("file_names")
        if self.cache_text:
            assert self.text_key in values_dict
            text = values_dict.pop(self.text_key)
            text_token = values_dict.pop(self.text_token_key)
        else:
            text = values_dict.pop(self.text_key)
            if len(text) == 1:
                text_token = np.asarray(self.text_tokenizer(text)["input_ids"], dtype=np.int32)
            else:
                text_token = [np.asarray(self.text_tokenizer(_text)["input_ids"], dtype=np.int32) for _text in text]

        # list of list
        input_ids_list = []
        input_len_list = []
        output_ids_list = []
        input_type_list = []
        for i, (asr_text, _text_token) in enumerate(zip(text, text_token)):
            asr_input_ids = self.text_tokenizer.encode(asr_text, add_special_tokens=False)
            _input_ids, _output_ids, _input_type = self.layout_data(asr_input_ids, _text_token)

            input_ids_list.append(_input_ids)
            output_ids_list.append(_output_ids)
            input_type_list.append(_input_type)
            input_len_list.append(len(_input_ids))

        values_dict["input_ids"] = input_ids_list
        values_dict["output_ids"] = output_ids_list
        values_dict["input_type"] = input_type_list
        values_dict["input_lengths"] = input_len_list

        return values_dict

    def get_index_offset(self, indice):
        # For a given dataset item and shift, return song index and offset within song
        file_indice = []
        for index in indice:
            file_indice.append(self.files[index])
        return file_indice

    def get_item(self, item):
        index = self.get_index_offset(item)
        return self.get_audio_chunk(index, item)


@DATASETS.register_module()
class HuggingfaceMinmoASRLhotseDataset(HuggingfaceMinmoASRDataset):
    classes_dict: Union[Dict[str, int], None]

    def __init__(
        self,
        opt,
    ):
        super().__init__(opt)

    def normalize_dataset(self, dataset, **kwargs):
        sr = kwargs.get("sr", 16000)
        dataset = dataset.filter(
            lambda eg: eg["duration"] < self.max_duration - 1 / sr, num_proc=16, batch_size=32, writer_batch_size=192
        )

        dataset = dataset.map(
            lambda eg: {
                "speech_token_length": int(round(eg["duration"] * self.token_frame_rate, 3)),
            },
            num_proc=16,
            batch_size=32,
            writer_batch_size=192,
            desc="add length",
        )
        dataset = dataset.map(
            lambda eg: {
                "text": eg["supervisions"][0]["text"],
            },
            num_proc=16,
            batch_size=32,
            writer_batch_size=192,
            desc="add text",
        )
        dataset = dataset.map(
            lambda eg: {
                "source": eg["recording"]["sources"][0]["source"],
            },
            num_proc=16,
            batch_size=32,
            writer_batch_size=192,
            desc="add source",
        )
        return dataset

    def get_item(self, item):

        return self.dataset[item]


if __name__ == "__main__":

    dataset_params = {
        "name": "Emilia_24k",
        "n_workers": 0,
        "batch_size": 256,
        "path": "/mnt/bd/arnold-bytedrive/data/raw/LibriTTS_R/",
        "json_file": "/home/wumenglin/repo-dev/Questar-Speech-dev2/merged_output_p3_clear.jsonl",
        "phase": "train",
        "cache_dir": "/home/wumenglin/.cache/",
        "meta_path": "/home/wumenglin/.cache/Emilia_24k/train_meta.pth",
        "read_keys": ["text", "speech_token"],
        "carry_filename": True,
        "buffer_batch_group": 16,
        "bucket_batch_volume": 32,
        "file_names": None,
        "text_length": None,
        "speech_token_length": None,
        "text_key": "text",
        "text_token_key": "text_token",
        "speech_token_key": "speech_token",
        "data_cfg": {
            "speech_token": {"padding_val": 0},
            "text": {"padding_val": 0},
            "speech_token_length": {"padding_val": 0},
            "text_length": {"padding_val": 0},
        },
    }
    dataset = HuggingfaceMinmoASRDataset(dataset_params)
    from data.audio.xdata_collate import Collator
    from data.builder import create_dataloader

    dataloader = create_dataloader(dataset, dataset_params, collate_fn=Collator())
    dataloader.sampler.set_epoch(1)
    i = 0
    for b in tqdm(dataloader):
        print(b["text"])
        print(b["speech_token"])
        i += 1
        if i > 20:
            break
