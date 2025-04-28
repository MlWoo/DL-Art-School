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
        self.logger.info(f"load {phase} dataset time: {end_time - start_time}")

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
                num_proc=16,
                batch_size=32,
                writer_batch_size=192,
                desc="cache text",
            )

        self.dataset = dataset
        self.dataset_len = len(dataset)

        self.phases_indice_dict = self.create_datasets(
            phase,
            self.dataset_len,
            phases_ratios=opt_get(opt, ["phases_ratios"], [0.985]),
            seed=opt_get(opt, ["seed"], 1234),
        )

        self.info_dict = dict()
        info_list = []
        for key in info_list:
            self.info_dict[key] = dataset[key]

        self.info_dict["text_token_length"] = dataset["text_token_length"]
        self.info_dict["speech_token_length"] = dataset["speech_token_length"]

        self.carry_filename = opt_get(opt, ["carry_filename"], False)
        self.selected_names = opt_get(opt, ["selected_names"], None)
        self.meta_path = opt_get(opt, ["meta_path"], None)
        self.stand_dict = opt_get(opt, ["stand_dict"], None)
        if opt_get(opt, ["shrink"], False):
            dataset = dataset.select_columns(self.read_keys + info_list)
        self.sorted = opt_get(opt, ["sorted"], False)
        if self.sorted:
            self.logger.info("It's dangerouts to sort sampler idx when using buffer_batch_group > 1.")
        self.dataset = dataset
        self.file_path_key = opt_get(opt, ["file_path_key"], "id")

    def normalize_dataset(self, dataset, **kwargs):
        if "speech_token" in dataset.column_names:
            dataset = dataset.map(
                lambda eg: {
                    "speech_token_length": len(eg["speech_token"]),
                },
                num_proc=16,
                batch_size=32,
                writer_batch_size=192,
                desc="add speech token length",
            )
        elif "duration" in dataset.column_names:
            dataset = dataset.map(
                lambda eg: {
                    "speech_token_length": int(round(eg["duration"] * self.token_frame_rate, 3)),
                },
                num_proc=16,
                batch_size=32,
                writer_batch_size=192,
                desc="add speech token length",
            )
        elif "begin_time" in dataset.column_names and "end_time" in dataset.column_names:
            dataset = dataset.map(
                lambda eg: {
                    "speech_token_length": int(round((eg["end_time"] - eg["begin_time"]) * self.token_frame_rate, 3)),
                },
                num_proc=16,
                batch_size=32,
                writer_batch_size=192,
                desc="add speech token length",
            )
        else:
            raise ValueError("No duration or begin_time and end_time in dataset")

        if kwargs.get("rm_punc", False):
            dataset = dataset.map(
                partial(rm_punctuation, dataset_name=self.name),
                num_proc=32,
                batch_size=32,
                writer_batch_size=192,
                desc="rm punctuation",
            )
        return dataset

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
        if "source" in values_dict:
            source = values_dict.pop("source")
            audios = []
            for _source in source:
                audio, sr = torchaudio.load(_source)
                audios.append(audio)
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

    def get_item(self, item):
        return self.dataset[item]


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
