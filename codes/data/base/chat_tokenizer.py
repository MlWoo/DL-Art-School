import copy
from functools import partial
from itertools import zip_longest
from typing import List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence


class ChatTokenizer:
    def __init__(
        self,
        tokenizer,
        system_prompt: str = None,
        instruction_first: bool = True,
        q_audio_token: str = "<|q-audio|>",
        a_audio_token: str = "<|a-audio|>",
        label_token: str = "<|label|>",
        hint_text_token: str = "<|hint-text|>",
        a_audio_eos_token: str = "<|a-audio-eos|>",
    ):
        self.system_prompt = system_prompt
        self.instruction_first = instruction_first
        special_tokens = {
            "q_audio_token": q_audio_token,
            "a_audio_token": a_audio_token,
            "label_token": label_token,
            "hint_text_token": hint_text_token,
            "a_audio_eos_token": a_audio_eos_token,
        }
        for name, token in special_tokens.items():
            setattr(self, name, token)
        tokenizer.add_special_tokens({"additional_special_tokens": list(special_tokens.values())})
        special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens.values())
        for name, token_id in zip(special_tokens.keys(), special_token_ids):
            setattr(self, f"{name}_id", token_id)
        self.tokenizer = tokenizer

    def mask(
        self,
        input_ids: torch.Tensor,
        target_id: int,
        mask_bos: bool = False,
        mask_eos: bool = False,
        valid: bool = True,
    ) -> torch.Tensor:
        mask = input_ids == target_id
        if mask_bos:
            mask |= torch.roll(mask, -1, dims=1)
        if mask_eos:
            mask |= torch.roll(mask, 1, dims=1)
        return mask if valid else ~mask

    def fill_labels(self, label_ids: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids[self.mask(input_ids, self.label_token_id)] = label_ids[
            self.mask(label_ids, self.tokenizer.pad_token_id, valid=False)
        ]
        return input_ids

    def pad_token_ids(
        self, token_ids: List[List[int]], batch_first: bool = True, device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_lens = torch.tensor([len(ids) for ids in token_ids], device=device, dtype=torch.long)
        token_ids = [torch.tensor(ids, device=device, dtype=torch.long) for ids in token_ids]
        token_ids = pad_sequence(token_ids, padding_value=self.tokenizer.pad_token_id, batch_first=batch_first).long()
        return token_ids, token_lens

    def batch_tokenize_label(
        self, labels: List[str], batch_first: bool = True, device: torch.device = torch.device("cpu")
    ) -> List[List[int]]:
        label_ids = [self.tokenizer(label)["input_ids"] for label in labels]
        return self.pad_token_ids(label_ids, batch_first, device)

    def tokenize(
        self,
        q_audio_lens: Union[int, List[int]],
        labels: Optional[Union[str, List[str]]] = None,
        instructions: Union[str, List[str]] = "",
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> Tuple[List[int], List[int]]:
        if labels is None:
            assert add_generation_prompt
        if isinstance(q_audio_lens, int):
            assert labels is None or isinstance(labels, str)
            q_audio_lens, labels = [q_audio_lens], [labels]
        elif add_generation_prompt:
            assert len(labels) == len(q_audio_lens) - 1
            labels.append(None)
        if isinstance(instructions, str):
            instructions = [instructions] * len(q_audio_lens)
        assert len(instructions) == len(q_audio_lens)

        chat = []
        label_ids = []
        if self.system_prompt is not None:
            chat.append({"role": "system", "content": self.system_prompt})
        if len(q_audio_lens) - len(labels) == 1:
            labels_temp = copy.deepcopy(labels)
            labels_temp.append(None)
            instructions_temp = copy.deepcopy(instructions)
            instructions_temp.append("")
        else:
            labels_temp = labels
            instructions_temp = instructions

        for q_audio_len, label, instruction in zip(q_audio_lens, labels_temp, instructions_temp):
            q_audio_placeholder = self.q_audio_token * q_audio_len
            content = " ".join(
                [instruction, q_audio_placeholder] if self.instruction_first else [q_audio_placeholder, instruction]
            )
            chat.append({"role": "user", "content": content.strip()})
            if label is not None:
                label_ids.append(self.tokenizer(label)["input_ids"])
                label_placeholder = self.label_token * len(label_ids[-1])
                chat.append({"role": "assistant", "content": label_placeholder})

        return self.tokenizer.apply_chat_template(
            chat, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        ), sum(label_ids, [])

    @staticmethod
    def split(n: int, chunk_size: int) -> List[int]:
        return [chunk_size] * (n // chunk_size) + ([n % chunk_size] if n % chunk_size else [])

    def intermix(
        self,
        q_audio_lens: Union[int, List[int]],
        a_audio_lens: Union[int, List[int]],
        labels: Union[str, List[str]],
        label_chunk_size: int = 5,
        audio_chunk_size: int = 15,
        sep: str = "",
        append_text_turn: bool = False,
    ) -> Tuple[List[int], List[int]]:
        if isinstance(q_audio_lens, int):
            assert isinstance(a_audio_lens, int) and isinstance(labels, str)
            q_audio_lens, a_audio_lens, labels = [q_audio_lens], [a_audio_lens], [labels]
        assert len(a_audio_lens) == len(labels)
        assert len(q_audio_lens) - len(a_audio_lens) <= 1
        if len(q_audio_lens) - len(a_audio_lens) == 1:
            last_q_audio_len = q_audio_lens.pop()
        else:
            last_q_audio_len = 0

        chat = []
        label_ids = []
        for q_audio_len, a_audio_len, label in zip(q_audio_lens, a_audio_lens, labels):
            chat.extend([self.q_audio_token_id] * q_audio_len)
            chat.extend(self.tokenizer(sep)["input_ids"])
            if append_text_turn:
                label += self.tokenizer.eos_token
            label_ids.append(self.tokenizer(label)["input_ids"])
            label_chunks = self.split(len(label_ids[-1]), label_chunk_size)
            a_audio_chunks = self.split(a_audio_len, audio_chunk_size)
            for idx, (label_chunk, a_audio_chunk) in enumerate(zip_longest(label_chunks, a_audio_chunks, fillvalue=0)):
                chat.extend([self.label_token_id] * label_chunk)
                a_audio_placeholder = [self.a_audio_token_id] * a_audio_chunk
                chat.extend(a_audio_placeholder)
                if idx == len(a_audio_chunks) - 1:
                    chat.append(self.a_audio_eos_token_id)
        if last_q_audio_len > 0:
            chat.extend([self.q_audio_token_id] * last_q_audio_len)
        return chat, sum(label_ids, [])

    def batch_tokenize(
        self,
        q_audio_lens: List[Union[int, List[int]]],
        labels: Optional[List[Union[str, List[str]]]] = None,
        instructions: Union[str, List[Union[str, List[str]]]] = "",
        batch_first: bool = True,
        add_generation_prompt: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        batch_size = len(q_audio_lens)
        if labels is None:
            assert add_generation_prompt
            labels = [None] * batch_size
        if isinstance(instructions, str):
            instructions = [instructions] * batch_size
        assert len(instructions) == batch_size
        assert all(
            len(instruction) == len(audio_len)
            for instruction, audio_len in zip(instructions, q_audio_lens)
            if isinstance(instruction, list)
        )
        tokenize = partial(self.tokenize, add_generation_prompt=add_generation_prompt)
        input_ids, label_ids = zip(*map(tokenize, q_audio_lens, labels, instructions))
        input_ids, input_lens = self.pad_token_ids(input_ids, batch_first, device)
        label_ids, label_lens = self.pad_token_ids(label_ids, batch_first, device)
        return input_ids, input_lens, label_ids, label_lens

    def batch_intermix(
        self,
        q_audio_lens: List[Union[int, List[int]]],
        a_audio_lens: List[Union[int, List[int]]],
        labels: List[Union[str, List[str]]],
        label_chunk_size: int = 5,
        audio_chunk_size: int = 15,
        sep: str = "",
        append_text_turn: bool = True,
        batch_first: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        assert len(q_audio_lens) == len(a_audio_lens) == len(labels)
        intermix = partial(
            self.intermix,
            label_chunk_size=label_chunk_size,
            audio_chunk_size=audio_chunk_size,
            sep=sep,
            append_text_turn=append_text_turn,
        )
        input_ids, label_ids = zip(*map(intermix, q_audio_lens, a_audio_lens, labels))
        input_ids, input_lens = self.pad_token_ids(input_ids, batch_first, device)
        label_ids, label_lens = self.pad_token_ids(label_ids, batch_first, device)
        return input_ids, input_lens, label_ids, label_lens

    def template(self, instruction: str = "") -> Tuple[List[int], List[int]]:
        chat = []
        if self.system_prompt is not None:
            chat.append({"role": "system", "content": self.system_prompt})
        content = " ".join(
            [instruction, self.q_audio_token] if self.instruction_first else [self.q_audio_token, instruction]
        )
        for _ in range(2):
            chat.append({"role": "user", "content": content.strip()})
            chat.append({"role": "assistant", "content": self.label_token})
        template = self.tokenizer.apply_chat_template(chat)
        q_audio_token_index = template.index(self.q_audio_token_id)
        prefix = template[:q_audio_token_index]
        label_token_index = template.index(self.label_token_id)
        infix = template[q_audio_token_index + 1 : label_token_index]
        template = template[label_token_index + 1 :]
        q_audio_token_index = template.index(self.q_audio_token_id)
        suffix = template[:q_audio_token_index]
        return prefix, infix, suffix
