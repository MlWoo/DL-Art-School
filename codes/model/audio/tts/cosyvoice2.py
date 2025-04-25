# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import lru_cache
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import BaseModule
from model.module.base.mask import sequence_mask
from model.module.wenet.label_smoothing_loss import LabelSmoothingLoss
from trainer.networks import register_model
from transformers import AutoConfig, AutoTokenizer, Qwen2ForCausalLM
from utils.util import opt_get

IGNORE_ID = -100


def torch_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor, ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1), pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator / denominator).detach()


class QwenTokenizer:
    def __init__(self, token_path, skip_special_tokens=True):
        super().__init__()
        # NOTE: non-chat model, all these special tokens keep randomly initialized.
        special_tokens = {
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "additional_special_tokens": [
                "<|im_start|>",
                "<|im_end|>",
                "<|endofprompt|>",
                "[breath]",
                "<strong>",
                "</strong>",
                "[noise]",
                "[laughter]",
                "[cough]",
                "[clucking]",
                "[accent]",
                "[quick_breath]",
                "<laughter>",
                "</laughter>",
                "[hissing]",
                "[sigh]",
                "[vocalized-noise]",
                "[lipsmack]",
                "[mn]",
                "<|prompt_text_token|>",
                "<|prompt_speech_token|>",
                "<|text_token|>",
                "<|speech_token|>",
                "<|sos_eos_task_token|>",
                "<|fill_token|>",
            ],
        }
        self.special_tokens = special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.tokenizer.add_special_tokens(special_tokens)
        self.skip_special_tokens = skip_special_tokens
        if not skip_special_tokens:
            assert len(self.tokenizer.all_special_tokens) == len(self.tokenizer.all_special_ids)
            self.special_token2id = {
                k: v for k, v in zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids)
            }
            self.special_id2token = {v: k for k, v in self.special_token2id.items()}

        for additional_special_token in self.special_tokens["additional_special_tokens"]:
            if additional_special_token.endswith("token|>"):
                token_entity = additional_special_token[2:-2]
                setattr(self, token_entity.replace("token", "id"), self.tokenizer.encode(additional_special_token)[0])

    def specialtoken2id(self, token):
        assert not self.skip_special_tokens
        return self.special_token2id[token]

    def id2specialtoken(self, id):
        assert not self.skip_special_tokens
        return self.special_id2token[id]

    def encode(self, text, **kwargs):
        tokens = self.tokenizer([text], return_tensors="pt")
        tokens = tokens["input_ids"][0].cpu().tolist()
        return tokens

    def decode(self, tokens):
        tokens = torch.tensor(tokens, dtype=torch.int64)
        text = self.tokenizer.batch_decode([tokens], skip_special_tokens=self.skip_special_tokens)[0]
        return text


@lru_cache(maxsize=None)
def get_qwen_tokenizer(token_path: str, skip_special_tokens: bool) -> QwenTokenizer:
    return QwenTokenizer(token_path=token_path, skip_special_tokens=skip_special_tokens)


class Qwen2TTSLM(BaseModule):
    def __init__(
        self,
        pretrain_path: str = "Qwen/Qwen2-0.5B-Instruct",
        speech_token_size: int = 6561,
        sampling: Optional[Callable] = None,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        chunk_text_len: int = -1,
        text_type: int = 0,
        speech_type: int = 1,
        track_inputs: bool = True,
        num_attn_visual_debug: int = 1,
        debug_val_keys: List[str] = [],
    ):
        super().__init__(
            track_inputs=track_inputs,
            num_attn_visual_debug=num_attn_visual_debug,
            debug_val_keys=debug_val_keys,
        )
        config = AutoConfig.from_pretrained(pretrain_path)
        self.speech_token_size = speech_token_size

        # 1. build text token language model related modules
        self.tokenizer = get_qwen_tokenizer(pretrain_path, skip_special_tokens=False)

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = nn.Embedding(2, config.hidden_size)
        self.llm = Qwen2ForCausalLM.from_pretrained(pretrain_path)
        self.llm_decoder = nn.Linear(config.hidden_size, speech_token_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(
            speech_token_size + 3, config.hidden_size, padding_idx=speech_token_size + 2
        )

        # 4. sampling method
        self.sampling = sampling

        self.pretrained_path = pretrain_path
        self.text_type = text_type
        self.speech_type = speech_type
        self.chunk_text_len = chunk_text_len

    def from_vllm_model(self, prefix=""):
        from safetensors.torch import load_file

        state_dict = load_file(self.pretrained_path + "/model.safetensors")
        model_state_dict = self.state_dict()
        prefix_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict:
                prefix_state_dict[k] = v
            elif prefix + k in model_state_dict:
                prefix_state_dict[prefix + k] = v
            else:
                print(k)
        self.load_state_dict(prefix_state_dict, strict=True)

    def from_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict, strict=True)

    def forward(
        self,
        batch_template,
        batch_speech_token,
        batch_text_token,
        batch_input_lengths,
        batch_speech_token_lengths,
        batch_text_token_lengths,
        batch_prompt_speech_token,
        batch_prompt_speech_token_lengths,
        return_loss=True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        bs = batch_template.size(0)

        # 1. fill text_token
        text_token_mask = batch_template == self.tokenizer.text_id
        batch_template = batch_template.clone()
        batch_text_token_mask = sequence_mask(batch_text_token_lengths, valid=True)
        batch_template[text_token_mask] = batch_text_token[batch_text_token_mask]
        text_emb = self.llm.model.embed_tokens(batch_template)

        # 2. fill sos_eos_task_id
        sos_eos_task_id = torch.LongTensor([self.sos_eos, self.task_id]).to(batch_template.device).reshape(1, 2)
        sos_eos_task_emb = self.llm_embedding(sos_eos_task_id).expand(bs, -1, -1)
        sos_eos_task_mask = batch_template == self.tokenizer.sos_eos_task_id
        text_emb[sos_eos_task_mask] = sos_eos_task_emb.reshape(-1, sos_eos_task_emb.size(-1))

        # 3. fill prompt_speech_token
        prompt_speech_token_mask = batch_template == self.tokenizer.prompt_speech_id
        prompt_speech_emb = self.speech_embedding(batch_prompt_speech_token)
        prompt_speech_mask = sequence_mask(batch_prompt_speech_token_lengths, valid=True)
        text_emb[prompt_speech_token_mask] = prompt_speech_emb[prompt_speech_mask]

        # 4. fill speech_token
        speech_token_mask = batch_template == self.tokenizer.speech_id
        # batch_speech_token append eos
        batch_speech_token1 = torch.cat(
            [batch_speech_token, torch.zeros_like(batch_speech_token[:, :1]) + self.speech_token_size + 2], dim=1
        )
        eos_pure_speech_mask = sequence_mask(batch_speech_token_lengths + 1, end=True)
        batch_speech_token1[eos_pure_speech_mask] = self.speech_token_size + 1
        speech_emb = self.speech_embedding(batch_speech_token1)
        pure_speech_mask = sequence_mask(batch_speech_token_lengths + 1, valid=True)
        text_emb[speech_token_mask] = speech_emb[pure_speech_mask]

        # 4. calculate attention mask
        attention_mask = sequence_mask(batch_input_lengths, valid=True)

        # 5. run lm forward
        llm_out = self.llm(
            inputs_embeds=text_emb, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
        )
        last_hidden_state = llm_out.hidden_states[-1]
        logits = self.llm_decoder(last_hidden_state)
        output_speech_mask = torch.roll(speech_token_mask, -1, dims=1)
        output_logits = logits[output_speech_mask]
        output_ids = batch_speech_token1[output_speech_mask]

        if return_loss:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
                loss = self.criterion_ce(output_logits.float(), output_ids)
            # calculate accuracy
            acc = torch_accuracy(output_logits.view(-1, self.speech_token_size + 1), output_ids, ignore_label=IGNORE_ID)
            if self.debug_info is not None:
                self.debug_info["input_size"] = batch_template.shape[0] * batch_template.shape[1]
                self.debug_info["input_length"] = batch_template.shape[1]
                self.debug_info["acc"] = acc
            return loss
        else:
            output_speech_mask2 = torch.roll(output_speech_mask, -1, dims=1) & output_speech_mask
            output_logits2 = logits[output_speech_mask2]
            pure_speech_mask2 = sequence_mask(batch_speech_token_lengths)
            target_prob = torch.gather(
                output_logits2.log_softmax(-1), dim=1, index=batch_speech_token[pure_speech_mask2].unsqueeze(1)
            ).squeeze(1)
            samples_prob = torch.split(target_prob, batch_speech_token_lengths.tolist())
            return samples_prob


@register_model
def register_cosyvoice2(opt_net, opt):
    return Qwen2TTSLM(**opt_get(opt_net, ["kwargs"], {}))


class CosyVoice2DPO(Qwen2TTSLM):
    def __init__(
        self,
        pretrain_path: str = "Qwen/Qwen2-0.5B-Instruct",
        speech_token_size: int = 6561,
        sampling: Optional[Callable] = None,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        chunk_text_len: int = -1,
        text_type: int = 0,
        speech_type: int = 1,
        track_inputs: bool = False,
        num_attn_visual_debug: int = 1,
        debug_val_keys: List[str] = [],
        name: Optional[str] = "racer",
        dpo_beta: float = 0.0,
    ):
        super().__init__(
            pretrain_path,
            speech_token_size,
            sampling,
            length_normalized_loss,
            lsm_weight,
            chunk_text_len,
            text_type,
            speech_type,
            track_inputs,
            num_attn_visual_debug,
            debug_val_keys,
        )
        self.name = name
        self.dpo_beta = dpo_beta

    def forward(
        self,
        batch_template,
        batch_speech_token,
        batch_text_token,
        batch_input_lengths,
        batch_speech_token_lengths,
        batch_text_token_lengths,
        batch_prompt_speech_token,
        batch_prompt_speech_token_lengths,
        reference_log_prob=None,
    ):
        samples_prob = super().forward(
            batch_template,
            batch_speech_token,
            batch_text_token,
            batch_input_lengths,
            batch_speech_token_lengths,
            batch_text_token_lengths,
            batch_prompt_speech_token,
            batch_prompt_speech_token_lengths,
            return_loss=False,
        )

        policy_logprobs = []
        for i in range(batch_template.size(0)):
            policy_logprobs.append(samples_prob[i].sum())

        policy_logprobs = torch.stack(policy_logprobs)
        if reference_log_prob is None:
            return policy_logprobs
        else:
            chosen_policy_logprobs, rejected_policy_logprobs = torch.chunk(policy_logprobs, 2, dim=0)
            chosen_ref_logprobs, rejected_ref_logprobs = torch.chunk(reference_log_prob, 2, dim=0)
            if self.debug_info is not None:
                chosen_rewards = self.dpo_beta * (chosen_policy_logprobs - chosen_ref_logprobs).detach().mean()
                rejected_rewards = self.dpo_beta * (rejected_policy_logprobs - rejected_ref_logprobs).detach().mean()
                self.debug_info["chosen_rewards"] = chosen_rewards
                self.debug_info["rejected_rewards"] = rejected_rewards
            logprobs_diff = (chosen_policy_logprobs - rejected_policy_logprobs) - (
                chosen_ref_logprobs - rejected_ref_logprobs
            )
            loss = -F.logsigmoid(self.dpo_beta * logprobs_diff).mean()
            return loss


@register_model
def register_cosyvoice2_dpo(opt_net, opt):
    return CosyVoice2DPO(**opt_get(opt_net, ["kwargs"], {}))


if __name__ == "__main__":
    from utils.registry import construct_from_kwargs

    opt = opt_get(None, None, None)
    model = construct_from_kwargs(CosyVoice2DPO, opt)
    print(model)
