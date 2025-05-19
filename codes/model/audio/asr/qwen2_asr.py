import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import BaseModule
from model.module.base.conv import SamePad
from model.module.base.mask import sequence_mask
from trainer.networks import register_model
from trainer.util import set_requires_grad
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from utils.checkpoint import load_checkpoint  # noqa: F401
from utils.options import opt_get
from utils.registry import construct_from_kwargs
from utils.util import default  # noqa: F401

from data.audio.whisper_util import log_mel_spectrogram_batch


def apply_chat_template(speech_token_length, tokenizer, audio_token):
    chat = [
        {
            "role": "user",
            "content": f"Transcribe the speech.<|startofspeech|>{audio_token * speech_token_length}<|endofspeech|>",
        }
    ]
    tokens = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
    return tokens


class ProjectorConv1d(nn.Module):

    def __init__(self, encoder_dim, k, hidden_size, llm_dim, causal=False):
        super().__init__()
        kernel_size = min(k * 2 + 1, 7)
        self.conv1d = nn.Sequential(
            SamePad(kernel_size=kernel_size, causal=causal, stride=k, padding_mode="zeros", simple=False),
            nn.Conv1d(in_channels=encoder_dim, out_channels=llm_dim, kernel_size=kernel_size, stride=k, padding=0),
        )
        self.ln = nn.LayerNorm(llm_dim)
        self.linear1 = nn.Linear(llm_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, llm_dim)
        self.act = nn.SiLU()

    def forward(self, x, inference=False):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        res = x
        x = self.ln(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = x + res
        return x


class FoldedProjector(nn.Module):

    def __init__(self, encoder_dim, stride, llm_dim):
        super().__init__()
        self.linear = nn.Linear(encoder_dim * stride, llm_dim)
        self.stride = stride

    def forward(self, x):
        B, T, C = x.shape
        folded_T = T // self.stride
        unfolded_T = folded_T * self.stride
        x = x[:, :unfolded_T]
        x = x.reshape(B, folded_T, C * self.stride)
        x = self.linear(x)
        return x


class ASRModel(BaseModule):
    def __init__(
        self,
        pretrain_path,
        text_type,
        speech_type,
        asr_text_type,
        stream: bool = False,
        context_chunks: int = 2,
        feature_size: int = 1,
        track_inputs: bool = True,
        num_attn_visual_debug: int = 1,
        projector_type: str = "conv1d",
        speech_encoder_type: str = "whisper",
        encoder: Optional[dict] = None,
        encoder_pretrained: Optional[str] = None,
        encoder_layer_idx: Optional[int] = None,
        enable_output_ln: bool = False,
        enable_ctc: bool = False,
        encoder_freeze: bool = False,
        debug_val_keys: List[str] = ["input_size", "input_length", "acc"],
    ):
        super(ASRModel, self).__init__(
            track_inputs=track_inputs,
            num_attn_visual_debug=num_attn_visual_debug,
            debug_val_keys=debug_val_keys,
        )

        self.llm = AutoModelForCausalLM.from_pretrained(pretrain_path, attn_implementation="flash_attention_2")
        self.llm.lm_head.weight = nn.Parameter(self.llm.lm_head.weight.clone())
        self.llm.to(torch.bfloat16)
        set_requires_grad(self.llm, False, set_to_none=True)
        for param in self.llm.parameters():
            param.DO_NOT_TRAIN = True
            param.grad = None

        self.llm.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        config = AutoConfig.from_pretrained(pretrain_path)

        self.audio_token = "<|speech|>"
        self.start_of_speech_token = "<|startofspeech|>"
        self.end_of_speech_token = "<|endofspeech|>"
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.audio_token, self.start_of_speech_token, self.end_of_speech_token]}
        )
        self.tokenizer = tokenizer
        self.audio_token_id = tokenizer.convert_tokens_to_ids([self.audio_token])[0]

        if enable_ctc:
            self.ctc_fc = nn.Linear(config.hidden_size, tokenizer.vocab_size)
            self.ctc_loss = nn.CTCLoss()
        self.enable_ctc = enable_ctc

        self.feature_size = feature_size

        self.input_sr = 16000
        self.hop_length = 160
        if speech_encoder_type == "whisper":
            import whisper

            self.context_chunks = context_chunks
            self.stride = 2 * context_chunks
            self.encoder = whisper.load_model("medium").encoder.eval()
            self.codec = "whisper"
        elif speech_encoder_type == "conformer":
            assert encoder is not None, "encoder must be provided for conformer"
            self.codec = "conformer"
            self.encoder = construct_from_kwargs(encoder)
            if encoder_pretrained is not None:
                ckpt = torch.load(encoder_pretrained)
                encoder_state_dict = {}
                prefix = "encoder."
                for k, v in ckpt.items():
                    if k.startswith(prefix):
                        encoder_state_dict[k[len(prefix) :]] = v
                self.encoder.load_state_dict(encoder_state_dict, strict=True)
            if encoder_layer_idx is not None:
                for i, layer in enumerate(self.encoder.conformer.layers.layers):
                    if i > encoder_layer_idx:
                        self.encoder.conformer.layers.layers[i] = None
            self.encoder_layer_idx = encoder_layer_idx
            self.stride = 2
            self.encoder_out_ln = nn.LayerNorm(encoder["num_model_dim"]) if enable_output_ln else nn.Identity()
            if encoder_freeze:
                self.encoder.eval()
                set_requires_grad(self.encoder, False, set_to_none=True)
                for p in self.encoder.parameters():
                    p.requires_grad = False
                    p.DO_NOT_TRAIN = True
            self.encoder_freeze = encoder_freeze
        else:
            raise ValueError(f"Invalid speech encoder type: {speech_encoder_type}")

        if projector_type == "conv1d":
            self.projector = ProjectorConv1d(1024, self.stride, 2048, config.hidden_size)
        elif projector_type == "folded":
            self.projector = FoldedProjector(1024, self.stride, config.hidden_size)
        else:
            raise ValueError(f"Invalid projector type: {projector_type}")

        self.codebook_size = 2048
        self.text_type = text_type
        self.speech_type = speech_type
        self.asr_text_type = asr_text_type
        self.stream = stream

    def get_embeddings(self, input_ids, speech, speech_lengths):
        if self.codec == "whisper":
            with torch.no_grad():
                if self.stream:
                    audios = speech[:, :].to(torch.bfloat16)
                    chunk_size = 32000
                    stride = 3840  # 240ms
                    mel_center_stride = (440 // 2 - 1) // 160 + 1
                    mel_center_offset = mel_center_stride * 160
                    B, T = audios.shape
                    num_windows = (T - 1) // stride + 1
                    audios = audios.view(B, 1, 1, T)
                    audios = F.pad(audios, (mel_center_offset, mel_center_offset, 0, 0, 0, 0), mode="replicate")
                    audios = audios.view(B, -1)
                    audios = F.pad(
                        audios,
                        (chunk_size - stride, stride - T % stride, 0, 0),
                    )
                    B, T = audios.shape
                    windows = audios.as_strided(
                        size=(B, num_windows, chunk_size + mel_center_offset * 2), stride=(T, stride, 1)
                    )
                    windows = windows.reshape(B * num_windows, chunk_size + mel_center_offset * 2)
                    mels = log_mel_spectrogram_batch(windows, n_mels=80)[..., mel_center_stride:-mel_center_stride]
                    context_chunk_len = self.context_chunks * 12
                    speech_embeds = self.encoder(mels)[:, -context_chunk_len:].reshape(B, -1, 1024)
                else:
                    speech_tokens = log_mel_spectrogram_batch(speech, n_mels=80)
                    speech_down_lengths = speech_lengths // self.hop_length // 2
                    speech_embeds = self.encoder(speech_tokens, speech_down_lengths)

            stride = self.stride // self.context_chunks
            speech_embeds = self.projector(speech_embeds)
            speech_mask = sequence_mask(
                speech_lengths // self.hop_length // (stride * 2), max_len=speech_embeds.shape[1]
            )
        elif self.codec == "conformer":
            if self.encoder_freeze:
                with torch.no_grad():
                    speech_embeds = self.encoder(
                        speech.permute(0, 2, 1), speech_lengths, layer_idx=self.encoder_layer_idx
                    ).last_hidden_state
                    speech_embeds = self.encoder_out_ln(speech_embeds)
                    speech_embeds = speech_embeds.detach()
            else:
                speech_embeds = self.encoder(
                    speech.permute(0, 2, 1), speech_lengths, layer_idx=self.encoder_layer_idx
                ).last_hidden_state
                speech_embeds = self.encoder_out_ln(speech_embeds)

            stride = self.stride
            reduction_factor = self.encoder.reduction_factors
            speech_embeds = self.projector(speech_embeds)
            # speech_mask = sequence_mask(speech_lengths // reduction_factor // stride, max_len=speech_embeds.shape[1])
        else:
            speech_tokens = speech
            speech_mask = speech_tokens < self.codebook_size
            speech_tokens[~speech_mask] = 0
            with torch.no_grad():
                # audios: (B, T, C) => (B, C, T)
                # audio_embeds: (B, T, C)
                speech_tokens = speech_tokens[..., : self.feature_size]
                speech_embeds = self.encode(speech_tokens.transpose(1, 2))

                speech_mask = speech_mask[..., 0]
            speech_embeds = self.projector(speech_embeds * speech_mask.unsqueeze(-1)) * speech_mask.unsqueeze(-1)
        input_speech_mask = input_ids == self.audio_token_id
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        speech_lengths_new = input_speech_mask.sum(-1)
        speech_mask = sequence_mask(speech_lengths_new, max_len=speech_embeds.shape[1])
        text_embeds[input_speech_mask] = speech_embeds[speech_mask]
        return text_embeds, speech_embeds, input_speech_mask, speech_mask

    def forward(self, input_ids, output_ids, input_type, speech=None, speech_lengths=None):
        if speech is not None:
            inputs_embeds, speech_embeds, input_speech_mask, speech_mask = self.get_embeddings(
                input_ids, speech[:, : self.feature_size], speech_lengths
            )
            del speech
            del speech_lengths
        else:
            inputs_embeds, speech_embeds, input_speech_mask, speech_mask = self.get_embeddings(input_ids, None, None)
        input_mask = input_type != self.asr_text_type + 1
        output_mask = output_ids != -100
        output_ids = output_ids[output_mask]

        outputs = self.llm(inputs_embeds=inputs_embeds, input_mask=input_mask)

        logits = outputs.logits
        del outputs
        logits = logits[output_mask]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
            loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), output_ids.view(-1))

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == output_ids).float().mean()

        if self.enable_ctc:
            speech_lengths = speech_mask.sum(-1)
            output_lengths = output_mask.sum(-1)
            logits = self.ctc_fc(speech_embeds).log_softmax(2).transpose(0, 1)
            ctc_loss = self.ctc_loss(logits, output_ids, speech_lengths, output_lengths)
            ctc_loss = torch.zeros_like(loss_ce)
        else:
            ctc_loss = torch.zeros_like(loss_ce)

        if self.debug_info is not None:
            self.debug_info["input_size"] = input_ids.numel()
            self.debug_info["input_length"] = input_ids.shape[1]
            self.debug_info["acc"] = acc

        return loss_ce, acc, ctc_loss

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(self, input_speech):
        speech_token_length = input_speech.shape[-1] // 4
        input_ids = apply_chat_template(speech_token_length, self.tokenizer, self.audio_token)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to("cuda")

        with torch.no_grad():
            inputs_embeds, speech_embeds, _, speech_mask = self.get_embeddings(
                input_ids, input_speech, torch.tensor([input_speech.shape[-1]]).to("cuda")
            )
            input_mask = torch.ones_like(input_ids)
            output_tokens = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=input_mask,
                do_sample=False,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=100,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(output_tokens[0][:-1])
        return text

    def from_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict, strict=True)
        if self.config.tie_weights:
            self.llm.lm_head.weight = self.llm.model.embed_tokens.weight

    def inference(self, input_speech):
        import torchaudio

        audio, ori_sr = torchaudio.load(input_speech)
        if audio.shape[1] / ori_sr > 29.99:
            return False, None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        audio = audio.to(device)
        # resample to 16000
        audio = torchaudio.functional.resample(audio, ori_sr, 16000)
        from trainer.injectors.audio_injectors import CanonicalTorchMelSpectrogram

        mel_cfg = {
            "mel_fmin": 0,
            "mel_fmax": 8000,
            "sampling_rate": 16000,
            "n_mel_channels": 128,
            "filter_length": 1024,
            "hop_length": 160,
            "win_length": 640,
            "true_normalization": False,
            "mel_norm_file": "/home/wumenglin/repo-dev/dl-art-school/codes/torch_mels.pt",
        }
        injector = CanonicalTorchMelSpectrogram(opt=mel_cfg, env=None)
        audio = injector({"in": audio})["mel"]
        return True, self.generate(audio)


@register_model
def register_qwen2_asr(opt_net, opt):
    return ASRModel(**opt_get(opt_net, ["kwargs"], {}))


if __name__ == "__main__":
    import argparse
    import json
    import os.path as osp

    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default="checkpoint.pt")
    parser.add_argument("--input-speech", type=str, default="test.wav")
    parser.add_argument("--input-json", type=str, default=None)
    parser.add_argument("--output-json", type=str, default="output.jsonl")
    parser.add_argument("--out-dir", type=str, default="test-result")
    parser.add_argument("--num-codec", type=int, default=32)
    args = parser.parse_args()

    # model = ASRModel(
    #     pretrain_path="Qwen/Qwen2.5-1.5B-Instruct",
    #     text_type=0,
    #     speech_type=1,
    #     asr_text_type=2,
    #     feature_size=args.feature_size,
    #     projector_type="conv1d_v5",
    #     stream=True,
    #     #causal=True,
    # )
    model = ASRModel(
        pretrain_path="Qwen/Qwen2.5-1.5B-Instruct",
        text_type=0,
        speech_type=1,
        asr_text_type=2,
        # feature_size=args.feature_size,
        projector_type="folded",
        speech_encoder_type="conformer",
        stream=True,
        encoder={
            "class_name": "model.audio.module.encoders.StreamConformerEncoder",
            "attn_type": "wo-pos",
            "num_preconformer_layers": 3,
            "num_preconformer_dim": 384,
            "sub_type": "2d",
            "sub_layers": 2,
            "num_input_dim": 128,
            "num_sub_dim": [8, 32],
            "num_model_dim": 1024,
            "proj_dropout_p": 0.1,
            "num_conformer_layers": 20,
            "num_heads": 8,
            "attn_dropout_p": 0.1,
            "cxt_dropout_p": 0.1,
            "ffn_dropout_p": 0.1,
            "conv_module_kernel_size": 4,
            "causal": True,
            "window_size": 128,
            "max_len": 2048,
            "norm_groups": 8,
            "conv_activation": "gelu",
        },
        encoder_pretrained="/home/wumenglin/repo-dev/dl-art-school/experiments/streamconformer-bestrq-large_lr_h100x4_gelu8-librilight/models1/250000_generator.pth",  # noqa E501
        # encoder_pretrained="/home/wumenglin/repo-dev/dl-art-school/experiments/streamconformer-bestrq-large_lr_h100x4_gelu8-v2/models/120000_generator.pth",
        encoder_freeze=False,
        encoder_layer_idx=12,
        enable_output_ln=False,
        feature_size=128,
    )
    model.from_checkpoint(args.checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # .to(torch.bfloat16)
    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.input_json is not None:
        results = []
        with open(args.input_json, "r") as f:
            for i, line in enumerate(tqdm(f)):
                data = json.loads(line)
                input_speech = data["audio"]
                success, output_text = model.inference(input_speech)
                if not success:
                    continue
                data["text"] = output_text
                print("--------------------------------")
                print(data["text"].lower())
                print(data["gt"].lower())
                print("--------------------------------")
                results.append(data)
                # if i > 10:
                #     break
        with open(osp.join(args.out_dir, args.output_json), "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        results = {
            "test_clean_text": [],
            "test_other_text": [],
            "dev_clean_text": [],
            "dev_other_text": [],
            "test_clean_ref": [],
            "test_other_ref": [],
            "dev_clean_ref": [],
            "dev_other_ref": [],
        }

        with open(osp.join(args.out_dir, args.output_json), "r") as f:
            for line in tqdm(f):
                data = json.loads(line)
                fid = data["audio"].split("/")[-1]

                if "test-clean" in data["audio"]:
                    results["test_clean_text"].append(f"{fid}\t{data['text']}")
                    results["test_clean_ref"].append(f"{fid}\t{data['gt']}")
                elif "test-other" in data["audio"]:
                    results["test_other_text"].append(f"{fid}\t{data['text']}")
                    results["test_other_ref"].append(f"{fid}\t{data['gt']}")
                elif "dev-clean" in data["audio"]:
                    results["dev_clean_text"].append(f"{fid}\t{data['text']}")
                    results["dev_clean_ref"].append(f"{fid}\t{data['gt']}")
                elif "dev-other" in data["audio"]:
                    results["dev_other_text"].append(f"{fid}\t{data['text']}")
                    results["dev_other_ref"].append(f"{fid}\t{data['gt']}")

        for key in results:
            results[key] = "\n".join(results[key])
            with open(osp.join(args.out_dir, f"{key}.txt"), "w") as f:
                f.write(results[key])
    else:
        output_text = model.inference(args.input_speech)
        print(output_text)
