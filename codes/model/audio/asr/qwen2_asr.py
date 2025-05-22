import os
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import BaseModule
from model.module.base.conv import Conv1d, LConv1DBlock
from model.module.base.mask import sequence_mask
from model.module.base.transformer import Transformer
from model.module.wenet.label_smoothing_loss import LabelSmoothingLoss
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

    def __init__(self, encoder_dim, stride, llm_dim, hidden_size=None):
        super().__init__()
        kernel_size = min(stride * 2 + 1, 5)
        self.conv1d = Conv1d(
            in_channels=encoder_dim,
            out_channels=llm_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            causal=True,
        )
        self.ln = nn.LayerNorm(llm_dim)
        hidden_size = default(hidden_size, llm_dim * 4)
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


class ProjectorLConv1d(nn.Module):

    def __init__(self, encoder_dim, stride, llm_dim, hidden_size=None):
        super().__init__()
        self.linear = nn.Linear(encoder_dim * stride, llm_dim)
        self.stride = stride
        self.conv1d = LConv1DBlock(
            llm_dim,
            conv_kernel_size=5,
            conv_padding_l=4,
            conv_heads=8,
            conv_dynamic=True,
            weight_softmax=True,
            ff_ops_seq=("linear", "relu", "dropout"),
            ff_dropout_p=0.0,
            norm_before=True,
            norm_after=False,
        )

    def forward(self, x, inference=False):
        B, T, C = x.shape
        folded_T = T // self.stride
        unfolded_T = folded_T * self.stride
        x = x[:, :unfolded_T]
        x = x.reshape(B, folded_T, C * self.stride)
        x = self.linear(x)
        # lconv + ffn
        x = self.conv1d(x)

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


class FoldedFFNProjector(nn.Module):

    def __init__(self, encoder_dim, stride, llm_dim, hidden_size=None):
        super().__init__()
        self.linear = nn.Linear(encoder_dim * stride, llm_dim)
        self.stride = stride
        hidden_size = default(hidden_size, llm_dim * 4)
        self.ln = nn.LayerNorm(llm_dim)
        self.linear1 = nn.Linear(llm_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, llm_dim)
        self.act = nn.SiLU()

    def forward(self, x, **kwargs):
        B, T, C = x.shape
        folded_T = T // self.stride
        unfolded_T = folded_T * self.stride
        x = x[:, :unfolded_T]
        x = x.reshape(B, folded_T, C * self.stride)
        x = self.linear(x)

        res = x
        x = self.ln(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = x + res
        return x


class FoldedMLPProjector(nn.Module):

    def __init__(self, encoder_dim, stride, llm_dim, hidden_size=None):
        super().__init__()
        self.linear = nn.Linear(encoder_dim * stride, llm_dim)
        self.stride = stride
        hidden_size = default(hidden_size, llm_dim * 4)
        self.linear1 = nn.Linear(llm_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, llm_dim)
        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()

    def forward(self, x):
        B, T, C = x.shape
        folded_T = T // self.stride
        unfolded_T = folded_T * self.stride
        x = x[:, :unfolded_T]
        x = x.reshape(B, folded_T, C * self.stride)
        x = self.linear(x)
        x = self.act1(x)
        x = self.linear1(x)
        x = self.act2(x)
        x = self.linear2(x)
        return x


class FoldedLSTMProjector(nn.Module):

    def __init__(self, encoder_dim, stride, llm_dim, hidden_size=None):
        super().__init__()
        self.linear = nn.Linear(encoder_dim * stride, llm_dim)
        self.stride = stride
        hidden_size = default(hidden_size, llm_dim * 4)
        self.lstm = nn.LSTM(llm_dim, llm_dim, num_layers=2, batch_first=True)

    def forward(self, x):
        B, T, C = x.shape
        folded_T = T // self.stride
        unfolded_T = folded_T * self.stride
        x = x[:, :unfolded_T]
        x = x.reshape(B, folded_T, C * self.stride)
        x = self.linear(x)
        x, _ = self.lstm(x)
        return x


class TransformerProjector(nn.Module):

    def __init__(
        self,
        encoder_dim: int,
        stride: int,
        llm_dim: int,
        hidden_size: int,
        encoder_cfg: dict,
        num_layers: int = 4,
        num_heads: int = 8,
        shift: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(encoder_dim * stride, llm_dim)
        hidden_size = default(hidden_size, llm_dim * 4)
        num_attn_head_dim = llm_dim // num_heads
        attn_func_type = 7
        self.transformer = Transformer(
            attn_type="rope",
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_attn_in_dim=llm_dim,
            num_attn_out_dim=llm_dim,
            num_attn_head_dim=num_attn_head_dim,
            ffn_dim=hidden_size,
            attn_func_type=attn_func_type,
            ffn_kernel_size=1,
            ffn_act_func="swish",
            attn_dropout_p=0.0,
            cxt_dropout_p=0.0,
            ffn_dropout_p=0.0,
            pre_LN=True,
            block_LN=False,
            final_LN=True,
            use_macaron=False,
            conv_module_type="none",
            causal=True,
            padding_mode="none",
            chunkwise_size=None,
            stream_chunk_size=None,
            window_size=encoder_cfg["window_size"] // stride,
            max_len=None,
            lookforward_size=encoder_cfg["lookforward_size"] // stride,
            gradient_checkpointing=False,  # encoder_cfg["gradient_checkpointing"],
        )
        self.stride = stride
        # self.stream_chunk_size = stream_chunk_size
        # self.downsampled_chunk_size = stream_chunk_size // stride

    def forward(self, x, x_lengths):
        # step 1: enhance the chunk look-ahead ability
        # step 2: enhance the chunk inter-dependence
        # step 3: project to the llm dimension
        # return 13.5hz downsampled features with 12.5hz and inserted 1 frame

        B, T, C = x.shape
        folded_T = T // self.stride
        unfolded_T = folded_T * self.stride
        x = x[:, :unfolded_T]
        x = x.reshape(B, folded_T, C * self.stride)
        x = self.linear(x)
        x_lengths = x_lengths // self.stride
        mask = sequence_mask(x_lengths, max_len=x.shape[1]).unsqueeze(-1)
        output_values, attn_tuple, score_mask_tuple = self.transformer(x, mask)
        last_hidden_state = output_values[-1]

        return last_hidden_state


class TransformerProjectorV2(nn.Module):

    def __init__(
        self,
        encoder_dim: int,
        stride: int,
        llm_dim: int,
        hidden_size: int,
        encoder_cfg: dict,
        num_layers: int = 4,
        num_heads: int = 4,
        shift: bool = False,
    ):
        super().__init__()
        transformer_dim = min(encoder_dim * stride, 2048)  # 1280
        self.linear = nn.Linear(encoder_dim * stride, transformer_dim)

        hidden_size = default(hidden_size, llm_dim * 4)
        num_attn_head_dim = transformer_dim // num_heads
        attn_func_type = 7
        self.transformer = Transformer(
            attn_type="rope",
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_attn_in_dim=transformer_dim,
            num_attn_out_dim=transformer_dim,
            num_attn_head_dim=num_attn_head_dim,
            ffn_dim=hidden_size,
            attn_func_type=attn_func_type,
            ffn_kernel_size=1,
            ffn_act_func="swish",
            attn_dropout_p=0.0,
            cxt_dropout_p=0.0,
            ffn_dropout_p=0.0,
            pre_LN=True,
            block_LN=False,
            final_LN=True,
            use_macaron=False,
            conv_module_type="none",
            causal=True,
            padding_mode="none",
            chunkwise_size=None,
            stream_chunk_size=None,
            window_size=min(encoder_cfg["window_size"] // stride, 25),
            max_len=None,
            lookforward_size=encoder_cfg["lookforward_size"] // stride,
            gradient_checkpointing=False,  # encoder_cfg["gradient_checkpointing"],
        )
        self.stride = stride
        # self.stream_chunk_size = stream_chunk_size
        # self.downsampled_chunk_size = stream_chunk_size // stride
        self.linear_out = nn.Linear(transformer_dim, llm_dim)

    def forward(self, x, x_lengths):
        # step 1: enhance the chunk look-ahead ability
        # step 2: enhance the chunk inter-dependence
        # step 3: project to the llm dimension
        # return 13.5hz downsampled features with 12.5hz and inserted 1 frame

        B, T, C = x.shape
        folded_T = T // self.stride
        unfolded_T = folded_T * self.stride
        x = x[:, :unfolded_T]
        x = x.reshape(B, folded_T, C * self.stride)
        x = self.linear(x)
        x_lengths = x_lengths // self.stride
        mask = sequence_mask(x_lengths, max_len=x.shape[1]).unsqueeze(-1)
        output_values, attn_tuple, score_mask_tuple = self.transformer(x, mask)
        last_hidden_state = output_values[-1]
        last_hidden_state = self.linear_out(last_hidden_state)
        return last_hidden_state


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
        encoder_freeze_end_layer_idx: Optional[int] = None,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 256,
        lora_dropout: float = 0.0,
        lora_bias: str = "none",
        lora_modules: List[str] = ["gate_proj", "up_proj", "down_proj"],
        inference_mode: bool = False,
        attn_implementation: str = "flash_attention_2",
        dtype: torch.dtype = torch.bfloat16,
        stride: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ):
        super(ASRModel, self).__init__(
            track_inputs=track_inputs,
            num_attn_visual_debug=num_attn_visual_debug,
            debug_val_keys=debug_val_keys,
        )
        config = AutoConfig.from_pretrained(pretrain_path)
        self.llm_config = config
        llm = AutoModelForCausalLM.from_pretrained(
            pretrain_path,
            attn_implementation=attn_implementation,
            cache_dir="/home/wumenglin/cache/models",
            torch_dtype=dtype,
        )
        if config.tie_word_embeddings:
            llm.lm_head.weight = nn.Parameter(llm.lm_head.weight.clone())
        set_requires_grad(llm, False, set_to_none=True)
        for param in llm.parameters():
            param.DO_NOT_TRAIN = True
            param.grad = None

        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            # 在target_modules中，我们只需要指定这些层的通用名称。
            # mlp_target_modules = ["gate_proj", "up_proj", "down_proj"]
            # --- 可选：如果你还想对注意力机制的QKV层也应用LoRA ---
            # attention_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            # target_modules = mlp_target_modules + attention_target_modules
            # target_modules = mlp_target_modules  # 本示例仅针对MLP
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                task_type=TaskType.CAUSAL_LM,
                lora_dropout=lora_dropout,
                bias=lora_bias,
                target_modules=lora_modules,
                inference_mode=inference_mode,
            )
            self.llm = get_peft_model(llm, lora_config)
            self.llm.print_trainable_parameters()
        else:
            self.llm = llm

        self.llm.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained(pretrain_path)

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
                if isinstance(encoder_layer_idx, list):
                    last_encoder_layer_idx = encoder_layer_idx[-1]
                    self.encoder_layer_idx = encoder_layer_idx
                else:
                    last_encoder_layer_idx = encoder_layer_idx
                    self.encoder_layer_idx = [last_encoder_layer_idx]
                self.last_encoder_layer_idx = last_encoder_layer_idx

                for i, layer in enumerate(self.encoder.conformer.layers.layers):
                    if i > last_encoder_layer_idx:
                        self.encoder.conformer.layers.layers[i] = None
            else:
                self.last_encoder_layer_idx = None
                self.encoder_layer_idx = None

            self.stride = stride
            self.encoder_out_ln = nn.LayerNorm(encoder["num_model_dim"]) if enable_output_ln else nn.Identity()
            if encoder_freeze_end_layer_idx == -2:
                self.encoder.eval()
                set_requires_grad(self.encoder, False, set_to_none=True)
                for p in self.encoder.parameters():
                    p.requires_grad = False
                    p.DO_NOT_TRAIN = True
            else:
                for i in range(encoder_freeze_end_layer_idx + 1):
                    set_requires_grad(self.encoder.conformer.layers.layers[i], False, set_to_none=True)
                    for p in self.encoder.conformer.layers.layers[i].parameters():
                        p.requires_grad = False
                        p.DO_NOT_TRAIN = True
            self.encoder_freeze = encoder_freeze
        else:
            raise ValueError(f"Invalid speech encoder type: {speech_encoder_type}")

        if projector_type == "conv1d":
            self.projector = ProjectorConv1d(1024, self.stride, config.hidden_size, 4096)
        elif projector_type == "lconv1d":
            self.projector = ProjectorLConv1d(1024, self.stride, config.hidden_size, 4096)
        elif projector_type == "folded":
            self.projector = FoldedProjector(1024, self.stride, config.hidden_size)
        elif projector_type == "folded_ffn":
            self.projector = FoldedFFNProjector(1024, self.stride, config.hidden_size, 4096)
        elif projector_type == "folded_mlp":
            self.projector = FoldedMLPProjector(1024, self.stride, config.hidden_size, 4096)
        elif projector_type == "folded_lstm":
            self.projector = FoldedLSTMProjector(1024, self.stride, config.hidden_size, 4096)
        elif projector_type == "transformer":
            self.projector = TransformerProjector(
                1024 * len(self.encoder_layer_idx), self.stride, config.hidden_size, 4096, encoder
            )
        elif projector_type == "transformer_v2":
            self.projector = TransformerProjectorV2(
                1024 * len(self.encoder_layer_idx), self.stride, config.hidden_size, 2560, encoder
            )
        else:
            raise ValueError(f"Invalid projector type: {projector_type}")

        self.codebook_size = 2048
        self.text_type = text_type
        self.speech_type = speech_type
        self.asr_text_type = asr_text_type
        self.stream = stream
        self.dtype = dtype

        if lsm_weight > 0:
            self.criterion_ce = LabelSmoothingLoss(
                size=self.llm_config.vocab_size,
                padding_idx=-100,
                smoothing=lsm_weight,
                normalize_length=True,  # length_normalized_loss,
            )
        else:
            self.criterion_ce = nn.CrossEntropyLoss()

    def _merge_input_ids_with_audio_features(
        self, audio_features, num_audio_tokens, inputs_embeds, input_ids, attention_mask, labels
    ):
        """
        Merge input_ids with with audio features into final embeddings

        Args:
            audio_features (`torch.Tensor` of shape `(num_audios, max_audio_tokens, embed_dim)`):
                All audio vectors of all audios in the batch
            num_audio_tokens (`torch.LongTensor` of shape `(num_audios)`):
                The length of audio embeddings of each audio as stacked in `audio_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with audio embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with audio token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                labels need to be recalculated to support training (if provided)
        Returns:
            final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

        Explanation:
            each audio has variable length embeddings, with length specified by num_audio_tokens
            audio_features is concatenation of all audio embed vectors
            task: fill each <|AUDIO|> with the correct number of audio embeddings
            Example:
                X (5 tokens), Y (3 tokens), Z (8 tokens)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but audio token sizes are different, then cannot infer left or right padding
                ```python
                url1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
                audio1, _ = librosa.load(BytesIO(urlopen(url1).read()), sr=processor.feature_extractor.sampling_rate)
                url2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"
                audio2, _ = librosa.load(BytesIO(urlopen(url2).read()), sr=processor.feature_extractor.sampling_rate)
                prompts = [
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                ]
                inputs = processor(text=prompts, audios=[audio1, audio2], return_tensors='pt', padding=True).to("cuda")
                    audio1 has 101 tokens, while audio2 has 72 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(
            num_audio_tokens.device
        ) < num_audio_tokens.unsqueeze(1)
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)
        batch_size, sequence_length = input_ids.shape
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)

        left_padding = True
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                # both side is 1, so cannot tell
                left_padding = self.padding_side == "left"
            else:
                # invalid attention_mask
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")
        # 1. Create a mask to know where special audio tokens are
        special_audio_token_mask = input_ids == self.config.audio_token_index
        num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)

        # In case the Audio model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)
        num_audio_tokens = num_audio_tokens.to(target_device)
        batch_indices, non_audio_indices = torch.where(
            (input_ids != self.config.audio_token_index) & (attention_mask == 1)
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged audio-text sequence.
        # `special_audio_token_mask` identifies audio tokens. Each audio token will be replaced by `audio_feat_lengths - 1` text tokens.
        # `torch.cumsum` computes how each audio token shifts subsequent text token positions.
        token_placeholder_num = torch.zeros_like(input_ids)
        token_placeholder_num[special_audio_token_mask] = num_audio_tokens.long() - 1
        token_placeholder_num = token_placeholder_num + 1
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_token_num, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_token_num, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_token_num), self.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<audio>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the audio features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full_like(final_attention_mask, self.config.ignore_index).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]

        # 5. Fill the embeddings corresponding to the audios. Anything that is still zeros needs filling
        audio_to_overwrite = torch.full(
            (batch_size, max_token_num), True, dtype=torch.bool, device=inputs_embeds.device
        )
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        seq_indices = torch.arange(max_token_num).unsqueeze(0).to(target_device)
        seq_indices = seq_indices.expand(batch_size, max_token_num)

        if left_padding:
            # exclude padding on the left
            max_token_num = max_token_num.to(target_device)
            val = (max_token_num - seq_indices) <= (
                token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
            )[:, None]
        else:
            # exclude padding on the right
            val = seq_indices < (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]

        audio_to_overwrite &= val

        if audio_to_overwrite.sum() != num_audio_tokens.sum():
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {num_special_audio_tokens} while"
                f" the number of audio given to the model is {num_audios}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = (
            masked_audio_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

    def get_embeddings(self, input_ids, speech, speech_lengths):
        if self.codec == "whisper":
            with torch.no_grad():
                if self.stream:
                    audios = speech[:, :].to(self.dtype)
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
                    encoder_output = self.encoder(
                        speech.permute(0, 2, 1),
                        speech_lengths,
                        layer_idx=self.last_encoder_layer_idx,
                        out_hidden_states=True,
                    )
                    if len(self.encoder_layer_idx) == 1:
                        speech_embeds = encoder_output.last_hidden_state
                    else:
                        speech_embeds_list = []
                        for i in self.encoder_layer_idx:
                            speech_embeds_list.append(encoder_output.hidden_states[i])
                        speech_embeds = torch.cat(speech_embeds_list, dim=-1)
                    speech_embeds = self.encoder_out_ln(speech_embeds)
                    speech_embeds = speech_embeds.detach().to(self.dtype)
                    del encoder_output
            else:
                trim_size = 0

                encoder_output = self.encoder(
                    speech.permute(0, 2, 1),
                    speech_lengths,
                    layer_idx=self.last_encoder_layer_idx,
                    out_hidden_states=True,
                )
                if len(self.encoder_layer_idx) == 1:
                    speech_embeds = encoder_output.last_hidden_state
                else:
                    speech_embeds_list = []
                    for i in self.encoder_layer_idx:
                        speech_embeds_list.append(encoder_output.hidden_states[i])
                    speech_embeds = torch.cat(speech_embeds_list, dim=-1)
                del encoder_output
                if trim_size > 0:
                    speech_embeds = speech_embeds[:, trim_size:]

                speech_embeds = self.encoder_out_ln(speech_embeds).to(self.dtype)

            stride = self.stride
            reduction_factor = self.encoder.reduction_factor
            speech_embeds = self.projector(speech_embeds, x_lengths=speech_lengths).to(self.dtype)
            speech_mask = sequence_mask(speech_lengths // reduction_factor // stride, max_len=speech_embeds.shape[1])
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
        # speech_lengths_new = input_speech_mask.sum(-1)
        # speech_mask = sequence_mask(speech_lengths_new, max_len=speech_embeds.shape[1])
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
            loss_ce = self.criterion_ce(logits.view(-1, logits.size(-1)).float(), output_ids.view(-1))

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

    # @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(self, input_speech):
        speech_token_length = input_speech.shape[-1] // self.stride // self.encoder.reduction_factor
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
                # top_p=1.0,
                num_beams=1,
                max_new_tokens=256,
                # eos_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(output_tokens[0][:-1])
        return text

    def from_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict, strict=False)
        # if self.llm_config.tie_weights:
        #     self.llm.lm_head.weight = self.llm.model.embed_tokens.weight

    def inference(self, input_speech):
        import torchaudio

        audio, ori_sr = torchaudio.load(input_speech)
        if audio.shape[1] / ori_sr > 39.99:
            return False, None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        audio = audio.to(device)
        # resample to 16000
        audio = torchaudio.functional.resample(audio, ori_sr, 16000)
        from trainer.injectors.audio2mels import CanonicalTorchMelSpectrogram

        mel_cfg = {
            "in": "input",
            "out": "output",
            "mel_fmin": 0,
            "mel_fmax": 8000,
            "sampling_rate": 16000,
            "n_mel_channels": 128,
            "filter_length": 400,
            "hop_length": 160,
            "win_length": 400,
            "true_normalization": False,
            "mel_norm_file": "/home/wumenglin/cache/peoples_speech/dirty/mel_stat_10001.pt",
        }
        injector = CanonicalTorchMelSpectrogram(opt=mel_cfg, env=None)
        audio = injector({"input": audio})["output"]
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
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/home/wumenglin/repo-dev/DL-Art-School-dev/experiments/minmo_asr_librispeech+gigaspeech_conformer_chunk240ms-v1-look_forward-sft4/models-bkp/90000_gpt.pth",
    )
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

    # pretrain_path = "Qwen/Qwen2.5-1.5B-Instruct"

    pretrain_path = "Qwen/Qwen2.5-7B-Instruct"

    # encoder_pretrained = "/home/wumenglin/repo-dev/DL-Art-School-dev/experiments/streamconformer-bestrq_gelu-causal-sub1-stat-ft240ms-v2-no-final-ln-fix-mask-25hz/models-clean/250000_generator.pth"  # noqa E501
    # encoder_pretrained = "/home/wumenglin/repo-dev/DL-Art-School-dev/experiments/streamconformer-bestrq_gelu-causal-sub1-stat-ft240ms-v2-no-final-ln-fix-mask-25hz/models-clean/590000_generator.pth"  # noqa E501
    encoder_pretrained = "/home/wumenglin/repo-dev/DL-Art-School-dev/experiments/streamconformer-bestrq_gelu-causal-sub1-stat-ft240ms-v2-no-final-ln-fix-mask-25hz/models/1010000_generator.pth"  # noqa E501
    model = ASRModel(
        pretrain_path=pretrain_path,
        text_type=0,
        speech_type=1,
        asr_text_type=2,
        # feature_size=args.feature_size,
        # projector_type="folded_mlp",
        projector_type="transformer_v2",
        speech_encoder_type="conformer",
        stream=True,
        # encoder={
        #     "class_name": "model.audio.module.encoders.StreamConformerEncoder",
        #     "attn_type": "wo-pos",
        #     "num_preconformer_layers": 3,
        #     "num_preconformer_dim": 384,
        #     "sub_type": "2d",
        #     "sub_layers": 2,
        #     "num_input_dim": 128,
        #     "num_sub_dim": [8, 32],
        #     "num_model_dim": 1024,
        #     "proj_dropout_p": 0.1,
        #     "num_conformer_layers": 20,
        #     "num_heads": 8,
        #     "attn_dropout_p": 0.1,
        #     "cxt_dropout_p": 0.1,
        #     "ffn_dropout_p": 0.1,
        #     "conv_module_kernel_size": 4,
        #     "causal": True,
        #     "window_size": 128,
        #     "max_len": 2048,
        #     "norm_groups": 8,
        #     "conv_activation": "gelu",
        # },
        encoder={
            "class_name": "model.audio.module.bestrq_flex.BestRqConformerEncoder",
            "input_dim": 128,
            "input_channels": 1,
            "num_attention_heads": 8,
            "hidden_size": 1024,
            "ffn_dim": 4096,
            "num_hidden_layers": 24,
            "conv_depthwise_kernel_size": 4,
            "feat_proj_dropout": 0.0,
            "activation_dropout": 0.0,
            "hidden_dropout": 0.0,
            "max_source_positions": 3000,
            "no_scale_embedding": False,
            "hidden_act": "swish",
            "conformer_conv_dropout": 0.0,
            "position_embeddings_type": "relative",
            "attention_dropout": 0.0,
            "rotary_embedding_base": 10000,
            "layerdrop": 0.0,
            "final_dropout": 0.0,
            "num_preconformer_layers": 0,
            "num_preconformer_heads": 4,
            "preconformer_hidden_size": 384,
            "preconformer_ffn_dim": 1536,
            "preconformer_input_feature_projection": False,
            "causal": True,
            "sub_layers": 2,
            "conv_hidden_size": [8, 32],
            "compile": False,
            "window_size": 256,
            "stream_chunk_size": None,  # 160ms
            "lookforward_size": 6,  # 240ms  240 / 10 / 2 = 12
            "final_LN": False,
        },
        encoder_pretrained=encoder_pretrained,
        # encoder_pretrained="/home/wumenglin/repo-dev/dl-art-school/experiments/streamconformer-bestrq-large_lr_h100x4_gelu8-v2/models/120000_generator.pth",
        encoder_freeze=True,
        encoder_layer_idx=11,
        encoder_freeze_end_layer_idx=-1,
        enable_output_ln=False,
        feature_size=128,
        use_lora=True,
        attn_implementation="sdpa",
        lora_alpha=32,
        lora_dropout=0.1,
        lora_r=8,
        inference_mode=True,
        dtype=torch.float32,
        stride=2,
    )
    model.from_checkpoint(args.checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)
    count_dict = {
        "dev-clean": 0,
        "dev-other": 0,
        "test-clean": 0,
        "test-other": 0,
    }
    if args.input_json is not None:
        results = []
        with open(args.input_json, "r") as f:
            for i, line in enumerate(tqdm(f)):
                data = json.loads(line)
                input_speech = data["audio"]
                # if "dev-clean" in input_speech:
                #     count_dict["dev-clean"] += 1
                #     if count_dict["dev-clean"] > 1000:
                #         continue
                # elif "dev-other" in input_speech:
                #     count_dict["dev-other"] += 1
                #     if count_dict["dev-other"] > 1000:
                #         continue
                # else:
                #     continue
                if "test-clean" in input_speech:
                    count_dict["test-clean"] += 1
                    if count_dict["test-clean"] > 1000:
                        continue
                elif "test-other" in input_speech:
                    count_dict["test-other"] += 1
                    if count_dict["test-other"] > 1000:
                        continue
                else:
                    continue
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
