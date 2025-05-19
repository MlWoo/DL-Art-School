import gc
import random
from typing import Optional

import jiwer
import torch
from einops import rearrange
from model.audio.module.usm_encoder import USMSpeechEncoder
from model.base import BaseModule
from model.module.base.mask import sequence_mask
from model.util.ops import safe_stack  # noqa F401
from model.vqvae.quantizers import RandomProjectionQuantizer
from torch import Tensor, nn
from torch.nn import functional as F
from trainer.networks import register_model
from trainer.util import set_requires_grad
from utils.checkpoint import load_checkpoint
from utils.options import opt_get
from utils.registry import construct_from_kwargs


class BestRqFramework(BaseModule):
    def __init__(
        self,
        input_dim: int = 128,
        num_codebooks: int = 16,
        codebook_size: int = 4096,
        codebook_dim: int = 512,
        chunkwise_size: Optional[int] = None,
        encoder: Optional[dict] = None,
        enable_input_norm: bool = False,
        mask_time: int = 400,
        mask_prob: float = 0.01,
        stride_time: int = 10,
        num_visual_debug: int = 1,
        track_inputs: bool = True,
        num_attn_visual_debug: int = 0,
        debug_val_keys: list[str] = ["input_length", "input_size"],
        *args,
        **kwargs,
    ):
        super(BestRqFramework, self).__init__(
            track_inputs=track_inputs,
            num_attn_visual_debug=num_attn_visual_debug,
            num_visual_debug=num_visual_debug,
            debug_val_keys=debug_val_keys,
        )
        if encoder is None:
            heads: int = (8,)
            ff_dim: int = (128,)
            depth: int = (5,)
            depthwise_conv_kernel_size: int = (5,)
            hidden_dim: int = (512,)
            sub_layers: int = (2,)
            sub_stride: int = (2,)
            sub_num_resnet_blocks: int = (3,)
            activation: str = ("relu",)
            dropout: float = (0.0,)
            use_group_norm: bool = (False,)
            conv_first: bool = (False,)
            self.encoder = USMSpeechEncoder(
                in_dim=input_dim,
                heads=heads,
                ffn_dim=ff_dim,
                depth=depth,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                hidden_dim=hidden_dim,
                sub_layers=sub_layers,
                sub_stride=sub_stride,
                sub_num_resnet_blocks=sub_num_resnet_blocks,
                activation=activation,
                dropout=dropout,
                use_group_norm=use_group_norm,
                conv_first=conv_first,
                *args,
                **kwargs,
            )
        else:
            encoder_additional_parameters = {"input_dim": input_dim, "chunkwise_size": chunkwise_size}
            self.encoder = construct_from_kwargs(encoder, additional_parameters=encoder_additional_parameters)

        self.codebook_size = codebook_size

        self.reduction_factors = self.encoder.reduction_factors
        # features are only normalized but not projected!

        self.enable_input_norm = enable_input_norm

        self.out_linears = nn.ModuleList()
        self.random_projection_quantizers = nn.ModuleList()
        for _ in range(num_codebooks):
            out_linear = nn.Linear(self.encoder.output_dim, codebook_size)
            rpq = RandomProjectionQuantizer(
                input_feature_size=input_dim,
                reduction_factors=self.reduction_factors,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
            )
            self.out_linears.append(out_linear)
            self.random_projection_quantizers.append(rpq)

        self.num_time_steps = int(mask_time // (stride_time * self.reduction_factors))
        self.mask_prob = mask_prob
        self.chunkwise_size = chunkwise_size

    def masking(self, input_values: Tensor, input_lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            input_values (Tensor): with shape `(B, L, D)`
            input_lengths (Tensor): with shape `(B)'

        Returns:
            tuple(
            Tensor with shape `(B, L, D)`
            Tensor with shape `(B, L)`
            )
        """
        batch_size, num_steps, hidden_size = input_values.size()

        # non mask: 0, maks: 1
        time_mask_indices = torch.zeros(
            batch_size, num_steps + self.num_time_steps, device=input_values.device, dtype=torch.bool
        )

        for batch in range(batch_size):
            time_mask_idx_candidates = list(range(int(input_lengths[batch])))
            k = max(1, int((self.mask_prob * input_lengths[batch]).round()))
            start_time_mask_idx_array = torch.tensor(
                random.sample(time_mask_idx_candidates, k=k), device=input_values.device, dtype=torch.long
            )
            for i in range(self.num_time_steps):
                # mask的帧数
                time_mask_indices[batch, start_time_mask_idx_array + i] = 1

        time_mask_indices = time_mask_indices[:, : -self.num_time_steps]
        num_masks = sum(time_mask_indices.flatten())

        # Replace to random value where mask
        random_values = torch.normal(
            mean=0, std=0.1, size=(num_masks, hidden_size), device=input_values.device, dtype=input_values.dtype
        )

        target_values = input_values[time_mask_indices == 1]
        input_values[time_mask_indices == 1] = random_values

        return input_values, target_values, time_mask_indices

    def cal_mel_feats(self, input_values: Tensor, input_lengths: Tensor = None):
        """
        Args:
            input_values (Tensor): with shape `(B, T, D)`
            input_lengths (Tensor): with shape `(B)`
        """
        pass

    def forward(self, input_values: Tensor, input_lengths: Tensor = None):
        """
        Args:
            input_values (Tensor): with shape `(B, T, D)`
            input_lengths (Tensor): with shape `(B)`

        Returns:

        """
        with torch.no_grad():
            # [B, L, 80]
            batch_size, dim, num_steps = input_values.size()

            if self.debug_info is not None and self.debug_info.get("will_visual", False) and self.num_visual_debug > 0:
                log_num = min(16, batch_size)
                if log_num < 4:
                    self.debug_info["input_values"] = input_values.detach()[:1].reshape(1, 1, 1, dim, num_steps).cpu()
                    self.debug_info["input_lengths"] = input_lengths.detach()[:1].cpu()
                    self.debug_info["dim_lengths"] = torch.LongTensor([dim]).cpu()
                else:
                    log_rows = log_num // 4
                    self.debug_info["input_values"] = (
                        input_values.detach()[: (4 * log_rows)].reshape(1, log_rows, 4, dim, num_steps).cpu()
                    )
                    self.debug_info["input_lengths"] = input_lengths.detach()[: (4 * log_rows)].cpu()
                    self.debug_info["dim_lengths"] = torch.LongTensor(
                        [
                            dim,
                        ]
                        * (4 * log_rows)
                    ).cpu()

            if self.chunkwise_size is not None:
                chunks = (num_steps - 1) // self.chunkwise_size + 1
                new_num_steps = chunks * self.chunkwise_size
                if new_num_steps != num_steps:
                    input_values = F.pad(input_values, [0, new_num_steps - num_steps])
                    num_steps = new_num_steps

            input_values = rearrange(input_values, "b c l -> b l c")

            if not num_steps % self.reduction_factors == 0:
                num_steps = (num_steps // self.reduction_factors) * self.reduction_factors
                input_values = input_values[:, :num_steps]

            # [B, L//4 * 4, 128] => # [B, L//4, 512]
            input_values = input_values.reshape(batch_size, -1, self.reduction_factors * dim)

            if self.enable_input_norm:
                input_values = F.normalize(input_values, p=2.0, dim=-1)

            quantized_input_lengths = input_lengths // self.reduction_factors
            recover_input_lengths = quantized_input_lengths * self.reduction_factors

            masked_input_values, masked_target_values, time_mask_indices = self.masking(
                input_values, quantized_input_lengths
            )
            # [B, L//4 * 4, 512]
            masked_input_values = masked_input_values.reshape(batch_size, num_steps, -1)

            labels_list = []
            for rpq in self.random_projection_quantizers:
                labels = rpq(masked_target_values)
                labels_list.append(labels)

            del masked_target_values

        encoder_out = self.encoder(
            masked_input_values,
            input_lengths=recover_input_lengths,
            num_attn=1 if self.run_info.get("will_visual", False) else 0,
            out_score_mask=True if self.run_info.get("will_visual", False) else False,
        )

        if self.debug_info is not None:
            self.debug_info["input_size"] = input_values.shape[0] * input_values.shape[1]
            self.debug_info["input_length"] = input_values.shape[1]
            if self.debug_info.get("will_visual", False):
                self.debug_info["attn"] = safe_stack(encoder_out.attentions, dim=1)
                self.debug_info["reduced_lengths"] = input_lengths.detach() // self.reduction_factors
                self.debug_info["score_mask"] = safe_stack(encoder_out.score_mask, dim=1)

        last_hidden_state = encoder_out.last_hidden_state
        del encoder_out
        targets = last_hidden_state[time_mask_indices].contiguous()

        losses = 0.0
        for out_linear, labels in zip(self.out_linears, labels_list):
            pred_labels = out_linear(targets)
            loss = F.cross_entropy(pred_labels, labels)
            losses = losses + loss

        return pred_labels, losses

    def apply_compile(self):
        self.encoder.apply_compile()

    def infer(self, input_values, input_lengths, get_hidden=False, get_labels=True):
        # batch_size, dim, _  = input_values.size()
        input_values = rearrange(input_values, "b c l -> b l c")
        # [B, L, 80]
        batch_size, num_steps, dim = input_values.size()

        if not num_steps % self.reduction_factors == 0:
            num_steps = (num_steps // self.reduction_factors) * self.reduction_factors
            input_values = input_values[:, :num_steps]

        # [B, L//4 * 4, 128] => # [B, L//4, 512]
        input_values = input_values.reshape(batch_size, -1, self.reduction_factors * dim)

        if self.enable_input_norm:
            input_values = self.layer_norm(input_values)

        quantized_input_lengths = input_lengths // self.reduction_factors
        recover_input_lengths = quantized_input_lengths * self.reduction_factors

        masked_input_values, masked_target_values, time_mask_indices = self.masking(
            input_values, quantized_input_lengths
        )

        # [B, L//4 * 4, 512]
        masked_input_values = masked_input_values.reshape(batch_size, num_steps, -1)

        enc1_out, enc2_out, _, _ = self.encoder(masked_input_values, input_lengths=recover_input_lengths)

        if get_hidden:
            return enc1_out

        if get_labels:
            labels = self.random_projection_quantizer(input_values, time_mask_indices)
            targets = enc2_out[time_mask_indices].contiguous()
            targets_out = self.out_linear(targets)
            logits = torch.softmax(targets_out, 1)
            logits_ids = torch.argmax(logits, 1)

        return labels, logits_ids

    def visual_cfg(self):
        plot_cfg = dict(
            input_values=dict(
                tensor_keys=[
                    "input_values",
                ],
                shapes_keys=[
                    (
                        "input_lengths",
                        "dim_lengths",
                    ),
                ],
                t_labels=["col"],
                l_labels=["row"],
                visual_methods=["show"],
                color_info=[None],
                align_direction="h",
                width=4,
                height=4,
            ),
            token_attn=dict(
                tensor_keys=[
                    "attn",
                ],
                shapes_keys=[
                    ("reduced_lengths", "reduced_lengths"),
                ],
                t_labels=["head"],
                l_labels=["layers"],
                visual_methods=["show"],
                color_info=[None],
                align_direction="h",
                width=4,
                height=4,
            ),
            score_mask=dict(
                tensor_keys=[
                    "score_mask",
                ],
                shapes_keys=[
                    ("reduced_lengths", "reduced_lengths"),
                ],
                t_labels=["head"],
                l_labels=["layers"],
                visual_methods=["show"],
                color_info=[None],
                align_direction="v",
                width=4,
                height=4,
            ),
        )
        return plot_cfg


@register_model
def register_best_rq(opt_net, opt):
    return BestRqFramework(**opt_get(opt_net, ["kwargs"], {}))


class BestRqCTC(BestRqFramework):
    def __init__(
        self,
        tokenizer,
        input_dim: int = 128,
        num_codebooks: int = 16,
        codebook_size: int = 4096,
        codebook_dim: int = 512,
        chunkwise_size: Optional[int] = None,
        encoder=dict(),
        decoder=None,
        enable_input_norm: bool = False,
        mask_time: int = 400,
        mask_prob: float = 0.01,
        stride_time: int = 10,
        num_visual_debug=1,
        track_inputs: bool = True,
        layer_idx: int = -1,
        probe: bool = True,
        vocab_size=28,
        ctc_zero_infinity=False,
        ctc_loss_reduction="sum",
        pad_token_id: int = 0,
        pretrained_model: str = None,
        check_pretrain: bool = False,
        enable_output_ln: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            input_dim,
            num_codebooks,
            codebook_size,
            codebook_dim,
            chunkwise_size,
            encoder,
            enable_input_norm,
            mask_time,
            mask_prob,
            stride_time,
            num_visual_debug,
            track_inputs,
            *args,
            **kwargs,
        )

        if pretrained_model is not None:
            try:
                load_checkpoint(self, pretrained_model, raw_model=True)
            except Exception:
                import warnings

                warnings.warn("pretrained model failed to load")
        if not check_pretrain:
            del self.out_linears
            del self.random_projection_quantizers
        if probe:
            set_requires_grad(self.encoder, False, set_to_none=None)
            for p in self.encoder.parameters():
                p.requires_grad = False
                p.DO_NOT_TRAIN = True

        if enable_output_ln:
            self.out_ln = nn.LayerNorm(self.encoder.num_model_dim)
        else:
            self.out_ln = nn.Identity()

        if decoder is None:
            self.decoder = nn.Linear(self.encoder.num_model_dim, vocab_size)
        else:
            decoder_additional_parameters = {"num_input_dim": self.encoder.num_model_dim}
            self.decoder = construct_from_kwargs(decoder, additional_parameters=decoder_additional_parameters)

        self.tokenizer = tokenizer

        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.pad_token_id = pad_token_id
        self.layer_idx = layer_idx
        self.check_pretrain = check_pretrain

    def cer_compute(self, predictions, references, chunk_size=None):
        if chunk_size is None:
            return jiwer.cer(references, predictions)
        start = 0
        end = chunk_size
        HIT, SUB, DEL, INS = 0, 0, 0, 0
        while start < len(references):
            chunk_metrics = jiwer.cer(references[start:end], predictions[start:end], return_dict=True)
            HIT = HIT + chunk_metrics["hits"]
            SUB = SUB + chunk_metrics["substitutions"]
            DEL = DEL + chunk_metrics["deletions"]
            INS = INS + chunk_metrics["insertions"]
            start += chunk_size
            end += chunk_size
            del chunk_metrics
            gc.collect()
        return float(HIT + DEL + INS) / float(HIT + SUB + DEL + INS)

    def wer_compute(self, predictions, references, chunk_size=None):
        if chunk_size is None:
            return jiwer.wer(references, predictions)
        start = 0
        end = chunk_size
        HIT, SUB, DEL, INS = 0, 0, 0, 0
        while start < len(references):
            chunk_metrics = jiwer.compute_measures(references[start:end], predictions[start:end])
            HIT = HIT + chunk_metrics["hits"]
            SUB = SUB + chunk_metrics["substitutions"]
            DEL = DEL + chunk_metrics["deletions"]
            INS = INS + chunk_metrics["insertions"]
            start += chunk_size
            end += chunk_size
        return float(SUB + DEL + INS) / float(HIT + SUB + DEL + INS)

    def compute_metrics(self, pred_logits, labels):
        pred_ids = torch.argmax(pred_logits, axis=-1)
        pred_str = self.tokenizer.batch_decode(pred_ids)

        labels[labels == -1] = self.tokenizer.tokenizer.pad_token_id
        # we do not want to group tokens when computing the metrics
        label_str = self.tokenizer.batch_decode(labels, group_tokens=False)

        wer = self.wer_compute(predictions=pred_str, references=label_str, chunk_size=1000)
        # label_str = self.tokenizer.batch_decode(labels)
        cer = self.cer_compute(predictions=pred_str, references=label_str, chunk_size=1000)
        return torch.Tensor([wer]), torch.Tensor([cer])

    def forward_body(self, input_values, input_lengths, labels):
        with torch.no_grad():
            batch_size, dim, num_steps = input_values.size()
            if self.chunkwise_size is not None:
                chunks = (num_steps - 1) // self.chunkwise_size + 1
                new_num_steps = chunks * self.chunkwise_size
                if new_num_steps != num_steps:
                    input_values = F.pad(input_values, [0, new_num_steps - num_steps])
                    num_steps = new_num_steps

            input_values = rearrange(input_values, "b c l -> b l c")

            if not num_steps % self.reduction_factors == 0:
                num_steps = (num_steps // self.reduction_factors) * self.reduction_factors
                input_values = input_values[:, :num_steps]

            # [B, L//4 * 4, 128] => # [B, L//4, 512]
            input_values = input_values.reshape(batch_size, -1, self.reduction_factors * dim)

            if self.enable_input_norm:
                input_values = self.layer_norm(input_values)

            recover_input_values = input_values.reshape(batch_size, num_steps, -1)

            quantized_input_lengths = input_lengths // self.reduction_factors
            recover_input_lengths = quantized_input_lengths * self.reduction_factors
            self.encoder.eval()
            enc2_out, _ = self.encoder(recover_input_values, recover_input_lengths, layer_idx=self.layer_idx)
        enc2_out = self.out_ln(enc2_out)
        logits = self.decoder(enc2_out)

        # assuming that padded tokens are filled with -100
        # when not being attended to
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = labels.masked_select(labels_mask)

        # ctc_loss doesn't support fp16
        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        with torch.backends.cudnn.flags(enabled=False):
            loss = nn.functional.ctc_loss(
                log_probs,
                flattened_targets,
                quantized_input_lengths,
                target_lengths,
                blank=self.pad_token_id,
                reduction=self.ctc_loss_reduction,
                zero_infinity=self.ctc_zero_infinity,
            )
        with torch.no_grad():
            input_mask = sequence_mask(quantized_input_lengths, max_len=logits.size(1))
            logits = logits * input_mask.unsqueeze(-1)
        return loss, logits

    def forward(self, input_values, input_lengths, labels):
        if self.training:
            loss, _ = self.forward_body(input_values, input_lengths, labels)
            return loss
        else:
            loss, pred_logits = self.forward_body(input_values, input_lengths, labels)
            wer, cer = self.compute_metrics(pred_logits, labels)
            return loss, wer, cer


@register_model
def register_best_rq_ctc(opt_net, opt):
    return BestRqCTC(**opt_get(opt_net, ["kwargs"], {}))


if __name__ == "__main__":
    batch_size = 4
    length = 256 * 4
    channel = 80
    input = torch.randn(batch_size, channel, length).cuda(0)
    input_length = torch.randint(length - 16, length, (batch_size,)).cuda(0)
    input_length[-1] = length
    print(f"input_length: {input_length}")
    model = BestRqFramework(channel, mask_prob=0.05).cuda(0)
    model(input, input_length)
