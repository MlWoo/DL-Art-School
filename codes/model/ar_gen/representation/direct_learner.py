
import math
import random
from math import sqrt
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from model.module.base.gradient_reversal import GradientReversal
from model.module.base.mask import sequence_mask
from model.util.arch import eval_decorator
from model.util.ops import rand_slice_segments
from torch import Tensor, einsum, nn
from trainer.networks import register_model
from trainer.util import set_requires_grad
from utils.options import opt_get
from utils.registry import construct_from_kwargs


class OrthogonalEmbeddingLoss(nn.Module):
    __constants__ = ["dim", "eps", "sim_eps"]
    dim: int
    eps: float
    sim_eps: float

    def __init__(self, dim: int = 1, eps: float = 1e-8, sim_eps=1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.sim_eps = sim_eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        sim = F.cosine_similarity(x1, x2, self.dim, self.eps)
        return torch.mean(torch.clamp(torch.abs(sim), min=self.sim_eps))


class RepresentationLearner(nn.Module):
    """A basic representation learner.

    The data goes through:
    1. vae_encoder (extracts letent representation)
    2. res_encoder (extracts residual info like noise)
    3. bottleneck
    4. reconstructor (reconstructs the inputs in an autoregressive or
                      deconvolutional or other way)
    """

    def __init__(
        self,
        in_channels=80,
        rand_crop_len=-1,
        codebook_dim=64,
        vae_encoder=dict(),
        vae_encoder_codebook=False,
        res_encoder=None,
        bottleneck=dict(),
        reconstructor=dict(),
        asr_head=dict(),
        global_extractor=None,
        global_extractor_crop_len=50,
        seperator=None,
        temperature=0.07,
        diversity_loss=None,
        noise_paired: bool = False,
        noise_smooth_l1_loss: bool = False,
        recon_smooth_l1_loss: bool = False,
        record_codes=False,
        record_debug_info=False,
        normalization=None,  # ((0.5,) * 3, (0.5,) * 3),
        mask_time: int = 400,
        mask_prob: float = 0.0,
        stride_time: int = 10,
        reconstructor_training: bool = True,
        ctc_zero_infinity: bool = True,
        ctc_loss_reduction: str = "mean",
        **kwargs,
    ):
        super(RepresentationLearner, self).__init__()
        self.in_channels = in_channels
        if isinstance(in_channels, (list, tuple)):
            assert (
                len(in_channels) == 1 or len(in_channels) == 2
            ), "The params `channels` should be a number or two numbers"
            if len(in_channels) == 2:
                num_channels = in_channels[1] - in_channels[0]
            else:
                num_channels = in_channels[0]
        else:
            num_channels = in_channels
        self.num_channels = num_channels
        self.rand_crop_len = rand_crop_len

        if res_encoder is None:
            self.res_encoder = None
        else:
            self.res_encoder = construct_from_kwargs(res_encoder)

        vae_encoder_additional_parameters = {"in_channels": num_channels}
        if vae_encoder_codebook:
            vae_encoder_additional_parameters["codebook_dim"] = codebook_dim

        self.vae_encoder = construct_from_kwargs(vae_encoder, additional_parameters=vae_encoder_additional_parameters)
        # prevent affecting the encoder by the dummy minibatch
        self.vae_encoder.eval()

        if getattr(self.vae_encoder, "channel_last", False):
            input_val = torch.empty((1, 256, in_channels))
            out_dim = -1
            self.vae_encoder_channel_last = True
        else:
            input_val = torch.empty((1, in_channels, 256))
            out_dim = 1
            self.vae_encoder_channel_last = False

        enc_outputs = self.vae_encoder(input_val)
        if len(enc_outputs) > 1:
            enc_out_shape = enc_outputs[0].size()
        else:
            enc_out_shape = enc_outputs.size()

        self.reduction_steps = self.vae_encoder.reduction_steps
        self.num_time_steps = int(mask_time // (stride_time * self.reduction_steps))
        self.mask_prob = mask_prob
        self.reduction_steps = self.vae_encoder.reduction_steps

        self.bottleneck = construct_from_kwargs(
            bottleneck, additional_parameters=dict(in_dim=enc_out_shape[out_dim], codebook_dim=codebook_dim)
        )

        rec_params = {"in_channels": codebook_dim, "out_channels": num_channels}

        # Compatibility with single-reconstructor checkpoints
        if "class_name" in reconstructor:
            self.reconstructor = construct_from_kwargs(reconstructor, additional_parameters=rec_params)
            self.reconstructors = {"": self.reconstructor}
        else:
            self.reconstructors = nn.ModuleDict(
                {
                    name: construct_from_kwargs(rec, additional_parameters=rec_params)
                    for name, rec in reconstructor.items()
                }
            )

        # if output multiple outputs from latents, it is ok
        self.reconstructor = construct_from_kwargs(
            reconstructor, additional_parameters={"in_channels": codebook_dim, "out_channels": num_channels}
        )
        if asr_head is None:
            self.asr_head = None
        else:
            self.asr_head = construct_from_kwargs(
                asr_head,
                additional_parameters={
                    "in_channels": codebook_dim,
                },
            )

        self.ctc_zero_infinity = ctc_zero_infinity
        self.ctc_loss_reduction = ctc_loss_reduction

        if global_extractor is None:
            self.global_extractor = None
            self.grl = None
            self.similairy_loss = None
        else:
            self.global_extractor = construct_from_kwargs(
                global_extractor, additional_parameters={"in_channels": enc_out_shape[out_dim]}
            )
            self.grl = GradientReversal()
            self.similairy_loss = OrthogonalEmbeddingLoss(sim_eps=1e-4)
        self.global_crop_len = global_extractor_crop_len

        if seperator is None:
            self.seperator = None
            self.seperator_a = None
            self.seperator_b = None
        else:
            self.seperator = construct_from_kwargs(
                seperator, additional_parameters={"in_channels": enc_out_shape[out_dim]}
            )
            self.seperator_a = nn.Conv1d(codebook_dim, codebook_dim, 1)
            self.seperator_b = nn.Conv1d(codebook_dim, codebook_dim, 1)
            # The learnable temperature parameter was initialized to the equivalent of 0.07 from (Wu et al., 2018)
            self.temperature = nn.Parameter(torch.tensor([np.log(1 / temperature)]))

        self.noise_paired = noise_paired
        self.noise_loss_fn = F.smooth_l1_loss if noise_smooth_l1_loss else F.mse_loss
        if diversity_loss is None or not hasattr(self.bottleneck, "codebook_size"):
            self.diversity = None
            self.bottleneck.diversity_enabled = False
        else:
            self.diversity = construct_from_kwargs(
                diversity_loss, additional_parameters={"discrete_bins": self.bottleneck.codebook_size}
            )
            self.bottleneck.diversity_enabled = True

        self.side_info_encoder = None
        self.recon_loss_fn = F.smooth_l1_loss if recon_smooth_l1_loss else F.mse_loss
        # self.add_probes()

        # take care of normalization within class
        self.normalization = normalization
        self.record_codes = record_codes
        if record_codes:
            self.codes = torch.zeros((1228800,), dtype=torch.long)
            self.code_ind = 0
            self.total_codes = 0
        self.internal_step = 0
        self.debug_info = {} if record_debug_info else None
        self.reduction_steps = self.vae_encoder.reduction_steps
        store_run_info = kwargs.get("store_run_info", False)
        if store_run_info:
            self.run_info = dict(epoch=0, iter=0, will_log=False)
        else:
            self.run_info = None

        if not reconstructor_training:
            set_requires_grad(self.reconstructor, False)

    def norm_data(self, x):
        if self.normalization is None:
            return x
        x = torch.norm(x - torch.mean(x)) / np.sqrt(np.prod(x.shape))
        return x

    def global_info_extract(self, common, local):
        # common B T C
        temp = self.temperature.exp()
        temp = torch.clamp(temp, max=100)
        common_len = common.size(1)
        if self.global_crop_len is None or self.global_crop_len <= 0 or self.global_crop_len >= common_len:
            positive = self.global_extractor(common)
            if self.global_extractor.pool_type == "direct":
                positive = torch.mean(positive, dim=1)
            g_common0 = F.normalize(positive, p=2, dim=-1)
            g_common1 = g_common0
        else:
            assert self.global_extractor.pool_type == "direct", "Random cropping should use raw ouput of transformer."
            output = self.global_extractor(common)

            rand_starts = np.random.randint(0, common_len - self.global_crop_len, 2)
            common0 = output[:, rand_starts[0] : (rand_starts[0] + self.global_crop_len)].mean(dim=1)
            common1 = output[:, rand_starts[1] : (rand_starts[1] + self.global_crop_len)].mean(dim=1)
            positive = (common0 + common0) * 0.5

            g_common0 = F.normalize(common0, p=2, dim=-1)
            g_common1 = F.normalize(common1, p=2, dim=-1)

        labels = torch.arange(positive.shape[0], device=positive.device)
        sim = einsum("i d, j d -> i j", g_common0, g_common1) * temp
        contrast_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) * 0.5

        local_grl = self.grl(local)
        set_requires_grad(self.global_extractor, False)
        negative = self.global_extractor(local_grl)
        if self.global_extractor.pool_type == "direct":
            negative = torch.mean(negative, dim=1)
        set_requires_grad(self.global_extractor, True)
        sim_loss = self.similairy_loss(positive, negative)

        return positive, sim_loss, contrast_loss

    def store_run_info(self, epoch, step, will_log, **kwargs):
        if self.run_info is not None:
            self.run_info["epoch"] = epoch
            self.run_info["iter"] = step
            self.run_info["will_log"] = will_log
        if hasattr(self.bottleneck, "store_run_info"):
            self.bottleneck.store_run_info(epoch, step, will_log, **kwargs)

    def get_debug_values(self, step, __):
        if self.debug_info is None:
            debug_info = {}
        else:
            debug_info = self.debug_info
        if self.record_codes and self.total_codes > 0:
            # Report annealing schedule
            debug_info.update({"histogram_codes": self.codes[: self.total_codes]})
        else:
            debug_info.update({})
        return debug_info

    def log_codes(self, codes):
        # This is so we can debug the distribution of codes being learned.
        if self.record_codes and self.internal_step % 10 == 0:
            codes = codes.flatten()
            l = codes.shape[0]
            i = self.code_ind if (self.codes.shape[0] - self.code_ind) > l else self.codes.shape[0] - l
            self.codes[i : i + l] = codes.cpu()
            self.code_ind = self.code_ind + l
            if self.code_ind >= self.codes.shape[0]:
                self.code_ind = 0
            self.total_codes += 1
        self.internal_step += 1

    def seperate_feature(self, x):
        if self.seperator is None:
            x = x.permute((0, 2, 3, 1) if len(x.shape) == 4 else (0, 2, 1))
            return x, x
        else:
            a = self.seperator_a(x)
            b = self.seperator_b(x)
            common = torch.cat([a, b], dim=0)
            feature = self.seperator(common)  # B C T
            feature = feature.permute(0, 2, 1)
            a, b = torch.chunk(feature, 2, dim=0)
            return a, b

    def asr_loss(self, logits, logits_length, text, text_lengths):
        with torch.backends.cudnn.flags(enabled=False):
            loss_pos = F.ctc_loss(
                logits,
                text,
                logits_length,
                text_lengths,
                blank=self.ctc_blank_id,
                reduction=self.ctc_loss_reduction,
                zero_infinity=self.ctc_zero_infinity,
            )

        return loss_pos

    def reconstruction_loss(self, batch, conds, needs_rec_image):
        if self.side_info_encoder is not None:
            side_info = self.side_info_encoder(batch["side_info"])
            side_info = side_info.unsqueeze(1).unsqueeze(1)
            conds = conds + (side_info,)

        details = {}
        per_pix = {}
        all_inputs = []
        mean_field = []

        for (name, rec), rec_field in zip(self.reconstructors.items(), self.reconstructors_fields):
            assert rec_field in batch, f"The field to be reconstructed '{rec_field}' not found in batch. Failing."

            assert isinstance(batch[rec_field], torch.Tensor), (
                f"The field '{rec_field}' in the batch is not a torch.Tensor. Possible failing cases:"
                f"the field has not been added to 'varlen_fields' in the yaml of the PaddedDatasetLoader."
            )

            feats = batch[rec_field]
            feat_lens = batch.get(f"{rec_field}_len")
            if feat_lens is None:

                def apply_mask(_loss):
                    _loss = _loss.mean()
                    return _loss, _loss  # nats/pix

            else:

                def apply_mask(_loss):
                    mask = sequence_mask(feat_lens, mask_length=_loss.size(1))
                    mask = mask / mask.sum()
                    _loss = _loss * mask.unsqueeze(-1).unsqueeze(-1)
                    height_x_chanels = _loss.size(2) * _loss.size(3)
                    _loss = _loss.sum()
                    # The mask broadcasts over dim 2&3, hence we need to manually normalize
                    _loss_per_pix = _loss / height_x_chanels
                    return _loss, _loss_per_pix

            inputs, targets = rec.get_inputs_and_targets(feats, feat_lens)
            logits = rec(inputs, conds)
            loss = rec.loss(logits, targets)
            loss, loss_per_pix = apply_mask(loss)
            name = "_" + name if name else name
            details[f"rec{name}_loss"] = loss
            per_pix[f"rec{name}_loss_per_pix"] = loss_per_pix

            if needs_rec_image:
                all_inputs.append(inputs)
                mean_field.append(rec.get_mean_field_preds(logits.detach()))

        details["rec_loss"] = sum(details.values())
        per_pix["rec_loss_per_pix"] = sum(per_pix.values())
        return (details["rec_loss"], {**details, **per_pix}, all_inputs, mean_field)

    def masking(self, input_values: Tensor, input_lengths: Tensor) -> tuple[Tensor, Tensor]:
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
            k = int(self.mask_prob * input_lengths[batch])
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
        input_values[time_mask_indices == 1] = random_values

        return input_values, time_mask_indices

    # Note: This module is not meant to be run in forward() except while training. It has special logic which performs
    # evaluation using quantized values when it detects that it is being run in eval() mode, which will be substantially
    # more lossy (but useful for determining network performance).
    def forward(
        self,
        input_values: Tensor,
        input_lengths: Optional[Tensor] = None,
        file_name=None,
        ctc_codes=None,
    ):
        # input_values B C T or B C H W
        # preprocessing data

        input_values = self.norm_data(input_values)
        num_channels = input_values.shape[1]
        if num_channels != self.num_channels:
            if isinstance(self.in_channels, (list, tuple)) and len(self.in_channels) == 2:
                input_values = input_values[:, self.in_channels[0] : self.in_channels[1]]
            else:
                input_values = input_values[:, : self.in_channels]

        if input_lengths is not None and self.rand_crop_len > 0:
            assert input_values.ndim == 3, "Random cropping is only applied to 1d signal."
            input_values, ids_start, slice_lengths = rand_slice_segments(
                input_values, input_lengths, self.rand_crop_len
            )

        if self.vae_encoder_channel_last:
            input_values = input_values.permute(0, 2, 1)

        if input_lengths is not None and self.mask_prob > 0.0:
            batch_size, num_steps, dim = input_values.size()
            if not num_steps % self.reduction_steps == 0:
                num_steps = (num_steps // self.reduction_steps) * self.reduction_steps
                input_values = input_values[:, :num_steps]

            quantized_input_lengths = input_lengths // self.reduction_steps
            recover_input_lengths = quantized_input_lengths * self.reduction_steps
            masked_input_values, time_mask_indices = self.masking(input_values.clone(), quantized_input_lengths)
            masked_input_values = masked_input_values.reshape(batch_size, num_steps, -1)
            lengths = recover_input_lengths
        else:
            masked_input_values = input_values
            lengths = None

        # encode and decode to latent (like mels or other processed features)
        enc = self.vae_encoder(masked_input_values, lengths)
        if len(enc) == 2:
            enc = enc[0]
            attn = enc[1]

        if self.vae_encoder_channel_last:
            enc = enc.permute(0, 2, 1)

        content, speaker = self.seperate_feature(enc)
        sampled, codes, commitment_loss, confidence, aux_dict = self.bottleneck(content)

        if self.res_encoder is None:
            g_noise = None
        else:
            g_noise = self.res_encoder(input_values)

        if self.global_extractor is None:
            g_clean, sim_loss, contrast_loss = None, None, None
        else:
            g_clean, sim_loss, contrast_loss = self.global_info_extract(speaker, sampled)

        if self.noise_paired:
            # assume the first half is original data and the second half is added with noise
            origin, noised = torch.chunk(speaker, 2, dim=0)
            origin = origin.detach()
            noise_loss = self.noise_loss_fn(origin, noised)
        else:
            noise_loss = torch.zeros_like(commitment_loss)

        if self.diversity is None or confidence is None:
            diversity_loss = torch.zeros_like(commitment_loss)
        else:
            diversity_loss = self.diversity(confidence)

        if not self.training:
            # Be extra paranoid and make sure the encoder weights can't
            # change from straight-through estimator
            codes = codes.detach()
        sampled = sampled.permute((0, 3, 1, 2) if len(input_values.shape) == 4 else (0, 2, 1))

        if self.training:
            # set_requires_grad(self.reconstructor, False)
            recon_out = self.reconstructor(x=sampled, g_c=g_clean, g_n=g_noise)
            self.log_codes(codes)
        else:
            # This is non-differentiable, but gives a better idea of how the network is actually performing.
            recon_out, _ = self.reconstructor(codes)

        if self.asr_head is not None:
            asr_enc = self.asr_head(enc)

        if self.vae_encoder_channel_last:
            target = input_values.permute(0, 2, 1)
        else:
            target = input_values

        # reconstruction loss
        recon_loss = self.recon_loss_fn(target, recon_out)
        # if torch.isnan(recon_loss).any() or torch.isinf(recon_loss).any():
        #    import pdb; pdb.set_trace()
        if self.debug_info is not None and aux_dict is not None:
            self.debug_info.update(**aux_dict)
            # if aux_dict['used_curr'] < 2000:
            #    import pdb; pdb.set_trace()

        if isinstance(diversity_loss, dict):
            return (
                recon_loss,
                commitment_loss,
                sim_loss,
                contrast_loss,
                noise_loss,
                diversity_loss["entropy_loss"],
                diversity_loss["consistency_loss"],
                input_values,
                recon_out,
                codes,
            )
        else:
            consistency_loss = torch.zeros(()).cuda(recon_loss.device)
            return (
                recon_loss,
                commitment_loss,
                sim_loss,
                contrast_loss,
                noise_loss,
                diversity_loss,
                consistency_loss,
                input_values,
                recon_out,
                codes,
            )

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, input_values):
        input_values = self.norm_data(input_values)
        num_channels = input_values.shape[1]
        if num_channels != self.num_channels:
            if isinstance(self.in_channels, (list, tuple)) and len(self.in_channels) == 2:
                input_values = input_values[:, self.in_channels[0] : self.in_channels[1]]
            else:
                input_values = input_values[:, : self.in_channels]
        enc = self.vae_encoder(input_values)
        content, speaker = self.seperate_feature(enc)

        sampled, codes, _, _, _ = self.bottleneck(content)
        self.log_codes(codes)
        return codes


@register_model
def register_representation_learner(opt_net, opt):
    return RepresentationLearner(**opt_get(opt_net, ["kwargs"], {}))


if __name__ == "__main__":
    model = RepresentationLearner(
        in_channels=80,
        rand_crop_len=256,
        codebook_dim=2048,
        vae_encoder_codebook=True,
        vae_encoder=dict(
            class_name="modelaudio.module.encoders.LucidrainsEncoder",
            num_layers=3,
            num_resnet_blocks=0,
            hidden_dim=64,
            stride=2,
            kernel_size=3,
            encoder_norm=False,
            activation="relu",
            positional_dims=1,
        ),
        res_encoder=dict(
            class_name="modelaudio.module.encoders.ReferenceEncoderWang",
        ),
        bottleneck=dict(
            class_name="modelvqvae.vqvae.JukeboxQuantize",
            codebook_size=4096,
            decay=0.99,
            eps=1e-5,
            debug=False,
            max_threshold=1.0,
            min_threshold=1.0,
            new_return_order=False,
            norm_type=None,
        ),
        reconstructor=dict(
            class_name="modelaudio.module.encoders.LucidrainsDecoder",
            num_layers=3,
            num_resnet_blocks=0,
            hidden_dim=256,
            out_channels=80,
            stride=2,
            kernel_size=3,
            activation="relu",
            use_transposed_convs=False,
            positional_dims=1,
            g_clean_dim=0,
            g_noise_dim=0,
        ),
        global_extractor=dict(
            class_name="modelaudio.module.encoders.SelfAttentionSpeakerEncoder",
            num_layers=2,
            num_heads=4,
            pool_type="attn",
        ),
    )

    # r,l,o,c= model(torch.randn(1, 80, 256))
    # model.decode(torch.randint(0, 8192, (1,256)))
    # print(r.item(), l.item(), o.shape, c.shape)
    model(torch.randn(1, 80, 256))

    s_model = RepresentationLearner(
        in_channels=80,
        rand_crop_len=256,
        codebook_dim=2048,
        vae_encoder_codebook=True,
        vae_encoder=dict(
            class_name="modelaudio.module.encoders.LucidrainsEncoder",
            num_layers=3,
            num_resnet_blocks=0,
            hidden_dim=64,
            stride=2,
            kernel_size=3,
            encoder_norm=False,
            activation="relu",
            positional_dims=1,
        ),
        res_encoder=dict(
            class_name="modelaudio.module.encoders.ReferenceEncoderWang",
        ),
        bottleneck=dict(
            class_name="modelvqvae.vqvae.JukeboxQuantize",
            codebook_size=4096,
            decay=0.99,
            eps=1e-5,
            debug=False,
            max_threshold=1.0,
            min_threshold=1.0,
            new_return_order=False,
            norm_type=None,
        ),
        reconstructor=dict(
            class_name="modelaudio.module.encoders.LucidrainsDecoder",
            num_layers=3,
            num_resnet_blocks=0,
            hidden_dim=256,
            out_channels=80,
            stride=2,
            kernel_size=3,
            activation="relu",
            use_transposed_convs=False,
            positional_dims=1,
            g_clean_dim=0,
            g_noise_dim=0,
        ),
        global_extractor=dict(
            class_name="modelaudio.module.encoders.SelfAttentionSpeakerEncoder",
            num_layers=2,
            num_heads=4,
            pool_type="attn",
            num_attn_in_dim=256,
        ),
        seperator=dict(
            class_name="modelaudio.module.encoders.StackedResBlock",
            num_blocks=2,
            activation="relu",
            dropout_p=0.1,
            ln=True,
        ),
    )
    s_model(torch.randn(1, 80, 256))
