# Copyright 2019 RnD at Spoon Radio
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
# ==============================================================================
"""SpecAugment Implementation for Tensorflow.
Related paper : https://arxiv.org/pdf/1904.08779.pdf
In this paper, show summarized parameters by each open datasets in Tabel 1.
-----------------------------------------
Policy | W  | F  | m_F |  T  |  p  | m_T
-----------------------------------------
None   |  0 |  0 |  -  |  0  |  -  |  -
-----------------------------------------
LB     | 80 | 27 |  1  | 100 | 1.0 | 1
-----------------------------------------
LD     | 80 | 27 |  2  | 100 | 1.0 | 2
-----------------------------------------
SM     | 40 | 15 |  2  |  70 | 0.2 | 2
-----------------------------------------
SS     | 40 | 27 |  2  |  70 | 0.2 | 2
-----------------------------------------
LB : LibriSpeech basic
LD : LibriSpeech double
SM : Switchboard mild
SS : Switchboard strong
"""
import random
from typing import Any, Dict, Optional

import librosa
import librosa.display
import numpy as np
import torch

from data.builder import TRANSFORMS

from .sparse_time_warp_core import sparse_image_warp


def time_warp(spec, W=5):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    # assert len(horizontal_line_at_ctr) == spec_len

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
    # assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts = torch.tensor([[[y, point_to_warp]]])
    dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)


def spec_augment(
    mel_spectrogram,
    time_warping_para=80,
    frequency_masking_para=27,
    time_masking_para=100,
    frequency_mask_num=1,
    time_mask_num=1,
):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[2]

    # Step 1 : Time warping
    warped_mel_spectrogram = time_warp(mel_spectrogram, W=time_warping_para)

    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v - f)
        warped_mel_spectrogram[:, f0 : f0 + f, :] = 0

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau - t)
        warped_mel_spectrogram[:, :, t0 : t0 + t] = 0

    return warped_mel_spectrogram


@TRANSFORMS.register_module()
class SpecAugment:
    def __init__(
        self,
        in_key: str = "mel",
        out_key: Optional[str] = None,
        time_mask_width: int = 20,
        time_mask_width_rand: bool = False,
        frequency_mask_width: int = 27,
        frequency_mask_width_rand: bool = False,
        time_mask_application_prob: float = 0.05,
        frequency_mask_application_prob: float = 0.05,
        mask_value_method: str = "zero",
    ):
        """
        Applies time masking to a batch of log-Mel spectrograms (PyTorch Tensors).

        For each spectrogram in the batch, there is a `mask_application_prob` chance
        that a single continuous block of `time_mask_width` time steps will be masked.

        Args:
            batch_log_mel_spectrograms (torch.Tensor):
                The input batch of log-Mel spectrograms.
                Shape: (batch_size, num_mel_filters, num_time_steps)
            time_mask_width (int):
                The width of the time mask (T in SpecAugment notation, number of time steps).
                Default is 20, as per the description.
            mask_application_prob (float):
                The probability (0.0 to 1.0) of applying a time mask to
                any given spectrogram in the batch. Default is 0.05 (5%).
            mask_value_method (str):
                Method to determine the mask value.
                "mean": mask with the mean value of the current spectrogram.
                "zero": mask with zeros.
                Default is "mean".

        Returns:
            torch.Tensor: The batch of augmented log-Mel spectrograms.
                        Shape: (batch_size, num_mel_filters, num_time_steps)
        """
        self.time_mask_width = time_mask_width
        self.time_mask_width_rand = time_mask_width_rand
        self.frequency_mask_width = frequency_mask_width
        self.frequency_mask_width_rand = frequency_mask_width_rand
        self.time_mask_application_prob = time_mask_application_prob
        self.frequency_mask_application_prob = frequency_mask_application_prob
        self.mask_value_method = mask_value_method

    def apply_frequency_masking(
        self,
        batch_log_mel_spectrograms: torch.Tensor,
    ) -> torch.Tensor:
        augmented_batch = batch_log_mel_spectrograms
        batch_size, num_mel_filters, total_time_steps = augmented_batch.shape

        if self.frequency_mask_width <= 0:
            print("Warning: frequency_mask_width is non-positive. No frequency masking will be applied.")
            return augmented_batch

        idx_mask_applied = torch.rand(batch_size) < self.frequency_mask_application_prob

        for i in range(batch_size):
            if idx_mask_applied[i]:
                if num_mel_filters > self.frequency_mask_width:
                    mask_value: float = 0.0
                    if self.mask_value_method == "mean":
                        mask_value = torch.mean(augmented_batch[i, :, :]).item()
                    elif self.mask_value_method == "zero":
                        mask_value = 0.0
                    else:
                        print(f"Warning: Unknown mask_value_method '{self.mask_value_method}'. Defaulting to zero.")
                        mask_value = 0.0
                    if self.frequency_mask_width_rand:
                        f = torch.randint(low=0.0, high=self.frequency_mask_width, size=(1,)).item()
                    else:
                        f = self.frequency_mask_width
                    f0 = random.randint(0, num_mel_filters - f)
                    augmented_batch[i, f0 : f0 + f, :] = mask_value

        return augmented_batch

    def apply_time_masking(
        self,
        batch_log_mel_spectrograms: torch.Tensor,
    ) -> torch.Tensor:
        # Create a clone to avoid modifying the original tensor in place
        # .clone() also copies autograd history if input requires_grad.
        # If you want a true detached copy for augmentation, use .clone().detach()
        augmented_batch = batch_log_mel_spectrograms
        batch_size, num_mel_filters, total_time_steps = augmented_batch.shape

        if self.time_mask_width <= 0:
            print("Warning: time_mask_width is non-positive. No time masking will be applied.")
            return augmented_batch

        idx_mask_applied = torch.rand(batch_size) < self.time_mask_application_prob

        for i in range(batch_size):
            # Step 1: Decide whether to apply a mask to this specific spectrogram
            # torch.rand(1) produces a tensor, .item() gets the Python number
            if idx_mask_applied[i]:
                # Ensure the mask width is not greater than the total time steps
                if total_time_steps >= self.time_mask_width:
                    # Step 2: Determine the mask value
                    mask_value: float = 0.0
                    if self.mask_value_method == "mean":
                        # Calculate the mean of the current spectrogram
                        mask_value = torch.mean(augmented_batch[i, :, :]).item()
                    elif self.mask_value_method == "zero":
                        mask_value = 0.0
                    else:
                        # Default or warn for unknown method
                        print(f"Warning: Unknown mask_value_method '{self.mask_value_method}'. Defaulting to zero.")
                        mask_value = 0.0

                    # Step 3: Choose a random start time 't0' for the mask
                    # t0 can be from 0 to (total_time_steps - time_mask_width)
                    # torch.randint high is exclusive, so add 1 to the upper bound.
                    # torch.randint returns a tensor, so use .item()
                    t0 = torch.randint(0, total_time_steps - self.time_mask_width + 1, (1,)).item()
                    if self.time_mask_width_rand:
                        t = torch.randint(low=0.0, high=self.time_mask_width, size=(1,)).item()
                    else:
                        t = self.time_mask_width
                    t_end = t0 + t

                    # Step 4: Apply the mask
                    # The mask is applied across all Mel frequency bins for the selected time steps
                    augmented_batch[i, :, t0:t_end] = mask_value
                # else:
                # print(f"Warning: Spectrogram {i} is too short ({total_time_steps} steps) for a mask of width {time_mask_width}.")

        return augmented_batch

    def __call__(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.out_key is None:
            self.out_key = self.in_key
        batch_dict[self.out_key] = self.apply_time_masking(batch_dict[self.in_key])
        return batch_dict
