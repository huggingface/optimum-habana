# coding=utf-8
# Copyright 2022 Stanford University Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput


class GaudiDDIMScheduler(DDIMScheduler):
    """
    Extends [Diffusers' DDIMScheduler](https://huggingface.co/docs/diffusers/api/schedulers#diffusers.DDIMScheduler) to run optimally on Gaudi:
    - All time-dependent parameters are generated at the beginning
    - At each time step, tensors are rolled to update the values of the time-dependent parameters

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        set_alpha_to_one (`bool`, defaults to `True`):
            Each diffusion step uses the alphas product value at that step and at the previous one. For the final step
            there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the alpha value at step 0.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps. You can use a combination of `offset=1` and
            `set_alpha_to_one=False` to make the last step use step 0 for the previous alpha product like in Stable
            Diffusion.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
    ):
        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            clip_sample,
            set_alpha_to_one,
            steps_offset,
            prediction_type,
            thresholding,
            dynamic_thresholding_ratio,
            clip_sample_range,
            sample_max_value,
            timestep_spacing,
            rescale_betas_zero_snr,
        )

        self.reset_timestep_dependent_params()

    def reset_timestep_dependent_params(self):
        self.are_timestep_dependent_params_set = False
        self.alpha_prod_t_list = []
        self.alpha_prod_t_prev_list = []
        self.variance_list = []

    def get_params(self):
        if not self.are_timestep_dependent_params_set:
            prev_timesteps = self.timesteps - self.config.num_train_timesteps // self.num_inference_steps
            for timestep, prev_timestep in zip(self.timesteps, prev_timesteps):
                alpha_prod_t = self.alphas_cumprod[timestep]
                alpha_prod_t_prev = (
                    self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
                )

                self.alpha_prod_t_list.append(alpha_prod_t)
                self.alpha_prod_t_prev_list.append(alpha_prod_t_prev)
                self.variance_list.append(self._get_variance(alpha_prod_t, alpha_prod_t_prev))

            self.alpha_prod_t_list = torch.stack(self.alpha_prod_t_list)
            self.alpha_prod_t_prev_list = torch.stack(self.alpha_prod_t_prev_list)
            self.variance_list = torch.stack(self.variance_list)
            self.are_timestep_dependent_params_set = True

        alpha_prod_t = self.alpha_prod_t_list[0]
        self.alpha_prod_t_list = torch.roll(self.alpha_prod_t_list, shifts=-1, dims=0)
        alpha_prod_t_prev = self.alpha_prod_t_prev_list[0]
        self.alpha_prod_t_prev_list = torch.roll(self.alpha_prod_t_prev_list, shifts=-1, dims=0)
        variance = self.variance_list[0]
        self.variance_list = torch.roll(self.variance_list, shifts=-1, dims=0)

        return alpha_prod_t, alpha_prod_t_prev, variance

    # def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
    #     """
    #     Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
    #     current timestep.
    #     Args:
    #         sample (`torch.FloatTensor`): input sample
    #         timestep (`int`, optional): current timestep
    #     Returns:
    #         `torch.FloatTensor`: scaled input sample
    #     """
    #     return sample

    def _get_variance(self, alpha_prod_t, alpha_prod_t_prev):
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def step(
        self,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.FloatTensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~diffusers.schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`diffusers.schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~diffusers.schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        # Done in self.get_params() below

        # 2. compute alphas, betas
        alpha_prod_t, alpha_prod_t_prev, variance = self.get_params()
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            device = model_output.device
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                # torch.randn is broken on HPU so running it on CPU
                variance_noise = torch.randn(
                    model_output.shape, dtype=model_output.dtype, device="cpu", generator=generator
                )
                if device.type == "hpu":
                    variance_noise = variance_noise.to(device)

            prev_sample = prev_sample + std_dev_t * variance_noise

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    # def add_noise(
    #     self,
    #     original_samples: torch.FloatTensor,
    #     noise: torch.FloatTensor,
    #     timesteps: torch.IntTensor,
    # ) -> torch.FloatTensor:
    #     # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    #     self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
    #     timesteps = timesteps.to(original_samples.device)

    #     sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
    #     sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    #     while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
    #         sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    #     sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
    #     sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    #     while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
    #         sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    #     noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    #     return noisy_samples
