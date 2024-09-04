# Copyright 2023 Katherine Crowson and The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import EulerDiscreteScheduler
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor

from optimum.utils import logging


logger = logging.get_logger(__name__)


class GaudiEulerDiscreteScheduler(EulerDiscreteScheduler):
    """
    Extends [Diffusers' EulerDiscreteScheduler](https://huggingface.co/docs/diffusers/api/schedulers#diffusers.EulerDiscreteScheduler) to run optimally on Gaudi:
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
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        interpolation_type(`str`, defaults to `"linear"`, *optional*):
            The interpolation type to compute intermediate sigmas for the scheduler denoising steps. Should be on of
            `"linear"` or `"log_linear"`.
        use_karras_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use Karras sigmas for step sizes in the noise schedule during the sampling process. If `True`,
            the sigmas are determined according to a sequence of noise levels {σi}.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps. You can use a combination of `offset=1` and
            `set_alpha_to_one=False` to make the last step use step 0 for the previous alpha product like in Stable
            Diffusion.
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
        prediction_type: str = "epsilon",
        interpolation_type: str = "linear",
        use_karras_sigmas: Optional[bool] = False,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        timestep_spacing: str = "linspace",
        timestep_type: str = "discrete",  # can be "discrete" or "continuous"
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
    ):
        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            prediction_type,
            interpolation_type,
            use_karras_sigmas,
            sigma_min,
            sigma_max,
            timestep_spacing,
            timestep_type,
            steps_offset,
            rescale_betas_zero_snr,
        )

        self._initial_timestep = None
        self.reset_timestep_dependent_params()
        self.hpu_opt = False

    def reset_timestep_dependent_params(self):
        self.are_timestep_dependent_params_set = False
        self.sigma_list = []
        self.sigma_next_list = []

    def get_params(self, timestep: Union[float, torch.FloatTensor]):
        if self.step_index is None:
            self._init_step_index(timestep)

        if not self.are_timestep_dependent_params_set:
            sigmas = self.sigmas[self.step_index : -1]
            sigmas_next = self.sigmas[(self.step_index + 1) :]

            for sigma, sigma_next in zip(sigmas, sigmas_next):
                self.sigma_list.append(sigma)
                self.sigma_next_list.append(sigma_next)

            self.sigma_list = torch.stack(self.sigma_list)
            self.sigma_next_list = torch.stack(self.sigma_next_list)
            self.are_timestep_dependent_params_set = True

        sigma = self.sigma_list[0]
        sigma_next = self.sigma_next_list[0]

        return sigma, sigma_next

    def roll_params(self):
        """
        Roll tensors to update the values of the time-dependent parameters at each timestep.
        """
        if self.are_timestep_dependent_params_set:
            self.sigma_list = torch.roll(self.sigma_list, shifts=-1, dims=0)
            self.sigma_next_list = torch.roll(self.sigma_next_list, shifts=-1, dims=0)
        else:
            raise ValueError("Time-dependent parameters should be set first.")
        return

    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """

        if self.hpu_opt:
            if self.step_index is None:
                self._init_step_index(timestep)
            self.sigmas = self.sigmas.to(sample.dtype)
            sigma = self.sigmas[self.step_index]
        else:
            sigma, _ = self.get_params(timestep)
        sample = sample / ((sigma**2 + 1) ** 0.5)
        self.is_scale_input_called = True
        return sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerDiscreteSchedulerOutput, Tuple]:
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
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if self.hpu_opt and self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        if self.hpu_opt:
            sigma = self.sigmas[self.step_index]
        else:
            sigma, sigma_next = self.get_params(timestep)

        if self.hpu_opt and sigma.device.type == "hpu":
            gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        else:
            gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        if self.hpu_opt:
            noise = randn_tensor(
                model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
            )
        else:
            device = model_output.device

            # torch.randn is broken on HPU so running it on CPU
            noise = torch.randn(model_output.shape, dtype=model_output.dtype, device="cpu", generator=generator)
            if device.type == "hpu":
                noise = noise.to(device)

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
        elif self.config.prediction_type == "v_prediction":
            # denoised = model_output * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat

        if self.hpu_opt:
            dt = self.sigmas[self.step_index + 1] - sigma_hat
        else:
            dt = sigma_next - sigma_hat

        prev_sample = sample + derivative * dt

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1
        if not self.hpu_opt:
            self.roll_params()

        if not return_dict:
            return (prev_sample,)

        return EulerDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
