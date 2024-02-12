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
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteSchedulerOutput

from optimum.utils import logging


logger = logging.get_logger(__name__)


class GaudiEulerAncestralDiscreteScheduler(EulerAncestralDiscreteScheduler):
    """
    Extends [Diffusers' EulerAncestralDiscreteScheduler](https://huggingface.co/docs/diffusers/en/api/schedulers/euler_ancestral) to run optimally on Gaudi:
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
        timestep_spacing: str = "linspace",
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
            timestep_spacing,
            steps_offset,
        )

        self._initial_timestep = None
        self.reset_timestep_dependent_params()

    def reset_timestep_dependent_params(self):
        self.are_timestep_dependent_params_set = False
        self.sigma_t_list = []
        self.sigma_up_t_list = []
        self.sigma_down_t_list = []

    def get_params(self, timestep: Union[float, torch.FloatTensor]):
        """
        Initialize the time-dependent parameters, and retrieve the time-dependent
        parameters at each timestep. The tensors are rolled in a separate function
        at the end of the scheduler step in case parameters are retrieved multiple
        times in a timestep, e.g., when scaling model inputs and in the scheduler step.

        Args:
            timestep (`float`):
                The current discrete timestep in the diffusion chain. Optionally used to
                initialize parameters in cases which start in the middle of the
                denoising schedule (e.g. for image-to-image)
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        if not self.are_timestep_dependent_params_set:
            sigmas_from = self.sigmas[self.step_index : -1]
            sigmas_to = self.sigmas[(self.step_index + 1) :]

            for sigma_from, sigma_to in zip(sigmas_from, sigmas_to):
                sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
                sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

                self.sigma_t_list.append(sigma_from)
                self.sigma_up_t_list.append(sigma_up)
                self.sigma_down_t_list.append(sigma_down)

            self.sigma_t_list = torch.stack(self.sigma_t_list)
            self.sigma_up_t_list = torch.stack(self.sigma_up_t_list)
            self.sigma_down_t_list = torch.stack(self.sigma_down_t_list)
            self.are_timestep_dependent_params_set = True

        sigma = self.sigma_t_list[0]
        sigma_up = self.sigma_up_t_list[0]
        sigma_down = self.sigma_down_t_list[0]

        return sigma, sigma_up, sigma_down

    def roll_params(self):
        """
        Roll tensors to update the values of the time-dependent parameters at each timestep.
        """
        if self.are_timestep_dependent_params_set:
            self.sigma_t_list = torch.roll(self.sigma_t_list, shifts=-1, dims=0)
            self.sigma_up_t_list = torch.roll(self.sigma_up_t_list, shifts=-1, dims=0)
            self.sigma_down_t_list = torch.roll(self.sigma_down_t_list, shifts=-1, dims=0)
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

        sigma, _, _ = self.get_params(timestep)
        sample = sample / ((sigma**2 + 1) ** 0.5)
        self.is_scale_input_called = True
        return sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
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
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

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

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma, sigma_up, sigma_down = self.get_params(timestep)

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt

        device = model_output.device

        # torch.randn is broken on HPU so running it on CPU
        noise = torch.randn(model_output.shape, dtype=model_output.dtype, device="cpu", generator=generator)
        if device.type == "hpu":
            noise = noise.to(device)

        prev_sample = prev_sample + noise * sigma_up

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1
        self.roll_params()

        if not return_dict:
            return (prev_sample,)

        return EulerAncestralDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )
