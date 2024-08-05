# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from diffusers.models import UNet2DModel
from diffusers.pipelines import DDPMPipeline
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from optimum.habana.diffusers.pipelines.pipeline_utils import GaudiDiffusionPipeline
from optimum.habana.transformers.gaudi_configuration import GaudiConfig
from optimum.utils import logging


logger = logging.get_logger(__name__)


@dataclass
class GaudiDDPMPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    throughput: float


class GaudiDDPMPipeline(GaudiDiffusionPipeline, DDPMPipeline):
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddpm/pipeline_ddpm.py
    Changes are:
    - Markstep for non-graph mode
    - Support GaudiDDIMScheduler

    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: Union[DDPMScheduler, DDIMScheduler],
        use_habana: bool = True,
        use_hpu_graphs: bool = False,
        gaudi_config: Union[str, GaudiConfig] = None,
        bf16_full_eval: bool = False,
    ):
        GaudiDiffusionPipeline.__init__(self, use_habana, use_hpu_graphs, gaudi_config, bf16_full_eval)

        DDPMPipeline.__init__(self, unet, scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[GaudiDDPMPipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self._device.type == "hpu":  # Patch random tensor
            image = torch.randn(image_shape, generator=generator, device="cpu")
            image = image.to(self._device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # Set step values
        self.scheduler.set_timesteps(num_inference_steps, device="cpu")  # Patch timesteps
        timesteps = self.scheduler.timesteps.to(self._device)
        if isinstance(self.scheduler, DDIMScheduler):
            self.scheduler.reset_timestep_dependent_params()
        num_inference_steps = [1] * len(self.scheduler.timesteps)

        if self.use_hpu_graphs:
            self.unet = self.ht.hpu.wrap_in_hpu_graph(self.unet, disable_tensor_cache=True)

        if self.use_habana:
            self.unet = self.unet.to(self._device)

        start_time = time.time()
        for i in self.progress_bar(num_inference_steps):
            timestep = timesteps[0]
            timesteps = torch.roll(timesteps, shifts=-1, dims=0)

            # 1. Predict noise model_output
            with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=self.gaudi_config.use_torch_autocast):
                model_output = self.unet(image, timestep).sample
            # 2. Compute previous image: x_t -> x_t-1
            image = self.scheduler.step(
                model_output.to(torch.float32), timestep, image, generator=generator
            ).prev_sample

            if not self.use_hpu_graphs:  # For checking output resutls
                self.htcore.mark_step()

        if self.gaudi_config.use_torch_autocast:
            image = image.float()

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        end_time = time.time()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        throughput = (end_time - start_time) / batch_size
        return GaudiDDPMPipelineOutput(images=image, throughput=throughput)
