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
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLImg2ImgPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from optimum.utils import logging

from ....transformers.gaudi_configuration import GaudiConfig
from ....utils import HabanaProfile, speed_metrics, warmup_inference_steps_time_adjustment
from ..pipeline_utils import GaudiDiffusionPipeline
from ..stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from .pipeline_stable_diffusion_xl import GaudiStableDiffusionXLPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class GaudiStableDiffusionXLImg2ImgPipeline(GaudiDiffusionPipeline, StableDiffusionXLImg2ImgPipeline):
    """
    Pipeline for image-to-image generation using Stable Diffusion XL on Gaudi devices
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py

    Extends the [`StableDiffusionXLImg2ImgPipeline`] class:
        - Generation is performed by batches
        - Two `mark_step()` were added to add support for lazy mode
        - Added support for HPU graphs

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        requires_aesthetics_score (`bool`, *optional*, defaults to `"False"`):
            Whether the `unet` requires an `aesthetic_score` condition to be passed during inference. Also see the
            config of `stabilityai/stable-diffusion-xl-refiner-1-0`.
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
        use_habana (bool, defaults to `False`):
            Whether to use Gaudi (`True`) or CPU (`False`).
        use_hpu_graphs (bool, defaults to `False`):
            Whether to use HPU graphs or not.
        gaudi_config (Union[str, [`GaudiConfig`]], defaults to `None`):
            Gaudi configuration to use. Can be a string to download it from the Hub.
           Or a previously initialized config can be passed.
        bf16_full_eval (bool, defaults to `False`):
            Whether to use full bfloat16 evaluation instead of 32-bit.
            This will be faster and save memory compared to fp32/mixed precision but can harm generated images.
        sdp_on_bf16 (bool, defaults to `False`):
            Whether to allow PyTorch to use reduced precision in the SDPA math backend.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        requires_aesthetics_score: bool = False,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
        use_habana: bool = False,
        use_hpu_graphs: bool = False,
        gaudi_config: Union[str, GaudiConfig] = None,
        bf16_full_eval: bool = False,
        sdp_on_bf16: bool = False,
    ):
        GaudiDiffusionPipeline.__init__(
            self,
            use_habana,
            use_hpu_graphs,
            gaudi_config,
            bf16_full_eval,
            sdp_on_bf16,
        )

        StableDiffusionXLImg2ImgPipeline.__init__(
            self,
            vae,
            text_encoder,
            text_encoder_2,
            tokenizer,
            tokenizer_2,
            unet,
            scheduler,
            image_encoder,
            feature_extractor,
            requires_aesthetics_score,
            force_zeros_for_empty_prompt,
            add_watermarker,
        )

        self.to(self._device)

    @classmethod
    def _split_inputs_into_batches(
        cls,
        batch_size,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        add_text_embeds,
        negative_pooled_prompt_embeds,
        add_time_ids,
        negative_add_time_ids,
    ):
        # Use torch.split to generate num_batches batches of size batch_size
        latents_batches = list(torch.split(latents, batch_size))
        prompt_embeds_batches = list(torch.split(prompt_embeds, batch_size))
        if negative_prompt_embeds is not None:
            negative_prompt_embeds_batches = list(torch.split(negative_prompt_embeds, batch_size))
        if add_text_embeds is not None:
            add_text_embeds_batches = list(torch.split(add_text_embeds, batch_size))
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds_batches = list(torch.split(negative_pooled_prompt_embeds, batch_size))
        if add_time_ids is not None:
            add_time_ids_batches = list(torch.split(add_time_ids, batch_size))
        if negative_add_time_ids is not None:
            negative_add_time_ids_batches = list(torch.split(negative_add_time_ids, batch_size))

        # If the last batch has less samples than batch_size, pad it with dummy samples
        num_dummy_samples = 0
        if latents_batches[-1].shape[0] < batch_size:
            num_dummy_samples = batch_size - latents_batches[-1].shape[0]
            # Pad latents_batches
            sequence_to_stack = (latents_batches[-1],) + tuple(
                torch.zeros_like(latents_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
            )
            latents_batches[-1] = torch.vstack(sequence_to_stack)
            # Pad prompt_embeds_batches
            sequence_to_stack = (prompt_embeds_batches[-1],) + tuple(
                torch.zeros_like(prompt_embeds_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
            )
            prompt_embeds_batches[-1] = torch.vstack(sequence_to_stack)
            # Pad negative_prompt_embeds_batches if necessary
            if negative_prompt_embeds is not None:
                sequence_to_stack = (negative_prompt_embeds_batches[-1],) + tuple(
                    torch.zeros_like(negative_prompt_embeds_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                negative_prompt_embeds_batches[-1] = torch.vstack(sequence_to_stack)
            # Pad add_text_embeds_batches if necessary
            if add_text_embeds is not None:
                sequence_to_stack = (add_text_embeds_batches[-1],) + tuple(
                    torch.zeros_like(add_text_embeds_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                add_text_embeds_batches[-1] = torch.vstack(sequence_to_stack)
            # Pad negative_pooled_prompt_embeds_batches if necessary
            if negative_pooled_prompt_embeds is not None:
                sequence_to_stack = (negative_pooled_prompt_embeds_batches[-1],) + tuple(
                    torch.zeros_like(negative_pooled_prompt_embeds_batches[-1][0][None, :])
                    for _ in range(num_dummy_samples)
                )
                negative_pooled_prompt_embeds_batches[-1] = torch.vstack(sequence_to_stack)
            # Pad add_time_ids_batches if necessary
            if add_time_ids is not None:
                sequence_to_stack = (add_time_ids_batches[-1],) + tuple(
                    torch.zeros_like(add_time_ids_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                add_time_ids_batches[-1] = torch.vstack(sequence_to_stack)
            # Pad negative_add_time_ids_batches if necessary
            if negative_add_time_ids is not None:
                sequence_to_stack = (negative_add_time_ids_batches[-1],) + tuple(
                    torch.zeros_like(negative_add_time_ids_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                negative_add_time_ids_batches[-1] = torch.vstack(sequence_to_stack)

        # Stack batches in the same tensor
        latents_batches = torch.stack(latents_batches)

        if negative_prompt_embeds is not None:
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            for i, (negative_prompt_embeds_batch, prompt_embeds_batch) in enumerate(
                zip(negative_prompt_embeds_batches, prompt_embeds_batches[:])
            ):
                prompt_embeds_batches[i] = torch.cat([negative_prompt_embeds_batch, prompt_embeds_batch])
        prompt_embeds_batches = torch.stack(prompt_embeds_batches)

        if add_text_embeds is not None:
            if negative_pooled_prompt_embeds is not None:
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                for i, (negative_pooled_prompt_embeds_batch, add_text_embeds_batch) in enumerate(
                    zip(negative_pooled_prompt_embeds_batches, add_text_embeds_batches[:])
                ):
                    add_text_embeds_batches[i] = torch.cat(
                        [negative_pooled_prompt_embeds_batch, add_text_embeds_batch]
                    )
            add_text_embeds_batches = torch.stack(add_text_embeds_batches)
        else:
            add_text_embeds_batches = None

        if add_time_ids is not None:
            if negative_add_time_ids is not None:
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                for i, (negative_add_time_ids_batch, add_time_ids_batch) in enumerate(
                    zip(negative_add_time_ids_batches, add_time_ids_batches[:])
                ):
                    add_time_ids_batches[i] = torch.cat([negative_add_time_ids_batch, add_time_ids_batch])
            add_time_ids_batches = torch.stack(add_time_ids_batches)
        else:
            add_time_ids_batches = None

        return latents_batches, prompt_embeds_batches, add_text_embeds_batches, add_time_ids_batches, num_dummy_samples

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        strength: float = 0.3,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        batch_size: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **kwargs,
    ):
        """
        Adapted from: https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py
        - Two `mark_step()` were added to add support for lazy mode
        - Added support for HPU graphs
        - Added batch_size args
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=self.gaudi_config.use_torch_autocast):
            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                prompt_2,
                strength,
                num_inference_steps,
                callback_steps,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds,
                negative_prompt_embeds,
                callback_on_step_end_tensor_inputs,
            )

            self._guidance_scale = guidance_scale
            self._guidance_rescale = guidance_rescale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs
            self._denoising_end = denoising_end
            self._denoising_start = denoising_start
            self._interrupt = False

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                num_prompts = 1
            elif prompt is not None and isinstance(prompt, list):
                num_prompts = len(prompt)
            else:
                num_prompts = prompt_embeds.shape[0]
            num_batches = ceil((num_images_per_prompt * num_prompts) / batch_size)
            logger.info(
                f"{num_prompts} prompt(s) received, {num_images_per_prompt} generation(s) per prompt,"
                f" {batch_size} sample(s) per batch, {num_batches} total batch(es)."
            )
            if num_batches < 3:
                logger.warning("The first two iterations are slower so it is recommended to feed more batches.")

            device = self._execution_device

            # 3. Encode input prompt
            text_encoder_lora_scale = (
                self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
            )
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )

            # 4. Preprocess image
            image = self.image_processor.preprocess(image)

            # 5. Prepare timesteps
            def denoising_value_valid(dnv):
                return isinstance(self.denoising_end, float) and 0 < dnv < 1

            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

            timesteps, num_inference_steps = self.get_timesteps(
                num_inference_steps,
                strength,
                device,
                denoising_start=self.denoising_start if denoising_value_valid else None,
            )
            timesteps = timesteps.to(device)
            latent_timestep = timesteps[:1].repeat(num_prompts * num_images_per_prompt)

            add_noise = True if self.denoising_start is None else False
            # 6. Prepare latent variables
            latents = self.prepare_latents(
                image,
                latent_timestep,
                num_prompts,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
                add_noise,
            )

            # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            height, width = latents.shape[-2:]
            height = height * self.vae_scale_factor
            width = width * self.vae_scale_factor

            original_size = original_size or (height, width)
            target_size = target_size or (height, width)

            # 8. Prepare added time ids & embeddings
            if negative_original_size is None:
                negative_original_size = original_size
            if negative_target_size is None:
                negative_target_size = target_size

            add_text_embeds = pooled_prompt_embeds
            if self.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

            add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                aesthetic_score,
                negative_aesthetic_score,
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            add_time_ids = add_time_ids.repeat(num_prompts * num_images_per_prompt, 1)
            if self.do_classifier_free_guidance:
                add_neg_time_ids = add_neg_time_ids.repeat(num_prompts * num_images_per_prompt, 1)
                add_neg_time_ids = add_neg_time_ids.to(device)

            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device)

            if ip_adapter_image is not None:
                image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image, device, num_prompts * num_images_per_prompt
                )

            # 7.5 Split into batches (HPU-specific step)

            (
                latents_batches,
                text_embeddings_batches,
                add_text_embeddings_batches,
                add_time_ids_batches,
                num_dummy_samples,
            ) = self._split_inputs_into_batches(
                batch_size,
                latents,
                prompt_embeds,
                negative_prompt_embeds,
                add_text_embeds,
                negative_pooled_prompt_embeds,
                add_time_ids,
                add_neg_time_ids,
            )

            outputs = {
                "images": [],
            }
            t0 = time.time()
            t1 = t0

            hb_profiler = HabanaProfile(
                warmup=profiling_warmup_steps,
                active=profiling_steps,
                record_shapes=False,
                name="stable_diffusion",
            )
            hb_profiler.start()

            # 9. Denoising loop
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

            # 9.1 Apply denoising_end
            if (
                self.denoising_end is not None
                and self.denoising_start is not None
                and denoising_value_valid(self.denoising_end)
                and denoising_value_valid(self.denoising_start)
                and self.denoising_start >= self.denoising_end
            ):
                raise ValueError(
                    f"`denoising_start`: {self.denoising_start} cannot be larger than or equal to `denoising_end`: "
                    + f" {self.denoising_end} when using type float."
                )
            elif self.denoising_end is not None and denoising_value_valid(self.denoising_end):
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                    )
                )
                num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
                timesteps = timesteps[:num_inference_steps]
            # 9.2 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                    batch_size * num_images_per_prompt
                )
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)

            self._num_timesteps = len(timesteps)

            # 8.3 Denoising loop
            throughput_warmup_steps = kwargs.get("throughput_warmup_steps", 3)
            use_warmup_inference_steps = (
                num_batches <= throughput_warmup_steps and num_inference_steps > throughput_warmup_steps
            )
            for j in self.progress_bar(range(num_batches)):
                # The throughput is calculated from the 3rd iteration
                # because compilation occurs in the first two iterations
                if j == throughput_warmup_steps:
                    t1 = time.time()
                if use_warmup_inference_steps:
                    t0_inf = time.time()

                latents_batch = latents_batches[0]
                latents_batches = torch.roll(latents_batches, shifts=-1, dims=0)
                text_embeddings_batch = text_embeddings_batches[0]
                text_embeddings_batches = torch.roll(text_embeddings_batches, shifts=-1, dims=0)
                add_text_embeddings_batch = add_text_embeddings_batches[0]
                add_text_embeddings_batches = torch.roll(add_text_embeddings_batches, shifts=-1, dims=0)
                add_time_ids_batch = add_time_ids_batches[0]
                add_time_ids_batches = torch.roll(add_time_ids_batches, shifts=-1, dims=0)

                if hasattr(self.scheduler, "_init_step_index"):
                    # Reset scheduler step index for next batch
                    self.scheduler._init_step_index(timesteps[0])

                for i in range(len(timesteps)):
                    if use_warmup_inference_steps and i == throughput_warmup_steps:
                        t1_inf = time.time()
                        t1 += t1_inf - t0_inf
                    if self.interrupt:
                        continue
                    timestep = timesteps[0]
                    timesteps = torch.roll(timesteps, shifts=-1, dims=0)

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents_batch] * 2) if self.do_classifier_free_guidance else latents_batch
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

                    # predict the noise residual
                    added_cond_kwargs = {"text_embeds": add_text_embeddings_batch, "time_ids": add_time_ids_batch}
                    if ip_adapter_image is not None:
                        added_cond_kwargs["image_embeds"] = image_embeds
                    noise_pred = self.unet_hpu(
                        latent_model_input,
                        timestep,
                        text_embeddings_batch,
                        timestep_cond,
                        self.cross_attention_kwargs,
                        added_cond_kwargs,
                    )

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(
                            noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_batch = self.scheduler.step(
                        noise_pred, timestep, latents_batch, **extra_step_kwargs, return_dict=False
                    )[0]

                    if not self.use_hpu_graphs:
                        self.htcore.mark_step()

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, timestep, callback_kwargs)

                        latents_batch = callback_outputs.pop("latents", latents_batch)
                        _prompt_embeds = callback_outputs.pop("prompt_embeds", None)
                        _negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", None)
                        if _prompt_embeds is not None and _negative_prompt_embeds is not None:
                            text_embeddings_batch = torch.cat([_negative_prompt_embeds, _prompt_embeds])
                        _add_text_embeds = callback_outputs.pop("add_text_embeds", None)
                        _negative_pooled_prompt_embeds = callback_outputs.pop("negative_pooled_prompt_embeds", None)
                        if _add_text_embeds is not None and _negative_pooled_prompt_embeds is not None:
                            add_text_embeddings_batch = torch.cat([_negative_pooled_prompt_embeds, _add_text_embeds])
                        _add_time_ids = callback_outputs.pop("add_time_ids", None)
                        _negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", None)
                        if _add_time_ids is not None and _negative_add_time_ids is not None:
                            add_time_ids_batch = torch.cat([_add_time_ids, _negative_add_time_ids])

                    # call the callback, if provided
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, timestep, latents)

                    hb_profiler.step()
                if use_warmup_inference_steps:
                    t1 = warmup_inference_steps_time_adjustment(
                        t1, t1_inf, num_inference_steps, throughput_warmup_steps
                    )

                if not output_type == "latent":
                    # make sure the VAE is in float32 mode, as it overflows in float16
                    needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

                    if needs_upcasting:
                        self.upcast_vae()
                        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

                    image = self.vae.decode(latents_batch / self.vae.config.scaling_factor, return_dict=False)[0]

                    # cast back to fp16 if needed
                    if needs_upcasting:
                        self.vae.to(dtype=torch.float16)

                else:
                    image = latents_batch

                outputs["images"].append(image)

                if not self.use_hpu_graphs:
                    self.htcore.mark_step()

            hb_profiler.stop()

            speed_metrics_prefix = "generation"
            speed_measures = speed_metrics(
                split=speed_metrics_prefix,
                start_time=t0,
                num_samples=num_batches * batch_size
                if t1 == t0 or use_warmup_inference_steps
                else (num_batches - throughput_warmup_steps) * batch_size,
                num_steps=num_batches * batch_size * num_inference_steps,
                start_time_after_warmup=t1,
            )
            logger.info(f"Speed metrics: {speed_measures}")

            # Remove dummy generations if needed
            if num_dummy_samples > 0:
                outputs["images"][-1] = outputs["images"][-1][:-num_dummy_samples]

            # Process generated images
            for i, image in enumerate(outputs["images"][:]):
                if i == 0:
                    outputs["images"].clear()

                if not output_type == "latent":
                    # apply watermark if available
                    if self.watermark is not None:
                        image = self.watermark.apply_watermark(image)

                image = self.image_processor.postprocess(image, output_type=output_type)

                if output_type == "pil" and isinstance(image, list):
                    outputs["images"] += image
                elif output_type in ["np", "numpy"] and isinstance(image, np.ndarray):
                    if len(outputs["images"]) == 0:
                        outputs["images"] = image
                    else:
                        outputs["images"] = np.concatenate((outputs["images"], image), axis=0)
                else:
                    if len(outputs["images"]) == 0:
                        outputs["images"] = image
                    else:
                        outputs["images"] = torch.cat((outputs["images"], image), 0)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return outputs["images"]

            return GaudiStableDiffusionXLPipelineOutput(
                images=outputs["images"],
                throughput=speed_measures[f"{speed_metrics_prefix}_samples_per_second"],
            )

    @torch.no_grad()
    def unet_hpu(
        self,
        latent_model_input,
        timestep,
        encoder_hidden_states,
        timestep_cond,
        cross_attention_kwargs,
        added_cond_kwargs,
    ):
        if self.use_hpu_graphs:
            return self.capture_replay(
                latent_model_input,
                timestep,
                encoder_hidden_states,
                timestep_cond,
                cross_attention_kwargs,
                added_cond_kwargs,
            )
        else:
            return self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

    @torch.no_grad()
    def capture_replay(
        self,
        latent_model_input,
        timestep,
        encoder_hidden_states,
        timestep_cond,
        cross_attention_kwargs,
        added_cond_kwargs,
    ):
        inputs = [
            latent_model_input,
            timestep,
            encoder_hidden_states,
            timestep_cond,
            cross_attention_kwargs,
            added_cond_kwargs,
        ]
        h = self.ht.hpu.graphs.input_hash(inputs)
        cached = self.cache.get(h)

        if cached is None:
            # Capture the graph and cache it
            with self.ht.hpu.stream(self.hpu_stream):
                graph = self.ht.hpu.HPUGraph()
                graph.capture_begin()

                outputs = self.unet(
                    sample=inputs[0],
                    timestep=inputs[1],
                    encoder_hidden_states=inputs[2],
                    timestep_cond=inputs[3],
                    cross_attention_kwargs=inputs[4],
                    added_cond_kwargs=inputs[5],
                    return_dict=False,
                )[0]

                graph.capture_end()
                graph_inputs = inputs
                graph_outputs = outputs
                self.cache[h] = self.ht.hpu.graphs.CachedParams(graph_inputs, graph_outputs, graph)
            return outputs

        # Replay the cached graph with updated inputs
        self.ht.hpu.graphs.copy_to(cached.graph_inputs, inputs)
        cached.graph.replay()
        self.ht.core.hpu.default_stream().synchronize()

        return cached.graph_outputs
