# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines import StableDiffusionUpscalePipeline
from diffusers.schedulers import DDPMScheduler, KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from optimum.utils import logging

from ....transformers.gaudi_configuration import GaudiConfig
from ....utils import speed_metrics, warmup_inference_steps_time_adjustment
from ..pipeline_utils import GaudiDiffusionPipeline


logger = logging.get_logger(__name__)

PipelineImageInput = Union[
    PIL.Image.Image, np.ndarray, torch.FloatTensor, List[PIL.Image.Image], List[np.ndarray], List[torch.FloatTensor]
]


@dataclass
class GaudiStableDiffusionPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
    throughput: float


class GaudiStableDiffusionUpscalePipeline(GaudiDiffusionPipeline, StableDiffusionUpscalePipeline):
    """
    Pipeline for text-guided image super-resolution using Stable Diffusion 2.

    Adapted from: https://github.com/huggingface/diffusers/blob/v0.23.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_upscale.py#L70
    - Generation is performed by batches
    - Two `mark_step()` were added to add support for lazy mode
    - Added support for HPU graphs

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        low_res_scheduler ([`SchedulerMixin`]):
            A scheduler used to add initial noise to the low resolution conditioning image. It must be an instance of
            [`DDPMScheduler`].
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
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
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        low_res_scheduler: DDPMScheduler,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: Optional[Any] = None,
        feature_extractor: Optional[CLIPImageProcessor] = None,
        watermarker: Optional[Any] = None,
        max_noise_level: int = 350,
        use_habana: bool = False,
        use_hpu_graphs: bool = False,
        gaudi_config: Union[str, GaudiConfig] = None,
        bf16_full_eval: bool = False,
    ):
        GaudiDiffusionPipeline.__init__(self, use_habana, use_hpu_graphs, gaudi_config, bf16_full_eval)

        # Workaround for Synapse 1.11 for full bf16
        if bf16_full_eval:
            unet.conv_in.float()

        StableDiffusionUpscalePipeline.__init__(
            self,
            vae,
            text_encoder,
            tokenizer,
            unet,
            low_res_scheduler,
            scheduler,
            safety_checker,
            feature_extractor,
            watermarker,
            max_noise_level,
        )

        self.to(self._device)

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height, width)
        if latents is None:
            # torch.randn is broken on HPU so running it on CPU
            rand_device = "cpu" if device.type == "hpu" else device
            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @classmethod
    def _split_inputs_into_batches(cls, batch_size, latents, text_embeddings, uncond_embeddings, image, noise_level):
        # Use torch.split to generate num_batches batches of size batch_size
        latents_batches = list(torch.split(latents, batch_size))
        text_embeddings_batches = list(torch.split(text_embeddings, batch_size))
        image_batches = list(torch.split(image, batch_size))
        noise_level_batches = list(torch.split(noise_level.view(-1, 1), batch_size))
        if uncond_embeddings is not None:
            uncond_embeddings_batches = list(torch.split(uncond_embeddings, batch_size))

        # If the last batch has less samples than batch_size, pad it with dummy samples
        num_dummy_samples = 0
        if latents_batches[-1].shape[0] < batch_size:
            num_dummy_samples = batch_size - latents_batches[-1].shape[0]
            # Pad latents_batches
            sequence_to_stack = (latents_batches[-1],) + tuple(
                torch.zeros_like(latents_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
            )
            latents_batches[-1] = torch.vstack(sequence_to_stack)
            # Pad image_batches
            sequence_to_stack = (image_batches[-1],) + tuple(
                torch.zeros_like(image_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
            )
            image_batches[-1] = torch.vstack(sequence_to_stack)
            # Pad noise_level_batches
            sequence_to_stack = (noise_level_batches[-1],) + tuple(
                torch.zeros_like(noise_level_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
            )
            noise_level_batches[-1] = torch.vstack(sequence_to_stack)
            # Pad text_embeddings_batches
            sequence_to_stack = (text_embeddings_batches[-1],) + tuple(
                torch.zeros_like(text_embeddings_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
            )
            text_embeddings_batches[-1] = torch.vstack(sequence_to_stack)
            # Pad uncond_embeddings_batches if necessary
            if uncond_embeddings is not None:
                sequence_to_stack = (uncond_embeddings_batches[-1],) + tuple(
                    torch.zeros_like(uncond_embeddings_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                uncond_embeddings_batches[-1] = torch.vstack(sequence_to_stack)

        # Stack batches in the same tensor
        latents_batches = torch.stack(latents_batches)
        if uncond_embeddings is not None:
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            for i, (uncond_embeddings_batch, text_embeddings_batch) in enumerate(
                zip(uncond_embeddings_batches, text_embeddings_batches[:])
            ):
                text_embeddings_batches[i] = torch.cat([uncond_embeddings_batch, text_embeddings_batch])
        text_embeddings_batches = torch.stack(text_embeddings_batches)
        image_batches = torch.stack(image_batches)
        noise_level_batches = torch.stack(noise_level_batches).squeeze(-1)

        return latents_batches, text_embeddings_batches, image_batches, noise_level_batches, num_dummy_samples

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        num_inference_steps: int = 75,
        guidance_scale: float = 9.0,
        noise_level: int = 20,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        batch_size: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image` or tensor representing an image batch to be upscaled.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            batch_size (`int`, *optional*, defaults to 1):
                The number of images in a batch.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated randomly.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.GaudiStableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Returns:
            [`~diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.GaudiStableDiffusionPipelineOutput`] or `tuple`:
            [`~diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.GaudiStableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.

        Examples:
        ```py
        >>> import requests   #TODO to test?
        >>> from PIL import Image
        >>> from io import BytesIO
        >>> from optimum.habana.diffusers import GaudiStableDiffusionUpscalePipeline
        >>> import torch

        >>> # load model and scheduler
        >>> model_id = "stabilityai/stable-diffusion-x4-upscaler"
        >>> pipeline = GaudiStableDiffusionUpscalePipeline.from_pretrained(
        ...     model_id, revision="fp16", torch_dtype=torch.bfloat16
        ... )
        >>> pipeline = pipeline.to("cuda")

        >>> # let's download an  image
        >>> url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
        >>> response = requests.get(url)
        >>> low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
        >>> low_res_img = low_res_img.resize((128, 128))
        >>> prompt = "a white cat"

        >>> upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
        >>> upscaled_image.save("upsampled_cat.png")
        ```
        """
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=self.gaudi_config.use_torch_autocast):
            # 0. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt, image, noise_level, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
            )
            if image is None:
                raise ValueError("`image` input cannot be undefined.")

            # 1. Define call parameters
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
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # 2. Encode input prompt
            text_encoder_lora_scale = (
                cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
            )
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=clip_skip,
            )

            # 3. Preprocess image
            image = self.image_processor.preprocess(image)
            image = image.to(dtype=prompt_embeds.dtype, device=device)

            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device="cpu")
            timesteps = self.scheduler.timesteps.to(device)
            self.scheduler.reset_timestep_dependent_params()

            # 5. Add noise to image
            noise_level = torch.tensor([noise_level], dtype=torch.long, device=device)
            noise = randn_tensor(image.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
            image = self.low_res_scheduler.add_noise(image, noise, noise_level)

            image = torch.cat([image] * num_images_per_prompt)
            noise_level = torch.cat([noise_level] * image.shape[0])

            # 6. Prepare latent variables
            height, width = image.shape[2:]
            num_channels_latents = self.vae.config.latent_channels
            latents = self.prepare_latents(
                num_prompts * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 7. Check that sizes of image and latents match
            num_channels_image = image.shape[1]
            if num_channels_latents + num_channels_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_image`: {num_channels_image} "
                    f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                    " `pipeline.unet` or your `image` input."
                )

            # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 9. Split into batches (HPU-specific step)
            (
                latents_batches,
                text_embeddings_batches,
                image_batches,
                noise_level_batches,
                num_dummy_samples,
            ) = self._split_inputs_into_batches(
                batch_size, latents, prompt_embeds, negative_prompt_embeds, image, noise_level
            )

            outputs = {"images": [], "has_nsfw_concept": []}
            t0 = time.time()
            t1 = t0

            # 10. Denoising loop
            throughput_warmup_steps = kwargs.get("throughput_warmup_steps", 3)
            use_warmup_inference_steps = (
                num_batches < throughput_warmup_steps and num_inference_steps > throughput_warmup_steps
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
                image_batch = image_batches[0]
                image_batches = torch.roll(image_batches, shifts=-1, dims=0)
                noise_level_batch = noise_level_batches[0]
                noise_level_batches = torch.roll(noise_level_batches, shifts=-1, dims=0)

                for i in range(len(timesteps)):
                    if use_warmup_inference_steps and i == throughput_warmup_steps:
                        t1_inf = time.time()
                        t1 += t1_inf - t0_inf

                    timestep = timesteps[0]
                    timesteps = torch.roll(timesteps, shifts=-1, dims=0)

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents_batch] * 2) if do_classifier_free_guidance else latents_batch
                    )
                    # latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep) #TODO why this has been removed?
                    image_input = torch.cat([image_batch] * 2) if do_classifier_free_guidance else image_batch
                    noise_level_input = (
                        torch.cat([noise_level_batch] * 2) if do_classifier_free_guidance else noise_level_batch
                    )
                    latent_model_input = torch.cat([latent_model_input, image_input], dim=1)

                    # predict the noise residual
                    noise_pred = self.unet_hpu(
                        latent_model_input,
                        timestep,
                        text_embeddings_batch,
                        cross_attention_kwargs,
                        class_labels=noise_level_input,
                    )

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_batch = self.scheduler.step(
                        noise_pred, timestep, latents_batch, **extra_step_kwargs, return_dict=False
                    )[0]

                    if not self.use_hpu_graphs:
                        self.htcore.mark_step()

                    # call the callback, if provided
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, timestep, latents_batch)

                if use_warmup_inference_steps:
                    t1 = warmup_inference_steps_time_adjustment(
                        t1, t1_inf, num_inference_steps, throughput_warmup_steps
                    )

                if not output_type == "latent":
                    # 8. Post-processing
                    # make sure the VAE is in float32 mode, as it overflows in bfloat16
                    needs_upcasting = self.vae.dtype == torch.bfloat16 and self.vae.config.force_upcast

                    if needs_upcasting:
                        self.upcast_vae()

                    # Ensure latents are always the same type as the VAE
                    latents_batch = latents_batch.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
                    image = self.vae.decode(latents_batch / self.vae.config.scaling_factor, return_dict=False)[0]

                    # cast back to fp16 if needed
                    if needs_upcasting:
                        self.vae.to(dtype=torch.bfloat16)

                    image, has_nsfw_concept, _ = self.run_safety_checker(image, device, prompt_embeds.dtype)

                else:
                    image = latents_batch
                outputs["images"].append(image)

                if not self.use_hpu_graphs:
                    self.htcore.mark_step()

            speed_metrics_prefix = "generation"
            speed_measures = speed_metrics(
                split=speed_metrics_prefix,
                start_time=t0,
                num_samples=num_batches * batch_size
                if t1 == t0 or use_warmup_inference_steps
                else (num_batches - throughput_warmup_steps) * batch_size,
                num_steps=num_batches,
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

                if output_type == "latent":
                    has_nsfw_concept = None
                else:
                    image, has_nsfw_concept, _ = self.run_safety_checker(image, device, prompt_embeds.dtype)

                if has_nsfw_concept is None:
                    do_denormalize = [True] * image.shape[0]
                else:
                    do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

                image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

                if output_type == "pil" and isinstance(image, list):
                    # Apply watermark
                    if self.watermarker is not None:
                        image = self.watermarker.apply_watermark(image)
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

                if has_nsfw_concept is not None:
                    outputs["has_nsfw_concept"] += has_nsfw_concept
                else:
                    outputs["has_nsfw_concept"] = None

            if not return_dict:
                return (outputs["images"], outputs["has_nsfw_concept"])

            return GaudiStableDiffusionPipelineOutput(
                images=outputs["images"],
                nsfw_content_detected=outputs["has_nsfw_concept"],
                throughput=speed_measures[f"{speed_metrics_prefix}_samples_per_second"],
            )

    @torch.no_grad()
    def unet_hpu(self, latent_model_input, timestep, encoder_hidden_states, cross_attention_kwargs, class_labels):
        if self.use_hpu_graphs:
            return self.capture_replay(latent_model_input, timestep, encoder_hidden_states, class_labels)
        else:
            return self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
                class_labels=class_labels,
            )[0]

    @torch.no_grad()
    def capture_replay(self, latent_model_input, timestep, encoder_hidden_states, class_labels):
        inputs = [latent_model_input, timestep, encoder_hidden_states, False, class_labels]
        h = self.ht.hpu.graphs.input_hash(inputs)
        cached = self.cache.get(h)

        if cached is None:
            # Capture the graph and cache it
            with self.ht.hpu.stream(self.hpu_stream):
                graph = self.ht.hpu.HPUGraph()
                graph.capture_begin()
                outputs = self.unet(
                    inputs[0],
                    timestep=inputs[1],
                    encoder_hidden_states=inputs[2],
                    return_dict=inputs[3],
                    class_labels=inputs[4],
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
