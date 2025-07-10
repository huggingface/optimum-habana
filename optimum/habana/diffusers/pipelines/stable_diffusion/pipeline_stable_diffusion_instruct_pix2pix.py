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
from math import ceil
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionInstructPix2PixPipeline, StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from optimum.utils import logging

from ....transformers.gaudi_configuration import GaudiConfig
from ....utils import HabanaProfile, speed_metrics, warmup_inference_steps_time_adjustment
from ..pipeline_utils import GaudiDiffusionPipeline
from .pipeline_stable_diffusion import GaudiStableDiffusionPipelineOutput


logger = logging.get_logger(__name__)


class GaudiStableDiffusionInstructPix2PixPipeline(GaudiDiffusionPipeline, StableDiffusionInstructPix2PixPipeline):
    """
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
    - Generation is performed by batches
    - Two `mark_step()` were added to add support for lazy mode
    - Added support for HPU graphs

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
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
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: Optional[CLIPVisionModelWithProjection] = None,
        requires_safety_checker: bool = True,
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

        # Workaround for Synapse 1.11 for full bf16
        if bf16_full_eval:
            unet.conv_in.float()

        StableDiffusionInstructPix2PixPipeline.__init__(
            self,
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            image_encoder,
            requires_safety_checker,
        )

        self.to(self._device)

    def prepare_latents(self, num_images, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (num_images, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != num_images:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective number"
                f" of images of {num_images}. Make sure the number of images matches the length of the generators."
            )

        if latents is None:
            # torch.randn is broken on HPU so running it on CPU
            rand_device = "cpu" if device.type == "hpu" else device
            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(num_images)
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
    def _split_inputs_into_batches(
        cls, batch_size, latents, prompt_embeds, image_latents, do_classifier_free_guidance
    ):
        # Use torch.split to generate num_batches batches of size batch_size
        latents_batches = list(torch.split(latents, batch_size))
        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.chunk(prompt_embeds, 3)[2]
            prompt_embeds = torch.chunk(prompt_embeds, 3)[0]
            uncond_image_latents = torch.chunk(image_latents, 3)[2]
            image_latents = torch.chunk(image_latents, 3)[0]
        else:
            negative_prompt_embeds = None
            uncond_image_latents = None

        prompt_embeds_batches = list(torch.split(prompt_embeds, batch_size))
        image_latents_batches = list(torch.split(image_latents, batch_size))
        if negative_prompt_embeds is not None:
            negative_prompt_embeds_batches = list(torch.split(negative_prompt_embeds, batch_size))
        if uncond_image_latents is not None:
            uncond_image_latents_batches = list(torch.split(uncond_image_latents, batch_size))

        # If the last batch has less samples than batch_size, pad it with dummy samples
        num_dummy_samples = 0
        if latents_batches[-1].shape[0] < batch_size:
            num_dummy_samples = batch_size - latents_batches[-1].shape[0]
            # Pad latents_batches
            sequence_to_stack = (latents_batches[-1],) + tuple(
                torch.zeros_like(latents_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
            )
            latents_batches[-1] = torch.vstack(sequence_to_stack)

            # Pad image latents_batches
            sequence_to_stack = (image_latents_batches[-1],) + tuple(
                torch.zeros_like(image_latents_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
            )
            image_latents_batches[-1] = torch.vstack(sequence_to_stack)

            # Pad prompt_embeds_batches
            sequence_to_stack = (prompt_embeds_batches[-1],) + tuple(
                torch.zeros_like(prompt_embeds_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
            )
            prompt_embeds_batches[-1] = torch.vstack(sequence_to_stack)

            if negative_prompt_embeds is not None:
                sequence_to_stack = (negative_prompt_embeds_batches[-1],) + tuple(
                    torch.zeros_like(negative_prompt_embeds_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                negative_prompt_embeds_batches[-1] = torch.vstack(sequence_to_stack)

            if uncond_image_latents is not None:
                sequence_to_stack = (uncond_image_latents_batches[-1],) + tuple(
                    torch.zeros_like(uncond_image_latents_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                uncond_image_latents_batches[-1] = torch.vstack(sequence_to_stack)

        # Stack batches in the same tensor
        latents_batches = torch.stack(latents_batches)
        if negative_prompt_embeds is not None:
            for i in range(len(negative_prompt_embeds_batches)):
                prompt_embeds_batches[i] = torch.cat(
                    [prompt_embeds_batches[i], negative_prompt_embeds_batches[i], negative_prompt_embeds_batches[i]]
                )
        prompt_embeds_batches = torch.stack(prompt_embeds_batches)
        if uncond_image_latents is not None:
            for i in range(len(uncond_image_latents_batches)):
                image_latents_batches[i] = torch.cat(
                    [image_latents_batches[i], image_latents_batches[i], uncond_image_latents_batches[i]]
                )
        image_latents_batches = torch.stack(image_latents_batches)
        return latents_batches, prompt_embeds_batches, image_latents_batches, num_dummy_samples

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        batch_size: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **kwargs,
    ):
        """
        Adapted from: https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
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
            # 0. Check inputs
            self.check_inputs(
                prompt,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                callback_on_step_end_tensor_inputs,
            )
            self._guidance_scale = guidance_scale
            self._image_guidance_scale = image_guidance_scale

            device = self._execution_device

            if ip_adapter_image is not None:
                output_hidden_state = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True
                image_embeds, negative_image_embeds = self.encode_image(
                    ip_adapter_image, device, num_images_per_prompt, output_hidden_state
                )
                if self.do_classifier_free_guidance:
                    image_embeds = torch.cat([image_embeds, negative_image_embeds, negative_image_embeds])

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

            # check if scheduler is in sigmas space
            scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")

            # 2. Encode input prompt
            prompt_embeds = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )

            # 3. Preprocess image
            image = self.image_processor.preprocess(image)

            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device="cpu")
            timesteps = self.scheduler.timesteps.to(device)

            # 5. Prepare Image latents
            image_latents = self.prepare_image_latents(
                image,
                num_prompts,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                self.do_classifier_free_guidance,
            )

            height, width = image_latents.shape[-2:]
            height = height * self.vae_scale_factor
            width = width * self.vae_scale_factor

            # 6. Prepare latent variables
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
            # 7. Check that shapes of latents and image match the UNet channels
            num_channels_image = image_latents.shape[1]
            if num_channels_latents + num_channels_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_image`: {num_channels_image} "
                    f" = {num_channels_latents + num_channels_image}. Please verify the config of"
                    " `pipeline.unet` or your `image` input."
                )

            # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 8.1 Add image embeds for IP-Adapter
            added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

            # 9. Split into batches (HPU-specific step)
            latents_batches, prompt_embeds_batches, image_latents_batches, num_dummy_samples = (
                self._split_inputs_into_batches(
                    batch_size,
                    latents,
                    prompt_embeds,
                    image_latents,
                    self.do_classifier_free_guidance,
                )
            )
            outputs = {
                "images": [],
                "has_nsfw_concept": [],
            }
            hb_profiler = HabanaProfile(
                warmup=profiling_warmup_steps,
                active=profiling_steps,
                record_shapes=False,
                name="stable_diffusion",
            )
            hb_profiler.start()

            # 10. Denoising loop
            t0 = time.time()
            t1 = t0
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
                image_latents_batch = image_latents_batches[0]
                image_latents_batches = torch.roll(image_latents_batches, shifts=-1, dims=0)
                prompt_embeds_batch = prompt_embeds_batches[0]
                prompt_embeds_batches = torch.roll(prompt_embeds_batches, shifts=-1, dims=0)

                for i in range(len(timesteps)):
                    if use_warmup_inference_steps and i == throughput_warmup_steps:
                        t1_inf = time.time()
                        t1 += t1_inf - t0_inf
                    t = timesteps[0]
                    timesteps = torch.roll(timesteps, shifts=-1, dims=0)
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents_batch] * 3) if self.do_classifier_free_guidance else latents_batch
                    )
                    scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents_batch], dim=1)

                    # predict the noise residual
                    noise_pred = self.unet_hpu(
                        scaled_latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds_batch,
                        added_cond_kwargs=added_cond_kwargs,
                    )

                    if scheduler_is_in_sigma_space:
                        step_index = (self.scheduler.timesteps == t).nonzero()[0].item()
                        sigma = self.scheduler.sigmas[step_index]
                        noise_pred = latent_model_input - sigma * noise_pred

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                        noise_pred = (
                            noise_pred_uncond
                            + self.guidance_scale * (noise_pred_text - noise_pred_image)
                            + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                        )

                    if scheduler_is_in_sigma_space:
                        noise_pred = (noise_pred - latents_batch) / (-sigma)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_batch = self.scheduler.step(
                        noise_pred, t, latents_batch, **extra_step_kwargs, return_dict=False
                    )[0]
                    if not self.use_hpu_graphs:
                        self.htcore.mark_step()

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents_batch = callback_outputs.pop("latents", latents_batch)
                        prompt_embeds_batch = callback_outputs.pop("prompt_embeds", prompt_embeds_batch)
                        image_latents_batch = callback_outputs.pop("image_latents", image_latents_batch)

                    # call the callback, if provided
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents_batch)
                    hb_profiler.step()

                if use_warmup_inference_steps:
                    t1 = warmup_inference_steps_time_adjustment(
                        t1, t1_inf, num_inference_steps, throughput_warmup_steps
                    )

                if not output_type == "latent":
                    image = self.vae.decode(latents_batch / self.vae.config.scaling_factor, return_dict=False)[0]
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

                if output_type == "latent":
                    has_nsfw_concept = None
                else:
                    image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

                if has_nsfw_concept is None:
                    do_denormalize = [True] * image.shape[0]
                else:
                    do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

                image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

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

                if has_nsfw_concept is not None:
                    outputs["has_nsfw_concept"] += has_nsfw_concept
                else:
                    outputs["has_nsfw_concept"] = None

            self.maybe_free_model_hooks()

            if not return_dict:
                return (outputs["images"], outputs["has_nsfw_concept"])

            return GaudiStableDiffusionPipelineOutput(
                images=outputs["images"],
                nsfw_content_detected=outputs["has_nsfw_concept"],
                throughput=speed_measures[f"{speed_metrics_prefix}_samples_per_second"],
            )

    @torch.no_grad()
    def unet_hpu(
        self,
        latent_model_input,
        timestep,
        encoder_hidden_states,
        added_cond_kwargs,
    ):
        if self.use_hpu_graphs:
            return self.capture_replay(latent_model_input, timestep, encoder_hidden_states)
        else:
            return self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

    @torch.no_grad()
    def capture_replay(self, latent_model_input, timestep, encoder_hidden_states):
        inputs = [latent_model_input, timestep, encoder_hidden_states, False]
        h = self.ht.hpu.graphs.input_hash(inputs)
        cached = self.cache.get(h)
        if cached is None:
            # Capture the graph and cache it
            with self.ht.hpu.stream(self.hpu_stream):
                graph = self.ht.hpu.HPUGraph()
                graph.capture_begin()
                outputs = self.unet(inputs[0], inputs[1], encoder_hidden_states=inputs[2], return_dict=inputs[3])[0]
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
