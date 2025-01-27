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

from dataclasses import dataclass
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from diffusers.models import AutoencoderKL, UNet3DConditionModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import TextToVideoSDPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from diffusers.utils.outputs import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer

from ....transformers.gaudi_configuration import GaudiConfig
from ..pipeline_utils import GaudiDiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class GaudiTextToVideoSDPipelineOutput(BaseOutput):
    videos: Union[List[PIL.Image.Image], np.ndarray]


class GaudiTextToVideoSDPipeline(GaudiDiffusionPipeline, TextToVideoSDPipeline):
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/pipelines/text_to_video_synthesis/pipeline_text_to_video_synth.py#L84

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer (`~transformers.CLIPTokenizer`):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
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
        unet: UNet3DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
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
        TextToVideoSDPipeline.__init__(
            self,
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
        )
        self.to(self._device)

    def enable_model_cpu_offload(self, *args, **kwargs):
        if self.use_habana:
            raise NotImplementedError("enable_model_cpu_offload() is not implemented for HPU")
        else:
            return super().enable_model_cpu_offload(*args, **kwargs)

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

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
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from optimum.habana.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.GaudiStableDiffusionPipeline._split_inputs_into_batches
    @classmethod
    def _split_inputs_into_batches(cls, batch_size, latents, prompt_embeds, negative_prompt_embeds):
        # Use torch.split to generate num_batches batches of size batch_size
        latents_batches = list(torch.split(latents, batch_size))
        prompt_embeds_batches = list(torch.split(prompt_embeds, batch_size))
        if negative_prompt_embeds is not None:
            negative_prompt_embeds_batches = list(torch.split(negative_prompt_embeds, batch_size))

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

        return latents_batches, prompt_embeds_batches, num_dummy_samples

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 16,
        batch_size: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide video generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated video.
            num_frames (`int`, *optional*, defaults to 16):
                The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds
                amounts to 2 seconds of video.
            batch_size (`int`, *optional*, defaults to 1):
                The number of videos in a batch.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality videos at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate videos closely linked to the text
                `prompt` at the expense of lower video quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in video generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_videos_per_prompt (`int`, defaults to 1):
                The number of videos to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`. Latents should be of shape
                `(batch_size, num_channel, num_frames, height, width)`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated video. Choose between `torch.FloatTensor` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] instead
                of a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        Examples:

        Returns:
            [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated frames.
        """
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=self.gaudi_config.use_torch_autocast):
            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                height,
                width,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
            )

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                num_prompts = 1
            elif prompt is not None and isinstance(prompt, list):
                num_prompts = len(prompt)
            else:
                num_prompts = prompt_embeds.shape[0]
            num_videos = num_videos_per_prompt * num_prompts
            num_batches = ceil((num_videos) / batch_size)
            logger.info(
                f"{num_prompts} prompt(s) received, {num_videos_per_prompt} generation(s) per prompt, "
                f"{batch_size} sample(s) per batch, {num_batches} total batch(es)."
            )
            if num_batches < 3:
                logger.warning("The first two iterations are slower so it is recommended to feed more batches.")

            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # 3. Encode input prompt
            text_encoder_lora_scale = (
                cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
            )
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=clip_skip,
            )

            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 5. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                num_prompts * num_videos_per_prompt,
                num_channels_latents,
                num_frames,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 7. Split into batches (HPU-specific step)
            latents_batches, text_embeddings_batches, num_dummy_samples = self._split_inputs_into_batches(
                batch_size,
                latents,
                prompt_embeds,
                negative_prompt_embeds,
            )

            # 8. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            outputs = []
            for j in self.progress_bar(range(num_batches)):
                latents_batch = latents_batches[0]
                latents_batches = torch.roll(latents_batches, shifts=-1, dims=0)
                text_embeddings_batch = text_embeddings_batches[0]
                text_embeddings_batches = torch.roll(text_embeddings_batches, shifts=-1, dims=0)
                for i in self.progress_bar(range(len(timesteps))):
                    t = timesteps[0]
                    timesteps = torch.roll(timesteps, shifts=-1, dims=0)
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents_batch] * 2) if do_classifier_free_guidance else latents_batch
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet_hpu(
                        latent_model_input,
                        t,
                        text_embeddings_batch,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # reshape latents
                    bsz, channel, frames, width, height = latents_batch.shape
                    latents_batch = latents_batch.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                    noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_batch = self.scheduler.step(noise_pred, t, latents_batch, **extra_step_kwargs).prev_sample

                    # reshape latents_batch back
                    latents_batch = (
                        latents_batch[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)
                    )

                    if not self.use_hpu_graphs:
                        self.htcore.mark_step()

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents_batch)
                if output_type == "latent":
                    video_tensor = latents_batch
                else:
                    video_tensor = self.decode_latents(latents_batch)
                outputs.append(video_tensor)

                if not self.use_hpu_graphs:
                    self.htcore.mark_step()

            # Remove dummy generations if needed
            if num_dummy_samples > 0:
                outputs[-1] = outputs[-1][:-num_dummy_samples]

            # 9. Post processing
            videos = []
            for video_tensor in outputs:
                if output_type == "latent":
                    videos.extend(list(video_tensor))
                    continue
                video_batch = self.video_processor.postprocess_video(video=video_tensor, output_type=output_type)

                if output_type == "pil" and isinstance(video_batch, list):
                    videos += video_batch
                elif output_type in ["np", "numpy"] and isinstance(video_batch, np.ndarray):
                    if len(videos) == 0:
                        videos = video_batch
                    else:
                        videos = np.concatenate((videos, video_batch), axis=0)
                else:  # Torch Tensor
                    if len(videos) == 0:
                        videos = video_batch
                    else:
                        videos = torch.cat((videos, video_batch), 0)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (videos,)

            return GaudiTextToVideoSDPipelineOutput(videos=videos)

    @torch.no_grad()
    def unet_hpu(self, latent_model_input, timestep, encoder_hidden_states, cross_attention_kwargs):
        if self.use_hpu_graphs:
            return self.capture_replay(latent_model_input, timestep, encoder_hidden_states, cross_attention_kwargs)
        else:
            return self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

    @torch.no_grad()
    def capture_replay(self, latent_model_input, timestep, encoder_hidden_states, cross_attention_kwargs):
        inputs = [latent_model_input, timestep, encoder_hidden_states, cross_attention_kwargs, False]
        h = self.ht.hpu.graphs.input_hash(inputs)
        cached = self.cache.get(h)

        if cached is None:
            # Capture the graph and cache it
            with self.ht.hpu.stream(self.hpu_stream):
                graph = self.ht.hpu.HPUGraph()
                graph.capture_begin()
                outputs = self.unet(
                    inputs[0], inputs[1], inputs[2], cross_attention_kwargs=inputs[3], return_dict=inputs[4]
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
