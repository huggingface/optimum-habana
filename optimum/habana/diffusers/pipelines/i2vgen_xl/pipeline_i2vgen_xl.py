# Copyright 2024 Alibaba DAMO-VILAB and The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKL
from diffusers.models.unets.unet_i2vgen_xl import I2VGenXLUNet
from diffusers.pipelines.i2vgen_xl.pipeline_i2vgen_xl import I2VGenXLPipeline, _center_crop_wide, _resize_bilinear
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import (
    BaseOutput,
    logging,
    replace_example_docstring,
)
from diffusers.video_processor import VideoProcessor
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ....transformers.gaudi_configuration import GaudiConfig
from ....utils import HabanaProfile, speed_metrics, warmup_inference_steps_time_adjustment
from ...models.attention_processor import (
    AttnProcessor2_0,
)
from ..pipeline_utils import GaudiDiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import I2VGenXLPipeline
        >>> from diffusers.utils import export_to_gif, load_image

        >>> pipeline = GaudiI2VGenXLPipeline.from_pretrained(
        ...     "ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16"
        ... )
        >>> pipeline.enable_model_cpu_offload()

        >>> image_url = (
        ...     "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"
        ... )
        >>> image = load_image(image_url).convert("RGB")

        >>> prompt = "Papers were floating in the air on a table in the library"
        >>> negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
        >>> generator = torch.manual_seed(8888)

        >>> frames = pipeline(
        ...     prompt=prompt,
        ...     image=image,
        ...     num_inference_steps=50,
        ...     negative_prompt=negative_prompt,
        ...     guidance_scale=9.0,
        ...     generator=generator,
        ... ).frames[0]
        >>> video_path = export_to_gif(frames, "i2v.gif")
        ```
"""


@dataclass
class GaudiI2VGenXLPipelineOutput(BaseOutput):
    r"""
     Output class for image-to-video pipeline.
     Copied from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/i2vgen_xl/pipeline_i2vgen_xl.py#L75
        - Add throughputs to the output class
    Args:
         frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
             List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
             denoised
     PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
    `(batch_size, num_frames, channels, height, width)`
    """

    frames: Union[torch.Tensor, np.ndarray, List[List[PIL.Image.Image]]]
    throughput: float


class GaudiI2VGenXLPipeline(
    GaudiDiffusionPipeline,
    I2VGenXLPipeline,
):
    r"""
    Copied from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/i2vgen_xl/pipeline_i2vgen_xl.py#L90
        - Use the GaudiDiffusionPipeline as the base class
        - Add the GaudiI2VGenXLPipelineOutput as the output class
        - Add the autocast into the __call__ method
        - Modify the __init__ method to inherit from GaudiDiffusionPipeline

    Pipeline for image-to-video generation as proposed in [I2VGenXL](https://i2vgen-xl.github.io/).

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer (`CLIPTokenizer`):
            A [`~transformers.CLIPTokenizer`] to tokenize text.
        unet ([`I2VGenXLUNet`]):
            A [`I2VGenXLUNet`] to denoise the encoded video latents.
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        image_encoder: CLIPVisionModelWithProjection,
        feature_extractor: CLIPImageProcessor,
        unet: I2VGenXLUNet,
        scheduler: DDIMScheduler,
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

        I2VGenXLPipeline.__init__(
            self,
            vae,
            text_encoder,
            tokenizer,
            image_encoder,
            feature_extractor,
            unet,
            scheduler,
        )

        if use_habana:
            self.unet.set_attn_processor(AttnProcessor2_0())

        if use_hpu_graphs:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph

            self.unet = wrap_in_hpu_graph(self.unet, disable_tensor_cache=True)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # `do_resize=False` as we do custom resizing.
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor, do_resize=False)
        self.to(self._device)

    @classmethod
    def _split_and_cat_tensors(cls, batch_size, input_a, input_b=None, do_classifier_free_guidance=True):
        if input_a is None:
            return None, 0

        input_a_batches = list(torch.split(input_a, batch_size))
        if input_b is not None:
            input_b_batches = list(torch.split(input_b, batch_size))

        num_dummy_samples = 0
        if input_a_batches[-1].shape[0] < batch_size:
            num_dummy_samples = batch_size - input_a_batches[-1].shape[0]
            # Pad input a
            sequence_to_stack = (input_a_batches[-1],) + tuple(
                torch.zeros_like(input_a_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
            )
            input_a_batches[-1] = torch.vstack(sequence_to_stack)

            if input_b is not None:
                # Pad input a
                sequence_to_stack = (input_b_batches[-1],) + tuple(
                    torch.zeros_like(input_b_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                input_b_batches[-1] = torch.vstack(sequence_to_stack)

        if input_b is not None and do_classifier_free_guidance:
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            for i, (input_b_batch, input_a_batch) in enumerate(zip(input_b_batches, input_a_batches[:])):
                input_a_batches[i] = torch.cat([input_b_batch, input_a_batch])

        input_a_batches = torch.stack(input_a_batches)
        return input_a_batches, num_dummy_samples

    """Copied from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/i2vgen_xl/pipeline_i2vgen_xl.py#L320
        - Commented out the code for do_classifier_free_guidance
    """

    def _encode_image(self, image, device, num_videos_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.video_processor.pil_to_numpy(image)
            image = self.video_processor.numpy_to_pt(image)

            # Normalize the image with CLIP training stats.
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # if self.do_classifier_free_guidance:
        #     negative_image_embeddings = torch.zeros_like(image_embeddings)
        #     image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    """Copied from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/i2vgen_xl/pipeline_i2vgen_xl.py#L441
        - Commented out the code for do_classifier_free_guidance
    """

    def prepare_image_latents(
        self,
        image,
        device,
        num_frames,
        num_videos_per_prompt,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.sample()
        image_latents = image_latents * self.vae.config.scaling_factor

        # Add frames dimension to image latents
        image_latents = image_latents.unsqueeze(2)

        # Append a position mask for each subsequent frame
        # after the intial image latent frame
        frame_position_mask = []
        for frame_idx in range(num_frames - 1):
            scale = (frame_idx + 1) / (num_frames - 1)
            frame_position_mask.append(torch.ones_like(image_latents[:, :, :1]) * scale)
        if frame_position_mask:
            frame_position_mask = torch.cat(frame_position_mask, dim=2)
            image_latents = torch.cat([image_latents, frame_position_mask], dim=2)

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1, 1)

        # if self.do_classifier_free_guidance:
        #     image_latents = torch.cat([image_latents] * 2)

        return image_latents

    """Copied from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/i2vgen_xl/pipeline_i2vgen_xl.py#L501
        - Add the autocast
        - Add the batching support
        - Add the throughput calculation
    """

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = 704,
        width: Optional[int] = 1280,
        target_fps: Optional[int] = 16,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        num_videos_per_prompt: Optional[int] = 1,
        decode_chunk_size: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = 1,
        batch_size: int = 1,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for image-to-video generation with [`I2VGenXLPipeline`].

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.Tensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            target_fps (`int`, *optional*):
                Frames per second. The rate at which the generated images shall be exported to a video after
                generation. This is also used as a "micro-condition" while generation.
            num_frames (`int`, *optional*):
                The number of video frames to generate.
            num_inference_steps (`int`, *optional*):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            eta (`float`, *optional*):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            num_videos_per_prompt (`int`, *optional*):
                The number of images to generate per prompt.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal
                consistency between frames, but also the higher the memory consumption. By default, the decoder will
                decode all frames at once for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            batch_size (`int`, *optional*, defaults to 1):
                The number of videos in a batch.

        Examples:

        Returns:
            [`pipelines.i2vgen_xl.pipeline_i2vgen_xl.I2VGenXLPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`pipelines.i2vgen_xl.pipeline_i2vgen_xl.I2VGenXLPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated frames.
        """
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=self.gaudi_config.use_torch_autocast):
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(prompt, image, height, width, negative_prompt, prompt_embeds, negative_prompt_embeds)
            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                num_prompts = 1
            elif prompt is not None and isinstance(prompt, list):
                num_prompts = len(prompt)
            else:
                num_prompts = prompt_embeds.shape[0]

            num_batches = ceil((num_videos_per_prompt * num_prompts) / batch_size)
            logger.info(
                f"{num_prompts} prompt(s) received, {num_videos_per_prompt} generation(s) per prompt,"
                f" {batch_size} sample(s) per batch, {num_batches} total batch(es)."
            )
            if num_batches < 3:
                logger.warning("The first two iterations are slower so it is recommended to feed more batches.")

            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            self._guidance_scale = guidance_scale

            # 3.1 Encode input text prompt
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                clip_skip=clip_skip,
            )
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            # if self.do_classifier_free_guidance:
            #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            # 3.2 Encode image prompt
            # 3.2.1 Image encodings.
            # https://github.com/ali-vilab/i2vgen-xl/blob/2539c9262ff8a2a22fa9daecbfd13f0a2dbc32d0/tools/inferences/inference_i2vgen_entrance.py#L114
            cropped_image = _center_crop_wide(image, (width, width))
            cropped_image = _resize_bilinear(
                cropped_image, (self.feature_extractor.crop_size["width"], self.feature_extractor.crop_size["height"])
            )
            image_embeddings = self._encode_image(cropped_image, device, num_videos_per_prompt)

            # 3.2.2 Image latents.
            resized_image = _center_crop_wide(image, (width, height))
            image = self.video_processor.preprocess(resized_image).to(device=device)
            needs_upcasting = (
                self.vae.dtype == torch.float16 or self.vae.dtype == torch.bfloat16
            ) and self.vae.config.force_upcast

            if needs_upcasting:
                cast_dtype = self.vae.dtype
                self.vae.to(dtype=torch.float32)

            image_latents = self.prepare_image_latents(
                image,
                device=device,
                num_frames=num_frames,
                num_videos_per_prompt=num_videos_per_prompt,
            )
            image_latents = image_latents.to(image_embeddings.dtype)

            # cast back to fp16/bf16 if needed
            if needs_upcasting:
                self.vae.to(dtype=cast_dtype)

            # 3.3 Prepare additional conditions for the UNet.
            # if self.do_classifier_free_guidance:
            #     fps_tensor = torch.tensor([target_fps, target_fps]).to(device)
            # else:
            fps_tensor = torch.tensor([target_fps]).to(device)
            fps_tensor = fps_tensor.repeat(num_prompts * num_videos_per_prompt, 1).ravel()

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

            # 7 Split into batches (HPU-specific step)
            latents_batches, num_dummy_samples = self._split_and_cat_tensors(batch_size, latents)
            prompt_embeds_batches, _ = self._split_and_cat_tensors(
                batch_size, prompt_embeds, negative_prompt_embeds, self.do_classifier_free_guidance
            )
            image_latents_batches, _ = self._split_and_cat_tensors(
                batch_size, image_latents, image_latents, self.do_classifier_free_guidance
            )
            image_embeddings_batches, _ = self._split_and_cat_tensors(
                batch_size, image_embeddings, torch.zeros_like(image_embeddings), self.do_classifier_free_guidance
            )
            fps_tensor_batches, _ = self._split_and_cat_tensors(
                batch_size, fps_tensor, fps_tensor, self.do_classifier_free_guidance
            )

            # 8. Denoising loop
            throughput_warmup_steps = kwargs.get("throughput_warmup_steps", 3)
            use_warmup_inference_steps = (
                num_batches <= throughput_warmup_steps and num_inference_steps > throughput_warmup_steps
            )

            outputs = {
                "videos": [],
            }
            t0 = time.time()
            t1 = t0

            hb_profiler = HabanaProfile(
                warmup=profiling_warmup_steps,
                active=profiling_steps,
                record_shapes=False,
            )
            hb_profiler.start()

            for j in self.progress_bar(range(num_batches)):
                # The throughput is calculated from the 3rd iteration
                # because compilation occurs in the first two iterations
                if j == throughput_warmup_steps:
                    t1 = time.time()
                if use_warmup_inference_steps:
                    t0_inf = time.time()

                latents_batch = latents_batches[0]
                latents_batches = torch.roll(latents_batches, shifts=-1, dims=0)
                prompt_embeds_batch = prompt_embeds_batches[0]
                prompt_embeds_batches = torch.roll(prompt_embeds_batches, shifts=-1, dims=0)
                fps_tensor_batch = fps_tensor_batches[0]
                fps_tensor_batches = torch.roll(fps_tensor_batches, shifts=-1, dims=0)
                image_latents_batch = image_latents_batches[0]
                image_latents_batches = torch.roll(image_latents_batches, shifts=-1, dims=0)
                image_embeddings_batch = image_embeddings_batches[0]
                image_embeddings_batches = torch.roll(image_embeddings_batches, shifts=-1, dims=0)

                if hasattr(self.scheduler, "_init_step_index"):
                    # Reset scheduler step index for next batch
                    self.scheduler._init_step_index(timesteps[0])

                for i in self.progress_bar(range(len(timesteps))):
                    if use_warmup_inference_steps and i == throughput_warmup_steps:
                        t1_inf = time.time()
                        t1 += t1_inf - t0_inf

                    # expand the latents if we are doing classifier free guidance
                    t = timesteps[0]
                    timesteps = torch.roll(timesteps, shifts=-1, dims=0)

                    latent_model_input = (
                        torch.cat([latents_batch] * 2) if self.do_classifier_free_guidance else latents_batch
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds_batch,
                        fps=fps_tensor_batch,
                        image_latents=image_latents_batch,
                        image_embeddings=image_embeddings_batch,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # reshape latents
                    bs, channel, frames, width, height = latents_batch.shape
                    latents_batch = latents_batch.permute(0, 2, 1, 3, 4).reshape(bs * frames, channel, width, height)
                    noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bs * frames, channel, width, height)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_batch = self.scheduler.step(noise_pred, t, latents_batch, **extra_step_kwargs).prev_sample

                    # reshape latents back
                    latents_batch = (
                        latents_batch[None, :].reshape(bs, frames, channel, width, height).permute(0, 2, 1, 3, 4)
                    )
                    # call the callback, if provided
                    # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    #     self.progress_bar.update()

                hb_profiler.step()

                if use_warmup_inference_steps:
                    t1 = warmup_inference_steps_time_adjustment(
                        t1, t1_inf, num_inference_steps, throughput_warmup_steps
                    )

                # 8. Post processing
                if output_type == "latent":
                    video = latents_batch
                else:
                    # cast back to fp16/bf16 if needed
                    if needs_upcasting:
                        self.vae.to(dtype=cast_dtype)

                    video_tensor = self.decode_latents(latents_batch, decode_chunk_size=decode_chunk_size)
                    video = self.video_processor.postprocess_video(video=video_tensor, output_type=output_type)

                outputs["videos"].append(video)

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
                outputs["videos"][-1] = outputs["videos"][-1][:-num_dummy_samples]

            # Process generated images
            for i, video in enumerate(outputs["videos"][:]):
                if i == 0:
                    outputs["videos"].clear()

                if output_type == "pil":
                    outputs["videos"] += video
                elif output_type in ["np", "numpy"] and isinstance(video, np.ndarray):
                    if len(outputs["videos"]) == 0:
                        outputs["videos"] = video
                    else:
                        outputs["videos"] = np.concatenate((outputs["videos"], video), axis=0)
                else:
                    outputs["videos"] += [*video]

            # 9. Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return outputs["videos"]

            return GaudiI2VGenXLPipelineOutput(
                frames=outputs["videos"],
                throughput=speed_measures[f"{speed_metrics_prefix}_samples_per_second"],
            )
