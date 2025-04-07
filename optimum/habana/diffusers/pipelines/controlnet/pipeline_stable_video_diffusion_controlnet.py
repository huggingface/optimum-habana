# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import PIL.Image
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.pipelines.stable_video_diffusion import StableVideoDiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _append_dims,
    _resize_with_antialiasing,
)
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ....transformers.gaudi_configuration import GaudiConfig
from ....utils import speed_metrics
from ...models import ControlNetSDVModel, UNetSpatioTemporalConditionControlNetModel
from ..pipeline_utils import GaudiDiffusionPipeline
from ..stable_video_diffusion.pipeline_stable_video_diffusion import (
    GaudiStableVideoDiffusionPipeline,
    GaudiStableVideoDiffusionPipelineOutput,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class GaudiStableVideoDiffusionControlNetPipeline(GaudiStableVideoDiffusionPipeline):
    r"""
    Adapted from: https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/765cd95c3659c54593ae36a9616121f00b3d7c29/pipeline/pipeline_stable_video_diffusion_controlnet.py#L99
    - Generation is performed by batches
    - Added support for HPU graphs

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionControlNetModel`]):
            A `UNetSpatioTemporalConditionControlNetModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
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
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionControlNetModel,
        controlnet: ControlNetSDVModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
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

        StableVideoDiffusionPipeline.__init__(
            self,
            vae,
            image_encoder,
            unet,
            scheduler,
            feature_extractor,
        )

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            controlnet=controlnet,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.to(self._device)

    def _encode_image(self, image, device, num_videos_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            """ This is different with statble_video_diffusion
            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values
            """

        # image = image.unsqueeze(0)
        image = _resize_with_antialiasing(image, (224, 224))

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    @classmethod
    def _split_inputs_into_batches(
        cls,
        batch_size,
        latents,
        image_latents,
        image_embeddings,
        controlnet_condition,
        added_time_ids,
        num_images,
        do_classifier_free_guidance,
    ):
        if do_classifier_free_guidance:
            negative_image_embeddings, image_embeddings = image_embeddings.chunk(2)
            negative_added_time_ids, added_time_ids = added_time_ids.chunk(2)
        else:
            negative_image_embeddings = None
            negative_added_time_ids = None

        # If the last batch has less samples than batch_size, compute number of dummy samples to pad
        last_samples = latents.shape[0] % batch_size
        num_dummy_samples = batch_size - last_samples if last_samples > 0 else 0

        # Generate num_batches batches of size batch_size
        latents_batches = cls._split_input_into_batches(latents, batch_size, num_dummy_samples)
        image_latents_batches = cls._split_image_latents_into_batches(
            image_latents, batch_size, num_dummy_samples, num_images, do_classifier_free_guidance
        )
        image_embeddings_batches = cls._split_input_into_batches(
            image_embeddings, batch_size, num_dummy_samples, negative_image_embeddings
        )
        controlnet_condition_batches = cls._split_input_into_batches(
            controlnet_condition, batch_size, num_dummy_samples
        )
        added_time_ids_batches = cls._split_input_into_batches(
            added_time_ids, batch_size, num_dummy_samples, negative_added_time_ids
        )

        return (
            latents_batches,
            image_latents_batches,
            image_embeddings_batches,
            controlnet_condition_batches,
            added_time_ids_batches,
            num_dummy_samples,
        )

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        controlnet_condition: Optional[torch.FloatTensor] = None,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        controlnet_cond_scale=1.0,
        batch_size=1,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=self.gaudi_config.use_torch_autocast):
            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
            decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(image, height, width)

            # 2. Define call parameters
            if isinstance(image, PIL.Image.Image):
                num_images = 1
            elif isinstance(image, list):
                num_images = len(image)
            else:
                num_images = image.shape[0]
            num_batches = ceil((num_videos_per_prompt * num_images) / batch_size)
            logger.info(
                f"{num_images} image(s) received, {num_videos_per_prompt} video(s) per prompt,"
                f" {batch_size} sample(s) per batch, {num_batches} total batch(es)."
            )
            if num_batches < 3:
                logger.warning("The first two iterations are slower so it is recommended to feed more batches.")

            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = max_guidance_scale > 1.0

            # 3. Encode input image
            image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)

            # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
            # is why it is reduced here.
            # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
            fps = fps - 1

            # 4. Encode input image using VAE
            image = self.video_processor.preprocess(image, height=height, width=width)
            # torch.randn is broken on HPU so running it on CPU
            rand_device = "cpu" if device.type == "hpu" else device
            noise = randn_tensor(image.shape, generator=generator, device=rand_device, dtype=image.dtype).to(device)
            image = image + noise_aug_strength * noise

            needs_upcasting = (
                self.vae.dtype == torch.float16 or self.vae.dtype == torch.bfloat16
            ) and self.vae.config.force_upcast

            if needs_upcasting:
                cast_dtype = self.vae.dtype
                self.vae.to(dtype=torch.float32)

            image_latents = self._encode_vae_image(
                image, device, num_videos_per_prompt, do_classifier_free_guidance=False
            )  # Override to return only conditional latents
            image_latents = image_latents.to(image_embeddings.dtype)

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)

            # Repeat the image latents for each frame so we can concatenate them with the noise
            # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
            image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

            # 5. Get Added Time IDs
            added_time_ids = self._get_add_time_ids(
                fps,
                motion_bucket_id,
                noise_aug_strength,
                image_embeddings.dtype,
                batch_size,
                num_videos_per_prompt,
                do_classifier_free_guidance,
            )
            added_time_ids = added_time_ids.to(device)
            # 6. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            self.scheduler.reset_timestep_dependent_params()

            # 7. Prepare latent variables

            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_frames,
                num_channels_latents,
                height,
                width,
                image_embeddings.dtype,
                device,
                generator,
                latents,
            )
            # prepare controlnet condition
            controlnet_condition = self.image_processor.preprocess(controlnet_condition, height=height, width=width)
            controlnet_condition = controlnet_condition.unsqueeze(0)
            controlnet_condition = controlnet_condition.to(device, latents.dtype)

            # 8. Prepare guidance scale
            guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
            guidance_scale = guidance_scale.to(device, latents.dtype)
            guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
            guidance_scale = _append_dims(guidance_scale, latents.ndim)

            self._guidance_scale = guidance_scale
            # 9. Split into batches (HPU-specific step)
            (
                latents_batches,
                image_latents_batches,
                image_embeddings_batches,
                controlnet_condition_batches,
                added_time_ids_batches,
                num_dummy_samples,
            ) = self._split_inputs_into_batches(
                batch_size,
                latents,
                image_latents,
                image_embeddings,
                controlnet_condition,
                added_time_ids,
                num_images,
                do_classifier_free_guidance,
                # self.do_classifier_free_guidance(),
            )
            outputs = {
                "frames": [],
            }
            t0 = time.time()
            t1 = t0

            # 10. Denoising loop
            throughput_warmup_steps = kwargs.get("throughput_warmup_steps", 3)
            self._num_timesteps = len(timesteps)
            for j in self.progress_bar(range(num_batches)):
                # The throughput is calculated from the 3rd iteration
                # because compilation occurs in the first two iterations
                if j == throughput_warmup_steps:
                    t1 = time.time()

                latents_batch = latents_batches[0]
                latents_batches = torch.roll(latents_batches, shifts=-1, dims=0)
                image_latents_batch = image_latents_batches[0]
                image_latents_batches = torch.roll(image_latents_batches, shifts=-1, dims=0)
                image_embeddings_batch = image_embeddings_batches[0]
                image_embeddings_batches = torch.roll(image_embeddings_batches, shifts=-1, dims=0)
                added_time_ids_batch = added_time_ids_batches[0]
                added_time_ids_batches = torch.roll(added_time_ids_batches, shifts=-1, dims=0)
                controlnet_condition_batch = controlnet_condition_batches[0]
                controlnet_condition_batches = torch.roll(controlnet_condition_batches, shifts=-1, dims=0)

                for i in self.progress_bar(range(num_inference_steps)):
                    timestep = timesteps[0]
                    timesteps = torch.roll(timesteps, shifts=-1, dims=0)

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents_batch] * 2) if do_classifier_free_guidance else latents_batch
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
                    if do_classifier_free_guidance:
                        controlnet_condition_input = torch.cat([controlnet_condition_batch] * 2)

                    # Concatenate image_latents over channels dimention
                    latent_model_input = torch.cat([latent_model_input, image_latents_batch], dim=2)
                    down_block_res_samples, mid_block_res_sample = self.controlnet_hpu(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=image_embeddings_batch,
                        added_time_ids=added_time_ids_batch,
                        controlnet_cond=controlnet_condition_input,
                        return_dict=False,
                        guess_mode=False,
                        conditioning_scale=controlnet_cond_scale,
                    )
                    # predict the noise residual
                    noise_pred = self.unet_hpu(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=image_embeddings_batch,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                        added_time_ids=added_time_ids_batch,
                    )
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_batch = self.scheduler.step(noise_pred, timestep, latents_batch).prev_sample
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, timestep, callback_kwargs)

                        latents_batch = callback_outputs.pop("latents", latents_batch)

                if not output_type == "latent":
                    # cast back to fp16 if needed
                    if needs_upcasting:
                        self.vae.to(dtype=cast_dtype)
                    frames = self.decode_latents(latents_batch, num_frames, decode_chunk_size)
                    frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
                else:
                    frames = latents_batch

                outputs["frames"].append(frames)

            speed_metrics_prefix = "generation"
            speed_measures = speed_metrics(
                split=speed_metrics_prefix,
                start_time=t0,
                num_samples=num_batches * batch_size
                if t1 == t0
                else (num_batches - throughput_warmup_steps) * batch_size,
                num_steps=num_batches,
                start_time_after_warmup=t1,
            )
            logger.info(f"Speed metrics: {speed_measures}")

            # Remove dummy generations if needed
            if num_dummy_samples > 0:
                outputs["frames"][-1] = outputs["frames"][-1][:-num_dummy_samples]

            # Process generated images
            for i, frames in enumerate(outputs["frames"][:]):
                if i == 0:
                    outputs["frames"].clear()

                if output_type == "pil":
                    outputs["frames"] += frames
                else:
                    outputs["frames"] += [*frames]

            self.maybe_free_model_hooks()

            if not return_dict:
                return outputs["frames"]

            return GaudiStableVideoDiffusionPipelineOutput(
                frames=outputs["frames"],
                throughput=speed_measures[f"{speed_metrics_prefix}_samples_per_second"],
            )

    @torch.no_grad()
    def controlnet_hpu(
        self,
        control_model_input,
        timestep,
        encoder_hidden_states,
        added_time_ids,
        controlnet_cond,
        return_dict,
        guess_mode,
        conditioning_scale,
    ):
        if self.use_hpu_graphs:
            return self.controlnet_capture_replay(
                control_model_input,
                timestep,
                encoder_hidden_states,
                added_time_ids,
                controlnet_cond,
                return_dict,
                guess_mode,
                conditioning_scale,
            )
        else:
            return self.controlnet(
                control_model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_time_ids=added_time_ids,
                controlnet_cond=controlnet_cond,
                return_dict=return_dict,
                guess_mode=guess_mode,
                conditioning_scale=conditioning_scale,
            )

    @torch.no_grad()
    def controlnet_capture_replay(
        self,
        control_model_input,
        timestep,
        encoder_hidden_states,
        added_time_ids,
        controlnet_cond,
        return_dict,
        guess_mode,
        conditioning_scale,
    ):
        inputs = [
            control_model_input,
            timestep,
            encoder_hidden_states,
            added_time_ids,
            controlnet_cond,
            return_dict,
            guess_mode,
            conditioning_scale,
        ]
        h = self.ht.hpu.graphs.input_hash(inputs)
        cached = self.cache.get(h)

        if cached is None:
            # Capture the graph and cache it
            with self.ht.hpu.stream(self.hpu_stream):
                graph = self.ht.hpu.HPUGraph()
                graph.capture_begin()
                outputs = self.controlnet(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    None,
                    inputs[5],
                    inputs[6],
                    inputs[7],
                )
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

    @torch.no_grad()
    def unet_hpu(
        self,
        latent_model_input,
        timestep,
        encoder_hidden_states,
        down_block_additional_residuals,
        mid_block_additional_residual,
        return_dict,
        added_time_ids,
    ):
        if self.use_hpu_graphs:
            return self.unet_capture_replay(
                latent_model_input,
                timestep,
                encoder_hidden_states,
                down_block_additional_residuals,
                mid_block_additional_residual,
                return_dict,
                added_time_ids,
            )
        else:
            return self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                return_dict=return_dict,
                added_time_ids=added_time_ids,
            )[0]

    @torch.no_grad()
    def unet_capture_replay(
        self,
        latent_model_input,
        timestep,
        encoder_hidden_states,
        down_block_additional_residuals,
        mid_block_additional_residual,
        return_dict,
        added_time_ids,
    ):
        inputs = [
            latent_model_input,
            timestep,
            encoder_hidden_states,
            down_block_additional_residuals,
            mid_block_additional_residual,
            return_dict,
            added_time_ids,
        ]
        h = self.ht.hpu.graphs.input_hash(inputs)
        cached = self.cache.get(h)

        if cached is None:
            # Capture the graph and cache it
            with self.ht.hpu.stream(self.hpu_stream):
                graph = self.ht.hpu.HPUGraph()
                graph.capture_begin()
                outputs = self.unet(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
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
