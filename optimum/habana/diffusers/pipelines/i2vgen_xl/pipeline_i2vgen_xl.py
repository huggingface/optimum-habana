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

import inspect
import time
from math import ceil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.pipelines.i2vgen_xl.pipeline_i2vgen_xl import I2VGenXLPipeline
from diffusers.pipelines.i2vgen_xl.pipeline_i2vgen_xl import _center_crop_wide, _resize_bilinear
from diffusers.models.unets.unet_i2vgen_xl import I2VGenXLUNet
from ....utils import HabanaProfile, speed_metrics, warmup_inference_steps_time_adjustment
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import (
    BaseOutput,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from ..pipeline_utils import GaudiDiffusionPipeline
from diffusers.pipelines.pipeline_utils import StableDiffusionMixin
from ....transformers.gaudi_configuration import GaudiConfig

from ...models.attention_processor import (
    AttentionProcessor,
    AttnProcessor2_0,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import I2VGenXLPipeline
        >>> from diffusers.utils import export_to_gif, load_image

        >>> pipeline = I2VGenXLPipeline.from_pretrained(
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
    ):
        GaudiDiffusionPipeline.__init__(
            self,
            use_habana,
            use_hpu_graphs,
            gaudi_config,
            bf16_full_eval,
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

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

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


    # def encode_prompt(
    #     self,
    #     prompt,
    #     device,
    #     num_videos_per_prompt,
    #     negative_prompt=None,
    #     prompt_embeds: Optional[torch.Tensor] = None,
    #     negative_prompt_embeds: Optional[torch.Tensor] = None,
    #     clip_skip: Optional[int] = None,
    # ):
    #     r"""
    #     Encodes the prompt into text encoder hidden states.

    #     Args:
    #         prompt (`str` or `List[str]`, *optional*):
    #             prompt to be encoded
    #         device: (`torch.device`):
    #             torch device
    #         num_videos_per_prompt (`int`):
    #             number of images that should be generated per prompt
    #         do_classifier_free_guidance (`bool`):
    #             whether to use classifier free guidance or not
    #         negative_prompt (`str` or `List[str]`, *optional*):
    #             The prompt or prompts not to guide the image generation. If not defined, one has to pass
    #             `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
    #             less than `1`).
    #         prompt_embeds (`torch.Tensor`, *optional*):
    #             Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
    #             provided, text embeddings will be generated from `prompt` input argument.
    #         negative_prompt_embeds (`torch.Tensor`, *optional*):
    #             Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
    #             weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
    #             argument.
    #         clip_skip (`int`, *optional*):
    #             Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
    #             the output of the pre-final layer will be used for computing the prompt embeddings.
    #     """
    #     if prompt is not None and isinstance(prompt, str):
    #         batch_size = 1
    #     elif prompt is not None and isinstance(prompt, list):
    #         batch_size = len(prompt)
    #     else:
    #         batch_size = prompt_embeds.shape[0]

    #     if prompt_embeds is None:
    #         text_inputs = self.tokenizer(
    #             prompt,
    #             padding="max_length",
    #             max_length=self.tokenizer.model_max_length,
    #             truncation=True,
    #             return_tensors="pt",
    #         )
    #         text_input_ids = text_inputs.input_ids
    #         untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    #         if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
    #             text_input_ids, untruncated_ids
    #         ):
    #             removed_text = self.tokenizer.batch_decode(
    #                 untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
    #             )
    #             logger.warning(
    #                 "The following part of your input was truncated because CLIP can only handle sequences up to"
    #                 f" {self.tokenizer.model_max_length} tokens: {removed_text}"
    #             )

    #         if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
    #             attention_mask = text_inputs.attention_mask.to(device)
    #         else:
    #             attention_mask = None

    #         if clip_skip is None:
    #             prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
    #             prompt_embeds = prompt_embeds[0]
    #         else:
    #             prompt_embeds = self.text_encoder(
    #                 text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
    #             )
    #             # Access the `hidden_states` first, that contains a tuple of
    #             # all the hidden states from the encoder layers. Then index into
    #             # the tuple to access the hidden states from the desired layer.
    #             prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
    #             # We also need to apply the final LayerNorm here to not mess with the
    #             # representations. The `last_hidden_states` that we typically use for
    #             # obtaining the final prompt representations passes through the LayerNorm
    #             # layer.
    #             prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

    #     if self.text_encoder is not None:
    #         prompt_embeds_dtype = self.text_encoder.dtype
    #     elif self.unet is not None:
    #         prompt_embeds_dtype = self.unet.dtype
    #     else:
    #         prompt_embeds_dtype = prompt_embeds.dtype

    #     prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    #     bs_embed, seq_len, _ = prompt_embeds.shape
    #     # duplicate text embeddings for each generation per prompt, using mps friendly method
    #     prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    #     prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

    #     # get unconditional embeddings for classifier free guidance
    #     if self.do_classifier_free_guidance and negative_prompt_embeds is None:
    #         uncond_tokens: List[str]
    #         if negative_prompt is None:
    #             uncond_tokens = [""] * batch_size
    #         elif prompt is not None and type(prompt) is not type(negative_prompt):
    #             raise TypeError(
    #                 f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
    #                 f" {type(prompt)}."
    #             )
    #         elif isinstance(negative_prompt, str):
    #             uncond_tokens = [negative_prompt]
    #         elif batch_size != len(negative_prompt):
    #             raise ValueError(
    #                 f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
    #                 f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
    #                 " the batch size of `prompt`."
    #             )
    #         else:
    #             uncond_tokens = negative_prompt

    #         max_length = prompt_embeds.shape[1]
    #         uncond_input = self.tokenizer(
    #             uncond_tokens,
    #             padding="max_length",
    #             max_length=max_length,
    #             truncation=True,
    #             return_tensors="pt",
    #         )

    #         if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
    #             attention_mask = uncond_input.attention_mask.to(device)
    #         else:
    #             attention_mask = None

    #         # Apply clip_skip to negative prompt embeds
    #         if clip_skip is None:
    #             negative_prompt_embeds = self.text_encoder(
    #                 uncond_input.input_ids.to(device),
    #                 attention_mask=attention_mask,
    #             )
    #             negative_prompt_embeds = negative_prompt_embeds[0]
    #         else:
    #             negative_prompt_embeds = self.text_encoder(
    #                 uncond_input.input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
    #             )
    #             # Access the `hidden_states` first, that contains a tuple of
    #             # all the hidden states from the encoder layers. Then index into
    #             # the tuple to access the hidden states from the desired layer.
    #             negative_prompt_embeds = negative_prompt_embeds[-1][-(clip_skip + 1)]
    #             # We also need to apply the final LayerNorm here to not mess with the
    #             # representations. The `last_hidden_states` that we typically use for
    #             # obtaining the final prompt representations passes through the LayerNorm
    #             # layer.
    #             negative_prompt_embeds = self.text_encoder.text_model.final_layer_norm(negative_prompt_embeds)

    #     if self.do_classifier_free_guidance:
    #         # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
    #         seq_len = negative_prompt_embeds.shape[1]

    #         negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    #         negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    #         negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    #     return prompt_embeds, negative_prompt_embeds

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

    # def decode_latents(self, latents, decode_chunk_size=None):
    #     latents = 1 / self.vae.config.scaling_factor * latents

    #     batch_size, channels, num_frames, height, width = latents.shape
    #     latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

    #     if decode_chunk_size is not None:
    #         frames = []
    #         for i in range(0, latents.shape[0], decode_chunk_size):
    #             frame = self.vae.decode(latents[i : i + decode_chunk_size]).sample
    #             frames.append(frame)
    #         image = torch.cat(frames, dim=0)
    #     else:
    #         image = self.vae.decode(latents).sample

    #     decode_shape = (batch_size, num_frames, -1) + image.shape[2:]
    #     video = image[None, :].reshape(decode_shape).permute(0, 2, 1, 3, 4)

    #     # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    #     video = video.float()
    #     return video

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    # def prepare_extra_step_kwargs(self, generator, eta):
    #     # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    #     # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    #     # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    #     # and should be between [0, 1]

    #     accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
    #     extra_step_kwargs = {}
    #     if accepts_eta:
    #         extra_step_kwargs["eta"] = eta

    #     # check if the scheduler accepts generator
    #     accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
    #     if accepts_generator:
    #         extra_step_kwargs["generator"] = generator
    #     return extra_step_kwargs

    # def check_inputs(
    #     self,
    #     prompt,
    #     image,
    #     height,
    #     width,
    #     negative_prompt=None,
    #     prompt_embeds=None,
    #     negative_prompt_embeds=None,
    # ):
    #     if height % 8 != 0 or width % 8 != 0:
    #         raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    #     if prompt is not None and prompt_embeds is not None:
    #         raise ValueError(
    #             f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
    #             " only forward one of the two."
    #         )
    #     elif prompt is None and prompt_embeds is None:
    #         raise ValueError(
    #             "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
    #         )
    #     elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
    #         raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    #     if negative_prompt is not None and negative_prompt_embeds is not None:
    #         raise ValueError(
    #             f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
    #             f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
    #         )

    #     if prompt_embeds is not None and negative_prompt_embeds is not None:
    #         if prompt_embeds.shape != negative_prompt_embeds.shape:
    #             raise ValueError(
    #                 "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
    #                 f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
    #                 f" {negative_prompt_embeds.shape}."
    #             )

    #     if (
    #         not isinstance(image, torch.Tensor)
    #         and not isinstance(image, PIL.Image.Image)
    #         and not isinstance(image, list)
    #     ):
    #         raise ValueError(
    #             "`image` has to be of type `torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
    #             f" {type(image)}"
    #         )

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

    # Copied from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth.TextToVideoSDPipeline.prepare_latents
    # def prepare_latents(
    #     self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    # ):
    #     shape = (
    #         batch_size,
    #         num_channels_latents,
    #         num_frames,
    #         height // self.vae_scale_factor,
    #         width // self.vae_scale_factor,
    #     )
    #     if isinstance(generator, list) and len(generator) != batch_size:
    #         raise ValueError(
    #             f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
    #             f" size of {batch_size}. Make sure the batch size matches the length of the generators."
    #         )

    #     if latents is None:
    #         latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    #     else:
    #         latents = latents.to(device)

    #     # scale the initial noise by the standard deviation required by the scheduler
    #     latents = latents * self.scheduler.init_noise_sigma
    #     return latents

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
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=(self.dtype != torch.float)):
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
            prompt_embeds_batches, _ = self._split_and_cat_tensors(batch_size, prompt_embeds, negative_prompt_embeds, self.do_classifier_free_guidance)
            image_latents_batches, _ = self._split_and_cat_tensors(batch_size, image_latents, image_latents, self.do_classifier_free_guidance)
            image_embeddings_batches, _ = self._split_and_cat_tensors(batch_size, image_embeddings, torch.zeros_like(image_embeddings), self.do_classifier_free_guidance)
            fps_tensor_batches, _ = self._split_and_cat_tensors(batch_size, fps_tensor, fps_tensor, self.do_classifier_free_guidance)


            # 8. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            throughput_warmup_steps = kwargs.get("throughput_warmup_steps", 3)
            use_warmup_inference_steps = (
                num_batches <= throughput_warmup_steps and num_inference_steps > throughput_warmup_steps
            )

            outputs = {
                "videos": [],
            }
            t0 = time.time()
            t1 = t0

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

                    latent_model_input = torch.cat([latents_batch] * 2) if self.do_classifier_free_guidance else latents_batch
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
                    latents_batch = latents_batch[None, :].reshape(bs, frames, channel, width, height).permute(0, 2, 1, 3, 4)
                    # call the callback, if provided
                    # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    #     self.progress_bar.update()

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
                else:
                    outputs["videos"] += [*video]

            # 9. Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return outputs["frames"]

            return GaudiI2VGenXLPipelineOutput(
                frames=outputs["videos"],
                throughput=speed_measures[f"{speed_metrics_prefix}_samples_per_second"],
            )