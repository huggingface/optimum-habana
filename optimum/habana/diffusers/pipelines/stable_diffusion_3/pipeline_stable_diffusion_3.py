# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
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
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    BaseOutput,
    replace_example_docstring,
)
from transformers import (
    BaseImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    PreTrainedModel,
    T5EncoderModel,
    T5TokenizerFast,
)

from optimum.utils import logging

from ....transformers.gaudi_configuration import GaudiConfig
from ....utils import HabanaProfile, speed_metrics, warmup_inference_steps_time_adjustment
from ..pipeline_utils import GaudiDiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class GaudiStableDiffusion3PipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    throughput: float


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from optimum.habana.diffusers import GaudiStableDiffusion3Pipeline

        >>> pipe = GaudiStableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers",
        ...     torch_dtype=torch.bfloat16,
        ...     use_habana=True,
        ...     use_hpu_graphs=True,
        ...     gaudi_config="Habana/stable-diffusion",
        ... )
        >>> image = pipe(
        ...     "A cat holding a sign that says hello world",
        ...     negative_prompt="",
        ...     num_inference_steps=28,
        ...     guidance_scale=7.0,
        ... ).images[0]
        >>> image.save("sd3.png")
        ```
"""


# ToDo: Look into FusedJointAttnProcessor2_0 usage for sd3 pipeline, and check its perf using fused sdpa
class GaudiJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections.
    Copied from JointAttnProcessor2_0.forward: https://github.com/huggingface/diffusers/blob/89e4d6219805975bd7d253a267e1951badc9f1c0/src/diffusers/models/attention_processor.py
        The only differences are:
        - applied Fused SDPA from Habana's framework.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        from habana_frameworks.torch.hpex.kernels import FusedSDPA

        hidden_states = FusedSDPA.apply(query, key, value, None, 0.0, False, None, "fast", None)

        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class GaudiStableDiffusion3Pipeline(GaudiDiffusionPipeline, StableDiffusion3Pipeline):
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.32.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L147

    Args:
        transformer ([`SD3Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
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

    _optional_components = ["image_encoder", "feature_extractor"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
        image_encoder: PreTrainedModel = None,
        feature_extractor: BaseImageProcessor = None,
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

        StableDiffusion3Pipeline.__init__(
            self,
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            text_encoder_3=text_encoder_3,
            tokenizer_3=tokenizer_3,
        )

        self.to(self._device)

    @classmethod
    def _split_inputs_into_batches(
        cls,
        batch_size,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ):
        # Use torch.split to generate num_batches batches of size batch_size
        latents_batches = list(torch.split(latents, batch_size))
        prompt_embeds_batches = list(torch.split(prompt_embeds, batch_size))

        if negative_prompt_embeds is not None:
            negative_prompt_embeds_batches = list(torch.split(negative_prompt_embeds, batch_size))
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds_batches = list(torch.split(pooled_prompt_embeds, batch_size))
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds_batches = list(torch.split(negative_pooled_prompt_embeds, batch_size))

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
            if pooled_prompt_embeds is not None:
                sequence_to_stack = (pooled_prompt_embeds_batches[-1],) + tuple(
                    torch.zeros_like(pooled_prompt_embeds_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                pooled_prompt_embeds_batches[-1] = torch.vstack(sequence_to_stack)
            # Pad negative_pooled_prompt_embeds_batches if necessary
            if negative_pooled_prompt_embeds is not None:
                sequence_to_stack = (negative_pooled_prompt_embeds_batches[-1],) + tuple(
                    torch.zeros_like(negative_pooled_prompt_embeds_batches[-1][0][None, :])
                    for _ in range(num_dummy_samples)
                )
                negative_pooled_prompt_embeds_batches[-1] = torch.vstack(sequence_to_stack)

        # Stack batches in the same tensor
        latents_batches = torch.stack(latents_batches)
        # if self.do_classifier_free_guidance:

        if negative_prompt_embeds is not None:
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            for i, (negative_prompt_embeds_batch, prompt_embeds_batch) in enumerate(
                zip(negative_prompt_embeds_batches, prompt_embeds_batches[:])
            ):
                prompt_embeds_batches[i] = torch.cat([negative_prompt_embeds_batch, prompt_embeds_batch])

        prompt_embeds_batches = torch.stack(prompt_embeds_batches)

        if pooled_prompt_embeds is not None:
            if negative_pooled_prompt_embeds is not None:
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                for i, (negative_pooled_prompt_embeds_batch, pooled_prompt_embeds_batch) in enumerate(
                    zip(negative_pooled_prompt_embeds_batches, pooled_prompt_embeds_batches[:])
                ):
                    pooled_prompt_embeds_batches[i] = torch.cat(
                        [negative_pooled_prompt_embeds_batch, pooled_prompt_embeds_batch]
                    )
            pooled_prompt_embeds_batches = torch.stack(pooled_prompt_embeds_batches)
        else:
            pooled_prompt_embeds_batches = None

        return latents_batches, prompt_embeds_batches, pooled_prompt_embeds_batches, num_dummy_samples

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        batch_size: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **kwargs,
    ):
        r"""
        Adapted from: https://github.com/huggingface/diffusers/blob/v0.29.2/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L634

        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            batch_size (`int`, *optional*, defaults to 1):
                The number of images in a batch.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (  ) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.
            profiling_warmup_steps (`int`, *optional*):
                Number of steps to ignore for profling.
            profiling_steps (`int`, *optional*):
                Number of steps to be captured when enabling profiling.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        import habana_frameworks.torch as ht
        import habana_frameworks.torch.core as htcore

        # Set dtype to BF16 only if --bf16 is used, else use device's default autocast precision
        # When --bf16 is used, bf16_full_eval=True, which disables use_torch_autocast
        with torch.autocast(
            device_type="hpu",
            enabled=self.gaudi_config.use_torch_autocast,
            dtype=torch.bfloat16 if not self.gaudi_config.use_torch_autocast else None,
        ):
            quant_mode = kwargs.get("quant_mode", None)
            if quant_mode == "measure" or quant_mode == "quantize":
                import os

                quant_config_path = os.getenv("QUANT_CONFIG")

                if not quant_config_path:
                    raise ImportError(
                        "Error: QUANT_CONFIG path is not defined. Please define path to quantization configuration JSON file."
                    )
                elif not os.path.isfile(quant_config_path):
                    raise ImportError(f"Error: QUANT_CONFIG path '{quant_config_path}' is not valid")

                htcore.hpu_set_env()
                from neural_compressor.torch.quantization import FP8Config, convert, prepare

                config = FP8Config.from_json_file(quant_config_path)
                if config.measure:
                    self.transformer = prepare(self.transformer, config)
                elif config.quantize:
                    self.transformer = convert(self.transformer, config)
                htcore.hpu_initialize(self.transformer, mark_only_scales_as_const=True)

            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                prompt_2,
                prompt_3,
                height,
                width,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

            self._guidance_scale = guidance_scale
            self._clip_skip = clip_skip
            self._joint_attention_kwargs = joint_attention_kwargs
            self._interrupt = False

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                num_prompts = 1
            elif prompt is not None and isinstance(prompt, list):
                num_prompts = len(prompt)
            else:
                num_prompts = prompt_embeds.shape[0]
            num_batches = ceil((num_images_per_prompt * num_prompts) / batch_size)

            device = self._execution_device

            lora_scale = kwargs.get("lora_scale", None) if kwargs is not None else None

            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_3=prompt_3,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                clip_skip=self.clip_skip,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

            # Pad the prompt embeddings ( text prompt feature space ) to the nearest multiple of the alignment size, the value which is compatible with softmax_hf8 kernels
            kernel_input_alignment_size = int(256 / prompt_embeds.element_size())

            pad_size = (
                ceil(prompt_embeds.shape[1] / kernel_input_alignment_size) * kernel_input_alignment_size
            ) - prompt_embeds.shape[1]
            prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, pad_size))
            negative_prompt_embeds = torch.nn.functional.pad(negative_prompt_embeds, (0, 0, 0, pad_size))

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)

            # 5. Prepare latent variables
            num_channels_latents = self.transformer.config.in_channels
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

            logger.info(
                f"{num_prompts} prompt(s) received, {num_images_per_prompt} generation(s) per prompt,"
                f" {batch_size} sample(s) per batch, {num_batches} total batch(es)."
            )
            if num_batches < 3:
                logger.warning("The first two iterations are slower so it is recommended to feed more batches.")

            throughput_warmup_steps = kwargs.get("throughput_warmup_steps", 3)
            use_warmup_inference_steps = (
                num_batches <= throughput_warmup_steps and num_inference_steps > throughput_warmup_steps
            )

            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index()

            hb_profiler = HabanaProfile(
                warmup=profiling_warmup_steps,
                active=profiling_steps,
                record_shapes=False,
            )

            hb_profiler.start()

            # 6. Split Input data to batches (HPU-specific step)
            latents_batches, text_embeddings_batches, pooled_prompt_embeddings_batches, num_dummy_samples = (
                self._split_inputs_into_batches(
                    batch_size,
                    latents,
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                )
            )

            outputs = {
                "images": [],
            }

            for block in self.transformer.transformer_blocks:
                block.attn.processor = GaudiJointAttnProcessor2_0()
            ht.hpu.synchronize()

            t0 = time.time()
            t1 = t0

            # 7. Denoising loop
            for j in range(num_batches):
                latents_batch = latents_batches[0]
                latents_batches = torch.roll(latents_batches, shifts=-1, dims=0)
                text_embeddings_batch = text_embeddings_batches[0]
                text_embeddings_batches = torch.roll(text_embeddings_batches, shifts=-1, dims=0)
                pooled_prompt_embeddings_batch = pooled_prompt_embeddings_batches[0]
                pooled_prompt_embeddings_batches = torch.roll(pooled_prompt_embeddings_batches, shifts=-1, dims=0)

                if hasattr(self.scheduler, "_init_step_index"):
                    # Reset scheduler step index for next batch
                    self.scheduler.timesteps = timesteps
                    self.scheduler._init_step_index(timesteps[0])

                # Throughput is calculated after warmup iterations
                if j == throughput_warmup_steps:
                    t1 = time.time()

                for i in self.progress_bar(range(len(timesteps))):
                    timestep = timesteps[0]
                    timesteps = torch.roll(timesteps, shifts=-1, dims=0)

                    if use_warmup_inference_steps and i == throughput_warmup_steps and j == num_batches - 1:
                        t1 = time.time()

                    if self.interrupt:
                        continue

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents_batch] * 2) if self.do_classifier_free_guidance else latents_batch
                    )
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep_batch = timestep.expand(latent_model_input.shape[0])

                    noise_pred = self.transformer_hpu(
                        latent_model_input,
                        timestep_batch,
                        text_embeddings_batch,
                        pooled_prompt_embeddings_batch,
                        self.joint_attention_kwargs,
                    )

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents_batch.dtype
                    latents_batch = self.scheduler.step(noise_pred, timestep, latents_batch, return_dict=False)[0]

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
                        _pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", None)
                        _negative_pooled_prompt_embeds = callback_outputs.pop("negative_pooled_prompt_embeds", None)
                        if _pooled_prompt_embeds is not None and _negative_pooled_prompt_embeds is not None:
                            pooled_prompt_embeddings_batch = torch.cat(
                                [_negative_pooled_prompt_embeds, _pooled_prompt_embeds]
                            )

                    hb_profiler.step()
                    htcore.mark_step(sync=True)

                if output_type == "latent":
                    image = latents_batch

                else:
                    latents_batch = (latents_batch / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    image = self.vae.decode(latents_batch, return_dict=False)[0]
                    image = self.image_processor.postprocess(image, output_type=output_type)

                outputs["images"].append(image)

            # End of Denoising loop

            hb_profiler.stop()

            ht.hpu.synchronize()
            speed_metrics_prefix = "generation"
            if use_warmup_inference_steps:
                t1 = warmup_inference_steps_time_adjustment(t1, t1, num_inference_steps, throughput_warmup_steps)
            speed_measures = speed_metrics(
                split=speed_metrics_prefix,
                start_time=t0,
                num_samples=batch_size
                if t1 == t0 or use_warmup_inference_steps
                else (num_batches - throughput_warmup_steps) * batch_size,
                num_steps=batch_size * num_inference_steps
                if use_warmup_inference_steps
                else (num_batches - throughput_warmup_steps) * batch_size * num_inference_steps,
                start_time_after_warmup=t1,
            )
            logger.info(f"Speed metrics: {speed_measures}")

            if quant_mode == "measure":
                from neural_compressor.torch.quantization import finalize_calibration

                finalize_calibration(self.transformer)

            # 8 Output Images
            # Remove dummy generations if needed
            if num_dummy_samples > 0:
                outputs["images"][-1] = outputs["images"][-1][:-num_dummy_samples]

            # Process generated images
            for i, image in enumerate(outputs["images"][:]):
                if i == 0:
                    outputs["images"].clear()

                # image = self.image_processor.postprocess(image, output_type=output_type)

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

            return GaudiStableDiffusion3PipelineOutput(
                images=outputs["images"],
                throughput=speed_measures[f"{speed_metrics_prefix}_samples_per_second"],
            )

    @torch.no_grad()
    def transformer_hpu(
        self,
        latent_model_input,
        timestep,
        text_embeddings_batch,
        pooled_prompt_embeddings_batch,
        joint_attention_kwargs,
    ):
        if self.use_hpu_graphs:
            return self.capture_replay(
                latent_model_input,
                timestep,
                text_embeddings_batch,
                pooled_prompt_embeddings_batch,
                joint_attention_kwargs,
            )
        else:
            return self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=text_embeddings_batch,
                pooled_projections=pooled_prompt_embeddings_batch,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]

    @torch.no_grad()
    def capture_replay(
        self,
        latent_model_input,
        timestep,
        encoder_hidden_states,
        pooled_prompt_embeddings_batch,
        joint_attention_kwargs,
    ):
        inputs = [
            latent_model_input,
            timestep,
            encoder_hidden_states,
            pooled_prompt_embeddings_batch,
            joint_attention_kwargs,
        ]
        h = self.ht.hpu.graphs.input_hash(inputs)
        cached = self.cache.get(h)

        if cached is None:
            # Capture the graph and cache it
            with self.ht.hpu.stream(self.hpu_stream):
                graph = self.ht.hpu.HPUGraph()
                graph.capture_begin()

                outputs = self.transformer(
                    hidden_states=inputs[0],
                    timestep=inputs[1],
                    encoder_hidden_states=inputs[2],
                    pooled_projections=inputs[3],
                    joint_attention_kwargs=inputs[4],
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
