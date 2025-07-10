# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, calculate_shift, retrieve_timesteps
from diffusers.utils import BaseOutput, replace_example_docstring
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from optimum.utils import logging

from ....transformers.gaudi_configuration import GaudiConfig
from ....utils import HabanaProfile, speed_metrics, warmup_inference_steps_time_adjustment
from ...models.attention_processor import GaudiFluxAttnProcessor2_0
from ...schedulers import GaudiFlowMatchEulerDiscreteScheduler
from ..pipeline_utils import GaudiDiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class GaudiFluxPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    throughput: float


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from optimum.habana.diffusers import GaudiFluxPipeline

        >>> pipe = GaudiFluxPipeline.from_pretrained(
        ...    "black-forest-labs/FLUX.1-schnell",
        ...     torch_dtype=torch.bfloat16,
        ...     use_habana=True,
        ...     use_hpu_graphs=True,
        ...     gaudi_config="Habana/stable-diffusion",
        ... )
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""


class GaudiFluxPipeline(GaudiDiffusionPipeline, FluxPipeline):
    r"""
    Adapted from https://github.com/huggingface/diffusers/blob/v0.32.0/src/diffusers/pipelines/flux/pipeline_flux.py#L140
        Added batch size control for inference, and support for HPU graphs and Gaudi quantization via Intel Neural Compressor

    The Flux pipeline for text-to-image generation.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
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

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: GaudiFlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        use_habana: bool = False,
        use_hpu_graphs: bool = False,
        gaudi_config: Union[str, GaudiConfig] = None,
        bf16_full_eval: bool = False,
        sdp_on_bf16: bool = False,
        is_training: bool = False,
    ):
        GaudiDiffusionPipeline.__init__(
            self,
            use_habana,
            use_hpu_graphs,
            gaudi_config,
            bf16_full_eval,
            sdp_on_bf16,
        )
        FluxPipeline.__init__(
            self,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
        )

        for block in self.transformer.single_transformer_blocks:
            block.attn.processor = GaudiFluxAttnProcessor2_0(is_training)
        for block in self.transformer.transformer_blocks:
            block.attn.processor = GaudiFluxAttnProcessor2_0(is_training)

        self.to(self._device)
        if use_hpu_graphs:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph

            transformer = wrap_in_hpu_graph(transformer)

    @classmethod
    def _split_inputs_into_batches(cls, batch_size, latents, prompt_embeds, pooled_prompt_embeds, guidance):
        # Use torch.split to generate num_batches batches of size batch_size
        latents_batches = list(torch.split(latents, batch_size))
        prompt_embeds_batches = list(torch.split(prompt_embeds, batch_size))
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds_batches = list(torch.split(pooled_prompt_embeds, batch_size))
        if guidance is not None:
            guidance_batches = list(torch.split(guidance, batch_size))

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

            # Pad pooled_prompt_embeds if necessary
            if pooled_prompt_embeds is not None:
                sequence_to_stack = (pooled_prompt_embeds_batches[-1],) + tuple(
                    torch.zeros_like(pooled_prompt_embeds_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                pooled_prompt_embeds_batches[-1] = torch.vstack(sequence_to_stack)

            # Pad guidance if necessary
            if guidance is not None:
                guidance_batches[-1] = guidance_batches[-1].unsqueeze(1)
                sequence_to_stack = (guidance_batches[-1],) + tuple(
                    torch.zeros_like(guidance_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                guidance_batches[-1] = torch.vstack(sequence_to_stack).squeeze(1)

        # Stack batches in the same tensor
        latents_batches = torch.stack(latents_batches)
        prompt_embeds_batches = torch.stack(prompt_embeds_batches)
        pooled_prompt_embeds_batches = torch.stack(pooled_prompt_embeds_batches)
        guidance_batches = torch.stack(guidance_batches) if guidance is not None else None

        return (
            latents_batches,
            prompt_embeds_batches,
            pooled_prompt_embeds_batches,
            guidance_batches,
            num_dummy_samples,
        )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        batch_size: int = 1,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **kwargs,
    ):
        r"""
        Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py#L531
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
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
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
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
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
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
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.
            profiling_warmup_steps (`int`, *optional*):
                Number of steps to ignore for profling.
            profiling_steps (`int`, *optional*):
                Number of steps to be captured when enabling profiling.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        import habana_frameworks.torch as ht
        import habana_frameworks.torch.core as htcore

        quant_mode = kwargs.get("quant_mode", None)

        if quant_mode == "quantize-mixed":
            import copy

            transformer_bf16 = copy.deepcopy(self.transformer).to(self._execution_device)

        if quant_mode in ("measure", "quantize", "quantize-mixed"):
            import os

            quant_config_path = os.getenv("QUANT_CONFIG")
            if not quant_config_path:
                raise ImportError(
                    "QUANT_CONFIG path is not defined. Please define path to quantization configuration JSON file."
                )
            elif not os.path.isfile(quant_config_path):
                raise ImportError(f"QUANT_CONFIG path '{quant_config_path}' is not valid")

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
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            num_prompts = 1
        elif prompt is not None and isinstance(prompt, list):
            num_prompts = len(prompt)
        else:
            num_prompts = prompt_embeds.shape[0]
        num_batches = math.ceil((num_images_per_prompt * num_prompts) / batch_size)

        device = self._execution_device

        # 3. Run text encoder
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            num_prompts * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

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

        ht.hpu.synchronize()
        t0 = time.time()
        t1 = t0

        hb_profiler = HabanaProfile(
            warmup=profiling_warmup_steps,
            active=profiling_steps,
            record_shapes=False,
            name="diffuser_pipeline",
        )
        hb_profiler.start()

        # 5.1. Split Input data to batches (HPU-specific step)
        (
            latents_batches,
            text_embeddings_batches,
            pooled_prompt_embeddings_batches,
            guidance_batches,
            num_dummy_samples,
        ) = self._split_inputs_into_batches(batch_size, latents, prompt_embeds, pooled_prompt_embeds, guidance)

        outputs = {
            "images": [],
        }

        # 6. Denoising loop
        for j in range(num_batches):
            # The throughput is calculated from the 4th iteration
            # because compilation occurs in the first 2-3 iterations
            if j == throughput_warmup_steps:
                ht.hpu.synchronize()
                t1 = time.time()

            latents_batch = latents_batches[0]
            latents_batches = torch.roll(latents_batches, shifts=-1, dims=0)
            text_embeddings_batch = text_embeddings_batches[0]
            text_embeddings_batches = torch.roll(text_embeddings_batches, shifts=-1, dims=0)
            pooled_prompt_embeddings_batch = pooled_prompt_embeddings_batches[0]
            pooled_prompt_embeddings_batches = torch.roll(pooled_prompt_embeddings_batches, shifts=-1, dims=0)
            guidance_batch = None if guidance_batches is None else guidance_batches[0]
            guidance_batches = None if guidance_batches is None else torch.roll(guidance_batches, shifts=-1, dims=0)

            if hasattr(self.scheduler, "_init_step_index"):
                # Reset scheduler step index for next batch
                self.scheduler.timesteps = timesteps
                self.scheduler._init_step_index(timesteps[0])

            # Mixed quantization
            quant_mixed_step = len(timesteps)
            if quant_mode == "quantize-mixed":
                # 10% of steps use higher precision in mixed quant mode
                quant_mixed_step = quant_mixed_step - (quant_mixed_step // 10)
                logger.info(f"Use FP8  Transformer at steps 0 to {quant_mixed_step - 1}")
                logger.info(f"Use BF16 Transformer at steps {quant_mixed_step} to {len(timesteps) - 1}")

            for i in self.progress_bar(range(len(timesteps))):
                if use_warmup_inference_steps and i == throughput_warmup_steps and j == num_batches - 1:
                    ht.hpu.synchronize()
                    t1 = time.time()

                if self.interrupt:
                    continue

                timestep = timesteps[0]
                timesteps = torch.roll(timesteps, shifts=-1, dims=0)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = timestep.expand(latents_batch.shape[0]).to(latents_batch.dtype)

                if i >= quant_mixed_step:
                    # Mixed quantization
                    noise_pred = transformer_bf16(
                        hidden_states=latents_batch,
                        timestep=timestep / 1000,
                        guidance=guidance_batch,
                        pooled_projections=pooled_prompt_embeddings_batch,
                        encoder_hidden_states=text_embeddings_batch,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    noise_pred = self.transformer(
                        hidden_states=latents_batch,
                        timestep=timestep / 1000,
                        guidance=guidance_batch,
                        pooled_projections=pooled_prompt_embeddings_batch,
                        encoder_hidden_states=text_embeddings_batch,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_batch = self.scheduler.step(noise_pred, timestep, latents_batch, return_dict=False)[0]

                hb_profiler.step()
                # htcore.mark_step(sync=True)
                if num_batches > throughput_warmup_steps:
                    ht.hpu.synchronize()

            if not output_type == "latent":
                latents_batch = self._unpack_latents(latents_batch, height, width, self.vae_scale_factor)
                latents_batch = (latents_batch / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents_batch, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)
            else:
                image = latents_batch

            outputs["images"].append(image)
            # htcore.mark_step(sync=True)

        # 7. Stage after denoising
        hb_profiler.stop()

        if quant_mode == "measure":
            from neural_compressor.torch.quantization import finalize_calibration

            finalize_calibration(self.transformer)

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

        # 8 Output Images
        if num_dummy_samples > 0:
            # Remove dummy generations if needed
            outputs["images"][-1] = outputs["images"][-1][:-num_dummy_samples]

        # Process generated images
        for i, image in enumerate(outputs["images"][:]):
            if i == 0:
                outputs["images"].clear()

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

        return GaudiFluxPipelineOutput(
            images=outputs["images"],
            throughput=speed_measures[f"{speed_metrics_prefix}_samples_per_second"],
        )
