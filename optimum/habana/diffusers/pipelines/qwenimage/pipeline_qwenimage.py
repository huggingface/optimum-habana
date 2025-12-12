# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline, calculate_shift, retrieve_timesteps
from diffusers.utils import BaseOutput, is_torch_xla_available, logging, replace_example_docstring
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from ....transformers.gaudi_configuration import GaudiConfig
from ....utils import HabanaProfile, speed_metrics, warmup_inference_steps_time_adjustment
from ...schedulers import GaudiFlowMatchEulerDiscreteScheduler
from ..pipeline_utils import GaudiDiffusionPipeline


if is_torch_xla_available():
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from optimum.habana.diffusers import GaudiQwenImagePipeline

        >>> pipe = GaudiQwenImagePipeline.from_pretrained(
        ...     "Qwen/Qwen-Image",
        ...     torch_dtype=torch.bfloat16,
        ...     use_habana=True,
        ...     use_hpu_graphs=True,
        ...     gaudi_config="Habana/stable-diffusion",
        ... )
        >>> prompt = "A cat holding a sign that says hello world"
        >>> negative_prompt = " "
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, negative_prompt, num_inference_steps=10).images[0]
        >>> image.save("qwenimage.png")
        ```
"""


@dataclass
class GaudiQwenImagePipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    throughput: float


class GaudiQwenImagePipeline(GaudiDiffusionPipeline, QwenImagePipeline):
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py#L132

    The QwenImage pipeline for text-to-image generation.

    Args:
        transformer ([`QwenImageTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Qwen2.5-VL-7B-Instruct`]):
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.
        tokenizer (`QwenTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: GaudiFlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        transformer: QwenImageTransformer2DModel,
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

        QwenImagePipeline.__init__(
            self,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )

        self.to(self._device)
        if use_hpu_graphs:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph

            transformer = wrap_in_hpu_graph(transformer)

    @classmethod
    def _split_inputs_into_batches(
        cls,
        batch_size,
        latents,
        prompt_embeds,
        prompt_embeds_mask,  # None
        negative_prompt_embeds,
        negative_prompt_embeds_mask,  # None
        guidance,
    ):
        # Use torch.split to generate num_batches batches of size batch_size
        latents_batches = list(torch.split(latents, batch_size))
        prompt_embeds_batches = list(torch.split(prompt_embeds, batch_size))
        if prompt_embeds_mask is not None:
            prompt_embeds_mask_batches = list(torch.split(prompt_embeds_mask, batch_size))
        if negative_prompt_embeds is not None:
            negative_prompt_embeds_batches = list(torch.split(negative_prompt_embeds, batch_size))
        else:
            negative_prompt_embeds_batches = None
        if negative_prompt_embeds_mask is not None:
            negative_prompt_embeds_mask_batches = list(torch.split(negative_prompt_embeds_mask, batch_size))
        else:
            negative_prompt_embeds_mask_batches = None
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

            # Pad prompt_embeds_mask if necessary
            if prompt_embeds_mask is not None:
                sequence_to_stack = (prompt_embeds_mask_batches[-1],) + tuple(
                    torch.zeros_like(prompt_embeds_mask_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                prompt_embeds_mask_batches[-1] = torch.vstack(sequence_to_stack)

            # Pad negative_prompt_embeds_batches
            sequence_to_stack = (negative_prompt_embeds_batches[-1],) + tuple(
                torch.zeros_like(negative_prompt_embeds_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
            )
            negative_prompt_embeds_batches[-1] = torch.vstack(sequence_to_stack)

            # Pad negative_prompt_embeds_mask if necessary
            if negative_prompt_embeds_mask is not None:
                sequence_to_stack = (negative_prompt_embeds_mask_batches[-1],) + tuple(
                    torch.zeros_like(negative_prompt_embeds_mask_batches[-1][0][None, :])
                    for _ in range(num_dummy_samples)
                )
                negative_prompt_embeds_mask_batches[-1] = torch.vstack(sequence_to_stack)

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
        prompt_embeds_mask_batches = torch.stack(prompt_embeds_mask_batches)
        if negative_prompt_embeds_batches is not None:
            negative_prompt_embeds_batches = torch.stack(negative_prompt_embeds_batches)
        if negative_prompt_embeds_mask_batches is not None:
            negative_prompt_embeds_mask_batches = torch.stack(negative_prompt_embeds_mask_batches)

        guidance_batches = torch.stack(guidance_batches) if guidance is not None else None

        return (
            latents_batches,
            prompt_embeds_batches,
            prompt_embeds_mask_batches,  # was NOne
            negative_prompt_embeds_batches,
            negative_prompt_embeds_mask_batches,  # was NOne
            guidance_batches,
            num_dummy_samples,
        )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.

                This parameter in the pipeline is there to support future guidance-distilled models when they come up.
                Note that passing `guidance_scale` to the pipeline is ineffective. To enable classifier-free guidance,
                please pass `true_cfg_scale` and `negative_prompt` (even an empty negative prompt like " ") should
                enable classifier-free guidance computations.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
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
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] or `tuple`:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
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
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
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

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        # 3. Run text encoder
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
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
        img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]] * batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
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
            prompt_embeds_batches,
            prompt_embeds_mask_batches,  #
            negative_prompt_embeds_batches,
            negative_prompt_embeds_mask_batches,  #
            guidance_batches,
            num_dummy_samples,
        ) = self._split_inputs_into_batches(
            batch_size,
            latents,
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,  #
            guidance,
        )

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
            prompt_embeds_batch = prompt_embeds_batches[0]
            prompt_embeds_batches = torch.roll(prompt_embeds_batches, shifts=-1, dims=0)
            prompt_embeds_mask_batch = prompt_embeds_mask_batches[0]
            prompt_embeds_mask_batches = torch.roll(prompt_embeds_mask_batches, shifts=-1, dims=0)

            if do_true_cfg:
                negative_prompt_embeds_batch = negative_prompt_embeds_batches[0]
                negative_prompt_embeds_batches = torch.roll(negative_prompt_embeds_batches, shifts=-1, dims=0)
                negative_prompt_embeds_mask_batch = negative_prompt_embeds_mask_batches[0]
                negative_prompt_embeds_mask_batches = torch.roll(
                    negative_prompt_embeds_mask_batches, shifts=-1, dims=0
                )

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

                # case cache_contxt("cond")
                if quant_mode == "quantize-mixed" and i >= quant_mixed_step:
                    # Mixed quantization
                    noise_pred = transformer_bf16(
                        hidden_states=latents_batch,  # latents,
                        timestep=timestep / 1000,
                        guidance=guidance_batch,
                        encoder_hidden_states_mask=prompt_embeds_mask_batch,  # WAS None prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds_batch,  # prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    noise_pred = self.transformer(
                        hidden_states=latents_batch,  # latents,
                        timestep=timestep / 1000,
                        guidance=guidance_batch,
                        encoder_hidden_states_mask=prompt_embeds_mask_batch,  # WAS None
                        encoder_hidden_states=prompt_embeds_batch,  # prompt_embeds_mask,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]

                    if do_true_cfg:
                        neg_noise_pred = self.transformer(
                            hidden_states=latents_batch,  # latents,
                            timestep=timestep / 1000,
                            guidance=guidance_batch,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask_batch,  # WAS None
                            encoder_hidden_states=negative_prompt_embeds_batch,  # prompt_embeds_mask,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                        comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                        noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents_batch = self.scheduler.step(noise_pred, timestep, latents_batch, return_dict=False)[0]

                hb_profiler.step()

                if num_batches > throughput_warmup_steps:
                    ht.hpu.synchronize()

            if output_type == "latent":
                image = latents_batch
            else:
                latents_batch = self._unpack_latents(latents_batch, height, width, self.vae_scale_factor)
                latents_batch = latents_batch.to(self.vae.dtype)
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents_batch.device, latents_batch.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(latents_batch.device, latents_batch.dtype)
                latents_batch = latents_batch / latents_std + latents_mean
                image = self.vae.decode(latents_batch, return_dict=False)[0][:, :, 0]
                image = self.image_processor.postprocess(image, output_type=output_type)

            outputs["images"].append(image)

        # 7. Stage after denoising
        hb_profiler.stop()

        if quant_mode == "measure":
            from neural_compressor.torch.quantization import finalize_calibration

            finalize_calibration(self.transformer)

        ht.hpu.synchronize()
        end_time = time.time()

        speed_metrics_prefix = "generation"
        if use_warmup_inference_steps:
            t1 = warmup_inference_steps_time_adjustment(t1, t1, num_inference_steps, throughput_warmup_steps)
        speed_measures = speed_metrics(
            split=speed_metrics_prefix,
            start_time=t0,
            end_time=end_time,
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
            return (image,)

        return GaudiQwenImagePipelineOutput(
            images=outputs["images"],
            throughput=speed_measures[f"{speed_metrics_prefix}_samples_per_second"],
        )
