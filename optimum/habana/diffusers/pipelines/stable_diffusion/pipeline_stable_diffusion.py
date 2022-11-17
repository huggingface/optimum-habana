import copy
import inspect
import time
from math import ceil
from typing import Callable, List, Optional, Union

import torch

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.schedulers import (
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate
from optimum.utils import logging
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from ....transformers.gaudi_configuration import GaudiConfig
from ...pipeline_utils import GaudiDiffusionPipeline


logger = logging.get_logger(__name__)


class GaudiStableDiffusionPipeline(GaudiDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
        ],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        use_habana: bool = False,
        use_lazy_mode: bool = False,
        use_hpu_graphs: bool = False,
        gaudi_config: Union[str, GaudiConfig] = None,
    ):
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None:
            logger.warn(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        super().__init__(
            use_habana,
            use_lazy_mode,
            use_hpu_graphs,
            gaudi_config,
        )

    # def enable_xformers_memory_efficient_attention(self):
    #     r"""
    #     Enable memory efficient attention as implemented in xformers.
    #     When this option is enabled, you should observe lower GPU memory usage and a potential speed up at inference
    #     time. Speed up at training time is not guaranteed.
    #     Warning: When Memory Efficient Attention and Sliced attention are both enabled, the Memory Efficient Attention
    #     is used.
    #     """
    #     self.unet.set_use_memory_efficient_attention_xformers(True)

    # def disable_xformers_memory_efficient_attention(self):
    #     r"""
    #     Disable memory efficient attention as implemented in xformers.
    #     """
    #     self.unet.set_use_memory_efficient_attention_xformers(False)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        batch_size: int = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
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
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            batch_size (`int`, *optional*, defaults to 1):
                The number of images in a batch.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        if isinstance(prompt, str):
            num_prompts = 1
        elif isinstance(prompt, list):
            num_prompts = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        num_batches = ceil((num_images_per_prompt * num_prompts) / batch_size)
        logger.info(
            f"{num_prompts} prompt(s) received, {num_images_per_prompt} generation(s) per prompt,"
            f" {batch_size} sample(s) per batch, {num_batches} total batch(es)."
        )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        batch_factor = 1
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * num_prompts
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif num_prompts != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has {len(negative_prompt)} elements, but `prompt`:"
                    f" {prompt} has {num_prompts}. Please make sure that passed `negative_prompt` matches"
                    " the size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(num_prompts * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            batch_factor = 2

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (num_prompts * num_images_per_prompt, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        if latents is None:
            if self.device.type == "hpu":
                # torch.randn is broken on HPU so running it on CPU
                latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(
                    self.device
                )
            else:
                latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        # Split into batches
        text_embeddings_batches = torch.stack(list(torch.split(text_embeddings, batch_factor * batch_size)))
        latents_batches = torch.stack(list(torch.split(latents, batch_size)))

        outputs = {
            "images": [],
            "has_nsfw_concept": [],
        }

        t0 = time.time()

        # Main loop
        for j in self.progress_bar(range(num_batches)):
            # if j == 0:
            #     # Keep a copy of the scheduler in its initial state to reset it later
            #     scheduler_copy = copy.deepcopy(self.scheduler)
            # else:
            #     # Reset the scheduler
            #     self.scheduler = copy.deepcopy(scheduler_copy)

            if j == 2:
                t1 = time.time()

            latents_batch = latents_batches[0]
            latents_batches = torch.roll(latents_batches, shifts=-1, dims=0)
            text_embeddings_batch = text_embeddings_batches[0]
            text_embeddings_batches = torch.roll(text_embeddings_batches, shifts=-1, dims=0)

            for i in range(num_inference_steps):
                timestep = timesteps_tensor[0]
                timesteps_tensor = torch.roll(timesteps_tensor, shifts=-1, dims=0)

                capture = True if self.use_hpu_graphs and i < 2 else False

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents_batch] * 2) if do_classifier_free_guidance else latents_batch
                # latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

                # predict the noise residual
                noise_pred = self.unet_hpu(latent_model_input, timestep, text_embeddings_batch, capture)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_batch = self.scheduler.step(noise_pred, latents_batch, **extra_step_kwargs).prev_sample

                if self.use_lazy_mode:
                    self.htcore.mark_step()

                # call the callback, if provided
                if callback is not None and i % callback_steps == 0:
                    callback(i, timestep, latents_batch)

            latents_batch = 1 / 0.18215 * latents_batch
            image = self.vae.decode(latents_batch).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            outputs["images"].append(image)

            self.scheduler.reset_timestep_dependent_params()

            if self.use_lazy_mode:
                self.htcore.mark_step()

        t2 = time.time()
        duration = t2 - t0
        if self.use_lazy_mode:
            throughput = (num_images_per_prompt * num_prompts - 2 * batch_size) / (t2 - t1)
        else:
            throughput = num_images_per_prompt * num_prompts / duration
        logger.info(f"Runtime: {duration} seconds, throughput: {throughput} samples/s.")

        # Process generated images
        for i, image in enumerate(outputs["images"][:]):
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            if self.safety_checker is not None:
                # Run the safety checker and void images if NSFW
                safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
                    self.device
                )
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
                )
            else:
                has_nsfw_concept = None

            if i == 0:
                outputs["images"].clear()

            if output_type == "pil":
                image = self.numpy_to_pil(image)

            outputs["images"] += image
            outputs["has_nsfw_concept"] += has_nsfw_concept

        if not return_dict:
            return (outputs["images"], outputs["has_nsfw_concept"])

        return StableDiffusionPipelineOutput(
            images=outputs["images"],
            nsfw_content_detected=outputs["has_nsfw_concept"],
        )

    @torch.no_grad
    def unet_hpu(self, latent_model_input, timestep, encoder_hidden_states, capture):
        if self.use_hpu_graphs:
            return self.capture_replay(latent_model_input, timestep, encoder_hidden_states, capture)
        else:
            return self.unet(latent_model_input, timestep, encoder_hidden_states).sample

    @torch.no_grad
    def capture_replay(self, latent_model_input, timestep, encoder_hidden_states, capture):
        if capture:
            self.static_inputs = [latent_model_input, timestep, encoder_hidden_states]
            with self.ht.hpu.stream(self.hpu_stream):
                self.hpu_graph.capture_begin()

                self.static_outputs = self.unet(
                    self.static_inputs[0], self.static_inputs[1], self.static_inputs[2], return_dict=False
                )[0]

                self.hpu_graph.capture_end()
        self.static_inputs[0].copy_(latent_model_input)
        self.static_inputs[1].copy_(timestep)
        self.static_inputs[2].copy_(encoder_hidden_states)
        self.ht.core.mark_step()
        self.ht.core.hpu.default_stream().synchronize()
        self.hpu_graph.replay()

        return self.static_outputs
