# Copyright 2023 DDPO-pytorch authors (Kevin Black), metric-space, The HuggingFace Team. All rights reserved.
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

import os
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Optional, Tuple
from warnings import warn

import torch
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from tqdm.auto import tqdm
from trl import DDPOTrainer
from trl.models import DDPOStableDiffusionPipeline
from trl.trainer import DDPOConfig
from trl.trainer.utils import PerPromptStatTracker

from optimum.habana import GaudiConfig
from optimum.habana.accelerate import GaudiAccelerator
from optimum.habana.utils import set_seed


logger = get_logger(__name__)


class GaudiDDPOTrainer(DDPOTrainer):
    def __init__(
        self,
        config: DDPOConfig,
        reward_function: Callable[[torch.Tensor, Tuple[str], Tuple[Any]], torch.Tensor],
        prompt_function: Callable[[], Tuple[str, Any]],
        sd_pipeline: DDPOStableDiffusionPipeline,
        image_samples_hook: Optional[Callable[[Any, Any, Any], Any]] = None,
        gaudi_config: GaudiConfig = None,
        use_habana: bool = True,
        use_hpu_graphs: bool = False,
    ):
        """
        Adapted from DDPOTrainer.__init__: https://github.com/huggingface/trl/blob/v0.7.8/trl/trainer/ddpo_trainer.py#L72
        The changes are:
        - add new args gaudi_config
        - support HPU graphs for trainable layers
        - use GaudiAccelerator instead of Accelerator
        - support FusedClipNorm
        """
        if image_samples_hook is None:
            warn("No image_samples_hook provided; no images will be logged")

        self.prompt_fn = prompt_function
        self.reward_fn = reward_function
        self.config = config
        self.image_samples_callback = image_samples_hook
        self.gaudi_config = gaudi_config
        self.use_hpu_graphs = use_hpu_graphs
        self.use_habana = use_habana

        if use_habana:
            try:
                import habana_frameworks.torch.core as htcore
            except ImportError as error:
                error.msg = f"Could not import habana_frameworks.torch.core. {error.msg}."
                raise error
            self.htcore = htcore

        accelerator_project_config = ProjectConfiguration(**self.config.project_kwargs)

        if self.config.resume_from:
            self.config.resume_from = os.path.normpath(os.path.expanduser(self.config.resume_from))
            if "checkpoint_" not in os.path.basename(self.config.resume_from):
                # get the most recent checkpoint in this directory
                checkpoints = list(
                    filter(
                        lambda x: "checkpoint_" in x,
                        os.listdir(self.config.resume_from),
                    )
                )
                if len(checkpoints) == 0:
                    raise ValueError(f"No checkpoints found in {self.config.resume_from}")
                checkpoint_numbers = sorted([int(x.split("_")[-1]) for x in checkpoints])
                self.config.resume_from = os.path.join(
                    self.config.resume_from,
                    f"checkpoint_{checkpoint_numbers[-1]}",
                )

                accelerator_project_config.iteration = checkpoint_numbers[-1] + 1

        # number of timesteps within each trajectory to train on
        self.num_train_timesteps = int(self.config.sample_num_steps * self.config.train_timestep_fraction)
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = GaudiAccelerator(
            log_with=self.config.log_with,
            mixed_precision="bf16" if config.mixed_precision == "bf16" else "no",
            project_config=accelerator_project_config,
            cpu=(not use_habana),
            kwargs_handlers=[kwargs],
            # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
            # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
            # the total number of optimizer steps to accumulate across.
            gradient_accumulation_steps=self.config.train_gradient_accumulation_steps * self.num_train_timesteps,
            **self.config.accelerator_kwargs,
        )

        is_okay, message = self._config_check()
        if not is_okay:
            raise ValueError(message)

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                self.config.tracker_project_name,
                config={"ddpo_trainer_config": config.to_dict()} if not is_using_tensorboard else config.to_dict(),
                init_kwargs=self.config.tracker_kwargs,
            )

        logger.info(f"\n{config}")

        set_seed(self.config.seed)

        self.sd_pipeline = sd_pipeline

        self.sd_pipeline.set_progress_bar_config(
            position=1,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16
        else:
            inference_dtype = torch.float32

        self.sd_pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.text_encoder.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)

        trainable_layers = self.sd_pipeline.get_trainable_layers()

        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)

        self.optimizer = self._setup_optimizer(
            trainable_layers.parameters() if not isinstance(trainable_layers, list) else trainable_layers
        )

        self.neg_prompt_embed = self.sd_pipeline.text_encoder(
            self.sd_pipeline.tokenizer(
                [""] if self.config.negative_prompts is None else self.config.negative_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
        )[0]

        if config.per_prompt_stat_tracking:
            self.stat_tracker = PerPromptStatTracker(
                config.per_prompt_stat_tracking_buffer_size,
                config.per_prompt_stat_tracking_min_count,
            )

        if hasattr(self.sd_pipeline, "use_lora") and self.sd_pipeline.use_lora:
            unet, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
            self.trainable_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
        else:
            self.trainable_layers, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)

        if self.gaudi_config.use_fused_clip_norm:
            try:
                from habana_frameworks.torch.hpex.normalization import FusedClipNorm
            except ImportError as error:
                error.msg = (
                    f"Could not import 'FusedClipNorm' from 'habana_frameworks.torch.hpex.normalization'. {error.msg}."
                )
                raise error
            self.FusedNorm = FusedClipNorm(
                self.trainable_layers.parameters()
                if not isinstance(self.trainable_layers, list)
                else self.trainable_layers,
                self.config.train_max_grad_norm,
            )

        if use_hpu_graphs:
            import habana_frameworks.torch as ht

            ht.hpu.ModuleCacher()(model=self.sd_pipeline.unet, inplace=True)

        if self.config.async_reward_computation:
            self.executor = futures.ThreadPoolExecutor(max_workers=config.max_workers)

        if config.resume_from:
            logger.info(f"Resuming from {config.resume_from}")
            self.accelerator.load_state(config.resume_from)
            self.first_epoch = int(config.resume_from.split("_")[-1]) + 1
        else:
            self.first_epoch = 0

    def step(self, epoch: int, global_step: int):
        """
        Adapted from https://github.com/huggingface/trl/blob/v0.7.8/trl/trainer/ddpo_trainer.py#L234
        - Add progress bar to track training epochs
        - Convert bfloat to float when creating to numpy arrays
        """
        samples, prompt_image_data = self._generate_samples(
            iterations=self.config.sample_num_batches_per_epoch,
            batch_size=self.config.sample_batch_size,
        )

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        rewards, rewards_metadata = self.compute_rewards(
            prompt_image_data, is_async=self.config.async_reward_computation
        )

        for i, image_data in enumerate(prompt_image_data):
            image_data.extend([rewards[i], rewards_metadata[i]])

        if self.image_samples_callback is not None:
            self.image_samples_callback(prompt_image_data, global_step, self.accelerator.trackers[0])

        rewards = torch.cat(rewards)

        if rewards.dtype == torch.bfloat16:
            rewards = rewards.float()  # bf16 not supported by numpy

        rewards = self.accelerator.gather(rewards).cpu().numpy()

        self.accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )

        if self.config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = self.accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = self.sd_pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = self.stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ungather advantages;  keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
            .to(self.accelerator.device)
        )

        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape

        pbar = tqdm(
            range(self.config.train_num_inner_epochs),
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_main_process,
        )
        for inner_epoch in pbar:
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=self.accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            # still trying to understand the code below
            perms = torch.stack(
                [torch.randperm(num_timesteps, device=self.accelerator.device) for _ in range(total_batch_size)]
            )

            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=self.accelerator.device)[:, None],
                    perms,
                ]

            original_keys = samples.keys()
            original_values = samples.values()
            # rebatch them as user defined train_batch_size is different from sample_batch_size
            reshaped_values = [v.reshape(-1, self.config.train_batch_size, *v.shape[1:]) for v in original_values]

            # Transpose the list of original values
            transposed_values = zip(*reshaped_values)
            # Create new dictionaries for each row of transposed values
            samples_batched = [dict(zip(original_keys, row_values)) for row_values in transposed_values]

            self.sd_pipeline.unet.train()
            global_step = self._train_batched_samples(inner_epoch, epoch, global_step, samples_batched)
            # ensure optimization step at the end of the inner epoch
            if not self.accelerator.sync_gradients:
                raise ValueError(
                    "Optimization step should have been performed by this point. Please check calculated gradient accumulation settings."
                )

        if epoch != 0 and epoch % self.config.save_freq == 0 and self.accelerator.is_main_process:
            self.accelerator.save_state()

        return global_step

    def calculate_loss(self, latents, timesteps, next_latents, log_probs, advantages, embeds):
        """
        Adapted from https://github.com/huggingface/trl/blob/v0.7.8/trl/trainer/ddpo_trainer.py#L340
        - Use accelerator autocast (original TRL implemenation uses nullcontext for LoRA training)
        - Convert logprob to float for loss calculation
        """
        with self.accelerator.autocast():
            if self.config.train_cfg:
                noise_pred = self.sd_pipeline.unet(
                    torch.cat([latents] * 2),
                    torch.cat([timesteps] * 2),
                    embeds,
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.sample_guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                noise_pred = self.sd_pipeline.unet(
                    latents,
                    timesteps,
                    embeds,
                ).sample
            # compute the log prob of next_latents given latents under the current model

            scheduler_step_output = self.sd_pipeline.scheduler_step(
                noise_pred,
                timesteps,
                latents,
                eta=self.config.sample_eta,
                prev_sample=next_latents,
            )

            log_prob = scheduler_step_output.log_probs.float()

        log_probs = log_probs.float()
        advantages = torch.clamp(
            advantages.float(),
            -self.config.train_adv_clip_max,
            self.config.train_adv_clip_max,
        )

        ratio = torch.exp(log_prob - log_probs)

        loss = self.loss(advantages, self.config.train_clip_range, ratio)

        approx_kl = 0.5 * torch.mean((log_prob - log_probs) ** 2)

        clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.config.train_clip_range).float())

        return loss, approx_kl, clipfrac

    def _setup_optimizer(self, trainable_layers_parameters):
        # Adapted from https://github.com/huggingface/trl/blob/v0.7.8/trl/trainer/ddpo_trainer.py#L422
        # Adds support for FusedAdamW
        if self.use_habana and self.gaudi_config.use_fused_adam:
            from habana_frameworks.torch.hpex.optimizers import FusedAdamW

            optimizer_cls = FusedAdamW
        else:
            optimizer_cls = torch.optim.AdamW

        return optimizer_cls(
            trainable_layers_parameters,
            lr=self.config.train_learning_rate,
            betas=(self.config.train_adam_beta1, self.config.train_adam_beta2),
            weight_decay=self.config.train_adam_weight_decay,
            eps=self.config.train_adam_epsilon,
        )

    def _generate_samples(self, iterations, batch_size):
        """
        Adapted from https://github.com/huggingface/trl/blob/v0.7.8/trl/trainer/ddpo_trainer.py#L446
        - Load timesteps to HPU
        """
        samples = []
        prompt_image_pairs = []
        self.sd_pipeline.unet.eval()

        sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)

        for i in range(iterations):
            prompts, prompt_metadata = zip(*[self.prompt_fn() for _ in range(batch_size)])

            prompt_ids = self.sd_pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
            prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]

            with self.accelerator.autocast():
                sd_output = self.sd_pipeline(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    output_type="pt",
                )

                images = sd_output.images
                latents = sd_output.latents
                log_probs = sd_output.log_probs

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, ...)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = self.sd_pipeline.scheduler.timesteps.repeat(batch_size, 1)
            timesteps = timesteps.to(latents.device)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # each entry is the latent before timestep t
                    "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "negative_prompt_embeds": sample_neg_prompt_embeds,
                }
            )
            prompt_image_pairs.append([images, prompts, prompt_metadata])

        return samples, prompt_image_pairs

    def _train_batched_samples(self, inner_epoch, epoch, global_step, batched_samples):
        """
        Adapted from https://github.com/huggingface/trl/blob/v0.7.8/trl/trainer/ddpo_trainer.py#L508
        - Reduce recompilations by avoiding constant variables in loops
        - Add `mark_step()` to support lazy mode
        """
        info = defaultdict(list)

        for _i, sample in enumerate(batched_samples):
            if self.config.train_cfg:
                # concat negative prompts to sample prompts to avoid two forward passes
                embeds = torch.cat([sample["negative_prompt_embeds"], sample["prompt_embeds"]])
            else:
                embeds = sample["prompt_embeds"]

            latents = sample["latents"]
            timesteps = sample["timesteps"]
            next_latents = sample["next_latents"]
            log_probs = sample["log_probs"]

            for j in range(self.num_train_timesteps):  # , desc=f"Epoch{i}"):
                with self.accelerator.accumulate(self.sd_pipeline.unet):
                    # Reduce recompilations by avoiding constant variables in loops
                    latent = latents[:, 0]
                    timestep = timesteps[:, 0]
                    next_latent = next_latents[:, 0]
                    log_prob = log_probs[:, 0]
                    latents = torch.roll(latents, shifts=-1, dims=1)
                    timesteps = torch.roll(timesteps, shifts=-1, dims=1)
                    next_latents = torch.roll(next_latents, shifts=-1, dims=1)
                    log_probs = torch.roll(log_probs, shifts=-1, dims=1)

                    loss, approx_kl, clipfrac = self.calculate_loss(
                        latent,
                        timestep,
                        next_latent,
                        log_prob,
                        sample["advantages"],
                        embeds,
                    )

                    info["approx_kl"].append(approx_kl)
                    info["clipfrac"].append(clipfrac)
                    info["loss"].append(loss)
                    self.accelerator.backward(loss)
                    if self.use_habana:
                        self.htcore.mark_step()

                    if self.accelerator.sync_gradients:
                        trainable_layers = (
                            self.trainable_layers.parameters()
                            if not isinstance(self.trainable_layers, list)
                            else self.trainable_layers
                        )
                        if self.gaudi_config.use_fused_clip_norm:
                            self.FusedNorm.clip_norm(trainable_layers)
                        else:
                            self.self.accelerator.clip_grad_norm_(
                                trainable_layers,
                                self.config.train_max_grad_norm,
                            )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.use_habana:
                        self.htcore.mark_step()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    # log training-related stuff
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = self.accelerator.reduce(info, reduction="mean")
                    info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                    self.accelerator.log(info, step=global_step)
                    global_step += 1
                    info = defaultdict(list)

        return global_step
