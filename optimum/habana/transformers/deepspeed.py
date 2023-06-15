# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
"""
Integration with Deepspeed
"""
import os
from copy import deepcopy
from dataclasses import make_dataclass

import torch
from transformers.deepspeed import (
    HfDeepSpeedConfig,
    HfTrainerDeepSpeedConfig,
    deepspeed_optim_sched,
    set_hf_deepspeed_config,
)
from transformers.dependency_versions_check import dep_version_check

from optimum.utils import logging


logger = logging.get_logger(__name__)


class GaudiTrainerDeepSpeedConfig(HfTrainerDeepSpeedConfig):
    """
    The `HfTrainerDeepSpeedConfig` object is meant to be created during `TrainingArguments` object creation and has the
    same lifespan as the latter.
    """

    def __init__(self, config_file_or_dict):
        set_hf_deepspeed_config(self)
        dep_version_check("accelerate")
        # dep_version_check("deepspeed")
        super(HfDeepSpeedConfig, self).__init__(config_file_or_dict)
        self._dtype = None
        self.mismatches = []

    def trainer_config_process(self, args):
        """
        Adjust the config with `TrainingArguments` values. This stage is run during `TrainingArguments` object
        creation.
        """
        # DeepSpeed does:
        # train_batch_size = world_size * train_micro_batch_size_per_gpu * gradient_accumulation_steps
        train_batch_size = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self.fill_match(
            "train_micro_batch_size_per_gpu", args.per_device_train_batch_size, "per_device_train_batch_size"
        )
        self.fill_match("gradient_accumulation_steps", args.gradient_accumulation_steps, "gradient_accumulation_steps")
        self.fill_match("train_batch_size", train_batch_size, "train_batch_size (calculated)")
        self.fill_match("gradient_clipping", args.max_grad_norm, "max_grad_norm")

        self.fill_match("optimizer.params.lr", args.learning_rate, "learning_rate")
        self.fill_match("optimizer.params.betas", [args.adam_beta1, args.adam_beta2], "adam_beta1+adam_beta2")
        self.fill_match("optimizer.params.eps", args.adam_epsilon, "adam_epsilon")
        self.fill_match("optimizer.params.weight_decay", args.weight_decay, "weight_decay")

        self.fill_only("scheduler.params.warmup_min_lr", 0)  # not a trainer arg
        self.fill_match("scheduler.params.warmup_max_lr", args.learning_rate, "learning_rate")
        # total_num_steps - will get set in trainer_config_finalize

        if args.save_on_each_node:
            # deepspeed uses shared storage by default. Let's override this setting if save_on_each_node == True
            self.config["checkpoint"] = self.config.get("checkpoint", {})
            self.config["checkpoint"]["use_node_local_storage"] = args.save_on_each_node

        # deepspeed's default mode is fp16 unless there is a config that says differently
        if self.is_true("bf16.enabled"):
            self._dtype = torch.bfloat16
        else:
            self._dtype = torch.float32


def deepspeed_init(trainer, num_training_steps, resume_from_checkpoint=None, inference=False):
    """
    Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.
    If `resume_from_checkpoint` was passed then an attempt to resume from a previously saved checkpoint will be made.
    Args:
        trainer: Trainer object
        num_training_steps: per single HPU
        resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load
        inference: launch in inference mode (no optimizer and no lr scheduler)
    Returns: model, optimizer, lr_scheduler
    We may use `deepspeed_init` more than once during the life of Trainer, when we do - it's a temp hack based on:
    https://github.com/microsoft/DeepSpeed/issues/1394#issuecomment-937405374 until Deepspeed fixes a bug where it
    can't resume from a checkpoint after it did some stepping https://github.com/microsoft/DeepSpeed/issues/1612
    """
    import deepspeed
    from deepspeed.utils import logger as ds_logger

    model = trainer.model
    args = trainer.args

    if hasattr(trainer, "hf_deepspeed_config_orig"):
        hf_deepspeed_config = deepcopy(trainer.hf_deepspeed_config_orig)
    else:
        hf_deepspeed_config = args.hf_deepspeed_config
        trainer.hf_deepspeed_config_orig = deepcopy(args.hf_deepspeed_config)

    # resume config update - some bits like `model` and `num_training_steps` only become available during train
    hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)
    config = hf_deepspeed_config.config

    # set the Deepspeed log level consistent with the Trainer
    ds_logger.setLevel(args.get_process_log_level())

    # TODO: temporary workaround
    # To remove when it is solved, see https://github.com/HabanaAI/Model-References/blob/17fbab7ceebca15b1560ffb2c4e15a3888bb5f33/PyTorch/nlp/pretraining/deepspeed-bert/run_pretraining.py#L527
    model.to(dtype=hf_deepspeed_config._dtype, device="hpu")

    if inference:
        # only Z3 makes sense for the inference
        if not hf_deepspeed_config.is_zero3():
            raise ValueError("ZeRO inference only makes sense with ZeRO Stage 3 - please adjust your config")

        # in case the training config is re-used for inference
        hf_deepspeed_config.del_config_sub_tree("optimizer")
        hf_deepspeed_config.del_config_sub_tree("lr_scheduler")
        optimizer, lr_scheduler = None, None
        model_parameters = None
    else:
        trainer.optimizer = None  # important for when deepspeed_init is used as re-init
        optimizer, lr_scheduler = deepspeed_optim_sched(trainer, hf_deepspeed_config, args, num_training_steps)
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    # keep for quick debug:
    # from pprint import pprint; pprint(config)

    HabanaArgs = make_dataclass("HabanaArgs", [("use_hpu", bool), ("no_cuda", bool)])
    habana_args = HabanaArgs(use_hpu=args.use_habana, no_cuda=args.no_cuda)
    if args.use_habana:
        # This env variable is initialized here to make sure it is set to "true"
        # It should be done by the launcher but it does not work for multi-node runs
        os.environ["DEEPSPEED_USE_HPU"] = "true"

    kwargs = {
        "args": habana_args,
        "model": model,
        "model_parameters": model_parameters,
        "config_params": config,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }

    deepspeed_engine, optimizer, _, lr_scheduler = deepspeed.initialize(**kwargs)

    if resume_from_checkpoint is not None:
        # it's possible that the user is trying to resume from model_path, which doesn't necessarily
        # contain a deepspeed checkpoint. e.g. examples just check if the dir exists and assume it's
        # a resume from a checkpoint and not just a local pretrained weight. So we check here if the
        # path contains what looks like a deepspeed checkpoint
        import glob

        deepspeed_checkpoint_dirs = sorted(glob.glob(f"{resume_from_checkpoint}/global_step*"))

        if len(deepspeed_checkpoint_dirs) > 0:
            logger.info(f"Attempting to resume from {resume_from_checkpoint}")
            # this magically updates self.optimizer and self.lr_scheduler
            load_path, _ = deepspeed_engine.load_checkpoint(
                resume_from_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True
            )
            if load_path is None:
                raise ValueError(f"[deepspeed] failed to resume from checkpoint {resume_from_checkpoint}")
        else:
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

    return deepspeed_engine, optimizer, lr_scheduler
