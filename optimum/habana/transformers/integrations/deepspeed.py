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

import torch
from transformers.dependency_versions_check import dep_version_check
from transformers.integrations.deepspeed import (
    HfDeepSpeedConfig,
    HfTrainerDeepSpeedConfig,
    deepspeed_optim_sched,
    set_hf_deepspeed_config,
)

from optimum.utils import logging


logger = logging.get_logger(__name__)


class GaudiTrainerDeepSpeedConfig(HfTrainerDeepSpeedConfig):
    """
    Adapted from: https://github.com/huggingface/transformers/blob/6da93f5580e109fad5f7b523cf2b6e8a5bafb623/src/transformers/integrations/deepspeed.py#L69

    The differences are:
    - disable DeepSpeed version check as we run a custom version on HPU
    - remove uncompatible args (e.g. fp16) in config processing
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


def deepspeed_init(trainer, num_training_steps, inference=False):
    """
    Adapted from: https://github.com/huggingface/transformers/blob/6da93f5580e109fad5f7b523cf2b6e8a5bafb623/src/transformers/integrations/deepspeed.py#L316

    The difference is:
    - add a workaround to cast the model to the target dtype
    """
    from deepspeed.utils import logger as ds_logger

    model = trainer.model
    args = trainer.args

    hf_deepspeed_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config

    # TODO: temporary workaround
    # To remove when it is solved, see https://github.com/HabanaAI/Model-References/blob/5df5baa261a677b63a2b20eab59981d05c191df5/PyTorch/nlp/DeepSpeedExamples/deepspeed-bert/run_pretraining.py#L608
    model.to(dtype=hf_deepspeed_config.dtype(), device="hpu")

    # resume config update - some bits like `model` and `num_training_steps` only become available during train
    hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)

    # set the Deepspeed log level consistent with the Trainer
    ds_logger.setLevel(args.get_process_log_level())

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
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer, lr_scheduler = deepspeed_optim_sched(
            trainer, hf_deepspeed_config, args, num_training_steps, model_parameters
        )

    # keep for quick debug:
    # from pprint import pprint; pprint(config)

    return optimizer, lr_scheduler


def unwrap_deepspeed_model(model):
    if hasattr(model, "module"):
        return model.module
    return model
