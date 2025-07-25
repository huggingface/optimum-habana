# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
The Gaudi Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import copy
import functools
import inspect
import json
import math
import os
import random
import shutil
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
from accelerate import DistributedType, skip_first_batches
from accelerate.data_loader import SeedableRandomSampler
from accelerate.utils import (
    DistributedDataParallelKwargs,
    TorchTensorParallelPlugin,
    load_fsdp_model,
    load_fsdp_optimizer,
    save_fsdp_model,
    save_fsdp_optimizer,
)
from huggingface_hub import upload_folder
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.deepspeed import (
    deepspeed_load_checkpoint,
    is_deepspeed_available,
    is_deepspeed_zero3_enabled,
)
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import _get_fsdp_ckpt_kwargs, _is_peft_model, safe_globals
from transformers.trainer_callback import ExportableState, TrainerCallback, TrainerState
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    find_batch_size,
    get_model_param_count,
    nested_concat,
    nested_detach,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalLoopOutput,
    EvalPrediction,
    HubStrategy,
    PredictionOutput,
    SaveStrategy,
    TrainOutput,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    is_accelerate_available,
    is_datasets_available,
    is_peft_available,
    is_safetensors_available,
)
from transformers.utils.deprecation import deprecate_kwarg

from optimum.utils import logging

from ..accelerate import GaudiAccelerator
from ..accelerate.utils import FP8ContextWrapper
from ..utils import (
    HabanaGenerationTime,
    HabanaProfile,
    get_hpu_memory_stats,
    set_seed,
    speed_metrics,
    to_device_dtype,
    warn0,
)
from .gaudi_configuration import GAUDI_CONFIG_NAME, GaudiConfig
from .integrations.deepspeed import deepspeed_init
from .trainer_utils import convert_into_dtypes, get_dtype
from .training_args import GaudiTrainingArguments


if is_datasets_available():
    import datasets

if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel
    from peft.utils import PeftType

if is_deepspeed_available():
    from accelerate.utils import DeepSpeedSchedulerWrapper

from accelerate.utils import DataLoaderConfiguration, is_torch_version


def _get_input_update_settings(model, lazy_mode: Optional[bool] = None) -> Tuple[bool, Dict]:
    """
    Determines whether the input settings need to be updated.

    Currently (attn_softmax_bf16, use_flash_attention, flash_attention_recompute,
    flash_attention_causal_mask) are enabled only for llama, qwen2, starcoder2, gemma, baichuan
    and chatglm

    lazy_mode for llama, qwen2, starcoder2 and mistral

    Args:
        model: The model instance for which the input update settings are being evaluated
        lazy_mode[Optional[bool]]: Whether to use lazy mode for the model (defaults to `None`)

    Returns:
        Tuple[bool, Dict]: A flag indicating whether the input settings should be updated.
        A dictionary containing the specific input settings that need to be updated, if any
    """
    inputs_update: Dict = {}

    should_update_inputs = (getattr(model, "generation_config", None) is not None) and (
        model.config.model_type in ("llama", "qwen2", "starcoder2", "gemma", "baichuan", "chatglm", "deepseek_v2")
    )
    if should_update_inputs:
        if model.generation_config.attn_softmax_bf16:
            inputs_update["attn_softmax_bf16"] = True
        if model.generation_config.use_flash_attention:
            inputs_update["use_flash_attention"] = True
        if model.generation_config.flash_attention_recompute:
            inputs_update["flash_attention_recompute"] = True
        if model.generation_config.flash_attention_causal_mask:
            inputs_update["flash_attention_causal_mask"] = True

    should_update_inputs = (
        (getattr(model, "generation_config", None) is not None)
        and (model.config.model_type in ("llama", "qwen2", "starcoder2", "mistral"))
        and (lazy_mode is not None)
    )
    if should_update_inputs:
        if _is_peft_model(model):
            forward_method = getattr(model.get_base_model(), "forward")
        else:
            forward_method = getattr(model, "forward")
        signature = inspect.signature(forward_method)
        if "lazy_mode" in signature.parameters:
            inputs_update["lazy_mode"] = lazy_mode

    should_update_inputs: bool = len(inputs_update) > 0

    return should_update_inputs, inputs_update


if TYPE_CHECKING:
    import optuna


DATA_SAMPLERS = [RandomSampler, SeedableRandomSampler]

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCALER_NAME = "scaler.pt"
SCHEDULER_NAME = "scheduler.pt"


class GaudiTrainer(Trainer):
    """
    GaudiTrainer is built on top of the tranformers' Trainer to enable
    deployment on Habana's Gaudi.
    """

    @deprecate_kwarg("tokenizer", new_name="processing_class", version="5.0.0", raise_if_both_names=True)
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module, None] = None,
        gaudi_config: GaudiConfig = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset], "datasets.Dataset"]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `GaudiTrainingArguments` passed, using `output_dir={output_dir}`.")
            args = GaudiTrainingArguments(output_dir=output_dir)

        self.use_hpu_amp = False
        self.use_cpu_amp = False
        if args.bf16 and not args.deepspeed:
            if args.half_precision_backend == "hpu_amp":
                self.use_hpu_amp = True
            else:
                self.use_cpu_amp = True

            # Workaround to not set amp backend again when calling super().__init__(...)
            # args.bf16 is not used after the __init__ anyway
            args.bf16 = False

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            processing_class,
            model_init,
            compute_loss_func,
            compute_metrics,
            callbacks,
            optimizers,
            optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics,
        )

        if gaudi_config is None:
            self.gaudi_config = GaudiConfig.from_pretrained(args.gaudi_config_name)
        else:
            self.gaudi_config = copy.deepcopy(gaudi_config)

        if self.args.use_habana:
            if self.args.use_hpu_graphs_for_inference:
                self.already_wrapped_for_hpu_graphs = False

            if self.args.deepspeed:
                # Mixed-precision backends are turned off when using DeepSpeed since it manages this itself
                self.gaudi_config.use_torch_autocast = False
                self.use_hpu_amp = False

            if self.args.deepspeed or self.args.dataloader_num_workers >= 1:
                # To avoid warnings about parallelism in tokenizers
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

            if self.gaudi_config.use_torch_autocast:
                if not self.use_hpu_amp and not self.use_cpu_amp:
                    self.use_hpu_amp = True
                    logger.warning(
                        "The argument `--bf16` was not given but `use_torch_autocast` is True in the Gaudi configuration so mixed-precision training with Torch Autocast is enabled."
                    )

            if self.use_hpu_amp and "PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST" not in os.environ:
                self.gaudi_config.declare_autocast_bf16_fp32_ops()

            if self.args.use_lazy_mode:
                try:
                    import habana_frameworks.torch.core as htcore
                except ImportError as error:
                    error.msg = f"Could not import habana_frameworks.torch.core. {error.msg}."
                    raise error
                self.htcore = htcore

                try:
                    import habana_frameworks.torch.hpu as hthpu
                except ImportError as error:
                    error.msg = f"Could not import habana_frameworks.torch.hpu. {error.msg}."
                    raise error
                if self.gaudi_config.use_dynamic_shapes:
                    hthpu.enable_dynamic_shape()
                else:
                    hthpu.disable_dynamic_shape()

            try:
                from habana_frameworks.torch.hpu import random as hpu_random
            except ImportError as error:
                error.msg = f"Could not import habana_frameworks.torch.hpu.random. {error.msg}."
                raise error
            self.hpu_random = hpu_random

        # Set the correct log level depending on the node
        # Already done in super().init() but we have to do it again
        # because we use optimum.utils.logging here and not
        # transformers.utils.logging
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)
        logging.enable_default_handler()
        logging.enable_explicit_format()

        # Suppress PyTorch autocast warnings with Wav2Vec2
        # This is a bug in PyTorch
        warnings.filterwarnings(
            "ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling"
        )

    def _move_model_to_device(self, model, device):
        model = model.to(device)
        # Moving a model to HPU disconnects the tied weights, so we have to retie them.
        if self.args.use_habana and hasattr(model, "tie_weights"):
            model.tie_weights()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = (
                self.processing_class.model_input_names[0] if self.processing_class is not None else None
            )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            num_samples = len(self.train_dataset)
            if (
                not self.args.dataloader_drop_last
                and num_samples % self.args.per_device_train_batch_size != 0
                and self.args.parallel_mode != ParallelMode.DISTRIBUTED
            ):
                # Make the total number of samples divisible by the batch size in lazy mode if needed
                num_samples += (
                    self.args.per_device_train_batch_size - num_samples % self.args.per_device_train_batch_size
                )
            return RandomSampler(self.train_dataset, num_samples=num_samples)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)

            optimizer_grouped_parameters = []
            for t_params, t_weight_decay in zip(
                [
                    [p for n, p in self.model.named_parameters() if n in decay_parameters and p.requires_grad],
                    [p for n, p in self.model.named_parameters() if n not in decay_parameters and p.requires_grad],
                ],
                [self.args.weight_decay, 0.0],
            ):
                # Empty groups of parameters are filtered because they make FusedAdamW crash
                if t_params:
                    optimizer_grouped_parameters.append(
                        {
                            "params": t_params,
                            "weight_decay": t_weight_decay,
                        }
                    )

            if self.gaudi_config.use_fused_adam and self.args.use_habana:
                try:
                    from habana_frameworks.torch.hpex.optimizers import FusedAdamW
                except ImportError as error:
                    error.msg = (
                        f"Could not import 'FusedAdamW' from 'habana_frameworks.torch.hpex.optimizers'. {error.msg}."
                    )
                    raise error
                optimizer_cls = FusedAdamW
                optimizer_kwargs = {
                    "lr": self.args.learning_rate,
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            elif self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _tune_save_checkpoint(self, checkpoint_dir: str):
        output_dir = os.path.join(checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        self.save_model(output_dir, _internal_call=True)
        if self.args.should_save:
            # TODO
            # Update the `TrainerControl` state to where we are currently
            # self.state.stateful_callbacks["TrainerControl"] = self.control.state()
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

    def _wrap_model(self, model, training=True, dataloader=None):
        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if self.accelerator.unwrap_model(model, keep_torch_compile=False) is not model:
            return model

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        if self.args.parallel_mode == ParallelMode.DISTRIBUTED and self.args.distribution_strategy == "ddp":
            kwargs = {}

            if self.args.ddp_find_unused_parameters is not None:
                kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
                if self.args.ddp_find_unused_parameters and self.args.gradient_checkpointing:
                    logger.warning(
                        "ddp_find_unused_parameters and gradient_checkpointing are both True, which may lead to an error:"
                        " https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021"
                    )
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb

            if self.args.use_habana:
                kwargs["gradient_as_bucket_view"] = True

            if self.args.ddp_broadcast_buffers is not None:
                kwargs["broadcast_buffers"] = self.args.ddp_broadcast_buffers

            self.accelerator.ddp_handler = DistributedDataParallelKwargs(**kwargs)

        if self.args.use_hpu_graphs_for_training:
            import habana_frameworks.torch as ht

            ht.hpu.ModuleCacher()(model=model, inplace=True)

        return model

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", dict[str, Any], None] = None,
        ignore_keys_for_eval: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.bf16_full_eval and not args.do_train and not self.is_model_parallel and self.model_init is None:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warn0(
                (
                    "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                    "instead."
                ),
                category=FutureWarning,
                state=self.accelerator.state,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            if self.args.full_determinism:
                enable_full_determinism(self.args.seed)
            else:
                set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )

        if args.push_to_hub:
            try:
                # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
                hf_hub_utils.disable_progress_bars()
                return inner_training_loop(
                    args=args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            finally:
                hf_hub_utils.enable_progress_bars()
        else:
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)
        if (
            self.accelerator.mpu.sequence_parallel_is_initialized()
            and self.accelerator.mpu.get_sequence_parallel_world_size() > 1
        ):
            total_train_batch_size = total_train_batch_size / self.accelerator.mpu.get_sequence_parallel_world_size()

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self.is_fsdp_enabled

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            import transformers.modeling_utils

            if args.deepspeed:
                from deepspeed.runtime.activation_checkpointing.checkpointing import (
                    CheckpointFunction,
                    non_reentrant_checkpoint,
                )

                # HACK because outputs should always be tuples
                def hpu_deepspeed_checkpointing(function, *checkpoint_args, use_reentrant: Optional[bool] = None):
                    """DeepSpeed activation checkpointing."""
                    if use_reentrant is None:
                        use_reentrant = True
                    if use_reentrant:
                        all_outputs = []
                        CheckpointFunction.apply(function, all_outputs, *checkpoint_args)
                    else:
                        logger.info("DeepSpeed activation checkpointing=non_reentrant_checkpoint")
                        all_outputs = non_reentrant_checkpoint(function, *checkpoint_args)

                    # Always return a tuple
                    # When all_outputs contains only one element, DeepSpeed returns this element instead of a tuple
                    # which is not consistent with some models. See https://github.com/microsoft/DeepSpeed/issues/1057.
                    return tuple(all_outputs)

                torch.utils.checkpoint.checkpoint = hpu_deepspeed_checkpointing
                transformers.modeling_utils.checkpoint = hpu_deepspeed_checkpointing
            elif args.use_lazy_mode:
                from .gradient_checkpointing import checkpoint as lazy_mode_checkpointing

                torch.utils.checkpoint.checkpoint = lazy_mode_checkpointing
                transformers.modeling_utils.checkpoint = lazy_mode_checkpointing

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

            # Wrap `_gradient_checkpointing_func` in the model with `transformer_engine` `activation_checkpointing` context.
            if self.accelerator.fp8_enabled:
                FP8ContextWrapper.gradient_checkpointing_wrap(self.model)
        else:
            # Hack because `RegressionModel` in test_trainer.py doesn't have `gradient_checkpointing_disable`
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        if self.gaudi_config.use_fused_clip_norm and self.args.use_habana:
            try:
                from habana_frameworks.torch.hpex.normalization import FusedClipNorm
            except ImportError as error:
                error.msg = f"Could not import habana_frameworks.torch.hpex.normalization. {error.msg}."
                raise error
            self.FusedNorm = FusedClipNorm(model.parameters(), args.max_grad_norm)
        else:
            self.FusedNorm = None

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        start_time_after_warmup = None
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # In multi-worker training: broadcast model parameters from worker:0 to all the others.
        # This must be done manually unless DistributedDataParallel is used.
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED and self.args.distribution_strategy == "fast_ddp":
            from ..distributed import all_reduce_gradients

            logger.debug(
                f"Broadcasting the model parameters to assure that each of {self.args.world_size} workers start the training from the same point."
            )
            for param in model.parameters():
                torch.distributed.broadcast(param.data, src=0)

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated every time .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self._zero_model_grad(model)

        # Gradient clipping
        grad_norm: Optional[float] = None
        _should_compute_grad_norm: bool = self.accelerator.distributed_type != DistributedType.DEEPSPEED and (
            args.max_grad_norm is not None and args.max_grad_norm > 0
        )

        learning_rate = None

        # attn_softmax_bf16 and use_flash_attention are enabled only for llama, qwen2, starcoder2, gemma and baichuan
        # lazy_mode for llama, qwen2, starcoder2 and mistral
        _should_update_inputs, _inputs_update = _get_input_update_settings(self.model, lazy_mode=args.use_lazy_mode)

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        if self.args.adjust_throughput:
            self.log_evaluate_save_time = 0
        else:
            self.log_evaluate_save_time = None

        hb_profiler = HabanaProfile(
            warmup=self.args.profiling_warmup_steps,
            active=self.args.profiling_steps,
            record_shapes=self.args.profiling_record_shapes,
            with_stack=self.args.profiling_with_stack,
            name="train",
        )
        hb_profiler.start()

        if _is_peft_model(self.model) and self.model.peft_type == PeftType.ADALORA:
            self.model.base_model.peft_config[self.model.trainable_adapter_name].total_step = max_steps
            if max_steps < self.model.base_model.peft_config[self.model.trainable_adapter_name].tfinal:
                self.model.base_model.peft_config[self.model.trainable_adapter_name].tfinal = 0

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            if args.gradient_accumulation_steps == 1:
                total_updates -= 1
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples_transformers(
                    epoch_iterator, num_batches, args.device
                )
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    if self.args.compile_from_sec_iteration and is_torch_version(">=", "2.6.0"):
                        torch.compiler.set_stance("force_eager" if step == 0 else "default")

                    if (
                        args.throughput_warmup_steps > 0
                        and (args.throughput_warmup_steps * args.gradient_accumulation_steps)
                        == epoch * steps_in_epoch + step
                    ):
                        start_time_after_warmup = time.time()

                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # attn_softmax_bf16 and use_flash_attention is enabled only for llama, qwen2, starcoder2, gemma, baichuan and chatglm
                    # lazy_mode for llama, qwen2, starcoder2 and mistral
                    if _should_update_inputs:
                        inputs.update(_inputs_update)

                    # TODO: keep syncs for fast DDP?
                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if (
                        args.parallel_mode == ParallelMode.DISTRIBUTED
                        and args.distribution_strategy == "fast_ddp"
                        and do_sync_step
                    ):
                        all_reduce_gradients(
                            model, use_hpu_graphs=True
                        )  # use HPU graphs for gradient fusion regardless of args.use_hpu_graphs_for_training setting

                    if args.logging_nan_inf_filter and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if args.use_lazy_mode:
                        self.htcore.mark_step()

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # If the condition is true, we need to compute grad_norm, deepspeed does its own clipping
                        if _should_compute_grad_norm:
                            # Gradient clipping
                            if self.FusedNorm is not None:
                                # TODO: to merge self.accelerator.clip_grad_norm_ when HMP is removed
                                grad_norm = self.FusedNorm.clip_norm(model.parameters())
                            else:
                                # Revert to normal clipping otherwise
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        # get leaning rate before update
                        learning_rate = self._get_learning_rate()

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        self._zero_model_grad(model)
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        if args.use_lazy_mode:
                            self.htcore.mark_step()
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                            learning_rate=learning_rate,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    hb_profiler.step()
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
            )

            if self.control.should_training_stop:
                break

        hb_profiler.stop()

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if args.parallel_mode == ParallelMode.DISTRIBUTED:
                torch.distributed.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        # Warmup steps are removed from the calculation of speed metrics
        num_samples_for_speed_metrics = num_train_samples - args.throughput_warmup_steps * total_train_batch_size
        num_steps_for_speed_metrics = self.state.max_steps - args.throughput_warmup_steps
        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_samples_for_speed_metrics,
            num_steps=num_steps_for_speed_metrics,
            num_tokens=num_train_tokens,
            start_time_after_warmup=start_time_after_warmup,
            log_evaluate_save_time=self.log_evaluate_save_time,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        best_safe_model_path = os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_NAME)
        best_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_WEIGHTS_NAME)
        best_safe_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        model = self.model
        if self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(
                self.model_wrapped,
                self.state.best_model_checkpoint,
                load_module_strict=not _is_peft_model(self.model),
            )
        elif self.is_fsdp_enabled:
            load_result = load_fsdp_model(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                model,
                self.state.best_model_checkpoint,
                **_get_fsdp_ckpt_kwargs(),
            )
        elif (
            os.path.exists(best_model_path)
            or os.path.exists(best_safe_model_path)
            or os.path.exists(best_adapter_model_path)
            or os.path.exists(best_safe_adapter_model_path)
        ):
            has_been_loaded = True
            if _is_peft_model(model):
                # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
                # TODO: in the future support only specific min PEFT versions
                if (hasattr(model, "active_adapter") or hasattr(model, "active_adapters")) and hasattr(
                    model, "load_adapter"
                ):
                    # For BC for older PEFT versions
                    if hasattr(model, "active_adapters"):
                        active_adapter = model.active_adapters[0]
                        if len(model.active_adapters) > 1:
                            logger.warning("Detected multiple active adapters, will only consider the first one")
                    else:
                        active_adapter = model.active_adapter

                    if os.path.exists(best_adapter_model_path) or os.path.exists(best_safe_adapter_model_path):
                        try:
                            model.load_adapter(self.state.best_model_checkpoint, active_adapter)
                        except RuntimeError as exc:
                            if model.peft_config[active_adapter].is_prompt_learning:
                                # for context: https://github.com/huggingface/peft/issues/2256
                                msg = (
                                    "When using prompt learning PEFT methods such as "
                                    f"{model.peft_config[active_adapter].peft_type.value}, setting "
                                    "load_best_model_at_end=True can lead to errors, it is recommended "
                                    "to set this to False and to load the model manually from the checkpoint "
                                    "directory using PeftModel.from_pretrained(base_model, <path>) after training "
                                    "has finished."
                                )
                                raise RuntimeError(msg) from exc
                            else:
                                raise
                        # Load_adapter has no return value present, modify it when appropriate.
                        from torch.nn.modules.module import _IncompatibleKeys

                        load_result = _IncompatibleKeys([], [])
                    else:
                        logger.warning(
                            "The intermediate checkpoints of PEFT may not be saved correctly, "
                            f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                            "Check some examples here: https://github.com/huggingface/peft/issues/96"
                        )
                        has_been_loaded = False
                else:
                    logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
                    has_been_loaded = False
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                    state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                else:
                    state_dict = torch.load(best_model_path, map_location="cpu", weights_only=True)

                # If the model is on the GPU, it still works!
                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                load_result = model.load_state_dict(state_dict, False)

            if has_been_loaded:
                self._issue_warnings_after_load(load_result)
        elif os.path.exists(os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_INDEX_NAME)) or os.path.exists(
            os.path.join(self.state.best_model_checkpoint, WEIGHTS_INDEX_NAME)
        ):
            load_result = load_sharded_checkpoint(model, self.state.best_model_checkpoint, strict=False)
            self._issue_warnings_after_load(load_result)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )

    def _maybe_log_save_evaluate(
        self, tr_loss, _grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        timer = HabanaGenerationTime()
        timer.start()
        if self.args.adjust_throughput:
            timer.step()

        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            # This grad_norm block was outside of _maybe_log_save_evaluate method causing perf degradation.
            # Moving it here so the grad tensor is only copied when it's needed.
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                grad_norm = model.get_global_grad_norm()
                # In some cases the grad norm may not return a float
                if hasattr(grad_norm, "item"):
                    grad_norm = grad_norm.item()
            else:
                if (
                    _grad_norm is not None
                    and self.accelerator.distributed_type != DistributedType.FSDP
                    and _grad_norm.size() == torch.Size([1])
                ):
                    grad_norm = _grad_norm.detach().item()
                else:
                    grad_norm = None

            if grad_norm is not None:
                logs["grad_norm"] = grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time=start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        if self.args.adjust_throughput:
            timer.step()
            self.log_evaluate_save_time += timer.last_duration

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        with safe_globals():
            checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if self.args.use_habana:
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                self.hpu_random.set_rng_state_all(checkpoint_rng_state["hpu"])
            else:
                try:
                    self.hpu_random.set_rng_state(checkpoint_rng_state["hpu"])
                except Exception as e:
                    logger.info(
                        f"Didn't manage to set back the RNG states of the HPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )

    def _save_rng_state(self, output_dir):
        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }

        if self.args.use_habana:
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # In non distributed, we save the global HPU RNG state
                rng_states["hpu"] = self.hpu_random.get_rng_state_all()
            else:
                rng_states["hpu"] = self.hpu_random.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

    def _save_optimizer_and_scheduler(self, output_dir):
        if self.is_deepspeed_enabled:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            accept_exclude_frozen_parameters = "exclude_frozen_parameters" in set(
                inspect.signature(self.model_wrapped.save_checkpoint).parameters.keys()
            )
            if accept_exclude_frozen_parameters and _is_peft_model(self.model):
                self.model_wrapped.save_checkpoint(output_dir, exclude_frozen_parameters=True)
            else:
                self.model_wrapped.save_checkpoint(output_dir)
        elif self.is_fsdp_enabled:
            if isinstance(self.model, torch._dynamo.eval_frame.OptimizedModule):
                # TODO: for some reason the fsdp model is not unwrapped correctly here, the self.mode
                # shouldn't be an OptimizedModule at this point.
                model = self.model._orig_mod
            else:
                model = self.model
            # save fsdp specific ckpt for resuming from ckpt
            save_fsdp_model(
                self.accelerator.state.fsdp_plugin, self.accelerator, model, output_dir, **_get_fsdp_ckpt_kwargs()
            )
            save_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, model, output_dir
            )
        elif self.args.should_save:
            # deepspeed.save_checkpoint above saves model/optim/sched
            # This block is executed by the main process only
            optim_dict = self.optimizer.state_dict()
            if self.args.use_habana:
                # Move the state dict from HPU to CPU before saving
                optim_dict = to_device_dtype(optim_dict, target_device=torch.device("cpu"))
            torch.save(optim_dict, os.path.join(output_dir, OPTIMIZER_NAME))

        # Save SCHEDULER & SCALER
        is_deepspeed_custom_scheduler = self.is_deepspeed_enabled and not isinstance(
            self.lr_scheduler, DeepSpeedSchedulerWrapper
        )
        if self.args.should_save and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler):
            if self.args.use_habana:
                # Move the state dict from HPU to CPU before saving
                scheduler_dict = self.lr_scheduler.state_dict()
                scheduler_dict = to_device_dtype(scheduler_dict, target_device=torch.device("cpu"))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if self.is_deepspeed_enabled:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            if not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.lr_scheduler.load_state_dict(
                        torch.load(os.path.join(checkpoint, SCHEDULER_NAME), weights_only=True)
                    )
                reissue_pt_warnings(caught_warnings)
            return

        checkpoint_file_exists = (
            os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME))
            or os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME_BIN))
            or (
                os.path.isdir(checkpoint)
                and any(
                    OPTIMIZER_NAME_BIN.split(".")[0] in folder_name
                    for folder_name in os.listdir(checkpoint)
                    if os.path.isdir(os.path.join(checkpoint, folder_name))
                )
            )
        )

        if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
            # We use the CPU when training on one GPU to avoid OOM for GPU RAM when training big models.
            # In distributed training however, we load directly on each GPU and risk the GPU OOM as it's more
            # likely to get OOM on CPU (since we load num_gpu times the optimizer state
            map_location = "cpu" if self.args.use_habana else self.args.device
            if self.is_fsdp_enabled:
                load_fsdp_optimizer(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    self.optimizer,
                    self.model,
                    checkpoint,
                    **_get_fsdp_ckpt_kwargs(),
                )
            else:
                self.optimizer.load_state_dict(
                    torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location=map_location, weights_only=True)
                )

            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(
                    torch.load(os.path.join(checkpoint, SCHEDULER_NAME), map_location=map_location, weights_only=True)
                )
            reissue_pt_warnings(caught_warnings)

            # Move optimizer state to HPU
            if self.args.use_habana:
                to_device_dtype(self.optimizer.state.values(), target_device=torch.device("hpu"))

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)

        mem_stats = get_hpu_memory_stats(self.args.device)
        logs.update(mem_stats)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        Compared to Transformers, it is also possible to enable non-blocking data copy.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            if (
                self.accelerator.mpu.sequence_parallel_is_initialized()
                and self.accelerator.mpu.get_sequence_parallel_world_size() > 1
            ):
                seq_parallel_world_rank = self.accelerator.mpu.get_sequence_parallel_rank()
                sub_seq_length = int(data.size()[1] / self.accelerator.mpu.get_sequence_parallel_world_size())
                data = data[
                    :, seq_parallel_world_rank * sub_seq_length : (seq_parallel_world_rank + 1) * sub_seq_length
                ]
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            if self.args.non_blocking_data_copy:
                return data.to(**kwargs, non_blocking=True)
            else:
                return data.to(**kwargs)
        return data

    def autocast_smart_context_manager(self, cache_enabled: Optional[bool] = True):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.

        Modified by Habana to enable using `autocast` on Gaudi devices.
        """
        if self.use_cpu_amp:
            ctx_manager = torch.autocast(device_type="cpu", dtype=torch.bfloat16, cache_enabled=cache_enabled)
        elif self.use_hpu_amp:
            ctx_manager = torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True)
        else:
            ctx_manager = contextlib.nullcontext()

        # Merge autocast context and `fp8_autocast` context if FP8 is enabled.
        # Currently FP8 is enabled only for training.
        if self.accelerator.fp8_enabled and self.model.training:
            ctx_manager = FP8ContextWrapper(ctx_manager, fp8_recipe=self.accelerator.fp8_recipe)

        return ctx_manager

    def training_step(
        self, model: torch.nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`torch.nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        # TODO
        # if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
        #     self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learning rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.use_lazy_mode and self.args.pipelining_fwd_bwd:
            self.htcore.mark_step()

        # Finally we need to normalize the loss for reporting
        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            # temporary fix to calculate loss correctly
            loss = loss / self.args.gradient_accumulation_steps

        # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
        # https://github.com/huggingface/transformers/pull/35808
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            kwargs["scale_wrt_gas"] = False

        if _is_peft_model(self.model) and self.model.peft_type == PeftType.ADALORA:
            assert not (self.accelerator.fp8_enabled and self.args.gradient_checkpointing), (
                "FP8 precision with gradient_checkpointing is currently not supported with PeftType.ADALORA"
            )
            if self.is_deepspeed_enabled and not is_deepspeed_zero3_enabled():
                self.accelerator.deepspeed_engine_wrapped.engine.backward(loss)
                self.model.base_model.update_and_allocate(self.state.global_step)
                self.accelerator.deepspeed_engine_wrapped.engine.step()
            else:
                self.accelerator.backward(loss, **kwargs)
                self.model.base_model.update_and_allocate(self.state.global_step)
        else:
            if self.accelerator.fp8_enabled and self.args.gradient_checkpointing:
                # The precision used in backward pass should be same as the one used in forward pass.
                # However when training with gradient_checkpointing and FP8 precision, recompute forward
                # in backward does not automatically run with FP8 precision. In order to handle this,
                # the backward is run in `fp8_autocast` context
                with FP8ContextWrapper.create_fp8_context(fp8_recipe=self.accelerator.fp8_recipe):
                    self.accelerator.backward(loss, **kwargs)
            else:
                self.accelerator.backward(loss, **kwargs)
        return loss.detach()

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.
        Will only save from the main process.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled:
            if "FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type):
                state_dict = self.accelerator.get_state_dict(self.model)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
        elif self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.deepspeed)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                accept_exclude_frozen_parameters = "exclude_frozen_parameters" in set(
                    inspect.signature(self.model_wrapped.save_checkpoint).parameters.keys()
                )
                if accept_exclude_frozen_parameters and _is_peft_model(self.model):
                    self.model_wrapped.save_checkpoint(output_dir, exclude_frozen_parameters=True)
                else:
                    self.model_wrapped.save_checkpoint(output_dir)
        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)

        if state_dict is None:
            state_dict = self.model.state_dict()
        if state_dict and self.args.use_habana:
            # state_dict items have to be saved on the CPU
            state_dict = to_device_dtype(state_dict, target_device=torch.device("cpu"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if isinstance(self.accelerator.unwrap_model(self.model, keep_torch_compile=False), supported_classes):
                self.accelerator.unwrap_model(self.model, keep_torch_compile=False).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            logger.info("Saving Trainer.data_collator.tokenizer by default as Trainer.processing_class is `None`")
            self.data_collator.tokenizer.save_pretrained(output_dir)

        self.gaudi_config.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        From https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/trainer.py#L3162 with the following modification
        1. use throughput_warmup_steps in evaluation throughput calculation
        """
        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        start_time = time.time()
        self.start_time_after_warmup = None

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        num_samples = output.num_samples - self.args.throughput_warmup_steps * total_batch_size
        num_steps = math.ceil(output.num_samples / total_batch_size) - self.args.throughput_warmup_steps

        eval_steps = math.ceil(output.num_samples / total_batch_size)
        if eval_steps <= self.args.throughput_warmup_steps:
            logger.warning(
                f" Warmup steps are taken into account for the throughput calculation because the number of evaluation steps ({eval_steps}) is smaller than the number of warmup steps ({self.args.throughput_warmup_steps})"
            )
            num_samples = output.num_samples
            num_steps = eval_steps

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_samples,
                num_steps=num_steps,
                start_time_after_warmup=self.start_time_after_warmup,
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[list[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        From https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/trainer.py#L3904 with the following modification
        1. comment out TPU related
        2. use throughput_warmup_steps in evaluation throughput calculation
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()
        self.start_time_after_warmup = None

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]

        num_samples = output.num_samples - self.args.throughput_warmup_steps * total_batch_size
        num_steps = math.ceil(output.num_samples / total_batch_size) - self.args.throughput_warmup_steps

        logger.info(f"num_samples : {num_samples}, num_steps: {num_steps}")

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_samples,
                num_steps=num_steps,
                start_time_after_warmup=self.start_time_after_warmup,
            )
        )

        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8" and not self.args.torch_compile)
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        model.eval()
        # if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
        #     self.optimizer.eval()

        # Do not use HPU graphs if the training is ongoing because it detaches gradients
        if args.use_hpu_graphs_for_inference and not self.is_in_train:
            logger.info("Using HPU graphs for inference.")
            # Do not wrap the model in HPU graphs if it has already been done
            if not self.already_wrapped_for_hpu_graphs:
                from habana_frameworks.torch.hpu import wrap_in_hpu_graph

                model = wrap_in_hpu_graph(
                    model, disable_tensor_cache=args.disable_tensor_cache_hpu_graphs, max_graphs=args.max_hpu_graphs
                )
                self.already_wrapped_for_hpu_graphs = True

        # if full bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # attn_softmax_bf16 and use_flash_attention are enabled only for llama, qwen2, starcoder2, gemma and baichuan
        _should_update_inputs, _inputs_update = _get_input_update_settings(self.model)

        # set a default dtype of logits
        logits_dtype: str = "float32"

        hb_profiler = HabanaProfile(
            warmup=self.args.profiling_warmup_steps_eval,
            active=self.args.profiling_steps_eval,
            record_shapes=self.args.profiling_record_shapes,
            with_stack=self.args.profiling_with_stack,
            name=description.lower(),
        )
        hb_profiler.start()

        # Main evaluation loop
        start_time_eval = time.time()
        for step, inputs in enumerate(dataloader):
            if (
                self.args.throughput_warmup_steps > 0
                and not self.is_in_train
                and step == self.args.throughput_warmup_steps
            ):
                self.start_time_after_warmup = time.time()
                self.compilation_time = self.start_time_after_warmup - start_time_eval

            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # attn_softmax_bf16 and use_flash_attention are enabled only for llama, qwen2, starcoder2, gemma, baichuan and chatglm
            if _should_update_inputs:
                inputs.update(_inputs_update)

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            # Update containers
            if losses is not None:
                losses = self.gather_function(losses.repeat(batch_size))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function(inputs_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            # Save the logits dtype since we need to convert them into floats during the process
            # They will be converted back into their original dtype right before computing metrics
            if logits is not None:
                logits_dtype = get_dtype(logits)
                if args.use_habana and logits_dtype != "float32":
                    logits = to_device_dtype(logits, target_dtype=torch.float32)
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                if self.args.context_parallel_size != 1:
                    labels = labels.clone()
                labels = self.gather_function(labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs

            # nested concat does accumulation on tensors of variable length.
            # Added mark step here to avoid graph recompile
            if args.use_lazy_mode:
                self.htcore.mark_step()

            hb_profiler.step()

        hb_profiler.stop()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Convert predictions back into their original dtype if necessary
        if all_preds is not None:
            all_preds = convert_into_dtypes(all_preds, logits_dtype)

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs)
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time
        if hasattr(self, "compilation_time"):
            metrics[f"{metric_key_prefix}_graph_compliation_duration"] = self.compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`torch.nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            try:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.detach().mean()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]
            except RuntimeError as error:
                if "cpu fallback is not supported during hpu graph capturing" in str(error):
                    error.args = (
                        f"{error}. You should run inference in lazy mode only with `use_lazy_mode=True` and `use_hpu_graphs_for_inference=False`.",
                    )
                raise error

        if self.args.use_lazy_mode and not (self.args.use_hpu_graphs_for_inference and not self.is_in_train):
            self.htcore.mark_step()

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def _push_from_checkpoint(self, checkpoint_folder):
        # Only push from one node.
        if not self.is_world_process_zero() or self.args.hub_strategy == HubStrategy.END:
            return
        # If we haven't finished the last push, we don't do this one unless args.hub_always_push=True.
        if not self.args.hub_always_push and self.push_in_progress is not None and not self.push_in_progress.is_done():
            return

        output_dir = self.args.output_dir
        # To avoid a new synchronization of all model weights, we just copy the file from the checkpoint folder
        modeling_files = [CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME, GAUDI_CONFIG_NAME]
        #  Add sharded checkpoints if we have an index
        for index_file in [WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME]:
            index_path = os.path.join(checkpoint_folder, index_file)
            if os.path.isfile(index_path):
                modeling_files.append(index_file)
                with open(index_path) as f:
                    index = json.loads(f.read())
                shard_files = list(set(index["weight_map"].values()))
                modeling_files.extend(shard_files)
        if is_peft_available():
            modeling_files.extend([ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME, ADAPTER_SAFE_WEIGHTS_NAME])
        for modeling_file in modeling_files:
            if os.path.isfile(os.path.join(checkpoint_folder, modeling_file)):
                shutil.copy(os.path.join(checkpoint_folder, modeling_file), os.path.join(output_dir, modeling_file))
        # Saving the processing class is fast and we don't know how many files it may have spawned, so we resave it to be sure.
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        # Same for the training arguments
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        if self.args.save_strategy == SaveStrategy.STEPS:
            commit_message = f"Training in progress, step {self.state.global_step}"
        else:
            commit_message = f"Training in progress, epoch {int(self.state.epoch)}"

        model_push_job = upload_folder(
            repo_id=self.hub_model_id,
            folder_path=output_dir,
            commit_message=commit_message,
            token=self.args.hub_token,
            run_as_future=True,
            ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
        )

        push_jobs = [model_push_job]

        if self.args.hub_strategy in [HubStrategy.CHECKPOINT, HubStrategy.ALL_CHECKPOINTS]:
            path_in_repo = (
                "last-checkpoint" if self.args.hub_strategy == HubStrategy.CHECKPOINT else Path(checkpoint_folder).name
            )
            checkpoint_push = upload_folder(
                repo_id=self.hub_model_id,
                folder_path=checkpoint_folder,
                path_in_repo=path_in_repo,
                commit_message=commit_message + ", checkpoint",
                token=self.args.hub_token,
                run_as_future=True,
            )
            push_jobs.append(checkpoint_push)

        if self.push_in_progress is None or self.push_in_progress.is_done():
            self.push_in_progress = PushInProgress(push_jobs)
        else:
            self.push_in_progress.jobs.extend(push_jobs)

    #
    # Deprecated code
    #
    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        if not has_length(dataloader):
            raise ValueError("dataloader must implement a working __len__")

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or self.is_fsdp_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        model.eval()
        # TODO
        # if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
        #     self.optimizer.eval()

        # Do not use HPU graphs if the training is ongoing because it detaches gradients
        if args.use_hpu_graphs_for_inference and not self.is_in_train:
            logger.info("Using HPU graphs for inference.")
            # Do not wrap the model in HPU graphs if it has already been done
            if not self.already_wrapped_for_hpu_graphs:
                from habana_frameworks.torch.hpu import wrap_in_hpu_graph

                model = wrap_in_hpu_graph(
                    model, disable_tensor_cache=args.disable_tensor_cache_hpu_graphs, max_graphs=args.max_hpu_graphs
                )
                self.already_wrapped_for_hpu_graphs = True

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = (
            dataloader.total_batch_size
            if getattr(dataloader, "_is_accelerate_prepared", False)
            else dataloader.batch_size
        )

        if batch_size is None:
            raise ValueError(
                "Batch size cannot be None. Ensure the dataloader has a valid batch_size or total_batch_size."
            )

        num_examples = self.num_examples(dataloader)
        logger.info(f"\n***** Running {description} *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Batch size = {batch_size}")

        losses_host: Optional[torch.Tensor] = None
        preds_host: Union[torch.Tensor, list[torch.Tensor], None] = None
        labels_host: Union[torch.Tensor, list[torch.Tensor], None] = None
        inputs_host: Union[torch.Tensor, list[torch.Tensor], None] = None
        metrics: Optional[dict] = None
        eval_set_kwargs: dict = {}

        world_size = max(1, args.world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, SequentialDistributedSampler):
                make_multiple_of = dataloader.sampler.batch_size
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            inputs_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)

        if args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and preds_host is not None and labels_host is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses_host if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs_host if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=preds_host, label_ids=labels_host, **batch_kwargs),
                        compute_result=is_last_step,
                    )

            if self.args.batch_eval_metrics or (
                args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0
            ):
                # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
                    inputs_gatherer.add_arrays(self._gather_and_numpify(inputs_host, "eval_inputs_ids"))

                # Set back to None to begin a new accumulation
                del losses_host, preds_host, labels_host, inputs_host
                losses_host, preds_host, labels_host, inputs_host = None, None, None, None

            # nested concat does accumulation on tensors of variable length.
            # Added mark step here to avoid graph recompile
            if args.use_lazy_mode:
                self.htcore.mark_step()

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
            inputs_gatherer.add_arrays(self._gather_and_numpify(inputs_host, "eval_inputs_ids"))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None
        inputs_ids = inputs_gatherer.finalize() if not prediction_loss_only else None

        if (
            self.compute_metrics is not None
            and preds is not None
            and label_ids is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs["losses"] = eval_loss if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = inputs_ids if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids, **eval_set_kwargs))
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=preds, label_ids=label_ids, metrics=metrics, num_samples=num_examples)

    def create_accelerator_and_postprocess(self):
        # We explicitly don't rely on the `Accelerator` to do gradient accumulation
        grad_acc_kwargs = {}
        if self.args.accelerator_config.gradient_accumulation_kwargs is not None:
            grad_acc_kwargs = self.args.accelerator_config.gradient_accumulation_kwargs

        # check if num_steps is attempted to be passed in gradient_accumulation_kwargs
        if "num_steps" in grad_acc_kwargs:
            if self.args.gradient_accumulation_steps > 1:
                # raise because we do not know which setting is intended.
                raise ValueError(
                    "The `AcceleratorConfig`'s `num_steps` is set but `gradient_accumulation_steps` is greater than 1 in the passed `TrainingArguments`"
                    "If using the passed `AcceleratorConfig` is desired, do not set the `TrainingArguments` `gradient_accumulation_steps`."
                )
            else:
                self.args.gradient_accumulation_steps = grad_acc_kwargs["num_steps"]

        accelerator_config = self.args.accelerator_config.to_dict()

        # Extract dataloader config params from accelerator config
        dataloader_params = ["split_batches", "dispatch_batches", "even_batches", "use_seedable_sampler"]
        dataloader_config = DataLoaderConfiguration(
            **{param: accelerator_config.pop(param) for param in dataloader_params}
        )
        if is_accelerate_available("1.1.0"):
            dataloader_config.data_seed = self.args.data_seed

        non_blocking = accelerator_config.pop("non_blocking")
        if non_blocking and not self.args.dataloader_pin_memory:
            logger.warning(
                "`non_blocking` is enabled but `dataloader_pin_memory` is not. For the best performance, it's recommended to enable both."
            )
        dataloader_config.non_blocking = non_blocking
        # this would have been updated above, no need for it anymore
        accelerator_config.pop("gradient_accumulation_kwargs")

        args = {
            "dataloader_config": dataloader_config,
            "deepspeed_plugins": self.args.deepspeed_plugin,
            # OH specific
            "distribution_strategy": self.args.distribution_strategy,
            "use_regional_compilation": self.args.use_regional_compilation,
            "compiled_autograd_enable": self.args.use_compiled_autograd,
        }
        # tp is initialized at Accelerator init phase so
        # args should be prepared here
        if self.args.tp_size > 1:
            self.is_tp_enabled = True
            args["torch_tp_plugin"] = TorchTensorParallelPlugin(tp_size=self.args.tp_size)

        # create accelerator object
        self.accelerator = GaudiAccelerator(**args)
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        if "use_gather_object" in inspect.signature(self.gather_function).parameters.keys():
            self.gather_function = functools.partial(
                self.gather_function, use_gather_object=self.args.eval_use_gather_object
            )

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        self.is_tp_enabled = getattr(self.accelerator.state, "torch_tp_plugin", None) is not None
        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            for param in ["limit_all_gathers", "activation_checkpointing"]:
                setattr(fsdp_plugin, param, self.args.fsdp_config.get(param, getattr(fsdp_plugin, param)))
            if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                raise ValueError(
                    "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                    "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                    "when using FSDP."
                )

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

        # `save_only_model` can't be used with DeepSpeed/FSDP along with `load_best_model_at_end`
        if (
            self.args.save_only_model
            and (self.is_deepspeed_enabled or self.is_fsdp_enabled)
            and self.args.load_best_model_at_end
        ):
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            raise ValueError(f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`.")

        # `auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3
        if (
            self.is_deepspeed_enabled
            and self.accelerator.state.deepspeed_plugin.zero_stage == 3
            and self.args.auto_find_batch_size
        ):
            raise ValueError(
                "`auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3. Please consider using Zero-2, Zero-1, or FSDP"
            )
        if (
            self.args.save_only_model
            and self.is_fsdp_enabled
            and "SHARDED_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)
        ):
            raise ValueError("save_only_model option is not compatible with FSDP state dict type 'SHARDED_STATE_DICT'")

    def propagate_args_to_deepspeed(self, auto_find_batch_size=False):
        """
        Sets values in the deepspeed plugin based on the Trainer args
        """
        from .integrations.deepspeed import GaudiTrainerDeepSpeedConfig

        ds_plugin = self.accelerator.state.deepspeed_plugin

        ds_plugin.hf_ds_config = GaudiTrainerDeepSpeedConfig(ds_plugin.hf_ds_config.config)
        ds_plugin.deepspeed_config = ds_plugin.hf_ds_config.config
        ds_plugin.hf_ds_config.trainer_config_process(self.args, auto_find_batch_size)

    def _zero_model_grad(self, model):
        if hasattr(model, "_zero_grad_kwargs"):
            model.zero_grad(**model._zero_grad_kwargs)
        else:
            # Optimization based on setting gradients to None (instead of zeroing them out) may only be used when gradients are not recorded using HPU graphs.
            # HPU graphs rely on fixed tensors - setting gradients to None will enforce their re-allocation during the backward pass each step.
            set_to_none = (
                self.args.parallel_mode != ParallelMode.DISTRIBUTED or self.args.distribution_strategy == "ddp"
            ) and not self.args.use_hpu_graphs_for_training

            try:
                model.zero_grad(set_to_none=set_to_none)
                model._zero_grad_kwargs = {"set_to_none": set_to_none}
            except TypeError:
                model.zero_grad()
                model._zero_grad_kwargs = {}

    def get_batch_samples_transformers(self, epoch_iterator, num_batches, device):
        batch_samples = []
        num_items_in_batch = None

        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break

        count_num_items_in_batch = (
            len(batch_samples) > 0
            and "labels" in batch_samples[0]
            and (
                # num_items_in_batch is passed to model forward
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3757
                self.model_accepts_loss_kwargs
                # num_items_in_batch is passed to compute_loss_func
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3773
                or self.compute_loss_func is not None
                # num_items_in_batch is also verified if (self.model_accepts_loss_kwargs or self.compute_loss_func)
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3790
            )
        )

        if count_num_items_in_batch:
            # For now we don't support object detection
            try:
                num_items_in_batch = torch.cat([batch["labels"] for batch in batch_samples]).ne(-100).sum()
            except (TypeError, AttributeError, RuntimeError):
                pass

        if num_items_in_batch is not None:
            if self.args.average_tokens_across_devices:
                num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()
            num_items_in_batch = num_items_in_batch.to(device)

        return batch_samples, num_items_in_batch

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        # Removed warning about usage of Trainer.tokenizer
        return self.processing_class

    @tokenizer.setter
    def tokenizer(self, processing_class) -> None:
        # Removed warning about usage of Trainer.tokenizer
        self.processing_class = processing_class
