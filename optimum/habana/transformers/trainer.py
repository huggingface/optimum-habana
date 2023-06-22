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

import copy
import math
import os
import random
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations import hp_params
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerState
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    find_batch_size,
    get_model_param_count,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    TrainOutput,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
)
from transformers.training_args import TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_datasets_available,
    is_safetensors_available,
)

from optimum.habana.distributed import all_reduce_gradients
from optimum.utils import logging

from ..utils import (
    HabanaProfile,
    get_hpu_memory_stats,
    set_seed,
    speed_metrics,
    to_device_dtype,
)
from .deepspeed import deepspeed_init
from .gaudi_configuration import GAUDI_CONFIG_NAME, GaudiConfig
from .trainer_utils import convert_into_dtypes, get_dtype
from .training_args import GaudiTrainingArguments


if is_datasets_available():
    import datasets


if is_safetensors_available():
    import safetensors.torch


if TYPE_CHECKING:
    import optuna


logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class GaudiTrainer(Trainer):
    """
    GaudiTrainer is built on top of the tranformers' Trainer to enable
    deployment on Habana's Gaudi.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        gaudi_config: GaudiConfig = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
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
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

        if gaudi_config is None:
            self.gaudi_config = GaudiConfig.from_pretrained(args.gaudi_config_name)
        else:
            self.gaudi_config = copy.deepcopy(gaudi_config)

        if self.args.use_habana:
            if self.args.use_lazy_mode:
                try:
                    import habana_frameworks.torch.core as htcore
                except ImportError as error:
                    error.msg = f"Could not import habana_frameworks.torch.core. {error.msg}."
                    raise error
                self.htcore = htcore

            if self.args.use_hpu_graphs_for_inference:
                self.already_wrapped_for_hpu_graphs = False

            if self.args.deepspeed:
                # Mixed-precision backends are turned off when using DeepSpeed since it manages this itself
                self.gaudi_config.use_habana_mixed_precision = False
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
            elif self.gaudi_config.use_habana_mixed_precision and self.use_hpu_amp:
                self.gaudi_config.use_habana_mixed_precision = False
                logger.warning(
                    "`--bf16` was given and `use_habana_mixed_precision` is True in the Gaudi configuration. Using Torch Autocast as mixed-precision backend."
                )

            if self.gaudi_config.use_habana_mixed_precision and not (self.use_hpu_amp or self.use_cpu_amp):
                try:
                    from habana_frameworks.torch.hpex import hmp
                except ImportError as error:
                    error.msg = f"Could not import habana_frameworks.torch.hpex. {error.msg}."
                    raise error
                self.hmp = hmp

                # Open temporary files to mixed-precision write ops
                with tempfile.NamedTemporaryFile() as hmp_bf16_file:
                    with tempfile.NamedTemporaryFile() as hmp_fp32_file:
                        # hmp.convert needs ops to be written in text files
                        self.gaudi_config.write_bf16_fp32_ops_to_text_files(
                            hmp_bf16_file.name,
                            hmp_fp32_file.name,
                        )
                        self.hmp.convert(
                            opt_level=self.gaudi_config.hmp_opt_level,
                            bf16_file_path=hmp_bf16_file.name,
                            fp32_file_path=hmp_fp32_file.name,
                            isVerbose=self.gaudi_config.hmp_is_verbose,
                        )

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

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

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
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                num_samples = len(self.train_dataset)
                if (
                    self.args.use_lazy_mode
                    and not self.args.dataloader_drop_last
                    and len(self.train_dataset) % self.args.per_device_train_batch_size != 0
                ):
                    # Make the total number of samples divisible by the batch size in lazy mode if needed
                    num_samples += (
                        self.args.per_device_train_batch_size
                        - len(self.train_dataset) % self.args.per_device_train_batch_size
                    )
                return RandomSampler(self.train_dataset, num_samples=num_samples, generator=generator)
            else:
                if self.args.use_lazy_mode and not self.args.dataloader_drop_last:
                    # Use a loop for HPUs when drop_last is False to have all batches have the same size
                    return DistributedSamplerWithLoop(
                        self.train_dataset,
                        batch_size=self.args.per_device_train_batch_size,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                    )
                else:
                    return DistributedSampler(
                        self.train_dataset,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                    )

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

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
            else:
                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _tune_save_checkpoint(self):
        from ray import tune

        if not self.use_tune_checkpoints:
            return
        with tune.checkpoint_dir(step=self.state.global_step) as checkpoint_dir:
            output_dir = os.path.join(checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            self.save_model(output_dir)
            if self.args.should_save:
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
                if not self.args.use_habana:
                    torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

    def _wrap_model(self, model, training=True, dataloader=None):
        if self.args.use_ipex:
            dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
            model = self.ipex_optimize_model(model, training, dtype=dtype)

        # already initialized its own DDP and AMP
        if self.deepspeed:
            return self.deepspeed

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        if self.args.local_rank != -1 and self.args.distribution_strategy == "ddp":
            kwargs = {}

            kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            if self.args.ddp_find_unused_parameters and self.args.gradient_checkpointing:
                logger.warning(
                    "ddp_find_unused_parameters and gradient_checkpointing are both True, which may lead to an error:"
                    " https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021"
                )
            kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb

            if self.args.use_habana:
                kwargs["gradient_as_bucket_view"] = True

            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank] if self.args._n_gpu != 0 and not self.args.use_habana else None,
                output_device=self.args.local_rank if self.args._n_gpu != 0 and not self.args.use_habana else None,
                **kwargs,
            )

        if self.args.use_hpu_graphs_for_training:
            import habana_frameworks.torch as ht

            ht.hpu.ModuleCacher()(model=model, inplace=True)

        # torch.compile() needs to be called after wrapping the model with FSDP or DDP
        # to ensure that it accounts for the graph breaks required by those wrappers
        if self.args.torch_compile:
            model = torch.compile(model, backend=self.args.torch_compile_backend, mode=self.args.torch_compile_mode)

        return model

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.args.adjust_throughput:
            save_start = time.perf_counter()

        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        if self.args.adjust_throughput:
            self.log_evaluate_save_time += time.perf_counter() - save_start

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
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
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                (
                    "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                    "instead."
                ),
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
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

        if resume_from_checkpoint is not None and args.deepspeed is None:
            self._load_from_checkpoint(resume_from_checkpoint)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )

        return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )

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
            kwargs = {"device": self.args.device}
            if self.deepspeed and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": self.args.hf_deepspeed_config.dtype()})
            if self.args.non_blocking_data_copy:
                return data.to(**kwargs, non_blocking=True)
            else:
                return data.to(**kwargs)
        return data

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
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
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.pipelining_fwd_bwd:
            self.htcore.mark_step()

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        else:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if args.deepspeed:
                from deepspeed.runtime.activation_checkpointing.checkpointing import CheckpointFunction

                # HACK because outputs should always be tuples
                def hpu_deepspeed_checkpointing(function, *checkpoint_args):
                    """DeepSpeed acitvation checkpointing."""
                    all_outputs = []
                    CheckpointFunction.apply(function, all_outputs, *checkpoint_args)
                    # Always return a tuple
                    # When all_outputs contains only one element, DeepSpeed returns this element instead of a tuple
                    # which is not consistent with some models. See https://github.com/microsoft/DeepSpeed/issues/1057.
                    return tuple(all_outputs)

                torch.utils.checkpoint.checkpoint = hpu_deepspeed_checkpointing
            elif args.use_lazy_mode:
                from .gradient_checkpointing import checkpoint as lazy_mode_checkpointing

                torch.utils.checkpoint.checkpoint = lazy_mode_checkpointing

            # HACK for gradient checkpointing with T5
            # For T5, checkpointing is imported with `from torch.utils.checkpoint import checkpoint`: https://github.com/huggingface/transformers/blob/04ab5605fbb4ef207b10bf2772d88c53fc242e83/src/transformers/models/t5/modeling_t5.py#L27
            # Whereas for other models we do `import torch.utils.checkpoint`
            # So monkey patching at Torch's level does not work
            if self.model.config.model_type == "t5":
                import transformers.models.t5.modeling_t5 as modeling_t5

                modeling_t5.checkpoint = torch.utils.checkpoint.checkpoint

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        if self.gaudi_config.use_fused_clip_norm:
            try:
                from habana_frameworks.torch.hpex.normalization import FusedClipNorm
            except ImportError as error:
                error.msg = (
                    f"Could not import 'FusedClipNorm' from 'habana_frameworks.torch.hpex.normalization'. {error.msg}."
                )
                raise error
            self.FusedNorm = FusedClipNorm(
                model.parameters(),
                args.max_grad_norm,
            )

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size:,}")
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
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
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
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # In multi-worker training: broadcast model parameters from worker:0 to all the others.
        # This must be done manually unless DistributedDataParallel is used.
        if self.args.local_rank != -1 and self.args.distribution_strategy == "fast_ddp":
            logger.debug(
                f"Broadcasting the model parameters to assure that each of {self.args.world_size} workers start the training from the same point."
            )
            for param in model.parameters():
                torch.distributed.broadcast(param.data, src=0)

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        self._zero_model_grad(model)

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if version.parse(torch.__version__) < version.parse("1.11") or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        if self.args.adjust_throughput:
            self.log_evaluate_save_time = 0
        else:
            self.log_evaluate_save_time = None

        hb_profiler = HabanaProfile(warmup=self.args.profiling_warmup_steps, active=self.args.profiling_steps)
        hb_profiler.start()

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1

            for step, inputs in enumerate(epoch_iterator):
                if (
                    args.throughput_warmup_steps > 0
                    and (args.throughput_warmup_steps * args.gradient_accumulation_steps)
                    == epoch * steps_in_epoch + step
                ):
                    start_time_after_warmup = time.time()

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

                # Proceed with forward and backward passes.
                if (
                    args.distribution_strategy == "ddp"
                    and ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                is_optimization_step = (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                )

                if args.local_rank != -1 and args.distribution_strategy == "fast_ddp" and is_optimization_step:
                    all_reduce_gradients(
                        model, use_hpu_graph=True
                    )  # use HPU graphs for gradient fusion regardless of args.use_hpu_graphs_for_training setting

                if args.logging_nan_inf_filter and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))
                if args.use_lazy_mode:
                    self.htcore.mark_step()

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if is_optimization_step:
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        elif self.gaudi_config.use_fused_clip_norm and args.use_habana:
                            self.FusedNorm.clip_norm(model.parameters())
                        else:
                            # Revert to normal clipping otherwise
                            if (
                                args.use_habana
                                and (not (self.use_hpu_amp or self.use_cpu_amp))
                                and self.gaudi_config.use_habana_mixed_precision
                            ):
                                with self.hmp.disable_casts():
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif (
                        args.use_habana
                        and self.gaudi_config.use_habana_mixed_precision
                        and (not self.gaudi_config.use_fused_adam)
                        and (not (self.use_hpu_amp or self.use_cpu_amp))
                    ):
                        with self.hmp.disable_casts():
                            self.optimizer.step()
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    self._zero_model_grad(model)

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    if args.use_lazy_mode:
                        self.htcore.mark_step()
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                hb_profiler.step()
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        hb_profiler.stop()

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if args.local_rank != -1:
                torch.distributed.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        # Warmup steps are removed from the calculation of speed metrics
        num_samples_for_speed_metrics = num_train_samples - args.throughput_warmup_steps * total_train_batch_size
        num_steps_for_speed_metrics = self.state.max_steps - args.throughput_warmup_steps
        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_samples_for_speed_metrics,
            num_steps=num_steps_for_speed_metrics,
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
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

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

        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if self.args.use_habana:
            if self.args.local_rank != -1:
                self.hpu_random.set_rng_state(checkpoint_rng_state["hpu"])
            else:
                try:
                    self.hpu_random.set_rng_state_all(checkpoint_rng_state["hpu"])
                except Exception as e:
                    logger.info(
                        f"Didn't manage to set back the RNG states of the HPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            # This block is exectuted by the main process only
            optim_dict = self.optimizer.state_dict()
            scheduler_dict = self.lr_scheduler.state_dict()
            if self.do_grad_scaling:
                scaler_dict = self.scaler.state_dict()
            if self.args.use_habana:
                # Move the state dict from HPU to CPU before saving
                optim_dict = to_device_dtype(optim_dict, target_device=torch.device("cpu"))
                scheduler_dict = to_device_dtype(scheduler_dict, target_device=torch.device("cpu"))
                if self.do_grad_scaling:
                    scaler_dict = to_device_dtype(scaler_dict, target_device=torch.device("cpu"))
            torch.save(optim_dict, os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(scheduler_dict, os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(scaler_dict, os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }

        if self.args.use_habana:
            if self.args.local_rank == -1:
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

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

        # Synchronize all processes after saving the current checkpoint
        if self.args.local_rank != -1 and self.args.use_habana:
            torch.distributed.barrier()

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if self.deepspeed:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            return

        if os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME)) and os.path.isfile(
            os.path.join(checkpoint, SCHEDULER_NAME)
        ):
            # Load in optimizer and scheduler states
            map_location = "cpu" if self.args.use_habana else self.args.device

            self.optimizer.load_state_dict(
                torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location=map_location)
            )

            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(
                    torch.load(os.path.join(checkpoint, SCHEDULER_NAME), map_location=map_location)
                )
            reissue_pt_warnings(caught_warnings)

            if self.do_grad_scaling and os.path.isfile(os.path.join(checkpoint, SCALER_NAME)):
                self.scaler.load_state_dict(
                    torch.load(os.path.join(checkpoint, SCALER_NAME), map_location=map_location)
                )

            # Move optimizer state to HPU
            if self.args.use_habana:
                to_device_dtype(self.optimizer.state.values(), target_device=torch.device("hpu"))
                if self.do_grad_scaling:
                    to_device_dtype(self.scaler.state.values(), target_device=torch.device("hpu"))

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        # Do not use HPU graphs if the training is ongoing because it detaches gradients
        if args.use_hpu_graphs_for_inference and not self.is_in_train:
            logger.info("Using HPU graphs for inference.")
            # Do not wrap the model in HPU graphs if it has already been done
            if not self.already_wrapped_for_hpu_graphs:
                from habana_frameworks.torch.hpu import wrap_in_hpu_graph

                model = wrap_in_hpu_graph(model)
                self.already_wrapped_for_hpu_graphs = True

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
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
        # losses/preds/labels on HPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            # Save the logits dtype since we need to convert them into floats during the process
            # They will be converted back into their original dtype right before computing metrics
            if logits is not None:
                logits_dtype = get_dtype(logits)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                if args.use_habana and logits_dtype != "float32":
                    logits = to_device_dtype(logits, target_dtype=torch.float32)
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    if args.use_habana and logits_dtype != "float32":
                        preds_host = to_device_dtype(preds_host, target_dtype=torch.float32)
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

            # nested concat does accumulation on tensors of variable length.
            # Added mark step here to avoid graph recompile
            if args.use_lazy_mode:
                self.htcore.mark_step()

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            if args.use_habana and logits_dtype != "float32":
                preds_host = to_device_dtype(preds_host, target_dtype=torch.float32)
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

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

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Convert predictions back into their original dtype if necessary
        if all_preds is not None:
            all_preds = convert_into_dtypes(all_preds, logits_dtype)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
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
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
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
                    loss = loss.mean().detach()

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

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
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

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        # Do not use HPU graphs if the training is ongoing because it detaches gradients
        if args.use_hpu_graphs_for_inference and not self.is_in_train:
            logger.info("Using HPU graphs for inference.")
            # Do not wrap the model in HPU graphs if it has already been done
            if not self.already_wrapped_for_hpu_graphs:
                from habana_frameworks.torch.hpu import wrap_in_hpu_graph

                model = wrap_in_hpu_graph(model)
                self.already_wrapped_for_hpu_graphs = True

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Batch size = {batch_size}")
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
        inputs_host: Union[torch.Tensor, List[torch.Tensor]] = None

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
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

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

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
                    inputs_gatherer.add_arrays(self._gather_and_numpify(inputs_host, "eval_inputs_ids"))

                # Set back to None to begin a new accumulation
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

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=preds, label_ids=label_ids, inputs=inputs_ids)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
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

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.
        Will only save from the main process.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)
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

        if state_dict is None:
            state_dict = self.model.state_dict()
        if state_dict and self.args.use_habana:
            # state_dict items have to be saved on the CPU
            state_dict = to_device_dtype(state_dict, target_device=torch.device("cpu"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        self.gaudi_config.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _push_from_checkpoint(self, checkpoint_folder):
        # Only push from one node.
        if not self.is_world_process_zero() or self.args.hub_strategy == HubStrategy.END:
            return
        # If we haven't finished the last push, we don't do this one.
        if self.push_in_progress is not None and not self.push_in_progress.is_done:
            return

        output_dir = self.args.output_dir
        # To avoid a new synchronization of all model weights, we just copy the file from the checkpoint folder
        modeling_files = [CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME, GAUDI_CONFIG_NAME]
        for modeling_file in modeling_files:
            if os.path.isfile(os.path.join(checkpoint_folder, modeling_file)):
                shutil.copy(os.path.join(checkpoint_folder, modeling_file), os.path.join(output_dir, modeling_file))
        # Saving the tokenizer is fast and we don't know how many files it may have spawned, so we resave it to be sure.
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        # Same for the training arguments
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        try:
            if self.args.hub_strategy == HubStrategy.CHECKPOINT:
                # Temporarily move the checkpoint just saved for the push
                tmp_checkpoint = os.path.join(output_dir, "last-checkpoint")
                # We have to remove the "last-checkpoint" dir if it exists, otherwise the checkpoint is moved as a
                # subfolder.
                if os.path.isdir(tmp_checkpoint):
                    shutil.rmtree(tmp_checkpoint)
                shutil.move(checkpoint_folder, tmp_checkpoint)

            if self.args.save_strategy == IntervalStrategy.STEPS:
                commit_message = f"Training in progress, step {self.state.global_step}"
            else:
                commit_message = f"Training in progress, epoch {int(self.state.epoch)}"
            _, self.push_in_progress = self.repo.push_to_hub(
                commit_message=commit_message, blocking=False, auto_lfs_prune=True
            )
        finally:
            if self.args.hub_strategy == HubStrategy.CHECKPOINT:
                # Move back the checkpoint to its place
                shutil.move(tmp_checkpoint, checkpoint_folder)

    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        best_safe_model_path = os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_NAME)
        model = self.model
        if os.path.exists(best_model_path) or os.path.exists(best_safe_model_path):
            # TODO: the code below does not work with Habana DeepSpeed
            # if self.deepspeed:

            #     if self.model_wrapped is not None:
            #         # this removes the pre-hooks from the previous engine
            #         self.model_wrapped.destroy()
            #         self.model_wrapped = None

            #     # temp hack until Deepspeed fixes the problem with resume from an existing engine that did some stepping
            #     deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
            #         self,
            #         num_training_steps=self.args.max_steps,
            #         resume_from_checkpoint=self.state.best_model_checkpoint,
            #     )
            #     self.model = deepspeed_engine.module
            #     self.model_wrapped = deepspeed_engine
            #     self.deepspeed = deepspeed_engine
            #     self.optimizer = optimizer
            #     self.lr_scheduler = lr_scheduler
            # else:
            # We load the model state dict on the CPU to avoid an OOM error.
            if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
            else:
                state_dict = torch.load(best_model_path, map_location="cpu")

            # If the model is on the GPU, it still works!
            load_result = model.load_state_dict(state_dict, strict=False)
            self._issue_warnings_after_load(load_result)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.
        Subclass and override this method to inject custom behavior.
        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        mem_stats = get_hpu_memory_stats()
        logs.update(mem_stats)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _move_model_to_device(self, model, device):
        model = model.to(device)
        # Moving a model to HPU disconnects the tied weights, so we have to retie them.
        if self.args.use_habana and hasattr(model, "tie_weights"):
            model.tie_weights()

    def autocast_smart_context_manager(self, cache_enabled: Optional[bool] = True):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation. Modified by Habana to enable using `autocast` on Gaudi devices.
        """
        if self.use_cpu_amp:
            ctx_manager = torch.cpu.amp.autocast(cache_enabled=cache_enabled, dtype=torch.bfloat16)
        elif self.use_hpu_amp:
            ctx_manager = torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True)
        else:
            import contextlib

            ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()
        return ctx_manager

    def _zero_model_grad(self, model):
        if hasattr(model, "_zero_grad_kwargs"):
            model.zero_grad(**model._zero_grad_kwargs)
        else:
            # Optimization based on setting gradients to None (instead of zeroing them out) may only be used when gradients are not recorded using HPU graphs.
            # HPU graphs rely on fixed tensors - setting gradients to None will enforce their re-allocation during the backward pass each step.
            set_to_none = (
                self.args.local_rank == -1 or self.args.distribution_strategy == "ddp"
            ) and not self.args.use_hpu_graphs_for_training

            try:
                model.zero_grad(set_to_none=set_to_none)
                model._zero_grad_kwargs = {"set_to_none": set_to_none}
            except TypeError:
                model.zero_grad()
                model._zero_grad_kwargs = {}
