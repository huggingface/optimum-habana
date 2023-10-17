# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

import contextlib
import math
import os
import sys
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import make_dataclass

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.scheduler import AcceleratedScheduler
from accelerate.state import GradientState
from accelerate.tracking import GeneralTracker, filter_trackers
from accelerate.utils import (
    AutocastKwargs,
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    DistributedType,
    DynamoBackend,
    FP8RecipeKwargs,
    FullyShardedDataParallelPlugin,
    GradientAccumulationPlugin,
    GradScalerKwargs,
    InitProcessGroupKwargs,
    KwargsHandler,
    LoggerType,
    MegatronLMPlugin,
    PrecisionType,
    ProjectConfiguration,
    RNGType,
    TorchDynamoPlugin,
    is_deepspeed_available,
    parse_choice_from_env,
)
from accelerate.utils.operations import _gpu_gather
from torch.optim.lr_scheduler import LRScheduler


if is_deepspeed_available():
    from accelerate.utils import (
        DeepSpeedEngineWrapper,
        DeepSpeedOptimizerWrapper,
        DeepSpeedSchedulerWrapper,
        DummyOptim,
        DummyScheduler,
    )

from .state import GaudiAcceleratorState, GaudiPartialState
from .utils import GaudiDistributedType


logger = get_logger(__name__)


class GaudiAccelerator(Accelerator):
    """
    Adapted from: https://github.com/huggingface/accelerate/blob/8514c35192ac9762920f1ab052e5cea4c0e46eeb/src/accelerate/accelerator.py#L145
    """

    def __init__(
        self,
        device_placement: bool = True,
        split_batches: bool = False,
        mixed_precision: PrecisionType | str | None = None,
        gradient_accumulation_steps: int = 1,
        cpu: bool = False,
        deepspeed_plugin: DeepSpeedPlugin | None = None,
        fsdp_plugin: FullyShardedDataParallelPlugin | None = None,
        megatron_lm_plugin: MegatronLMPlugin | None = None,
        rng_types: list[str | RNGType] | None = None,
        log_with: str | LoggerType | GeneralTracker | list[str | LoggerType | GeneralTracker] | None = None,
        project_dir: str | os.PathLike | None = None,
        project_config: ProjectConfiguration | None = None,
        gradient_accumulation_plugin: GradientAccumulationPlugin | None = None,
        dispatch_batches: bool | None = None,
        even_batches: bool = True,
        step_scheduler_with_optimizer: bool = True,
        kwargs_handlers: list[KwargsHandler] | None = None,
        dynamo_backend: DynamoBackend | str | None = None,
        distribution_strategy: str = None,
    ):
        if project_config is not None:
            self.project_configuration = project_config
        else:
            self.project_configuration = ProjectConfiguration(project_dir=project_dir)
        if project_dir is not None and self.project_dir is None:
            self.project_configuration.set_directories(project_dir)
        if mixed_precision is not None:
            mixed_precision = str(mixed_precision)
            if mixed_precision not in PrecisionType:
                raise ValueError(
                    f"Unknown mixed_precision mode: {mixed_precision}. Choose between {PrecisionType.list()}"
                )
            elif mixed_precision == "fp16":
                raise ValueError("fp16 is not supported on Habana Gaudi.")

        dynamo_plugin = TorchDynamoPlugin() if dynamo_backend is None else TorchDynamoPlugin(backend=dynamo_backend)

        if deepspeed_plugin is None:  # init from env variables
            deepspeed_plugin = (
                DeepSpeedPlugin() if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true" else None
            )
        else:
            if not isinstance(deepspeed_plugin, DeepSpeedPlugin):
                raise TypeError("`deepspeed_plugin` must be an `accelerate.utils.DeepSpeedPlugin` object.")
            os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"  # use DeepSpeed if plugin is provided
        if deepspeed_plugin:
            if not is_deepspeed_available():
                raise ImportError(
                    "DeepSpeed is not installed => run `pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.12.0`."
                )

            mixed_precision = (
                os.environ.get("ACCELERATE_MIXED_PRECISION", "no") if mixed_precision is None else mixed_precision
            )
            deepspeed_plugin.set_mixed_precision(mixed_precision)
            deepspeed_plugin.set_deepspeed_weakref()

        # Kwargs handlers
        self.ddp_handler = None
        self.scaler_handler = None
        self.init_handler = None
        self.fp8_recipe_handler = None
        self.autocast_handler = None
        if kwargs_handlers is not None:
            for handler in kwargs_handlers:
                assert isinstance(
                    handler, KwargsHandler
                ), f"Unsupported kwargs handler passed: {handler}, must be one that inherits `accelerate.utils.KwargsHandler`."
                if isinstance(handler, DistributedDataParallelKwargs):
                    if self.ddp_handler is not None:
                        raise ValueError("You can only pass one `DistributedDataParallelKwargs` in `kwargs_handler`.")
                    else:
                        self.ddp_handler = handler
                elif isinstance(handler, GradScalerKwargs):
                    if self.scaler_handler is not None:
                        raise ValueError("You can only pass one `GradScalerKwargs` in `kwargs_handler`.")
                    else:
                        self.scaler_handler = handler
                elif isinstance(handler, InitProcessGroupKwargs):
                    if self.init_handler is not None:
                        raise ValueError("You can only pass one `InitProcessGroupKwargs` in `kwargs_handler`.")
                    else:
                        self.init_handler = handler
                elif isinstance(handler, FP8RecipeKwargs):
                    if self.fp8_recipe_handler is not None:
                        raise ValueError("You can only pass one `FP8RecipeKwargs` in `kwargs_handler`.")
                    else:
                        self.fp8_recipe_handler = handler
                elif isinstance(handler, AutocastKwargs):
                    if self.autocast_handler is not None:
                        raise ValueError("You can only pass one `AutocastKwargs` in `kwargs_handler`.")
                    else:
                        self.autocast_handler = handler

        kwargs = self.init_handler.to_kwargs() if self.init_handler is not None else {}
        self.state = GaudiAcceleratorState(
            mixed_precision=mixed_precision,
            cpu=cpu,
            dynamo_plugin=dynamo_plugin,
            deepspeed_plugin=deepspeed_plugin,
            fsdp_plugin=fsdp_plugin,
            megatron_lm_plugin=megatron_lm_plugin,
            _from_accelerator=True,
            **kwargs,
        )

        trackers = filter_trackers(log_with, self.logging_dir)
        if len(trackers) < 1 and log_with is not None:
            warnings.warn(f"`log_with={log_with}` was passed but no supported trackers are currently installed.")
        self.log_with = trackers

        if (
            (mixed_precision != "bf16")
            and getattr(self.state, "downcast_bfloat", False)
            and (self.state.distributedType != DistributedType.TPU)
        ):
            raise ValueError("Can only use `downcast_bf16` when using `mixed_precision='bf16'` and on a TPU")

        if gradient_accumulation_plugin is not None:
            if gradient_accumulation_steps != 1:
                raise ValueError(
                    "You can only pass one of `gradient_accumulation_steps` and `gradient_accumulation_plugin`. Please only pass in the created `GradientAccumulationPlugin` object."
                )
        else:
            gradient_accumulation_steps = int(
                parse_choice_from_env("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", gradient_accumulation_steps)
            )
            gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=gradient_accumulation_steps)
        self.gradient_state = GradientState(
            gradient_accumulation_plugin=gradient_accumulation_plugin,
        )

        self.device_placement = device_placement
        self.split_batches = split_batches
        self.dispatch_batches = dispatch_batches
        self.even_batches = even_batches
        self.step_scheduler_with_optimizer = step_scheduler_with_optimizer

        # Mixed precision attributes
        self.scaler = None
        self.native_amp = self.state.mixed_precision == "bf16"

        # Start of internal step tracking
        self.step = 0

        # Internal references to the training objects
        self._optimizers = []
        self._models = []
        self._schedulers = []
        self._dataloaders = []
        self._custom_objects = []

        # Hooks
        self._load_model_state_pre_hook = OrderedDict()
        self._save_model_state_pre_hook = OrderedDict()

        # RNG Types
        self.rng_types = rng_types
        if self.rng_types is None:
            self.rng_types = ["generator"]

        # Set a flag tensor for early stopping and other breakpoints
        self.flag_tensor = None

        self._distribution_strategy = distribution_strategy

    @property
    def use_fp16(self):
        raise ValueError("fp16 is not supported on Habana Gaudi.")

    def prepare_model(self, model: torch.nn.Module, device_placement: bool = None, evaluation_mode: bool = False):
        """
        Prepares a PyTorch model for training in any distributed setup. It is recommended to use
        [`Accelerator.prepare`] instead.

        Args:
            model (`torch.nn.Module`):
                A PyTorch model to prepare. You don't need to prepare a model if it is used only for inference without
                any kind of mixed precision
            device_placement (`bool`, *optional*):
                Whether or not to place the model on the proper device. Will default to `self.device_placement`.
            evaluation_mode (`bool`, *optional*, defaults to `False`):
                Whether or not to set the model for evaluation only, by just applying mixed precision and
                `torch.compile` (if configured in the `Accelerator` object).

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> # Assume a model is defined
        >>> model = accelerator.prepare_model(model)
        ```
        """
        if device_placement is None:
            device_placement = self.device_placement and self.distributed_type != DistributedType.FSDP
            if not evaluation_mode and self.distributed_type == GaudiDistributedType.MULTI_HPU:
                device_placement = None
        self._models.append(model)

        # TODO: Look at enabling native TP training directly with a proper config
        if (
            self.verify_device_map(model)
            and self.distributed_type != DistributedType.NO
            and os.environ.get("ACCELERATE_BYPASS_DEVICE_MAP", "false") != "true"
        ):
            raise ValueError(
                "You can't train a model that has been loaded with `device_map='auto'` in any distributed mode."
                " Please rerun your script specifying `--num_processes=1` or by launching with `python {{myscript.py}}`."
            )

        if (getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)) and getattr(
            model, "hf_device_map", False
        ):
            model_devices = set(model.hf_device_map.values())
            if len(model_devices) > 1 and self.distributed_type != DistributedType.NO:
                raise ValueError(
                    "You can't train a model that has been loaded in 8-bit precision on multiple devices in any distributed mode."
                    " In order to use 8-bit models that have been loaded across multiple GPUs the solution is to use Naive Pipeline Parallelism."
                    " Therefore you should not specify that you are under any distributed regime in your accelerate config."
                )
            current_device = list(model_devices)[0]
            current_device_index = current_device.index if isinstance(current_device, torch.device) else current_device

            if torch.device(current_device_index) != self.device:
                # if on the first device (GPU 0) we don't care
                if (self.device.index is not None) or (current_device_index != 0):
                    raise ValueError(
                        "You can't train a model that has been loaded in 8-bit precision on a different device than the one "
                        "you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device()}"
                        "you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device() or device_map={'':torch.xpu.current_device()}"
                    )

            if "cpu" in model_devices or "disk" in model_devices:
                raise ValueError(
                    "You can't train a model that has been loaded in 8-bit precision with CPU or disk offload."
                )
        elif device_placement and not self.verify_device_map(model):
            model = model.to(self.device)

        # The following block is commented because forward+backward+loss is already wrapped with autocast in Trainer
        # if self.native_amp:
        #     model._original_forward = model.forward
        #     model_forward_func = model.forward.__func__ if hasattr(model.forward, "__func__") else model.forward
        #     if self.mixed_precision == "bf16":
        #         new_forward = torch.autocast(device_type=self.state.device.type, dtype=torch.bfloat16)(
        #             model_forward_func
        #         )

        #     if hasattr(model.forward, "__func__"):
        #         model.forward = MethodType(new_forward, model)
        #         model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)
        #     else:
        #         model.forward = convert_outputs_to_fp32(new_forward)
        # FP8 is not supported on Gaudi2 yet
        # elif self.mixed_precision == "fp8":
        #     if not has_transformer_engine_layers(model):
        #         with torch.no_grad():
        #             convert_model(model)
        #         model._converted_to_transformer_engine = True
        #     model._original_forward = model.forward

        #     kwargs = self.fp8_recipe_handler.to_kwargs() if self.fp8_recipe_handler is not None else {}
        #     if "fp8_format" in kwargs:
        #         kwargs["fp8_format"] = getattr(te_recipe.Format, kwargs["fp8_format"])
        #     fp8_recipe = te_recipe.DelayedScaling(**kwargs)
        #     cuda_device_capacity = torch.cuda.get_device_capability()
        #     fp8_enabled = cuda_device_capacity[0] >= 9 or (
        #         cuda_device_capacity[0] == 8 and cuda_device_capacity[1] >= 9
        #     )
        #     if not fp8_enabled:
        #         logger.warn(
        #             f"The current device has compute capability of {cuda_device_capacity} which is "
        #             "insufficient for FP8 mixed precision training (requires a GPU Hopper/Ada Lovelace "
        #             "or higher, compute capability of 8.9 or higher). Will use FP16 instead."
        #         )
        #     model.forward = fp8_autocast(enabled=fp8_enabled, fp8_recipe=fp8_recipe)(model.forward)
        if not evaluation_mode:
            if self.distributed_type == GaudiDistributedType.MULTI_HPU and self._distribution_strategy != "fast_ddp":
                if any(p.requires_grad for p in model.parameters()):
                    kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                    model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
        # torch.compile should be called last.
        if self.state.dynamo_plugin.backend != DynamoBackend.NO:
            model = torch.compile(model, **self.state.dynamo_plugin.to_kwargs())
        return model

    def _prepare_deepspeed(self, *args):
        import deepspeed

        deepspeed_plugin = self.state.deepspeed_plugin

        is_dataloader_present = any(isinstance(obj, torch.utils.data.DataLoader) for obj in args)
        if deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto" or is_dataloader_present:
            result = [
                self._prepare_one(obj, first_pass=True) if isinstance(obj, torch.utils.data.DataLoader) else obj
                for obj in args
            ]

            batch_sizes = [obj.batch_size for obj in args if hasattr(obj, "batch_size")]
            if self.split_batches:
                batch_sizes = [batch_size // self.num_processes for batch_size in batch_sizes]

            if any(bs is None for bs in batch_sizes):
                raise ValueError(
                    "At least one of the dataloaders passed to `accelerate.prepare()` has `None` as batch size."
                    "Please set an integer value in `train_micro_batch_size_per_gpu` in the deepspeed config file"
                    "or assign integer value to `AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']`."
                )
            if len(batch_sizes) == 0:
                raise ValueError(
                    "When using DeepSpeed `accelerate.prepare()` requires you to pass at least one of training or evaluation dataloaders "
                    "or alternatively set an integer value in `train_micro_batch_size_per_gpu` in the deepspeed config file"
                    "or assign integer value to `AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']`."
                )

            batch_size_per_device = min(batch_sizes) if deepspeed_plugin.is_train_batch_min else max(batch_sizes)
            if len(batch_sizes) > 1:
                logger.info(
                    "Since you passed both train and evaluation dataloader, `is_train_batch_min` (here "
                    f"{deepspeed_plugin.is_train_batch_min} will decide the `train_batch_size` ({batch_size_per_device})."
                )
        else:
            batch_size_per_device = deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"]
            result = list(args)

        # handle `gradient_accumulation_steps` when the value is `auto`
        deepspeed_plugin.fill_match(
            "gradient_accumulation_steps",
            must_match=False,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        config_kwargs = {
            "train_micro_batch_size_per_gpu": batch_size_per_device,
            "train_batch_size": batch_size_per_device
            * deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"]
            * self.num_processes,
            "gradient_clipping": 1.0,
            "zero_optimization.stage3_gather_16bit_weights_on_model_save": False,
        }

        model = None
        optimizer = None
        scheduler = None
        for obj in result:
            if isinstance(obj, torch.nn.Module):
                model = obj
            elif isinstance(obj, (torch.optim.Optimizer, DummyOptim)):
                optimizer = obj
            elif (isinstance(obj, (LRScheduler, DummyScheduler))) or (
                type(obj).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
            ):
                scheduler = obj

        if optimizer is not None:
            if "optimizer" in deepspeed_plugin.deepspeed_config and not isinstance(optimizer, (DummyOptim)):
                raise ValueError(
                    "You cannot specify an optimizer in the config file and in the code at the same time. "
                    "Please remove the optimizer from the config file or "
                    "create `accelerate.utils.DummyOptim` in the code."
                )
            elif "optimizer" not in deepspeed_plugin.deepspeed_config and isinstance(optimizer, (DummyOptim)):
                raise ValueError(
                    "You cannot create a `DummyOptim` without specifying an optimizer in the config file."
                )

            if isinstance(optimizer, (torch.optim.Optimizer)):
                deepspeed_plugin.deepspeed_config["zero_allow_untested_optimizer"] = True

        if scheduler is not None:
            if "scheduler" in deepspeed_plugin.deepspeed_config and not isinstance(scheduler, (DummyScheduler)):
                raise ValueError(
                    "You cannot specify a scheduler in the config file and in the code at the same time. "
                    "Please remove the scheduler from the config file or "
                    "create `accelerate.utils.DummyScheduler` in the code."
                )
            elif (
                "scheduler" not in deepspeed_plugin.deepspeed_config
                and isinstance(scheduler, (DummyScheduler))
                and scheduler.lr_scheduler_callable is None
            ):
                raise ValueError(
                    "Either specify a scheduler in the config file or "
                    "pass in the `lr_scheduler_callable` parameter when using `accelerate.utils.DummyScheduler`."
                )

        if optimizer is not None and scheduler is not None:
            if isinstance(optimizer, (DummyOptim)) and not isinstance(scheduler, (DummyScheduler)):
                raise ValueError(
                    "You can only specify `accelerate.utils.DummyScheduler` in the code when using "
                    "`accelerate.utils.DummyOptim`."
                )

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None:
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        }
                    )

            if isinstance(optimizer, (DummyOptim)):
                config_kwargs.update(
                    {"optimizer.params.lr": optimizer.lr, "optimizer.params.weight_decay": optimizer.weight_decay}
                )
            if isinstance(scheduler, (DummyScheduler)) and scheduler.lr_scheduler_callable is None:
                max_lr = (
                    getattr(scheduler.optimizer, "lr", None)
                    if getattr(scheduler.optimizer, "defaults", None) is None
                    else scheduler.optimizer.defaults["lr"]
                )
                config_kwargs.update(
                    {
                        "scheduler.params.warmup_min_lr": 0,
                        "scheduler.params.warmup_max_lr": max_lr,
                        "scheduler.params.warmup_num_steps": scheduler.warmup_num_steps,
                    }
                )
                if scheduler.total_num_steps is not None:
                    config_kwargs["scheduler.params.total_num_steps"] = (
                        math.ceil(scheduler.total_num_steps / self.num_processes)
                        if not self.split_batches
                        else scheduler.total_num_steps
                    )
            deepspeed_plugin.deepspeed_config_process(must_match=False, **config_kwargs)
            self.deepspeed_config = deepspeed_plugin.deepspeed_config
            kwargs = {"model": model, "config_params": self.deepspeed_config}
            if optimizer is not None:
                if isinstance(optimizer, (DummyOptim)):
                    kwargs["model_parameters"] = optimizer.params
                    if isinstance(scheduler, (DummyScheduler)) and scheduler.lr_scheduler_callable is not None:
                        kwargs["lr_scheduler"] = scheduler.lr_scheduler_callable
                else:
                    if self.deepspeed_config["zero_optimization"].get("offload_optimizer", {}).get(
                        "device", "none"
                    ) != "none" and self.deepspeed_config.get("zero_force_ds_cpu_optimizer", True):
                        from deepspeed.ops.adam import DeepSpeedCPUAdam

                        defaults = {k: v for k, v in optimizer.defaults.items() if k in ["lr", "weight_decay"]}
                        optimizer = DeepSpeedCPUAdam(optimizer.param_groups, **defaults)
                    kwargs["optimizer"] = optimizer
                    if scheduler is not None:
                        if (
                            isinstance(scheduler, LRScheduler)
                            or type(scheduler).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
                        ):
                            kwargs["lr_scheduler"] = scheduler

            HabanaArgs = make_dataclass("HabanaArgs", [("use_hpu", bool), ("no_cuda", bool)])
            habana_args = HabanaArgs(
                use_hpu=True if self.device.type == "hpu" else False,
                no_cuda=True if self.device.type == "cpu" else False,
            )
            if habana_args.use_hpu:
                # This env variable is initialized here to make sure it is set to "true"
                # It should be done by the launcher but it does not work for multi-node runs
                os.environ["DEEPSPEED_USE_HPU"] = "true"

            engine, optimizer, _, lr_scheduler = deepspeed.initialize(**kwargs)
            if optimizer is not None:
                optimizer = DeepSpeedOptimizerWrapper(optimizer)
            if scheduler is not None:
                if lr_scheduler is None:
                    scheduler = AcceleratedScheduler(
                        scheduler,
                        optimizer,
                        step_with_optimizer=self.step_scheduler_with_optimizer,
                        split_batches=self.split_batches,
                    )
                else:
                    scheduler = DeepSpeedSchedulerWrapper(lr_scheduler, optimizer)

            for i in range(len(result)):
                if isinstance(result[i], torch.nn.Module):
                    result[i] = engine
                elif isinstance(result[i], (torch.optim.Optimizer, DummyOptim)):
                    result[i] = optimizer
                elif (isinstance(result[i], (LRScheduler, DummyScheduler))) or (
                    type(result[i]).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
                ):
                    result[i] = scheduler
            # pointing for deepspeed_engine_wrapped.backward()
            self.deepspeed_engine_wrapped = DeepSpeedEngineWrapper(engine)
            self._models.append(engine)
            if optimizer is not None:
                self._optimizers.append(optimizer)
            if scheduler is not None:
                self._schedulers.append(scheduler)
            if len(self._models) > 1:
                raise AssertionError(
                    "You can't use same `Accelerator()` instance with multiple models when using DeepSpeed"
                )
        return tuple(result)

    def gather(self, tensor):
        """
        Gather the values in *tensor* across all processes and concatenate them on the first dimension. Useful to
        regroup the predictions from all processes when doing evaluation.

        Note:
            This gather happens in all processes.

        Args:
            tensor (`torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`):
                The tensors to gather across all processes.

        Returns:
            `torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`: The gathered tensor(s). Note that the
            first dimension of the result is *num_processes* multiplied by the first dimension of the input tensors.

        Example:

        ```python
        >>> # Assuming four processes
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> process_tensor = torch.tensor([accelerator.process_index])
        >>> gathered_tensor = accelerator.gather(process_tensor)
        >>> gathered_tensor
        tensor([0, 1, 2, 3])
        ```
        """
        if GaudiPartialState().distributed_type in [GaudiDistributedType.MULTI_HPU, GaudiDistributedType.DEEPSPEED]:
            return _gpu_gather(tensor)
        else:
            return tensor

    def get_state_dict(self, model, unwrap=True):
        """
        Returns the state dictionary of a model sent through [`Accelerator.prepare`] potentially without full
        precision.

        Args:
            model (`torch.nn.Module`):
                A PyTorch model sent through [`Accelerator.prepare`]
            unwrap (`bool`, *optional*, defaults to `True`):
                Whether to return the original underlying state_dict of `model` or to return the wrapped state_dict

        Returns:
            `dict`: The state dictionary of the model potentially without full precision.

        Example:

        ```python
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> net = torch.nn.Linear(2, 2)
        >>> net = accelerator.prepare(net)
        >>> state_dict = accelerator.get_state_dict(net)
        ```
        """

        if self.distributed_type == DistributedType.DEEPSPEED:
            if self.deepspeed_config["zero_optimization"]["stage"] == 3:
                if model.zero_gather_16bit_weights_on_model_save():
                    state_dict = model._zero3_consolidated_16bit_state_dict()
                else:
                    raise ValueError(
                        "Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. "
                        "To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or "
                        "set `zero3_save_16bit_model` to True when using `accelerate config`. "
                        "To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights."
                    )
            else:
                from deepspeed.checkpoint.utils import clone_tensors_for_torch_save

                state_dict = clone_tensors_for_torch_save(self.unwrap_model(model).state_dict())
        else:
            if unwrap:
                model = self.unwrap_model(model)
            state_dict = model.state_dict()

        return state_dict

    @contextmanager
    def autocast(self, cache_enabled: bool = False):
        """
        Will apply automatic mixed-precision inside the block inside this context manager, if it is enabled. Nothing
        different will happen otherwise.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(mixed_precision="fp16")
        >>> with accelerator.autocast():
        ...     train()
        ```
        """
        if self.native_amp:
            autocast_context = torch.autocast(device_type=self.state.device.type, dtype=torch.bfloat16)
        else:
            autocast_context = contextlib.nullcontext()

        autocast_context.__enter__()
        yield
        autocast_context.__exit__(*sys.exc_info())
