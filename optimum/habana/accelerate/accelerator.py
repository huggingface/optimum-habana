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
import functools
import math
import os
import sys
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import make_dataclass
from types import MethodType

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.scheduler import AcceleratedScheduler
from accelerate.state import GradientState
from accelerate.tracking import GeneralTracker, filter_trackers
from accelerate.utils import (
    AutocastKwargs,
    DataLoaderConfiguration,
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    DistributedType,
    GradientAccumulationPlugin,
    GradScalerKwargs,
    InitProcessGroupKwargs,
    KwargsHandler,
    LoggerType,
    MegatronLMPlugin,
    PrecisionType,
    ProfileKwargs,
    ProjectConfiguration,
    RNGType,
    check_os_kernel,
    convert_outputs_to_fp32,
    is_deepspeed_available,
    is_torch_version,
    parse_choice_from_env,
)
from accelerate.utils.constants import FSDP_PYTORCH_VERSION
from accelerate.utils.operations import _gpu_gather
from accelerate.utils.other import is_compiled_module
from torch.optim.lr_scheduler import LRScheduler


if is_deepspeed_available():
    from accelerate.utils import (
        DeepSpeedEngineWrapper,
        DeepSpeedOptimizerWrapper,
        DeepSpeedSchedulerWrapper,
        DummyOptim,
        DummyScheduler,
    )

from .data_loader import gaudi_prepare_data_loader
from .state import GaudiAcceleratorState, GaudiPartialState
from .utils import (
    GaudiDistributedType,
    GaudiDynamoBackend,
    GaudiFP8RecipeKwargs,
    GaudiFullyShardedDataParallelPlugin,
    GaudiTorchDynamoPlugin,
    convert_model,
    get_fp8_recipe,
)


logger = get_logger(__name__)

# Sentinel values for defaults
_split_batches = object()
_dispatch_batches = object()
_even_batches = object()
_use_seedable_sampler = object()


class GaudiAccelerator(Accelerator):
    """
    Adapted from: https://github.com/huggingface/accelerate/blob/8514c35192ac9762920f1ab052e5cea4c0e46eeb/src/accelerate/accelerator.py#L145
    """

    def __init__(
        self,
        device_placement: bool = True,
        split_batches: bool = _split_batches,
        mixed_precision: PrecisionType | str | None = None,
        gradient_accumulation_steps: int = 1,
        cpu: bool = False,
        dataloader_config: DataLoaderConfiguration | None = None,
        deepspeed_plugin: DeepSpeedPlugin | None = None,
        fsdp_plugin: GaudiFullyShardedDataParallelPlugin | None = None,
        megatron_lm_plugin: MegatronLMPlugin | None = None,
        rng_types: list[str | RNGType] | None = None,
        log_with: str | LoggerType | GeneralTracker | list[str | LoggerType | GeneralTracker] | None = None,
        project_dir: str | os.PathLike | None = None,
        project_config: ProjectConfiguration | None = None,
        gradient_accumulation_plugin: GradientAccumulationPlugin | None = None,
        dispatch_batches: bool | None = _dispatch_batches,
        even_batches: bool = _even_batches,
        use_seedable_sampler: bool = _use_seedable_sampler,
        step_scheduler_with_optimizer: bool = True,
        kwargs_handlers: list[KwargsHandler] | None = None,
        dynamo_backend: GaudiDynamoBackend | str | None = None,
        distribution_strategy: str = None,
        force_autocast: bool = False,
    ):
        self.trackers = []
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

        dynamo_plugin = (
            GaudiTorchDynamoPlugin() if dynamo_backend is None else GaudiTorchDynamoPlugin(backend=dynamo_backend)
        )

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
                    "DeepSpeed is not installed => run `pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.16.0`."
                )

            mixed_precision = (
                os.environ.get("ACCELERATE_MIXED_PRECISION", "no") if mixed_precision is None else mixed_precision
            )
            deepspeed_plugin.set_mixed_precision(mixed_precision)
            deepspeed_plugin.set_deepspeed_weakref()

        if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true" or isinstance(
            fsdp_plugin, GaudiFullyShardedDataParallelPlugin
        ):
            import importlib.metadata

            torch_version = importlib.metadata.version("torch")
            torch_version = torch_version[5:]
            if is_torch_version("<", FSDP_PYTORCH_VERSION + torch_version):
                raise ValueError(f"FSDP requires PyTorch >= {FSDP_PYTORCH_VERSION}")

        if fsdp_plugin is None:  # init from env variables
            fsdp_plugin = (
                GaudiFullyShardedDataParallelPlugin()
                if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true"
                else None
            )
        else:
            if not isinstance(fsdp_plugin, GaudiFullyShardedDataParallelPlugin):
                raise TypeError("`fsdp_plugin` must be a GaudiFullyShardedDataParallelPlugin object.")
            os.environ["ACCELERATE_USE_FSDP"] = "true"  # use FSDP if plugin is provided

        # Kwargs handlers
        self.ddp_handler = None
        self.scaler_handler = None
        self.init_handler = None
        self.fp8_recipe_handler = None
        self.autocast_handler = None
        self.profile_handler = None
        self.has_lomo_optimizer = False

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
                elif isinstance(handler, GaudiFP8RecipeKwargs):
                    if self.fp8_recipe_handler is not None:
                        raise ValueError("You can only pass one `GaudiFP8RecipeKwargs` in `kwargs_handler`.")
                    else:
                        self.fp8_recipe_handler = handler
                elif isinstance(handler, AutocastKwargs):
                    if self.autocast_handler is not None:
                        raise ValueError("You can only pass one `AutocastKwargs` in `kwargs_handler`.")
                    else:
                        self.autocast_handler = handler
                elif isinstance(handler, ProfileKwargs):
                    if self.profile_handler is not None:
                        raise ValueError("You can only pass one `ProfileKwargs` in `kwargs_handler`.")
                    else:
                        self.profile_handler = handler

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

        self.delayed_fp8_autocast = False
        if self.fp8_recipe_handler is not None:
            # We already check if FP8 is available during `self.state`
            if self.state.mixed_precision != "fp8":
                raise ValueError("Passing in a `FP8RecipeKwargs` object requires setting `mixed_precision='fp8'`.")
            self.delayed_fp8_autocast = self.fp8_recipe_handler.backend == "TE" and self.distributed_type in (
                DistributedType.MULTI_GPU,
                DistributedType.FSDP,
            )

        if self.state.is_fp8_enabled:
            if self.fp8_recipe_handler is None:
                self.fp8_recipe_handler = GaudiFP8RecipeKwargs()
            # Handling FP8 recipe creation in init since both `prepare_model` and `_prepare_deepspeed` require it.
            # (Base accelerator handles this in `prepare_model` function)
            self.fp8_recipe_handler = get_fp8_recipe(self.fp8_recipe_handler)

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
        if dataloader_config is None:
            dataloader_config = DataLoaderConfiguration()
        self.dataloader_config = dataloader_config
        # Deal with deprecated args
        # TODO: Remove in v1.0.0
        deprecated_dl_args = {}
        if dispatch_batches is not _dispatch_batches:
            deprecated_dl_args["dispatch_batches"] = dispatch_batches
            self.dataloader_config.dispatch_batches = dispatch_batches
        if split_batches is not _split_batches:
            deprecated_dl_args["split_batches"] = split_batches
            self.dataloader_config.split_batches = split_batches
        if even_batches is not _even_batches:
            deprecated_dl_args["even_batches"] = even_batches
            self.dataloader_config.even_batches = even_batches
        if use_seedable_sampler is not _use_seedable_sampler:
            deprecated_dl_args["use_seedable_sampler"] = use_seedable_sampler
            self.dataloader_config.use_seedable_sampler = use_seedable_sampler
        if len(deprecated_dl_args) > 0:
            values = ", ".join([f"{k}={v}" for k, v in deprecated_dl_args.items()])
            warnings.warn(
                f"Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: {deprecated_dl_args.keys()}. "
                "Please pass an `accelerate.DataLoaderConfiguration` instead: \n"
                f"dataloader_config = DataLoaderConfiguration({values})",
                FutureWarning,
            )
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

        self.force_autocast = force_autocast

        check_os_kernel()

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

        # The following block is executed only when force_autocast is True
        # because forward+backward+loss is already wrapped with autocast in Trainer
        if self.native_amp and self.force_autocast:
            model._original_forward = model.forward
            model_forward_func = model.forward.__func__ if hasattr(model.forward, "__func__") else model.forward
            new_forward = torch.autocast(device_type=self.state.device.type, dtype=torch.bfloat16)(model_forward_func)
            if hasattr(model.forward, "__func__"):
                model.forward = MethodType(new_forward, model)
                model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)
            else:
                model.forward = convert_outputs_to_fp32(new_forward)

        if self.state.is_fp8_enabled:
            model = convert_model(model)

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
            elif len(model_devices) == 1:
                current_device = list(model_devices)[0]
                current_device_index = (
                    current_device.index if isinstance(current_device, torch.device) else current_device
                )

                if torch.device(current_device_index) != self.device:
                    # if on the first device (GPU 0) we don't care
                    if (self.device.index is not None) or (current_device_index != 0):
                        raise ValueError(
                            "You can't train a model that has been loaded in 8-bit precision on a different device than the one "
                            "you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device() or device_map={'':torch.xpu.current_device()}"
                        )

            if "cpu" in model_devices or "disk" in model_devices:
                raise ValueError(
                    "You can't train a model that has been loaded in 8-bit precision with CPU or disk offload."
                )
        elif device_placement and not self.verify_device_map(model):
            model = model.to(self.device)
        if not evaluation_mode:
            if self.distributed_type == GaudiDistributedType.MULTI_HPU and self._distribution_strategy != "fast_ddp":
                if any(p.requires_grad for p in model.parameters()):
                    kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                    model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
                    if self.ddp_handler is not None:
                        self.ddp_handler.register_comm_hook(model)
            elif self.distributed_type == GaudiDistributedType.FSDP:
                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

                # Check if the model is already a FSDP model due to `Manual Wrapping` and if so,
                # don't wrap it again
                # In case the model is already compiled using PyTorch 2.0 and the wrapped model in it
                # is a FSDP model, don't wrap it again
                is_type_fsdp = isinstance(model, FSDP) or (
                    is_compiled_module(model) and isinstance(model._orig_mod, FSDP)
                )

                if not is_type_fsdp:
                    self.state.fsdp_plugin.set_auto_wrap_policy(model)
                    fsdp_plugin = self.state.fsdp_plugin
                    kwargs = {
                        "sharding_strategy": fsdp_plugin.sharding_strategy,
                        "cpu_offload": fsdp_plugin.cpu_offload,
                        "auto_wrap_policy": fsdp_plugin.auto_wrap_policy,
                        "mixed_precision": fsdp_plugin.mixed_precision_policy,
                        "sync_module_states": fsdp_plugin.sync_module_states,
                        "backward_prefetch": fsdp_plugin.backward_prefetch,
                        "forward_prefetch": fsdp_plugin.forward_prefetch,
                        "use_orig_params": fsdp_plugin.use_orig_params,
                        "param_init_fn": fsdp_plugin.param_init_fn,
                        "ignored_modules": fsdp_plugin.ignored_modules,
                        "limit_all_gathers": fsdp_plugin.limit_all_gathers,
                        "device_id": torch.device("hpu", torch.hpu.current_device()),
                    }
                    model = FSDP(model, **kwargs)
                    if fsdp_plugin.activation_checkpointing:
                        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                            CheckpointImpl,
                            apply_activation_checkpointing,
                            checkpoint_wrapper,
                        )

                        apply_activation_checkpointing(
                            model,
                            checkpoint_wrapper_fn=functools.partial(
                                checkpoint_wrapper,
                                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                            ),
                            auto_wrap_policy=fsdp_plugin.auto_wrap_policy,
                        )
                # In the event the model had been loaded in low precision, but
                # mixed precision had also been activated, then we follow DeepSpeed's
                # strategy to hold the parameters in full precision.
                # - assume that trainer.args.bf16 and trainer.args.fp16 are already checked against
                #   fsdp_plugin.mixed_precision_policy.
                # - NOTE: we do not check the mixed_precision attribute on the FSDP root wrapper.
                #   * this attribute will always set by init_utils.init_core_state so its always not None.
                #   * mixed_precision.param_dtype only regards _fwd_bwd_param_dtype
                #   * if model is loaded in 16bit, and even if mixed_precision.param_dtype is None,
                #     we sill want to upcast the flat_param.
                if self.mixed_precision != "no":  # if mixed precision is set
                    upcasted_log = []
                    for module in FSDP.fsdp_modules(model):
                        # Referencing DeepSpeed Zero3
                        # - in Init, params are converted to 16bit while partitioning.
                        # - in accelerator.prepare, deepspeed.initalize is called to:
                        #   * creates the DeepSpeeedEngine.
                        #   * since zero_optimization() is True , calls engine._configure_zero_optimizer.
                        #
                        # Inside the DeepSpeed Zero3 optimizer configuration, which initalizes
                        # DeepSpeedZeroOptimizer_Stage3, during which:
                        #   * trainable_param_groups are obtained from the attached optimizer
                        #     (already partitioned in 16bit).
                        #   * then _setup_for_real_optimizer -> _create_fp32_partitions
                        #     which performs the fp32 upcasting.

                        # To mimick DeepSeepds's casting in FSDP, we look at the (single) FlatParameter held
                        # within an FSDP wrapper. This FlatParameter will be seen by the optimizer.
                        #  - even though there is a torch.device('meta') guard below, we
                        #    expect _init_utils._init_param_handle_from_module to already
                        #    sync the parameter.

                        if not module._has_params:
                            continue  # skip if FSDP module not managing parameters
                        param = module._flat_param
                        if (
                            param.dtype != torch.float32
                            and param.device != torch.device("meta")
                            and param.requires_grad
                        ):
                            # keep log of names_params that was upcasted
                            # NOTE: resorted to this because warnings.simplefilter("once") is somehow not working
                            name_param_log = (module.module.__class__.__name__, ", ".join(module._flat_param._fqns))
                            if name_param_log not in upcasted_log:
                                upcasted_log.append(name_param_log)

                            # this works because of FSDP's _runtime_utils.lazy_init.
                            # Have to be careful not to call anything before this that
                            # triggers lazy_init (e.g., _is_fsdp_root).
                            param.data = param.data.to(torch.float32)  # upcasting
                            module._handle._orig_param_dtype = torch.float32  # update

                    # report the warnings
                    # some messages can be quite repetitive, especially when reporting about layers that have identical architecture.
                    if self.is_main_process:
                        for name_log, param_log in upcasted_log:
                            warnings.warn(
                                f"Upcasted low precision parameters in {name_log} because mixed precision turned on in FSDP. "
                                f"Affects: {param_log}."
                            )

                        if len(upcasted_log) > 0:
                            warnings.warn(
                                "FSDP upcast of low precision parameters may affect the precision of model checkpoints."
                            )

                # if the previous and current models are same, delete the previous one
                if len(self._models) > 1 and (self._models[-2] is self._models[-1]):
                    del self._models[-2]
                self._models[-1] = model
        # torch.compile should be called last and only if the model isn't already compiled.
        if self.state.dynamo_plugin.backend != GaudiDynamoBackend.NO and not is_compiled_module(model):
            model = torch.compile(model, **self.state.dynamo_plugin.to_kwargs())
        return model

    def _prepare_deepspeed(self, *args):
        import deepspeed

        deepspeed_plugin = self.state.deepspeed_plugin

        is_dataloader_present = any(isinstance(obj, torch.utils.data.DataLoader) for obj in args)
        result = [
            self._prepare_one(obj, first_pass=True)
            if isinstance(obj, torch.utils.data.DataLoader)
            else convert_model(obj)
            if isinstance(obj, torch.nn.Module) and self.state.is_fp8_enabled
            else obj
            for obj in args
        ]

        if deepspeed_plugin.is_auto("train_micro_batch_size_per_gpu"):
            if is_dataloader_present:
                batch_sizes = [obj.batch_size for obj in args if hasattr(obj, "batch_size")]
                if any(bs is None for bs in batch_sizes):
                    raise ValueError(
                        "At least one of the dataloaders passed to `accelerate.prepare()` has `None` as batch size. "
                        "Please set an integer value in `train_micro_batch_size_per_gpu` in the deepspeed config file "
                        "or assign integer value to `AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']`."
                    )
                if self.split_batches:
                    batch_sizes = [batch_size // self.num_processes for batch_size in batch_sizes]

                batch_size_per_device = min(batch_sizes) if deepspeed_plugin.is_train_batch_min else max(batch_sizes)
                if len(batch_sizes) > 1:
                    logger.info(
                        "Since you passed both train and evaluation dataloader, `is_train_batch_min` (here "
                        f"{deepspeed_plugin.is_train_batch_min} will decide the `train_batch_size` ({batch_size_per_device})."
                    )
            else:
                raise ValueError(
                    "When using DeepSpeed, `accelerate.prepare()` requires you to pass at least one of training or evaluation dataloaders "
                    "with `batch_size` attribute returning an integer value "
                    "or alternatively set an integer value in `train_micro_batch_size_per_gpu` in the deepspeed config file "
                    "or assign integer value to `AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']`."
                )
        else:
            batch_size_per_device = deepspeed_plugin.get_value("train_micro_batch_size_per_gpu")

        # handle `gradient_accumulation_steps` when the value is `auto`
        deepspeed_plugin.fill_match(
            "gradient_accumulation_steps",
            must_match=False,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        config_kwargs = {
            "train_micro_batch_size_per_gpu": batch_size_per_device,
            "train_batch_size": batch_size_per_device
            * deepspeed_plugin.get_value("gradient_accumulation_steps")
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
            # if the model is an MOE, set the appropriate MOE layers as leaf Z3 modules
            deepspeed_plugin.set_moe_leaf_modules(model)
            # deal with config keys that use `auto` value and rely on model's hidden_size
            hidden_size_based_keys = [
                "zero_optimization.reduce_bucket_size",
                "zero_optimization.stage3_prefetch_bucket_size",
                "zero_optimization.stage3_param_persistence_threshold",
            ]
            hidden_size_auto_keys = [x for x in hidden_size_based_keys if deepspeed_plugin.is_auto(x)]
            if len(hidden_size_auto_keys) > 0:
                reasoning = (
                    "therefore it's not possible to automatically fill out the following `auto` entries "
                    + f"in the DeepSpeed config file: {hidden_size_auto_keys}. You can fix that by replacing "
                    + "`auto` values for these keys with an integer value of your choice."
                )
                if not hasattr(model, "config"):
                    raise ValueError("Can't find `model.config` entry, " + reasoning)

                if hasattr(model.config, "hidden_size"):
                    hidden_size = model.config.hidden_size
                elif hasattr(model.config, "hidden_sizes"):
                    # if there are many hidden sizes pick the largest one
                    hidden_size = max(model.config.hidden_sizes)
                else:
                    raise ValueError(
                        "Can find neither `model.config.hidden_size` nor `model.config.hidden_sizes`, " + reasoning
                    )

                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": int(0.9 * hidden_size * hidden_size),
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
                        if type(scheduler).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES:
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
            # torch.compile should be called if dynamo plugin backend is set and only if the model isn't already compiled.
            if self.state.dynamo_plugin.backend == GaudiDynamoBackend.HPU_BACKEND and not is_compiled_module(
                kwargs["model"]
            ):
                engine.compile()
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

    def prepare_data_loader(
        self, data_loader: torch.utils.data.DataLoader, device_placement=None, slice_fn_for_dispatch=None
    ):
        """
        Prepares a PyTorch DataLoader for training in any distributed setup. It is recommended to use
        [`Accelerator.prepare`] instead.

        Args:
            data_loader (`torch.utils.data.DataLoader`):
                A vanilla PyTorch DataLoader to prepare
            device_placement (`bool`, *optional*):
                Whether or not to place the batches on the proper device in the prepared dataloader. Will default to
                `self.device_placement`.
            slice_fn_for_dispatch (`Callable`, *optional*`):
                If passed, this function will be used to slice tensors across `num_processes`. Will default to
                [`~utils.slice_tensors`]. This argument is used only when `dispatch_batches` is set to `True` and will
                be ignored otherwise.

        Example:

        ```python
        >>> import torch
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> data_loader = accelerator.prepare_data_loader(data_loader, device_placement=True)
        ```
        """
        # Ensure we can't double wrap a DataLoader due to `find_batch_size`
        if getattr(data_loader, "_is_accelerate_prepared", False):
            if data_loader not in self._dataloaders:
                self._dataloaders.append(data_loader)
            return data_loader
        if device_placement is None:
            device_placement = self.device_placement
        prepared_data_loader = gaudi_prepare_data_loader(
            data_loader,
            self.device,
            num_processes=self.num_processes,
            process_index=self.process_index,
            split_batches=self.split_batches,
            put_on_device=device_placement,
            rng_types=self.rng_types.copy(),
            dispatch_batches=self.dispatch_batches,
            even_batches=self.even_batches,
            slice_fn_for_dispatch=slice_fn_for_dispatch,
            use_seedable_sampler=self.use_seedable_sampler,
            non_blocking=self.non_blocking,
        )
        self._dataloaders.append(prepared_data_loader)
        return prepared_data_loader

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
        if GaudiPartialState().distributed_type in [
            GaudiDistributedType.MULTI_HPU,
            GaudiDistributedType.DEEPSPEED,
            GaudiDistributedType.FSDP,
        ]:
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
        # copied from https://github.com/huggingface/accelerate/blob/6f05bbd41a179cc9a86238c7c6f3f4eded70fbd8/src/accelerate/accelerator.py#L3057
        elif self.distributed_type == DistributedType.FSDP:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = model.state_dict()
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
