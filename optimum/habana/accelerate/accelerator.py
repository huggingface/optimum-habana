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

import functools
import math
import os
from dataclasses import make_dataclass
from types import MethodType

import torch
from accelerate import Accelerator
from accelerate.accelerator import _split_batches
from accelerate.data_loader import prepare_data_loader
from accelerate.logging import get_logger
from accelerate.scheduler import AcceleratedScheduler
from accelerate.tracking import GeneralTracker
from accelerate.utils import (
    DataLoaderConfiguration,
    DeepSpeedPlugin,
    DistributedType,
    DynamoBackend,
    FullyShardedDataParallelPlugin,
    GradientAccumulationPlugin,
    KwargsHandler,
    LoggerType,
    MegatronLMPlugin,
    PrecisionType,
    ProjectConfiguration,
    RNGType,
    TorchDynamoPlugin,
    TorchTensorParallelPlugin,
    convert_outputs_to_fp32,
    is_deepspeed_available,
)
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


import accelerate.utils.transformer_engine

from ..distributed import parallel_state
from .utils.dataclasses import GaudiTERecipeKwargs
from .utils.transformer_engine import convert_model, get_fp8_recipe


accelerate.utils.transformer_engine.convert_model = convert_model
accelerate.accelerator.convert_model = convert_model
accelerate.utils.convert_model = convert_model

accelerate.utils.dataclasses.TERecipeKwargs = GaudiTERecipeKwargs
accelerate.accelerator.TERecipeKwargs = GaudiTERecipeKwargs


logger = get_logger(__name__)


def compile_regions(model, compile_kwargs):
    if isinstance(model, torch.nn.ModuleList):
        for name, module in model.named_children():
            module = torch.compile(module, **compile_kwargs)
            module.__dict__.pop("_parameters", None)
            setattr(model, name, module)
    else:
        if model._modules:  # If model has submodules, recurse and reassign
            for name, module in model.named_children():
                compiled_module = compile_regions(module, compile_kwargs)
                if compiled_module is not None:  # Only reassign if something is returned
                    setattr(model, name, compiled_module)
        else:  # Leaf node
            model = torch.compile(model, **compile_kwargs)
            model.__dict__.pop("_parameters", None)
            return model


class GaudiAccelerator(Accelerator):
    def __init__(
        self,
        device_placement: bool = True,
        split_batches: bool = _split_batches,
        mixed_precision: PrecisionType | str | None = None,
        gradient_accumulation_steps: int = 1,
        cpu: bool = False,
        dataloader_config: DataLoaderConfiguration | None = None,
        deepspeed_plugin: DeepSpeedPlugin | dict[str, DeepSpeedPlugin] | None = None,
        fsdp_plugin: FullyShardedDataParallelPlugin | None = None,
        torch_tp_plugin: TorchTensorParallelPlugin | None = None,
        megatron_lm_plugin: MegatronLMPlugin | None = None,
        rng_types: list[str | RNGType] | None = None,
        log_with: str | LoggerType | GeneralTracker | list[str | LoggerType | GeneralTracker] | None = None,
        project_dir: str | os.PathLike | None = None,
        project_config: ProjectConfiguration | None = None,
        gradient_accumulation_plugin: GradientAccumulationPlugin | None = None,
        step_scheduler_with_optimizer: bool = True,
        kwargs_handlers: list[KwargsHandler] | None = None,
        dynamo_backend: DynamoBackend | str | None = None,
        dynamo_plugin: TorchDynamoPlugin | None = None,
        deepspeed_plugins: DeepSpeedPlugin | dict[str, DeepSpeedPlugin] | None = None,
        # TODO: remove these when the features are upstream or removed
        force_autocast: bool = False,
        distribution_strategy: str = None,
        use_regional_compilation: bool | None = None,
        compiled_autograd_enable: bool = False,
    ):
        self.use_regional_compilation = use_regional_compilation
        self.compiled_autograd_enable = compiled_autograd_enable
        self.distribution_strategy = distribution_strategy
        self.force_autocast = force_autocast
        self.mpu = parallel_state

        # This is to trigger the creation of te_recipe_handler when the env var is set to fp8
        # it will be fixed in upstream accelerate
        mixed_precision = mixed_precision or os.environ.get("ACCELERATE_MIXED_PRECISION", None)

        super().__init__(
            device_placement=device_placement,
            split_batches=split_batches,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            cpu=cpu,
            dataloader_config=dataloader_config,
            deepspeed_plugin=deepspeed_plugin,
            fsdp_plugin=fsdp_plugin,
            torch_tp_plugin=torch_tp_plugin,
            megatron_lm_plugin=megatron_lm_plugin,
            rng_types=rng_types,
            log_with=log_with,
            project_dir=project_dir,
            project_config=project_config,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
            step_scheduler_with_optimizer=step_scheduler_with_optimizer,
            kwargs_handlers=kwargs_handlers,
            dynamo_backend=dynamo_backend,
            dynamo_plugin=dynamo_plugin,
            deepspeed_plugins=deepspeed_plugins,
        )

        # This attribute works as a single source of truth about fp8 usage with the accelerator.
        # it will be added in upstream accelerate
        self.fp8_enabled = self.mixed_precision == "fp8" or mixed_precision == "fp8"

        # will be fixed in upstream accelerate
        self.has_fp8_handler = self.te_recipe_handler is not None or self.fp8_recipe_handler is not None

        # this is what will be used by the FP8ContextWrapper, avoiding recreating the recipe
        # we can clean this up later when the upstream accelerate is fixed
        self.fp8_recipe = None
        if self.has_fp8_handler:
            self.fp8_recipe = get_fp8_recipe(self.te_recipe_handler or self.fp8_recipe_handler)

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
            if not evaluation_mode and self.distributed_type == DistributedType.MULTI_HPU:
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

        if self.fp8_enabled:
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
            ###############################################################################################################
            if self.distributed_type == DistributedType.MULTI_HPU and self.distribution_strategy != "fast_ddp":
                if any(p.requires_grad for p in model.parameters()):
                    kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                    model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
                    if self.ddp_handler is not None:
                        self.ddp_handler.register_comm_hook(model)
            ###############################################################################################################
            elif self.distributed_type == DistributedType.FSDP:
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

                """
                TODO: Temporarily disable this upcast due to FSDP graph compile issue.
                Investigate why the parameters are loaded as bf16(autocast?) and why
                graph compile failure is seen due to upcast.
                Original accelerate PR: https://github.com/huggingface/accelerate/pull/2674

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
                        # - in accelerator.prepare, deepspeed.initialize is called to:
                        #   * creates the DeepSpeedEngine.
                        #   * since zero_optimization() is True , calls engine._configure_zero_optimizer.
                        #
                        # Inside the DeepSpeed Zero3 optimizer configuration, which initializes
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

                """

                # if the previous and current models are same, delete the previous one
                if len(self._models) > 1 and (self._models[-2] is self._models[-1]):
                    del self._models[-2]
                self._models[-1] = model
        # torch.compile should be called last and only if the model isn't already compiled.
        if self.state.dynamo_plugin.backend != DynamoBackend.NO and not is_compiled_module(model):
            compile_kwargs = self.state.dynamo_plugin.to_kwargs()
            ############################################################################################################
            if self.use_regional_compilation:
                compile_regions(model, compile_kwargs)
            else:
                model = torch.compile(model, **compile_kwargs)
            ############################################################################################################
        return model

    # TODO: Remove when compile_regions is removed
    def _prepare_deepspeed(self, *args):
        import deepspeed

        deepspeed_plugin = self.state.deepspeed_plugin

        is_dataloader_present = any(isinstance(obj, torch.utils.data.DataLoader) for obj in args)
        result = [
            self._prepare_one(obj, first_pass=True)
            if isinstance(obj, torch.utils.data.DataLoader)
            else convert_model(obj)
            if isinstance(obj, torch.nn.Module) and self.fp8_enabled
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
            if self.state.dynamo_plugin.backend != DynamoBackend.NO and not is_compiled_module(model):
                compile_kwargs = self.state.dynamo_plugin.to_kwargs()
                ###############################################################################################################
                if self.use_regional_compilation:
                    compile_regions(engine.module, compile_kwargs)
                else:
                    engine.compile(
                        backend=compile_kwargs.pop("backend"),
                        compile_kwargs=compile_kwargs,
                        compiled_autograd_enabled=self.compiled_autograd_enable,
                    )
                ###############################################################################################################
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
            if self.deepspeed_engine_wrapped is None:
                self.deepspeed_engine_wrapped = DeepSpeedEngineWrapper(engine)
            else:
                logger.warning(
                    "A wrapped DeepSpeed engine reference is currently tied for this `Accelerator()` instance. "
                    "If you want to call `accelerator.backward()` referencing a new model/engine, "
                    "please create a separate `Accelerator()` instance and call `accelerator.prepare()` on it."
                )
            self._models.append(engine)
            if optimizer is not None:
                self._optimizers.append(optimizer)
            if scheduler is not None:
                self._schedulers.append(scheduler)
        return tuple(result)

    # TODO: Remove when accelerate supports Sequence/Context parallelism
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
            device_placement = self.device_placement if self.distributed_type != DistributedType.XLA else False

        ###############################################################################################################
        # Patching the num_processes and process_index for sequence parallelism
        num_processes = self.num_processes
        process_index = self.process_index
        if num_processes is None:
            num_processes = self.state.num_processes
        if process_index is None:
            process_index = self.state.process_index
        if parallel_state.sequence_parallel_is_initialized() and parallel_state.get_sequence_parallel_world_size() > 1:
            num_processes = int(num_processes / parallel_state.get_sequence_parallel_world_size())
        if parallel_state.sequence_parallel_is_initialized() and parallel_state.get_sequence_parallel_world_size() > 1:
            process_index = int(process_index / parallel_state.get_sequence_parallel_world_size())
        ###############################################################################################################

        # To avoid training crash issue SW-207456 when num_worker > 0 in multi-node training tasks
        if int(os.environ.get("WORLD_SIZE", 1)) > 8 and data_loader.num_workers > 0:
            import multiprocessing

            multiprocessing_context = multiprocessing.get_context("spawn")
            data_loader.multiprocessing_context = multiprocessing_context

        prepared_data_loader = prepare_data_loader(
            data_loader,
            self.device,
            num_processes=num_processes,
            process_index=process_index,
            split_batches=self.split_batches,
            put_on_device=device_placement,
            rng_types=self.rng_types.copy(),
            dispatch_batches=self.dispatch_batches,
            even_batches=self.even_batches,
            slice_fn_for_dispatch=slice_fn_for_dispatch,
            use_seedable_sampler=self.use_seedable_sampler,
            data_seed=self.dataloader_config.data_seed,
            non_blocking=self.non_blocking,
            use_stateful_dataloader=self.use_stateful_dataloader,
            torch_device_mesh=self.state.torch_tp_plugin.torch_device_mesh if self.state.torch_tp_plugin else None,
        )
        self._dataloaders.append(prepared_data_loader)
        return prepared_data_loader


def patch_has_compiled_regions(*args, **kwargs):
    return False


accelerate.utils.other.has_compiled_regions = patch_has_compiled_regions
