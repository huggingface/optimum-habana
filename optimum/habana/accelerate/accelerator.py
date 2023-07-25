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
from torch.optim.lr_scheduler import LRScheduler


if is_deepspeed_available():
    import deepspeed
    from accelerate.utils import (
        DeepSpeedEngineWrapper,
        DeepSpeedOptimizerWrapper,
        DeepSpeedSchedulerWrapper,
        DummyOptim,
        DummyScheduler,
    )

from .state import GaudiAcceleratorState


logger = get_logger(__name__)


# We pass cloned tensors to torch.save() to avoid checkpoint bloat that occurs when torch.save()
# saves the underlying storage rather than the slice of the storage corresponding to individual tensors.
# This is a problem in DeepSpeed because we often allocate tensors using slices of large flattened buffers.
# Tensor cloning helps to avoid this problem because the storage of cloned tensors are closer to the true size.
# It is expected that the garbage collector will reclaim the cloned tensor storage to avoid memory bloat.
# See https://pytorch.org/docs/stable/notes/serialization.html#preserve-storage-sharing
# TODO: remove this method when it is available in Habana's DeepSpeed fork
def clone_tensors_for_torch_save(item, device=torch.device("cpu")):
    """
    Returns a copy of `item` with all enclosed tensors replaced by clones on a specified device.
    Works on individual tensors, and tensors contained/nested in lists, tuples, and dicts.

    Parameters:
        - `item`: tensor to clone or (possibly nested) container of tensors to clone.
        - `device`: target device (defaults to `cpu`)

    Returns:
        - copy of `item` with cloned tensors on target device
    """
    if torch.is_tensor(item):
        return item.detach().clone().to(device)
    elif isinstance(item, list):
        return [clone_tensors_for_torch_save(v, device) for v in item]
    elif isinstance(item, tuple):
        return tuple([clone_tensors_for_torch_save(v, device) for v in item])
    elif isinstance(item, dict):
        return type(item)({k: clone_tensors_for_torch_save(v, device) for k, v in item.items()})
    else:
        return item


class GaudiAccelerator(Accelerator):
    """
    Creates an instance of an accelerator for distributed training (on multi-GPU, TPU) or mixed precision training.

    Args:
        device_placement (`bool`, *optional*, defaults to `True`):
            Whether or not the accelerator should put objects on device (tensors yielded by the dataloader, model,
            etc...).
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If
            `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a
            round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set
            in your script multiplied by the number of processes.
        mixed_precision (`str`, *optional*):
            Whether or not to use mixed precision training. Choose from 'no','fp16','bf16 or 'fp8'. Will default to the
            value in the environment variable `ACCELERATE_MIXED_PRECISION`, which will use the default value in the
            accelerate config of the current system or the flag passed with the `accelerate.launch` command. 'fp16'
            requires pytorch 1.6 or higher. 'bf16' requires pytorch 1.10 or higher. 'fp8' requires the installation of
            transformers-engine.
        gradient_accumulation_steps (`int`, *optional*, default to 1):
            The number of steps that should pass before gradients are accumulated. A number > 1 should be combined with
            `Accelerator.accumulate`. If not passed, will default to the value in the environment variable
            `ACCELERATE_GRADIENT_ACCUMULATION_STEPS`. Can also be configured through a `GradientAccumulationPlugin`.
        cpu (`bool`, *optional*):
            Whether or not to force the script to execute on CPU. Will ignore GPU available if set to `True` and force
            the execution on one process only.
        deepspeed_plugin (`DeepSpeedPlugin`, *optional*):
            Tweak your DeepSpeed related args using this argument. This argument is optional and can be configured
            directly using *accelerate config*
        fsdp_plugin (`FullyShardedDataParallelPlugin`, *optional*):
            Tweak your FSDP related args using this argument. This argument is optional and can be configured directly
            using *accelerate config*
        megatron_lm_plugin (`MegatronLMPlugin`, *optional*):
            Tweak your MegatronLM related args using this argument. This argument is optional and can be configured
            directly using *accelerate config*
        rng_types (list of `str` or [`~utils.RNGType`]):
            The list of random number generators to synchronize at the beginning of each iteration in your prepared
            dataloaders. Should be one or several of:

            - `"torch"`: the base torch random number generator
            - `"cuda"`: the CUDA random number generator (GPU only)
            - `"xla"`: the XLA random number generator (TPU only)
            - `"generator"`: the `torch.Generator` of the sampler (or batch sampler if there is no sampler in your
              dataloader) or of the iterable dataset (if it exists) if the underlying dataset is of that type.

            Will default to `["torch"]` for PyTorch versions <=1.5.1 and `["generator"]` for PyTorch versions >= 1.6.
        log_with (list of `str`, [`~utils.LoggerType`] or [`~tracking.GeneralTracker`], *optional*):
            A list of loggers to be setup for experiment tracking. Should be one or several of:

            - `"all"`
            - `"tensorboard"`
            - `"wandb"`
            - `"comet_ml"`
            If `"all"` is selected, will pick up all available trackers in the environment and initialize them. Can
            also accept implementations of `GeneralTracker` for custom trackers, and can be combined with `"all"`.
        project_config (`ProjectConfiguration`, *optional*):
            A configuration for how saving the state can be handled.
        project_dir (`str`, `os.PathLike`, *optional*):
            A path to a directory for storing data such as logs of locally-compatible loggers and potentially saved
            checkpoints.
        dispatch_batches (`bool`, *optional*):
            If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process
            and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose
            underlying dataset is an `IterableDataset`, `False` otherwise.
        even_batches (`bool`, *optional*, defaults to `True`):
            If set to `True`, in cases where the total batch size across all processes does not exactly divide the
            dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among
            all workers.
        step_scheduler_with_optimizer (`bool`, *optional`, defaults to `True`):
            Set `True` if the learning rate scheduler is stepped at the same time as the optimizer, `False` if only
            done under certain circumstances (at the end of each epoch, for instance).
        kwargs_handlers (`list[KwargHandler]`, *optional*)
            A list of `KwargHandler` to customize how the objects related to distributed training or mixed precision
            are created. See [kwargs](kwargs) for more information.
        dynamo_backend (`str` or `DynamoBackend`, *optional*, defaults to `"no"`):
            Set to one of the possible dynamo backends to optimize your training with torch dynamo.
        gradient_accumulation_plugin (`GradientAccumulationPlugin`, *optional*):
            A configuration for how gradient accumulation should be handled, if more tweaking than just the
            `gradient_accumulation_steps` is needed.

    **Available attributes:**

        - **device** (`torch.device`) -- The device to use.
        - **distributed_type** ([`~utils.DistributedType`]) -- The distributed training configuration.
        - **local_process_index** (`int`) -- The process index on the current machine.
        - **mixed_precision** (`str`) -- The configured mixed precision mode.
        - **num_processes** (`int`) -- The total number of processes used for training.
        - **optimizer_step_was_skipped** (`bool`) -- Whether or not the optimizer update was skipped (because of
          gradient overflow in mixed precision), in which
        case the learning rate should not be changed.
        - **process_index** (`int`) -- The overall index of the current process among all processes.
        - **state** ([`~state.AcceleratorState`]) -- The distributed setup state.
        - **sync_gradients** (`bool`) -- Whether the gradients are currently being synced across all processes.
        - **use_distributed** (`bool`) -- Whether the current configuration is for distributed training.
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
                    "DeepSpeed is not installed => run `pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.10.0`."
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

    @property
    def use_fp16(self):
        raise ValueError("fp16 is not supported on Habana Gaudi.")

    def _prepare_deepspeed(self, *args):
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

        if self.gradient_accumulation_steps != deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"]:
            logger.info(
                f"Updating DeepSpeed's gradient accumulation steps to {self.gradient_accumulation_steps} from "
                f"{deepspeed_plugin.deepspeed_config['gradient_accumulation_steps']}."
            )
            deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = self.gradient_accumulation_steps
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
            elif "scheduler" not in deepspeed_plugin.deepspeed_config and isinstance(scheduler, (DummyScheduler)):
                raise ValueError(
                    "You cannot create a `DummyScheduler` without specifying a scheduler in the config file."
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
            if isinstance(scheduler, (DummyScheduler)):
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
                # from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
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
