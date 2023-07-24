from __future__ import annotations

import contextlib
import os
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import make_dataclass

import torch
from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
from accelerate.optimizer import AcceleratedOptimizer
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
    from accelerate.utils import DummyOptim, DummyScheduler, DeepSpeedOptimizerWrapper, DeepSpeedEngineWrapper

from .state import GaudiAcceleratorState


# We pass cloned tensors to torch.save() to avoid checkpoint bloat that occurs when torch.save()
# saves the underlying storage rather than the slice of the storage corresponding to individual tensors.
# This is a problem in DeepSpeed because we often allocate tensors using slices of large flattened buffers.
# Tensor cloning helps to avoid this problem because the storage of cloned tensors are closer to the true size.
# It is expected that the garbage collector will reclaim the cloned tensor storage to avoid memory bloat.
# See https://pytorch.org/docs/stable/notes/serialization.html#preserve-storage-sharing
# TODO: remove this method when it is available in Habana's DeepSpeed fork
def clone_tensors_for_torch_save(item, device=torch.device('cpu')):
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

    @contextmanager
    def accumulate(self, model):
        """
        A context manager that will lightly wrap around and perform gradient accumulation automatically

        Args:
            model (`torch.nn.Module`):
                PyTorch Module that was prepared with `Accelerator.prepare`

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(gradient_accumulation_steps=1)
        >>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

        >>> for input, output in dataloader:
        ...     with accelerator.accumulate(model):
        ...         outputs = model(input)
        ...         loss = loss_func(outputs)
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()
        ...         optimizer.zero_grad()
        ```
        """
        self._do_sync()
        if self.sync_gradients:
            context = contextlib.nullcontext
        else:
            context = self.no_sync

        with context(model):
            yield

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
        self._models.append(model)
        # We check only for models loaded with `accelerate`
        # Checks if any of the child module has the attribute `hf_device_map`.
        has_hf_device_map = False
        for m in model.modules():
            if hasattr(m, "hf_device_map"):
                has_hf_device_map = True
                break

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
        elif device_placement and not has_hf_device_map:
            model = model.to(self.device)

        if self.native_amp:
            model._original_forward = model.forward
            model_forward_func = model.forward.__func__ if hasattr(model.forward, "__func__") else model.forward
            if self.mixed_precision == "bf16":
                new_forward = torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)(model_forward_func)

            if hasattr(model.forward, "__func__"):
                model.forward = MethodType(new_forward, model)
                model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)
            else:
                model.forward = convert_outputs_to_fp32(new_forward)
        elif self.mixed_precision == "fp8":
            if not has_transformer_engine_layers(model):
                with torch.no_grad():
                    convert_model(model)
                model._converted_to_transformer_engine = True
            model._original_forward = model.forward

            kwargs = self.fp8_recipe_handler.to_kwargs() if self.fp8_recipe_handler is not None else {}
            if "fp8_format" in kwargs:
                kwargs["fp8_format"] = getattr(te_recipe.Format, kwargs["fp8_format"])
            fp8_recipe = te_recipe.DelayedScaling(**kwargs)
            cuda_device_capacity = torch.cuda.get_device_capability()
            fp8_enabled = cuda_device_capacity[0] >= 9 or (
                cuda_device_capacity[0] == 8 and cuda_device_capacity[1] >= 9
            )
            if not fp8_enabled:
                logger.warn(
                    f"The current device has compute capability of {cuda_device_capacity} which is "
                    "insufficient for FP8 mixed precision training (requires a GPU Hopper/Ada Lovelace "
                    "or higher, compute capability of 8.9 or higher). Will use FP16 instead."
                )
            model.forward = fp8_autocast(enabled=fp8_enabled, fp8_recipe=fp8_recipe)(model.forward)
        if not evaluation_mode:
            if self.distributed_type in (
                DistributedType.MULTI_GPU,
                DistributedType.MULTI_NPU,
                DistributedType.MULTI_XPU,
            ):
                if any(p.requires_grad for p in model.parameters()):
                    kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                    model = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[self.local_process_index], output_device=self.local_process_index, **kwargs
                    )
            elif self.distributed_type == DistributedType.FSDP:
                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

                # Check if the model is already a FSDP model due to `Manual Wrapping` and if so,
                # don't wrap it again
                if type(model) != FSDP:
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
                        "ignored_parameters": fsdp_plugin.ignored_parameters,
                        "limit_all_gathers": fsdp_plugin.limit_all_gathers,
                        "device_id": self.device,
                    }
                    model = FSDP(model, **kwargs)
                self._models[-1] = model
            elif self.distributed_type == DistributedType.MULTI_CPU:
                kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
            elif self.distributed_type == DistributedType.TPU and self.state.fork_launched:
                model = xmp.MpModelWrapper(model).to(self.device)
        # torch.compile should be called last.
        if self.state.dynamo_plugin.backend != DynamoBackend.NO:
            if not is_torch_version(">=", "2.0"):
                raise ValueError("Using `torch.compile` requires PyTorch 2.0 or higher.")
            model = torch.compile(model, **self.state.dynamo_plugin.to_kwargs())
        return model

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
            # # TODO: temporary workaround
            # # To remove when it is solved, see https://github.com/HabanaAI/Model-References/blob/17fbab7ceebca15b1560ffb2c4e15a3888bb5f33/PyTorch/nlp/pretraining/deepspeed-bert/run_pretraining.py#L527
            # model.to(dtype=deepspeed_plugin.hf_ds_config._dtype, device=self.device)

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

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        """
        Should be used in place of `torch.nn.utils.clip_grad_norm_`.

        Returns:
            `torch.Tensor`: Total norm of the parameter gradients (viewed as a single vector).

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(gradient_accumulation_steps=2)
        >>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

        >>> for input, target in dataloader:
        ...     optimizer.zero_grad()
        ...     output = model(input)
        ...     loss = loss_func(output, target)
        ...     accelerator.backward(loss)
        ...     if accelerator.sync_gradients:
        ...         accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        ...     optimizer.step()
        ```
        """
        if self.distributed_type == DistributedType.FSDP:
            self.unscale_gradients()
            parameters = list(parameters)
            for model in self._models:
                if parameters == list(model.parameters()):
                    return model.clip_grad_norm_(max_norm, norm_type)
        elif self.distributed_type == DistributedType.DEEPSPEED:
            # `accelerator.backward(loss)` is doing that automatically. Therefore, its implementation is not needed
            # We cannot return the gradient norm because DeepSpeed does it.
            return None
        self.unscale_gradients()
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    def save(self, obj, f):
        """
        Save the object passed to disk once per machine. Use in place of `torch.save`.

        Args:
            obj (`object`): The object to save.
            f (`str` or `os.PathLike`): Where to save the content of `obj`.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> arr = [0, 1, 2, 3]
        >>> accelerator.save(arr, "array.pkl")
        ```
        """
        save(obj, f)

    def save_model(
        self,
        model: torch.nn.Module,
        save_directory: Union[str, os.PathLike],
        max_shard_size: Union[int, str] = "10GB",
        safe_serialization: bool = False,
    ):
        """
        Save a model so that it can be re-loaded using load_checkpoint_in_model

        Arguments:
            model: (`torch.nn.Module`):
                Model to be saved. The model can be wrapped or unwraped.
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            safe_serialization (`bool`, *optional*, defaults to `False`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model = ...
        >>> accelerator.save_model(model, save_directory)
        ```
        """

        if safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # get the state_dict of the model
        state_dict = self.get_state_dict(model)

        if safe_serialization:
            # Safetensors does not allow tensor aliasing.
            # We're going to remove aliases before saving
            ptrs = collections.defaultdict(list)
            # when bnb serialization is used the weights in the state dict can be strings
            for name, tensor in state_dict.items():
                if not isinstance(tensor, str):
                    ptrs[id_tensor_storage(tensor)].append(name)

            # These are all the pointers of shared tensors.
            shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
            warn_names = set()
            for names in shared_ptrs.values():
                # When not all duplicates have been cleaned, still remove those keys, but put a clear warning.
                # If the link between tensors was done at runtime then `from_pretrained` will not get
                # the key back leading to random tensor. A proper warning will be shown
                # during reload (if applicable), but since the file is not necessarily compatible with
                # the config, better show a proper warning.
                found = 0
                for name in names:
                    if name in state_dict:
                        found += 1
                        if found > 1:
                            del state_dict[name]
                            warn_names.add(name)
            if len(warn_names) > 0:
                logger.warning_once(
                    f"Removed shared tensor {warn_names} while saving. This should be OK, but check by verifying that you don't receive any warning while reloading",
                )

        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME

        # Shard the model if it is too big.
        shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in shards.keys()
                and reg.fullmatch(filename_no_suffix) is not None
                and PartialState().is_main_process
            ):
                os.remove(full_filename)

        # Save the model
        for shard_file, shard in shards.items():
            self.save(shard, os.path.join(save_directory, shard_file))

        if index is None:
            path_to_weights = os.path.join(save_directory, WEIGHTS_NAME)
            logger.info(f"Model weights saved in {path_to_weights}")
        else:
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, save_index_file)
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

    def save_state(self, output_dir: str = None, **save_model_func_kwargs):
        """
        Saves the current states of the model, optimizer, scaler, RNG generators, and registered objects to a folder.

        If a `ProjectConfiguration` was passed to the `Accelerator` object with `automatic_checkpoint_naming` enabled
        then checkpoints will be saved to `self.project_dir/checkpoints`. If the number of current saves is greater
        than `total_limit` then the oldest save is deleted. Each checkpoint is saved in seperate folders named
        `checkpoint_<iteration>`.

        Otherwise they are just saved to `output_dir`.

        <Tip>

        Should only be used when wanting to save a checkpoint during training and restoring the state in the same
        environment.

        </Tip>

        Args:
            output_dir (`str` or `os.PathLike`):
                The name of the folder to save all relevant weights and states.
            save_model_func_kwargs (`dict`, *optional*):
                Additional keyword arguments for saving model which can be passed to the underlying save function, such
                as optional arguments for DeepSpeed's `save_checkpoint` function.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model, optimizer, lr_scheduler = ...
        >>> model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
        >>> accelerator.save_state(output_dir="my_checkpoint")
        ```
        """
        if self.project_configuration.automatic_checkpoint_naming:
            output_dir = os.path.join(self.project_dir, "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        if self.project_configuration.automatic_checkpoint_naming:
            folders = [os.path.join(output_dir, folder) for folder in os.listdir(output_dir)]
            if self.project_configuration.total_limit is not None and (
                len(folders) + 1 > self.project_configuration.total_limit
            ):

                def _inner(folder):
                    return list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[0]

                folders.sort(key=_inner)
                logger.warning(
                    f"Deleting {len(folders) + 1 - self.project_configuration.total_limit} checkpoints to make room for new checkpoint."
                )
                for folder in folders[: len(folders) + 1 - self.project_configuration.total_limit]:
                    shutil.rmtree(folder)
            output_dir = os.path.join(output_dir, f"checkpoint_{self.save_iteration}")
            if os.path.exists(output_dir):
                raise ValueError(
                    f"Checkpoint directory {output_dir} ({self.save_iteration}) already exists. Please manually override `self.save_iteration` with what iteration to start with."
                )
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving current state to {output_dir}")

        if self.distributed_type == DistributedType.TPU:
            # Finish running the previous step before checkpointing
            xm.mark_step()

        # Save the models taking care of FSDP and DeepSpeed nuances
        weights = []
        for i, model in enumerate(self._models):
            if self.distributed_type == DistributedType.FSDP:
                logger.info("Saving FSDP model")
                save_fsdp_model(self.state.fsdp_plugin, self, model, output_dir, i)
                logger.info(f"FSDP Model saved to output dir {output_dir}")
            elif self.distributed_type == DistributedType.DEEPSPEED:
                logger.info("Saving DeepSpeed Model and Optimizer")
                ckpt_id = f"{MODEL_NAME}" if i == 0 else f"{MODEL_NAME}_{i}"
                model.save_checkpoint(output_dir, ckpt_id, **save_model_func_kwargs)
                logger.info(f"DeepSpeed Model and Optimizer saved to output dir {os.path.join(output_dir, ckpt_id)}")
            elif self.distributed_type == DistributedType.MEGATRON_LM:
                logger.info("Saving Megatron-LM Model, Optimizer and Scheduler")
                model.save_checkpoint(output_dir)
                logger.info(f"Megatron-LM Model , Optimizer and Scheduler saved to output dir {output_dir}")
            else:
                weights.append(self.get_state_dict(model, unwrap=False))

        # Save the optimizers taking care of FSDP and DeepSpeed nuances
        optimizers = []
        if self.distributed_type == DistributedType.FSDP:
            for opt in self._optimizers:
                logger.info("Saving FSDP Optimizer")
                save_fsdp_optimizer(self.state.fsdp_plugin, self, opt, self._models[i], output_dir, i)
                logger.info(f"FSDP Optimizer saved to output dir {output_dir}")
        elif self.distributed_type not in [DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM]:
            optimizers = self._optimizers

        # Save the lr schedulers taking care of DeepSpeed nuances
        schedulers = []
        if self.distributed_type == DistributedType.DEEPSPEED:
            for i, scheduler in enumerate(self._schedulers):
                if isinstance(scheduler, DeepSpeedSchedulerWrapper):
                    continue
                schedulers.append(scheduler)
        elif self.distributed_type not in [DistributedType.MEGATRON_LM]:
            schedulers = self._schedulers

        # Call model loading hooks that might have been registered with
        # accelerator.register_model_state_hook
        for hook in self._save_model_state_pre_hook.values():
            hook(self._models, weights, output_dir)

        save_location = save_accelerator_state(
            output_dir, weights, optimizers, schedulers, self.state.process_index, self.scaler
        )
        for i, obj in enumerate(self._custom_objects):
            save_custom_state(obj, output_dir, i)
        self.project_configuration.iteration += 1
        return save_location

    def load_state(self, input_dir: str, **load_model_func_kwargs):
        """
        Loads the current states of the model, optimizer, scaler, RNG generators, and registered objects.

        <Tip>

        Should only be used in conjunction with [`Accelerator.save_state`]. If a file is not registered for
        checkpointing, it will not be loaded if stored in the directory.

        </Tip>

        Args:
            input_dir (`str` or `os.PathLike`):
                The name of the folder all relevant weights and states were saved in.
            load_model_func_kwargs (`dict`, *optional*):
                Additional keyword arguments for loading model which can be passed to the underlying load function,
                such as optional arguments for DeepSpeed's `load_checkpoint` function or a `map_location` to load the
                model and optimizer on.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> model, optimizer, lr_scheduler = ...
        >>> model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
        >>> accelerator.load_state("my_checkpoint")
        ```
        """
        # Check if folder exists
        input_dir = os.path.expanduser(input_dir)
        if not os.path.isdir(input_dir):
            raise ValueError(f"Tried to find {input_dir} but folder does not exist")
        logger.info(f"Loading states from {input_dir}")

        # Load the models taking care of FSDP and DeepSpeed nuances
        models = []
        for i, model in enumerate(self._models):
            if self.distributed_type == DistributedType.FSDP:
                logger.info("Loading FSDP model")
                load_fsdp_model(self.state.fsdp_plugin, self, model, input_dir, i)
                logger.info(f"FSDP Model loaded from input dir {input_dir}")
            elif self.distributed_type == DistributedType.DEEPSPEED:
                logger.info("Loading DeepSpeed Model and Optimizer")
                ckpt_id = f"{MODEL_NAME}" if i == 0 else f"{MODEL_NAME}_{i}"
                model.load_checkpoint(input_dir, ckpt_id, **load_model_func_kwargs)
                logger.info(f"DeepSpeed Model and Optimizer loaded from input dir {os.path.join(input_dir, ckpt_id)}")
            elif self.distributed_type == DistributedType.MEGATRON_LM:
                logger.info("Loading Megatron-LM Model, Optimizer and Scheduler")
                model.load_checkpoint(input_dir)
                logger.info(f"Megatron-LM Model , Optimizer and Scheduler loaded from input dir {input_dir}")
            else:
                models.append(model)

        # Load the optimizers taking care of FSDP and DeepSpeed nuances
        optimizers = []
        if self.distributed_type == DistributedType.FSDP:
            for i, opt in enumerate(self._optimizers):
                logger.info("Loading FSDP Optimizer")
                load_fsdp_optimizer(self.state.fsdp_plugin, self, opt, self._models[i], input_dir, i)
                logger.info(f"FSDP Optimizer loaded from input dir {input_dir}")
        elif self.distributed_type not in [DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM]:
            optimizers = self._optimizers

        # Load the lr schedulers taking care of DeepSpeed nuances
        schedulers = []
        if self.distributed_type == DistributedType.DEEPSPEED:
            for i, scheduler in enumerate(self._schedulers):
                if isinstance(scheduler, DeepSpeedSchedulerWrapper):
                    continue
                schedulers.append(scheduler)
        elif self.distributed_type not in [DistributedType.MEGATRON_LM]:
            schedulers = self._schedulers

        # Call model loading hooks that might have been registered with
        # accelerator.register_model_state_hook
        for hook in self._load_model_state_pre_hook.values():
            hook(models, input_dir)

        map_location = load_model_func_kwargs.pop("map_location", None)
        if map_location is None:
            if self.num_processes > 1 and self.distributed_type in (
                DistributedType.MULTI_GPU,
                DistributedType.MULTI_NPU,
            ):
                map_location = "on_device"
            else:
                map_location = "cpu"

        load_accelerator_state(
            input_dir,
            models,
            optimizers,
            schedulers,
            self.state.process_index,
            self.scaler,
            map_location,
            **load_model_func_kwargs,
        )
        custom_checkpoints = [
            f for f in os.listdir(input_dir) if re.search(r"^custom_checkpoint_\d+\.pkl$", f) is not None
        ]
        if len(custom_checkpoints) != len(self._custom_objects):
            err = "Number of custom checkpoints in folder {input_dir} does not match the number of registered objects:"
            err += f"\n\tFound checkpoints: {len(custom_checkpoints)}"
            err += f"\n\tRegistered objects: {len(self._custom_objects)}\n"
            err += "Please make sure to only load checkpoints from folders that were created with the same set of registered objects,"
            err += "or avoid using `custom_checkpoint` in the filename for files in that same directory and load them in manually."
            raise RuntimeError(err)
        else:
            logger.info(f"Loading in {len(custom_checkpoints)} custom states")
            for index, obj in enumerate(self._custom_objects):
                load_custom_state(obj, input_dir, index)

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
            autocast_context = torch.autocast(device_type=state.device.type, dtype=torch.bfloat16)
        else:
            autocast_context = contextlib.nullcontext()

        autocast_context.__enter__()
        yield
        autocast_context.__exit__(*sys.exc_info())
