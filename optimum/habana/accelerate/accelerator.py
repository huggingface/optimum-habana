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
import os
import re
from types import MethodType

import accelerate
import torch
from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedType,
    DynamoBackend,
    apply_fp8_autowrap,
    convert_outputs_to_fp32,
    ensure_weights_retied,
    get_mixed_precision_context_manager,
    model_has_dtensor,
)
from accelerate.utils.dataclasses import FP8BackendType
from accelerate.utils.other import compile_regions, is_compiled_module

from ..distributed import parallel_state
from .utils.dataclasses import GaudiTERecipeKwargs
from .utils.transformer_engine import convert_model, get_fp8_recipe


accelerate.utils.transformer_engine.convert_model = convert_model
accelerate.accelerator.convert_model = convert_model
accelerate.utils.convert_model = convert_model

accelerate.utils.dataclasses.TERecipeKwargs = GaudiTERecipeKwargs
accelerate.accelerator.TERecipeKwargs = GaudiTERecipeKwargs


logger = get_logger(__name__)


class GaudiAccelerator(Accelerator):
    def __init__(
        self,
        *args,
        # TODO: remove these when the features are upstream or removed
        force_autocast: bool = False,
        distribution_strategy: str = None,
        compiled_autograd_enabled: bool = False,
        ##############################################################
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mpu = parallel_state
        self.distribution_strategy = distribution_strategy
        self.compiled_autograd_enabled = compiled_autograd_enabled
        self.native_amp = self.native_amp and force_autocast

        # this is what will be used by the FP8ContextWrapper, avoiding recreating the recipe
        # we can clean this up later when the upstream accelerate is fixed
        self.fp8_recipe = None
        if self.has_fp8_handler:
            self.fp8_recipe = get_fp8_recipe(self.te_recipe_handler or self.fp8_recipe_handler)

    # NOTE: this is only kept here until FSDP upcast is fixed
    def prepare_model(
        self, model: torch.nn.Module, device_placement: bool | None = None, evaluation_mode: bool = False
    ):
        if device_placement is None:
            device_placement = self.device_placement and self.distributed_type != DistributedType.FSDP

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

        if self.native_amp:
            model._original_forward = model.forward
            autocast_context = get_mixed_precision_context_manager(self.native_amp, self.autocast_handler)
            # NOTE: MS-AMP adds `__func__` already to `model.forward`, so we should always use `model.forward`
            if self.fp8_backend == FP8BackendType.MSAMP or not hasattr(model.forward, "__func__"):
                model_forward_func = model.forward
                model.forward = convert_outputs_to_fp32(autocast_context(model_forward_func))
            else:
                model_forward_func = model.forward.__func__
                new_forward = autocast_context(model_forward_func)
                model.forward = MethodType(new_forward, model)
                model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)

        # We prepare TE after, allowing for bf16 autocast to happen first
        if self.fp8_backend == FP8BackendType.TE and not self.delayed_fp8_autocast:
            model = apply_fp8_autowrap(model, self.te_recipe_handler or self.fp8_recipe_handler)

        if device_placement and not self.verify_device_map(model):
            model = model.to(self.device)
        if not evaluation_mode and self.distribution_strategy != "fast_ddp":
            if self.multi_device and not (self.parallelism_config and self.parallelism_config.tp_enabled):
                if model_has_dtensor(model):
                    raise ValueError(
                        "Your model contains `DTensor` parameters, which is incompatible with DDP. Maybe you loaded your model with `device_map='auto'`? Specify `device_map='cuda'` or 'cpu' instead."
                    )
                if any(p.requires_grad for p in model.parameters()):
                    kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                    # TODO: Look at enabling native TP training directly with a proper config
                    if os.environ.get("ACCELERATE_BYPASS_DEVICE_MAP", "false") != "true":
                        if self.device.type == "hpu":
                            device_ids, output_device = [self.device.index], self.device.index
                        else:
                            device_ids, output_device = [self.local_process_index], self.local_process_index
                    else:
                        device_ids, output_device = None, None
                    model = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=device_ids, output_device=output_device, **kwargs
                    )
                    if self.ddp_handler is not None:
                        self.ddp_handler.register_comm_hook(model)
            elif self.parallelism_config and self.parallelism_config.tp_enabled:
                if not hasattr(model, "tp_size"):
                    raise NotImplementedError(
                        "Model should undergo tensor parallel before passing it to accelerate."
                        "You can use .from_pretrained(..., tp_plan='auto') if the model supports"
                    )
                if model.tp_size != self.parallelism_config.tp_size:
                    raise ValueError(
                        f"tp_size in the plugin {self.parallelism_config.tp_size} should be same as model's tp size {model.tp_size}"
                    )
            elif self.is_fsdp2:
                raise ValueError(
                    "FSDP2 preparation should be done via `accelerate.prepare()`, as it requires a model and an optimizer."
                )

            elif self.distributed_type == DistributedType.FSDP:
                # We need to fix the optimizer *before* sharding the model
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

                    # need to ensure that params are re-tied after running
                    # param_init_fn
                    fsdp_plugin.param_init_fn = ensure_weights_retied(
                        fsdp_plugin.param_init_fn,
                        model,
                        self.device,
                    )

                    kwargs = {
                        # We fallback to reshard_after_forward if sharding_strategy is not set.
                        # We prerfer sharding_strategy to not break the behavior of the existing code.
                        # Deprecation warning has already been issued in `utils.dataclasses.py`
                        "sharding_strategy": fsdp_plugin.sharding_strategy or fsdp_plugin.reshard_after_forward,
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
                        "device_id": self.device,
                    }

                    if isinstance(kwargs["ignored_modules"], str):
                        reg = re.compile(kwargs["ignored_modules"])
                        ignored = []
                        for name, module in model.named_modules():
                            if reg.fullmatch(name):
                                # ensure that the device for these modules is still set correctly
                                module.to(self.device)
                                ignored.append(module)
                        kwargs["ignored_modules"] = ignored

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

                # TODO: starting from transformers 4.43 and accelerate 0.33, upcasting was added for FSDP in mixed precision
                # https://github.com/huggingface/accelerate/pull/2674 making FSDP training more stable,
                # but was kept disabled in optimum-habana temporarily due to some failing tests.
                """
                # In the event the model had been loaded in low precision, but
                # mixed precision had also been activated, then we follow DeepSpeed's
                # strategy to hold the parameters in full precision.
                # - assume that trainer.args.bf16 and trainer.args.fp16 are already checked against
                #   fsdp_plugin.mixed_precision_policy.
                # - NOTE: we do not check the mixed_precision attribute on the FSDP root wrapper.
                #   * this attribute will always set by init_utils.init_core_state so its always not None.
                #   * mixed_precision.param_dtype only regards _fwd_bwd_param_dtype
                #   * if model is loaded in 16bit, and even if mixed_precision.param_dtype is None,
                #     we still want to upcast the flat_param.
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

                        # To mimic DeepSeepds's casting in FSDP, we look at the (single) FlatParameter held
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
            elif self.distributed_type == DistributedType.MULTI_CPU:
                kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler else {}
                model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
                if self.ddp_handler is not None:
                    self.ddp_handler.register_comm_hook(model)
        # Now we can apply the FP8 autocast
        if self.fp8_backend == FP8BackendType.TE and self.delayed_fp8_autocast:
            model = apply_fp8_autowrap(model, self.te_recipe_handler or self.fp8_recipe_handler)
        # torch.compile should be called last and only if the model isn't already compiled
        if self.state.dynamo_plugin.backend != DynamoBackend.NO and not is_compiled_module(model):
            if self.state.dynamo_plugin.use_regional_compilation:
                model = compile_regions(model, **self.state.dynamo_plugin.to_kwargs())
            else:
                model = torch.compile(model, **self.state.dynamo_plugin.to_kwargs())
        return model

    # INFO: this adds support for fast_ddp by not applying DDP wrapper
    # def prepare_model(self, model: torch.nn.Module, device_placement: bool = None, evaluation_mode: bool = False):
    #     if self.distribution_strategy == "fast_ddp":
    #         # with fast_ddp, we just skip ddp and fsdp model preparation
    #         model = super().prepare_model(model, device_placement=device_placement, evaluation_mode=True)
    #     else:
    #         model = super().prepare_model(model, device_placement=device_placement, evaluation_mode=evaluation_mode)
    #     return model

    # INFO: this adds support for autograd compilation to the deepspeed engine
    def _prepare_deepspeed(self, *args):
        orig_num_models = len(self._models)
        prepared_deepspeed = super()._prepare_deepspeed(*args)

        if len(self._models) > orig_num_models and self._models[-1]._is_compiled:
            # an engine was added and is compiled
            self._models[-1]._is_compiled_autograd_enabled = self.compiled_autograd_enabled

        return prepared_deepspeed

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
