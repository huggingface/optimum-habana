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

import accelerate
import torch
from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
from accelerate.logging import get_logger
from accelerate.utils import DistributedType

from ..distributed import parallel_state
from .utils import convert_model as gaudi_convert_model


logger = get_logger(__name__)


def patch_convert_model(func):
    """
    A decorator to patch the convert_model function in accelerate to use Gaudi specific conversion.
    This is used to avoid the need to revert the patch after the function is called.
    """

    def wrapper(self, *args, **kwargs):
        original_convert_model = accelerate.utils.convert_model
        accelerate.utils.convert_model = gaudi_convert_model
        result = func(self, *args, **kwargs)
        accelerate.utils.convert_model = original_convert_model

        return result

    return wrapper


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
        self.force_autocast = force_autocast
        self.distribution_strategy = distribution_strategy
        self.compiled_autograd_enabled = compiled_autograd_enabled
        self.native_amp = self.native_amp and self.force_autocast

    @patch_convert_model
    # TODO: Remove if ever Gaudi specific fp8 conversion is upstreamed in accelerate
    def prepare_model(self, model: torch.nn.Module, device_placement: bool = None, evaluation_mode: bool = False):
        if self.distribution_strategy == "fast_ddp":
            # with fast_ddp, we just skip ddp and fsdp model preparation
            model = super().prepare_model(model, device_placement=device_placement, evaluation_mode=True)
        else:
            model = super().prepare_model(model, device_placement=device_placement, evaluation_mode=evaluation_mode)
        return model

    @patch_convert_model
    # TODO: Remove if ever Gaudi specific fp8 conversion is upstreamed in accelerate
    # and compiled_autograd_enabled is upstreamed in deepspeed
    def _prepare_deepspeed(self, *args):
        orig_num_models = len(self._models)
        prepared_deepspeed = super()._prepare_deepspeed(*args)

        if len(self._models) > orig_num_models and self._models[-1]._is_compiled:
            # an engine was added and is compiled
            self._models[-1]._is_compiled_autograd_enabled = self.compiled_autograd_enabled

        return prepared_deepspeed

    # INFO: this adds support for sequence/context parallelism to the dataloader
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
