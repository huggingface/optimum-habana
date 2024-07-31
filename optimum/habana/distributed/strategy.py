# Copyright 2024 The Foundation Model Stack Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from its original version.
# The original version can be found at https://github.com/foundation-model-stack/foundation-model-stack

from abc import abstractmethod
from typing import List

import torch
import torch.distributed
from torch import nn


class DistributedStrategy:
    def __init__(self, from_meta=False):
        self.from_meta = from_meta

    def distribute_module(self, module: nn.Module, final_layers: bool = False) -> nn.Module:
        """
        Optionally a distributed strategy may distribute modules that are not
        numbered layers
        """
        return module

    @abstractmethod
    def distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        """
        Distribute each layer as-appropriate
        """
        pass


class NotDistributed(DistributedStrategy):
    def __init__(self, from_meta=False):
        super().__init__(from_meta)

    def distribute_module(self, module: nn.Module, final_layers: bool = False) -> nn.Module:
        return module

    def distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        return block


NoOpStrategy = NotDistributed()


class DeviceMover(nn.Module):
    def __init__(self, module: nn.Module, device):
        super().__init__()
        self.device = device
        # make this wrapper module behave as if it was the wrapped module.
        attr = module.__dict__
        attr["module"] = module.to(device)
        attr["device"] = device
        self.__dict__ = attr

    def forward(self, *args, **kwargs):
        device = self.device
        args = [arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {k: (kwargs[k].to(device) if isinstance(kwargs[k], torch.Tensor) else kwargs[k]) for k in kwargs}
        return self.module(*args, **kwargs)


class UniformModelParallelStrategy(DistributedStrategy):
    def __init__(self, devices: List[int], num_layers: int, from_meta=False):
        super().__init__(from_meta)
        num_dev = len(devices)
        layers_per_dev = num_layers // num_dev
        remainder = num_layers - (layers_per_dev * num_dev)
        self.layer_to_device = [0] * num_layers
        layer_id = 0
        for dev_idx in range(len(devices)):
            for i in range(layers_per_dev):
                self.layer_to_device[layer_id] = devices[dev_idx]
                layer_id = layer_id + 1
            if remainder > 0:
                self.layer_to_device[layer_id] = devices[dev_idx]
                layer_id = layer_id + 1
                remainder -= 1

    def distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        device = self.layer_to_device[layer]
        if self.from_meta:
            block.to_empty(device=device)  # type: ignore[arg-type]
        wrapped = DeviceMover(block, device)
        return wrapped

    def distribute_module(self, module: nn.Module, final_layers: bool = False) -> nn.Module:
        if final_layers:
            device = self.layer_to_device[len(self.layer_to_device) - 1]
        else:
            device = self.layer_to_device[0]
        if self.from_meta:
            return module.to_empty(device=device)  # type: ignore[arg-type]
        wrapped = DeviceMover(module, device)
        return wrapped


class TensorParallelStrategy(DistributedStrategy):
    def __init__(self, group=None, from_meta=False):
        super().__init__(from_meta)
        assert torch.distributed.is_initialized(), "must initialize a process group"
        self.group = group if group is not None else torch.distributed.GroupMember.WORLD

    def distribute_module(self, module: nn.Module, final_layers: bool = False) -> nn.Module:
        from optimum.habana.distributed import tp_wrapping

        return tp_wrapping.apply_tp(module, self.group)

    def distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        from optimum.habana.distributed import tp_wrapping

        return tp_wrapping.apply_tp(block, layer, self.group)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["group"] = None  # Remove ProcessGroup from state
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.group = None  # Restore to default state or reinitialize
