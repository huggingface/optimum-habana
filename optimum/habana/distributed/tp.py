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

import itertools
from abc import ABCMeta, abstractmethod
from typing import List

import torch
import torch.nn as nn
from torch.distributed.distributed_c10d import ProcessGroup

from .tensorparallel import (
    apply_colwise_tp,
    apply_embedding_tp,
    apply_rowwise_tp,
)


class TPModule(nn.Module, metaclass=ABCMeta):
    """
    This is an abstract class that any nn.Module can implement to enable
    Tensor Parallel. On top of inheriting from this class, the TP module
    will have to implement list_colwise_weights, list_rowwise_weights,
    list_embedding_weights, and import_module for their relevant weights.
    Finally, the module must call setup_tp at the end of their __init__
    function. See examples in attention.py, feedforward.py and embedding.py

    """

    rank: int
    world_size: int

    def setup_tp(self, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size

    def colwise_param_names(self) -> List[str]:
        return []

    def rowwise_param_names(self) -> List[str]:
        return []

    def embedding_param_names(self) -> List[str]:
        return []

    @staticmethod
    @abstractmethod
    def import_module(module, group: ProcessGroup):
        pass

    def import_weights(self, module: nn.Module):
        for weight in self.colwise_param_names():
            apply_colwise_tp(
                getattr(self, weight),
                getattr(module, weight),
                self.world_size,
                self.rank,
            )
        for weight in self.rowwise_param_names():
            apply_rowwise_tp(
                getattr(self, weight),
                getattr(module, weight),
                self.world_size,
                self.rank,
            )
        for weight in self.embedding_param_names():
            apply_embedding_tp(
                getattr(self, weight),
                getattr(module, weight),
                self.world_size,
                self.rank,
            )
        tp_sharded_modules = list(
            itertools.chain(
                self.colwise_param_names(),
                self.rowwise_param_names(),
                self.embedding_param_names(),
            )
        )
        with torch.no_grad():
            for mod_name, module in self.named_children():
                if mod_name not in tp_sharded_modules:
                    for param_name, param in module.named_parameters(recurse=False):
                        param.copy_(
                            getattr(getattr(module, mod_name), param_name),
                            non_blocking=True,
                        )
