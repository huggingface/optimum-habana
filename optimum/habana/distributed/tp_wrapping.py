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

from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from ..transformers.models.llama.modeling_llama import (
    GaudiLlamaAttention,
    GaudiLlamaMLP,
    TPGaudiLlamaAttention,
    TPGaudiLlamaMLP,
)


def _tp_wrapped(module: nn.Module, layer: int, group: ProcessGroup):
    if hasattr(module, "to_tp"):
        return module.to_tp(group)
    elif isinstance(module, GaudiLlamaAttention):
        return TPGaudiLlamaAttention.import_module(module, layer, group)
    elif isinstance(module, GaudiLlamaMLP):
        return TPGaudiLlamaMLP.import_module(module, group)
    else:
        return module


def apply_tp(model: nn.Module, layer_idx: int, group: ProcessGroup):
    wrapped = _tp_wrapped(model, layer_idx, group)
    if wrapped is not model:
        return wrapped

    for name, layer in model.named_children():
        tp_layer = apply_tp(layer, layer_idx, group)
        setattr(model, name, tp_layer)
    return model
