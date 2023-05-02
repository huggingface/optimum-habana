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
from typing import TYPE_CHECKING, List, Optional

import torch
from transformers.activations import FastGELUActivation, GELUActivation, NewGELUActivation
from transformers.utils.fx import HFTracer, check_if_model_is_supported, get_concrete_args


if TYPE_CHECKING:
    from torch.fx import GraphModule
    from transformers import PreTrainedModel


GELU_CLASSES_TO_TRACK = (GELUActivation, FastGELUActivation, NewGELUActivation)


class OptimumHabanaTracer(HFTracer):
    """
    Extends the [`HFTracer`](https://github.com/huggingface/transformers/blob/dfeb5aa6a9d0cb95c008854c4e67ceecfeff6ccc/src/transformers/utils/fx.py#L713) to be able to track down GELU modules.

    By default, only modules in `torch.nn` are leaf modules so tracing will go inside other modules to track down elementary operations.
    By making GELU modules leaf ones, we make sure tracking them down is easy as they won't be decomposed into elementary operations.
    """

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return isinstance(m, GELU_CLASSES_TO_TRACK) or super().is_leaf_module(m, module_qualified_name)


def symbolic_trace(
    model: "PreTrainedModel",
    input_names: Optional[List[str]] = None,
    disable_check: bool = False,
) -> "GraphModule":
    """
    Same as Transformers' [`symbolic_trace`](https://github.com/huggingface/transformers/blob/dfeb5aa6a9d0cb95c008854c4e67ceecfeff6ccc/src/transformers/utils/fx.py#L1206) but using `OptimumHabanaTracer` instead of `HFTracer`.
    """
    if input_names is None:
        input_names = model.dummy_inputs.keys()

    input_names = list(input_names)
    concrete_args = get_concrete_args(model, input_names)

    if not disable_check:
        check_if_model_is_supported(model)

    # Tracing.
    tracer = OptimumHabanaTracer()
    traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced = torch.fx.GraphModule(model, traced_graph)

    traced.config = model.config
    # The model class must be stored as an attribute to allow model deserialization, which uses trace, and thus
    # _generate_dummy_input, where the model class is needed.
    traced.class_for_deserialization = model.__class__
    traced.device = model.device

    return traced
