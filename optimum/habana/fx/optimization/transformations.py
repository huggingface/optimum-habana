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
from typing import TYPE_CHECKING

import torch
from transformers.activations import GELUActivation

from optimum.fx.optimization import ReversibleTransformation

from ..tracer import GELU_CLASSES_TO_TRACK


if TYPE_CHECKING:
    from torch.fx import GraphModule


class GeluToFusedGelu(ReversibleTransformation):
    """
    Transformation that replaces GELU modules by `torch.nn.functional.gelu` (i.e. `GELUActivation()`) if different.
    `torch.nn.functional.gelu` is mapped to a single fused GELU kernel on Gaudi so the transformed model will be faster.
    """

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        for node in graph_module.graph.nodes:
            if node.op == "call_module":
                module = graph_module.get_submodule(node.target)
                if isinstance(module, GELU_CLASSES_TO_TRACK):
                    if hasattr(module, "act") and getattr(module, "act") == torch.nn.functional.gelu:
                        # GELUActivation does not necessarily rely on torch.nn.functional.gelu
                        # If it does, no need to change anything so break the loop
                        break
                    parent_name, _, name = node.target.rpartition(".")
                    parent_module = graph_module.get_submodule(parent_name)
                    # Keep track of the original GELU module to be able to reverse the transformation
                    node.original_gelu_module = getattr(parent_module, name)
                    # Set the GELU module to GELUActivation()
                    setattr(parent_module, name, GELUActivation())
                    self.mark_as_transformed(node)
        return graph_module

    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        # If self.original_gelu_module is None, that means nothing was changed so the reverse transformation is the identity
        if self.original_gelu_module is not None:
            for node in self.get_transformed_nodes(graph_module):
                parent_name, _, name = node.target.rpartition(".")
                parent_module = graph_module.get_submodule(parent_name)
                # Set the GELU module back to what it originally was
                if hasattr(node, "original_gelu_module"):
                    setattr(parent_module, name, node.original_gelu_module)
                self.mark_as_restored(node)
        return graph_module
