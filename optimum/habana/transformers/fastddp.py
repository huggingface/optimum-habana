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

"""
    Fast and lightweight alternative of DistributeDataParallel for Habana Gaudi
"""

import functools

import torch
import torch.nn

_all_reduce_group_size = torch.distributed.group.WORLD.size()

_fusion_buffer = None
_fusion_views = []

_fusion_graph = None
_unfusion_graph = None


def FastDistributedDataParallel(model: torch.nn.Module, fusion_buffer_dtype: torch.dtype):
    global _fusion_buffer
    global _fusion_views
    global _fusion_graph
    global _unfusion_graph

    gradient_elem_count = 0
    for param in model.parameters():
        gradient_elem_count += torch.numel(param)

    _fusion_buffer = torch.zeros(size=(gradient_elem_count,), dtype=fusion_buffer_dtype, device='hpu:0')
    _fusion_views.clear()

    view_start_index = 0
    for param in model.parameters():
        view_next_index = view_start_index + torch.numel(param.data)
        _fusion_views.append(_fusion_buffer[view_start_index:view_next_index].reshape(param.data.shape))
        view_start_index = view_next_index

    model.all_reduce_gradients = functools.partial(_all_reduce_gradients, model)

    _fusion_graph = None
    _unfusion_graph = None

    return model


def _all_reduce_gradients(model: torch.nn.Module):
    global _all_reduce_group_size
    global _fusion_buffer
    global _fusion_views

    # Fuse all the gradient into the fusion buffer.
    global _fusion_graph
    if _fusion_graph is None:
        import habana_frameworks.torch as ht
        _fusion_graph = ht.hpu.HPUGraph()
        s = ht.hpu.Stream()
        with ht.hpu.stream(s):
            _fusion_graph.capture_begin()

            for param_index, param in enumerate(model.parameters()):
                grad = param.grad

                if param.grad is None:
                    continue

                grad = grad / _all_reduce_group_size

                if grad.dtype != _fusion_buffer.dtype:
                    grad = grad.to(_fusion_buffer.dtype)

                view = _fusion_views[param_index]

                view.copy_(grad, non_blocking=True)

            _fusion_graph.capture_end()

    _fusion_graph.replay()

    # All-reduce the gradients.
    torch.distributed.all_reduce(_fusion_buffer, group=torch.distributed.group.WORLD, async_op=True)

    # Unfuse the gradient from the fusion buffer.
    global _unfusion_graph
    if _unfusion_graph is None:
        import habana_frameworks.torch as ht
        _unfusion_graph = ht.hpu.HPUGraph()
        s = ht.hpu.Stream()
        with ht.hpu.stream(s):
            _unfusion_graph.capture_begin()

            for param_index, param in enumerate(model.parameters()):
                if param.grad is None:
                    continue

                view = _fusion_views[param_index]

                if view.dtype != param.grad.dtype:
                    view = view.to(param.grad.dtype)

                param.grad.copy_(view, non_blocking=True)

            _unfusion_graph.capture_end()

    _unfusion_graph.replay()
