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

###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

"""
Fast and lightweight alternative to DistributeDataParallel for Habana Gaudi
"""

import torch


def all_reduce_gradients(
    model: torch.nn.Module, fusion_buffer_dtype: torch.dtype = torch.bfloat16, use_hpu_graphs: bool = True
):
    """
    Invokes an all-reduce operation on the gradients supporting data parallel training.

    This function is meant to be called after forward+backward passes, where the gradient information is available in the model parameters.
    Once called, the list of gradients participating in the training process must remain the same.

    Args:
        model (torch.nn.Module): A model whose gradients are meant to be all-reduced.
        fusion_buffer_dtype (torch.dtype): The dtype of internally allocated gradient fusion buffer.
        use_hpu_graphs (bool): Determines whether HPU graph recording should be used for packing and unpacking the gradients.

    Raises:
        NotImplementedError: `all_reduce_gradients()` does not support changing the set of active gradients after first invocation.
    """

    # Try to get the existing fusion buffer created for the model.
    fusion_entries = model.__dict__.get("_all_reduce_fusion_entries", None)
    if fusion_entries is not None:
        if len(fusion_entries) == 0:
            # There is nothing to all-reduce, neither the fusion buffer.
            return

        fusion_buffer = model._all_reduce_fusion_buffer
        if use_hpu_graphs:
            pack_graph = model._all_reduce_gradient_pack_graph
            unpack_graph = model._all_reduce_gradient_unpack_graph
    else:
        # Count the total number of elements of the reduced gradients.
        grad_elem_count = 0
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_elem_count += torch.numel(param.grad)

        # There is nothing to all-reduce.
        if grad_elem_count == 0:
            model.__dict__["_all_reduce_fusion_entries"] = []
            return

        # Allocate the fusion buffer and associate it with the model.
        fusion_buffer = torch.zeros(size=(grad_elem_count,), dtype=fusion_buffer_dtype, device="hpu:0")
        model.__dict__["_all_reduce_fusion_buffer"] = fusion_buffer

        # Build the fusion information necessary for gradient packing and unpacking processes.
        grad_elem_count = 0
        fusion_entries = []
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_numel = torch.numel(param.grad)
            fused_view = fusion_buffer[grad_elem_count : grad_elem_count + grad_numel].reshape(param.grad.shape)
            fusion_entries.append((param, fused_view))
            grad_elem_count += grad_numel
        model.__dict__["_all_reduce_fusion_entries"] = fusion_entries

        # Instruct the following logic to record packing and unpacking HPU graphs based on the newly created fusion buffer.
        if use_hpu_graphs:
            pack_graph = None
            unpack_graph = None

    # Pack the gradients into the fusion buffer.
    def pack_grads():
        world_size_inv = 1.0 / torch.distributed.group.WORLD.size()
        for param, fused_view in fusion_entries:
            grad = param.grad

            if grad is None:
                raise NotImplementedError(
                    "`all_reduce_gradients()` does not support changing the set of active gradients after first invocation."
                )

            if grad.dtype != fusion_buffer_dtype:
                grad = grad.to(fusion_buffer_dtype)
            grad = grad * world_size_inv
            fused_view.copy_(grad, non_blocking=True)

    if use_hpu_graphs:
        if pack_graph is None:
            import habana_frameworks.torch as ht

            pack_graph = ht.hpu.HPUGraph()
            with ht.hpu.stream(ht.hpu.Stream()):
                pack_graph.capture_begin()
                pack_grads()
                pack_graph.capture_end()
            model.__dict__["_all_reduce_gradient_pack_graph"] = pack_graph

        pack_graph.replay()
    else:
        pack_grads()

    # Invoke an all-reduce operation of the fused gradients.
    torch.distributed.all_reduce(fusion_buffer, group=torch.distributed.group.WORLD, async_op=True)

    # Unpack the gradients back to the model parameters.
    def unpack_grads():
        for param, fused_view in fusion_entries:
            grad = param.grad

            if grad is None:
                raise NotImplementedError(
                    "`all_reduce_gradients()` does not support changing the set of active gradients after first invocation."
                )

            if fused_view.dtype != grad.dtype:
                fused_view = fused_view.to(grad.dtype)

            grad.copy_(fused_view, non_blocking=True)

    if use_hpu_graphs:
        if unpack_graph is None:
            import habana_frameworks.torch as ht

            unpack_graph = ht.hpu.HPUGraph()
            with ht.hpu.stream(ht.hpu.Stream()):
                unpack_graph.capture_begin()
                unpack_grads()
                unpack_graph.capture_end()
            model.__dict__["_all_reduce_gradient_unpack_graph"] = unpack_graph

        unpack_graph.replay()
    else:
        unpack_grads()
