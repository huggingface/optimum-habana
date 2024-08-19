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

import torch
import torch._inductor.ir as ir
import torch._inductor.lowering as lowering
import torch.distributed as dist
from torch import nn


# Workaround to overcome the accuracy/output correctnes issue in torch.compile all_reduce for batch sizes greater than 124.
# This is a temporary fix and needs to be addressed properly in future updates.
def disable_compiler(fn):
    if hasattr(torch, "compiler") and hasattr(torch.nn.Module, "compile"):
        return torch.compiler.disable(fn)
    return fn


def apply_colwise_tp(par_mod: nn.Linear, mod: nn.Linear, world_size, rank):
    # Divide the weight matrix along the last dimension.
    output_size_per_partition = mod.out_features // world_size
    with torch.no_grad():
        par_mod.weight.copy_(torch.split(mod.weight, output_size_per_partition, dim=0)[rank])
        if par_mod.bias is not None:
            par_mod.bias.copy_(torch.split(mod.bias, output_size_per_partition)[rank])


def apply_rowwise_tp(par_mod: nn.Linear, mod: nn.Linear, world_size, rank):
    # Divide the weight matrix along the last dimension.
    output_size_per_partition = mod.in_features // world_size
    with torch.no_grad():
        par_mod.weight.copy_(torch.split(mod.weight, output_size_per_partition, dim=1)[rank])
        if par_mod.bias is not None:
            if rank == 0:
                par_mod.bias.copy_(mod.bias)
            else:
                par_mod.bias.zero_()


def apply_embedding_tp(par_mod: nn.Embedding, mod: nn.Embedding, world_size, rank):
    # Divide the weight matrix along the last dimension.
    output_size_per_partition = mod.embedding_dim // world_size
    with torch.no_grad():
        par_mod.weight.copy_(torch.split(mod.weight, output_size_per_partition, dim=1)[rank])


## Fixes for PT 2.2 collectives until PT 2.3 is released


# Fix 1: https://github.com/pytorch/pytorch/issues/121311
def get_volatile_reads_fixed(self):
    inp = self.inputs[0]
    if isinstance(inp, ir._CollectiveKernel):
        # Out-of-place single-output
        return [inp.inputs[0]]
    elif isinstance(inp, ir.MultiOutput):
        # Out-of-place multi-output
        coll = inp.inputs[0]
        if isinstance(coll, ir._CollectiveKernel):
            _, idx = inp.indices[0]
            return [coll.inputs[idx]]
        return []  # e.g. regular FallbackKernel
    else:
        # In-place requires no additional deps handling for volatile
        # reads since the inputs are mutated.
        return []


ir._WaitKernel.get_volatile_reads = get_volatile_reads_fixed

# Fix 2: These are fixed already in nightlies and will be in 2.3
for overload in torch.ops._c10d_functional.all_reduce.overloads():
    other_fn = getattr(torch.ops._c10d_functional.all_reduce, overload)
    if other_fn in lowering.lowerings:
        del lowering.lowerings[other_fn]


@disable_compiler
def _all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    world_size = dist.get_world_size()

    if world_size == 1:
        return input_

    # Starting PT 2.3, we can go back to funcol.all_reduce
    return torch.ops._c10d_functional.wait_tensor(torch.ops._c10d_functional.all_reduce(input_, "sum", "default"))


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _all_reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _all_reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)
