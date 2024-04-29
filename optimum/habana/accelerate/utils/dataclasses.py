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

import os
from dataclasses import dataclass
from enum import Enum

import torch
from accelerate.utils import FullyShardedDataParallelPlugin
from accelerate.utils.constants import FSDP_BACKWARD_PREFETCH
from accelerate.utils.dataclasses import BaseEnum, TorchDynamoPlugin
from accelerate.utils.environment import str_to_bool


class GaudiDistributedType(str, Enum):
    """
    Represents a type of distributed environment.
    Adapted from: https://github.com/huggingface/accelerate/blob/8514c35192ac9762920f1ab052e5cea4c0e46eeb/src/accelerate/utils/dataclasses.py#L176

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **MULTI_HPU** -- Distributed on multiple HPUs.
        - **DEEPSPEED** -- Using DeepSpeed.
        - **FSDP** -- Using FSDP.
    """

    # Subclassing str as well as Enum allows the `GaudiDistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    MULTI_HPU = "MULTI_HPU"
    DEEPSPEED = "DEEPSPEED"
    FSDP = "FSDP"


class GaudiDynamoBackend(str, BaseEnum):
    """
    Represents a dynamo backend (see https://github.com/pytorch/torchdynamo).

    Values:

        - **NO** -- Do not use torch dynamo.
        - **EAGER** -- Uses PyTorch to run the extracted GraphModule. This is quite useful in debugging TorchDynamo
          issues.
        - **AOT_EAGER** -- Uses AotAutograd with no compiler, i.e, just using PyTorch eager for the AotAutograd's
          extracted forward and backward graphs. This is useful for debugging, and unlikely to give speedups.
        - **INDUCTOR** -- Uses TorchInductor backend with AotAutograd and cudagraphs by leveraging codegened Triton
          kernels. [Read
          more](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
        - **AOT_TS_NVFUSER** -- nvFuser with AotAutograd/TorchScript. [Read
          more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
        - **NVPRIMS_NVFUSER** -- nvFuser with PrimTorch. [Read
          more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
        - **CUDAGRAPHS** -- cudagraphs with AotAutograd. [Read more](https://github.com/pytorch/torchdynamo/pull/757)
        - **OFI** -- Uses Torchscript optimize_for_inference. Inference only. [Read
          more](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html)
        - **FX2TRT** -- Uses Nvidia TensorRT for inference optimizations. Inference only. [Read
          more](https://github.com/pytorch/TensorRT/blob/master/docsrc/tutorials/getting_started_with_fx_path.rst)
        - **ONNXRT** -- Uses ONNXRT for inference on CPU/GPU. Inference only. [Read more](https://onnxruntime.ai/)
        - **TENSORRT** -- Uses ONNXRT to run TensorRT for inference optimizations. [Read
          more](https://github.com/onnx/onnx-tensorrt)
        - **IPEX** -- Uses IPEX for inference on CPU. Inference only. [Read
          more](https://github.com/intel/intel-extension-for-pytorch).
        - **TVM** -- Uses Apach TVM for inference optimizations. [Read more](https://tvm.apache.org/)
        - **HPU_BACKEND** -- Uses Habana Gaudi.

    """

    # Subclassing str as well as Enum allows the `SageMakerDistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    EAGER = "EAGER"
    AOT_EAGER = "AOT_EAGER"
    INDUCTOR = "INDUCTOR"
    AOT_TS_NVFUSER = "AOT_TS_NVFUSER"
    NVPRIMS_NVFUSER = "NVPRIMS_NVFUSER"
    CUDAGRAPHS = "CUDAGRAPHS"
    OFI = "OFI"
    FX2TRT = "FX2TRT"
    ONNXRT = "ONNXRT"
    TENSORRT = "TENSORRT"
    IPEX = "IPEX"
    TVM = "TVM"
    HPU_BACKEND = "HPU_BACKEND"


@dataclass
class GaudiTorchDynamoPlugin(TorchDynamoPlugin):
    """
    This plugin is used to compile a model with PyTorch 2.0 on Gaudi.
    """

    def __post_init__(self):
        prefix = "ACCELERATE_DYNAMO_"
        if self.backend is None:
            self.backend = os.environ.get(prefix + "BACKEND", "no")
        self.backend = GaudiDynamoBackend(self.backend.upper())
        if self.mode is None:
            self.mode = os.environ.get(prefix + "MODE", "default")
        if self.fullgraph is None:
            self.fullgraph = str_to_bool(os.environ.get(prefix + "USE_FULLGRAPH", "False")) == 1
        if self.dynamic is None:
            self.dynamic = str_to_bool(os.environ.get(prefix + "USE_DYNAMIC", "False")) == 1


@dataclass
class GaudiFullyShardedDataParallelPlugin(FullyShardedDataParallelPlugin):
    def __post_init__(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch, CPUOffload, ShardingStrategy

        prefix = "FSDP_"
        if self.sharding_strategy is None:
            self.sharding_strategy = ShardingStrategy(int(os.environ.get(prefix + "SHARDING_STRATEGY", 1)))

        if self.cpu_offload is None:
            if str_to_bool(os.environ.get(prefix + "OFFLOAD_PARAMS", "False")) == 1:
                self.cpu_offload = CPUOffload(offload_params=True)
            else:
                self.cpu_offload = CPUOffload(offload_params=False)

        if self.backward_prefetch is None:
            prefetch_policy = os.environ.get(prefix + "BACKWARD_PREFETCH", "NO_PREFETCH")
            if prefetch_policy != FSDP_BACKWARD_PREFETCH[-1]:
                self.backward_prefetch = BackwardPrefetch(FSDP_BACKWARD_PREFETCH.index(prefetch_policy) + 1)

        if self.state_dict_type is None:
            state_dict_type_policy = os.environ.get(prefix + "STATE_DICT_TYPE", "FULL_STATE_DICT")
            self.set_state_dict_type(state_dict_type_policy)
        self.use_orig_params = str_to_bool(os.environ.get(prefix + "USE_ORIG_PARAMS", "False")) == 1
        self.sync_module_states = str_to_bool(os.environ.get(prefix + "SYNC_MODULE_STATES", "True")) == 1
        self.forward_prefetch = str_to_bool(os.environ.get(prefix + "FORWARD_PREFETCH", "False")) == 1
        self.activation_checkpointing = str_to_bool(os.environ.get(prefix + "ACTIVATION_CHECKPOINTING", "False")) == 1

        if self.sync_module_states:
            device = torch.device("hpu")
            self.param_init_fn = lambda x: x.to_empty(device=device, recurse=False)
