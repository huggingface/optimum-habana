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
    """

    # Subclassing str as well as Enum allows the `GaudiDistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    MULTI_HPU = "MULTI_HPU"
    DEEPSPEED = "DEEPSPEED"


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
    AOT_HPU_TRAINING_BACKEND = "AOT_HPU_TRAINING_BACKEND"


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
