# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import sys
from pathlib import Path

from optimum.configuration_utils import BaseConfig
from optimum.utils import logging


logger = logging.get_logger(__name__)

# Default bf16 and fp32 ops (BERT)
DEFAULT_BF16_OPS = [
    "add",
    "addmm",
    "bmm",
    "div",
    "dropout",
    "gelu",
    "iadd",
    "linear",
    "layer_norm",
    "matmul",
    "mm",
    "rsub",
    "softmax",
    "truediv",
]
DEFAULT_FP32_OPS = [
    "embedding",
    "nll_loss",
    "log_softmax",
]
GAUDI_CONFIG_NAME = "gaudi_config.json"


class GaudiConfig(BaseConfig):
    CONFIG_NAME = "gaudi_config.json"
    FULL_CONFIGURATION_FILE = "gaudi_config.json"

    def __init__(self, **kwargs):
        # Torch Autocast
        self.use_torch_autocast = kwargs.pop("use_torch_autocast", False)
        self.autocast_bf16_ops = kwargs.pop("autocast_bf16_ops", None)
        self.autocast_fp32_ops = kwargs.pop("autocast_fp32_ops", None)
        self.use_dynamic_shapes = kwargs.pop("use_dynamic_shapes", False)

        # Use Habana's custom AdamW implementation
        self.use_fused_adam = kwargs.pop("use_fused_adam", False)
        # Use Habana's custom fused clip norm implementation
        self.use_fused_clip_norm = kwargs.pop("use_fused_clip_norm", False)

        # TODO: to remove in a future version

    def write_bf16_fp32_ops_to_text_files(
        self,
        path_to_bf16_file: Path,
        path_to_fp32_file: Path,
    ):
        for path, ops in zip(
            [Path(path_to_bf16_file), Path(path_to_fp32_file)], [self.autocast_bf16_ops, self.autocast_fp32_ops]
        ):
            with path.open("w") as text_file:
                # writelines does not add new lines after each element so "\n" is inserted
                text_file.writelines(op + "\n" for op in ops)

    def declare_autocast_bf16_fp32_ops(self):
        if self.autocast_bf16_ops is not None and self.autocast_fp32_ops is not None:
            if "habana_frameworks.torch.core" in sys.modules:
                raise RuntimeError(
                    "Setting bf16/fp32 ops for Torch Autocast but `habana_frameworks.torch.core` has already been imported. "
                    "You should instantiate your Gaudi config and your training arguments before importing from `habana_frameworks.torch` or calling a method from `optimum.habana.utils`."
                )
            else:
                autocast_bf16_filename = "/tmp/lower_list.txt"
                autocast_fp32_filename = "/tmp/fp32_list.txt"

                self.write_bf16_fp32_ops_to_text_files(
                    autocast_bf16_filename,
                    autocast_fp32_filename,
                )
                os.environ["LOWER_LIST"] = autocast_bf16_filename
                os.environ["FP32_LIST"] = autocast_fp32_filename
