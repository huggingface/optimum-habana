#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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
        # Habana Mixed Precision (MHP) configuration
        self.use_habana_mixed_precision = kwargs.pop("use_habana_mixed_precision", False)
        self.hmp_opt_level = kwargs.pop("hmp_opt_level", "O1")
        self.hmp_bf16_ops = kwargs.pop("hmp_bf16_ops", DEFAULT_BF16_OPS)
        self.hmp_fp32_ops = kwargs.pop("hmp_fp32_ops", DEFAULT_FP32_OPS)
        self.hmp_is_verbose = kwargs.pop("hmp_is_verbose", False)

        # Use Habana's custom AdamW implementation
        self.use_fused_adam = kwargs.pop("use_fused_adam", False)
        # Use Habana's custom fused clip norm implementation
        self.use_fused_clip_norm = kwargs.pop("use_fused_clip_norm", False)

    def write_bf16_fp32_ops_to_text_files(
        self,
        path_to_bf16_file: Path,
        path_to_fp32_file: Path,
    ):
        for path, ops in zip(
            [Path(path_to_bf16_file), Path(path_to_fp32_file)], [self.hmp_bf16_ops, self.hmp_fp32_ops]
        ):
            with path.open("w") as text_file:
                # writelines does not add new lines after each element so "\n" is inserted
                text_file.writelines(op + "\n" for op in ops)
