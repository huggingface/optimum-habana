# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

import filecmp
import tempfile
import unittest
from pathlib import Path
from typing import List

from optimum.habana import GaudiConfig


BF16_OPS_REFERENCE_FILE = Path(__file__).parent.resolve() / Path("configs/bf16_ops.txt")
FP32_OPS_REFERENCE_FILE = Path(__file__).parent.resolve() / Path("configs/fp32_ops.txt")


def is_list_of_strings(my_list: List) -> bool:
    """
    This method assesses whether an object is a list of strings or not.

    Args:
        my_list (List): list to assess

    Returns:
        bool: whether the input argument is a list of strings or not
    """
    if my_list and isinstance(my_list, list):
        return all(isinstance(op, str) for op in my_list)
    else:
        return False


class GaudiConfigTester(unittest.TestCase):
    """
    Unit tests for Gaudi configuration class GaudiConfig.
    """

    def test_default_parameter_types(self):
        gaudi_config = GaudiConfig()

        self.assertIsInstance(gaudi_config.use_habana_mixed_precision, bool)
        self.assertIsInstance(gaudi_config.hmp_is_verbose, bool)
        self.assertIsInstance(gaudi_config.use_fused_adam, bool)
        self.assertIsInstance(gaudi_config.use_fused_clip_norm, bool)
        self.assertIsInstance(gaudi_config.use_torch_autocast, bool)

        self.assertTrue(is_list_of_strings(gaudi_config.hmp_bf16_ops))
        self.assertTrue(is_list_of_strings(gaudi_config.hmp_fp32_ops))
        self.assertIsNone(gaudi_config.autocast_bf16_ops)
        self.assertIsNone(gaudi_config.autocast_fp32_ops)

    def test_write_bf16_fp32_ops_to_text_files(self):
        gaudi_config = GaudiConfig()

        with tempfile.NamedTemporaryFile() as bf16_file:
            with tempfile.NamedTemporaryFile() as fp32_file:
                gaudi_config.write_bf16_fp32_ops_to_text_files(
                    bf16_file.name,
                    fp32_file.name,
                )

                self.assertTrue(
                    filecmp.cmp(
                        bf16_file.name,
                        BF16_OPS_REFERENCE_FILE,
                        shallow=False,
                    )
                )
                self.assertTrue(
                    filecmp.cmp(
                        fp32_file.name,
                        FP32_OPS_REFERENCE_FILE,
                        shallow=False,
                    )
                )
