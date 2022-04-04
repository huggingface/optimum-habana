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

import unittest
import torch.hpu

from transformers.file_utils import is_torch_available


def require_torch_multi_hpu(test_case):
    """
    Decorator marking a test that requires a multi-HPU setup (in PyTorch). These tests are skipped on a machine without
    multiple HPUs.
    To run *only* the multi_hpu tests, assuming all test names contain multi_hpu: $ pytest -sv ./tests -k "multi_hpu"
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    if torch.hpu.device_count() < 2:
        return unittest.skip("test requires multiple HPUs")(test_case)
    else:
        return test_case
