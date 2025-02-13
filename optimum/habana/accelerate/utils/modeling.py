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

"""
Device similarity check compatible with hpu
"""

import torch


def gaudi_check_device_same(first_device, second_device):
    """
    Copied from https://github.com/huggingface/accelerate/blob/6b2d968897c91bc3f96274b4679d84e9950ad908/src/accelerate/utils/modeling.py#L50
    difference is addition of HPU device checks

    Args:
        first_device (`torch.device`):
            First device to check
        second_device (`torch.device`):
            Second device to check
    """
    if first_device.type != second_device.type:
        return False

    if first_device.type == "cuda" and first_device.index is None:
        # In case the first_device is a cuda device and have
        # the index attribute set to `None`, default it to `0`
        first_device = torch.device("cuda", index=0)

    elif first_device.type == "hpu" and first_device.index is None:
        first_device = torch.device("hpu", index=0)

    if second_device.type == "cuda" and second_device.index is None:
        # In case the second_device is a cuda device and have
        # the index attribute set to `None`, default it to `0`
        second_device = torch.device("cuda", index=0)

    elif second_device.type == "hpu" and second_device.index is None:
        second_device = torch.device("hpu", index=0)

    return first_device == second_device
