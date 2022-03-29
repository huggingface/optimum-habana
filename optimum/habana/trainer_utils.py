# coding=utf-8
# Copyright 2022 the HuggingFace Inc. team.
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

import torch


def to_device_dtype(my_input, target_device: torch.device = None, target_dtype: torch.dtype = None):
    """
    Move a state_dict to the target device.

    Args:
        my_input: input to move to the target device
        target_device (torch.device, optional): target device. Defaults to torch.device("cpu").
    """
    if isinstance(my_input, torch.Tensor):
        if target_device is None:
            target_device = my_input.device
        if target_dtype is None:
            target_dtype = my_input.dtype
        return my_input.to(device=target_device, dtype=target_dtype)
    elif isinstance(my_input, list):
        return [to_device_dtype(i, target_device) for i in my_input]
    elif isinstance(my_input, tuple):
        return tuple(to_device_dtype(i, target_device) for i in my_input)
    elif isinstance(my_input, dict):
        return {k: to_device_dtype(v, target_device) for k, v in my_input.items()}
    else:
        return my_input
