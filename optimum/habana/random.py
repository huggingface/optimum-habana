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

from typing import Union

import torch
from torch.cuda import _lazy_init, _lazy_call


def get_hpu_rng_state(device: Union[int, str, torch.device] = "hpu") -> torch.Tensor:
    r"""Returns the random number generator state of the specified HPU as a ByteTensor.
    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'hpu'`` (i.e., ``torch.device('hpu')``, the current HPU).
    """
    _lazy_init()
    if device == -1:
        device = torch.device("hpu")
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("hpu", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = torch.default_generators[idx]
    return default_generator.get_state()


def set_hpu_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current HPU.

    Arguments:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 HPU cases.
    """
    if device == -1:
        device = torch.device("hpu")
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("hpu", device)

    def cb():
        idx = device.index
        if idx is None:
            idx = torch.current_device()
        default_generator = torch.default_generators[idx]
        default_generator.set_state(new_state)

    _lazy_call(cb)
