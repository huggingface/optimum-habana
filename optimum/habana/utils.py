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

import random
import time
from typing import Any, Dict, Union

import numpy as np
import torch
from habana_frameworks.torch.hpu import memory_stats
from habana_frameworks.torch.hpu import random as hpu_random
from transformers.utils import is_torch_available


def to_device_dtype(my_input: Any, target_device: torch.device = None, target_dtype: torch.dtype = None):
    """
    Move a state_dict to the target device and convert it into target_dtype.

    Args:
        my_input : input to transform
        target_device (torch.device, optional): target_device to move the input on. Defaults to None.
        target_dtype (torch.dtype, optional): target dtype to convert the input into. Defaults to None.

    Returns:
        : transformed input
    """
    if isinstance(my_input, torch.Tensor):
        if target_device is None:
            target_device = my_input.device
        if target_dtype is None:
            target_dtype = my_input.dtype
        return my_input.to(device=target_device, dtype=target_dtype)
    elif isinstance(my_input, list):
        return [to_device_dtype(i, target_device, target_dtype) for i in my_input]
    elif isinstance(my_input, tuple):
        return tuple(to_device_dtype(i, target_device, target_dtype) for i in my_input)
    elif isinstance(my_input, dict):
        return {k: to_device_dtype(v, target_device, target_dtype) for k, v in my_input.items()}
    else:
        return my_input


def speed_metrics(
    split: str,
    start_time: float,
    num_samples: int = None,
    num_steps: int = None,
    start_time_after_warmup: float = None,
) -> Dict[str, float]:
    """
    Measure and return speed performance metrics.
    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
        split (str): name to prefix metric (like train, eval, test...)
        start_time (float): operation start time
        num_samples (int, optional): number of samples processed. Defaults to None.
        num_steps (int, optional): number of steps performed. Defaults to None.
        start_time_after_warmup (float, optional): time after warmup steps have been performed. Defaults to None.

    Returns:
        Dict[str, float]: dictionary with performance metrics.
    """

    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}

    # Adjust runtime if there were warmup steps
    if start_time_after_warmup is not None:
        runtime = runtime + start_time - start_time_after_warmup

    # Compute throughputs
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)

    return result


def to_gb_rounded(mem: float) -> float:
    """
    Rounds and converts to GB.

    Args:
        mem (float): memory in bytes

    Returns:
        float: memory in GB rounded to the second decimal
    """
    return np.round(mem / 1024**3, 2)


def get_hpu_memory_stats() -> Dict[str, float]:
    """
    Returns memory stats of HPU as a dictionary:
    - current memory allocated (GB)
    - maximum memory allocated (GB)
    - total memory available (GB)

    Returns:
        Dict[str, float]: memory stats.
    """
    mem_stats = memory_stats()

    mem_dict = {
        "memory_allocated (GB)": to_gb_rounded(mem_stats["InUse"]),
        "max_memory_allocated (GB)": to_gb_rounded(mem_stats["MaxInUse"]),
        "total_memory_available (GB)": to_gb_rounded(mem_stats["Limit"]),
    }

    return mem_dict


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy` and `torch`.
    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        hpu_random.manual_seed_all(seed)


def input_shape_hash(inputs: Dict[str, Union[torch.Tensor, Any]]) -> int:
    """
    Returns a hash based on the shapes of the inputs.
    For instance, two different tensors of same shape will return the same hash.
    This is used for HPU graphs to know whether capturing the graph for the current input or not.

    Args:
        inputs (Dict[str, Union[torch.Tensor, Any]]): the inputs of the model

    Returns:
        int: hash of the inputs
    """

    if isinstance(inputs, dict):
        # Dictionaries are not hashable so turning them into tuples
        return input_shape_hash(tuple(inputs.items()))
    elif isinstance(inputs, list) or isinstance(inputs, tuple):
        # Get the hash of the tuple
        return hash(tuple(input_shape_hash(el) for el in inputs))
    elif torch.is_tensor(inputs):
        # Get the hash of the tensor shape
        return hash(inputs.shape)
    else:
        # Get the hash of the inputs
        return hash(inputs)


def copy_to(dst, src):
    """
    Copies the dat from the source object to the target object.

    Args:
        dst: target
        src: source
    """
    if type(dst) != type(src):
        raise TypeError(
            f"dst and src should have the same type, but dst is a {type(dst)} whereas src is a {type(src)}."
        )

    if isinstance(dst, dict):
        for (dst_key, dst_value), (src_key, src_value) in zip(dst.items(), src.items()):
            if dst_key == src_key:
                copy_to(dst_value, src_value)
            else:
                raise ValueError(f"dst_key and src_key should be equal but got {dst_key} and {src_key}.")
    elif isinstance(dst, list) or isinstance(dst, tuple):
        for d, s in zip(dst, src):
            copy_to(d, s)
    elif torch.is_tensor(dst):
        dst.copy_(src, non_blocking=True)


class CachedParams:
    """
    Manages cached inputs, outputs and graph for HPU graphs.
    """

    def __init__(self, graph_inputs, graph_outputs, graph):
        self.graph_inputs = graph_inputs
        self.graph_outputs = graph_outputs
        self.graph = graph
