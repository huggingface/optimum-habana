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

import time
from typing import Any, List, Tuple, Union

import numpy as np
import torch


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


def speed_metrics(split, start_time, num_samples=None, num_steps=None, start_time_after_warmup=None):
    """
    Measure and return speed performance metrics.
    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.
    Args:
    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    - start_time: operation start time
    - num_steps: number of steps performed
    - start_time_after_warmup: time after warmup steps have been performed
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


def get_dtype(logits: Union[torch.Tensor, Tuple[torch.Tensor]]) -> Union[str, List[str]]:
    """
    Extract the dtype of logits.

    Args:
        logits (Union[torch.Tensor, Tuple[torch.Tensor]]): input

    Raises:
        TypeError: only torch.Tensor and tuple are supported

    Returns:
        Union[str, List[str]]: logits' dtype
    """
    if isinstance(logits, torch.Tensor):
        # The dtype of a Torch tensor has the format 'torch.XXX', XXX being the actual dtype
        logits_dtype = str(logits.dtype).split(".")[-1]
        # If mixed-precision training was performed, dtype must be 'float32' to be understood by Numpy
        if logits_dtype == "bfloat16":
            logits_dtype = "float32"
        return logits_dtype
    elif isinstance(logits, tuple):
        return [get_dtype(logits_tensor) for logits_tensor in logits]
    else:
        raise TypeError(f"logits should be of type torch.Tensor or tuple, got {type(logits)} which is not supported")


def convert_into_dtypes(
    preds: Union[np.ndarray, Tuple[np.ndarray]], dtypes: Union[str, List[str]]
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """
    Convert preds into dtypes.

    Args:
        preds (Union[np.ndarray, Tuple[np.ndarray]]): predictions to convert
        dtypes (Union[str, List[str]]): dtypes used for the conversion

    Raises:
        TypeError: only torch.Tensor and tuple are supported

    Returns:
        Union[np.ndarray, Tuple[np.ndarray]]: converted preds
    """
    if isinstance(preds, np.ndarray):
        if preds.dtype == dtypes:
            return preds
        else:
            return preds.astype(dtypes)
    elif isinstance(preds, tuple):
        return tuple(convert_into_dtypes(preds_tensor, dtypes[i]) for i, preds_tensor in enumerate(preds))
    else:
        raise TypeError(f"preds should be of type np.ndarray or tuple, got {type(preds)} which is not supported")
