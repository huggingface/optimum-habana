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

from typing import List, Tuple, Union

import numpy as np
import torch


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
    elif isinstance(logits, dict):
        return {k: get_dtype(v) for k, v in logits.items()}
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
