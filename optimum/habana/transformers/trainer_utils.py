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

from typing import Any, Dict, List, Tuple, Union

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
            assert dst_key == src_key
            if dst_key == src_key:
                copy_to(dst_value, src_value)
            else:
                raise ValueError(f"dst_key and src_key should be equal but got {dst_key} and {src_key}.")
    elif isinstance(dst, list) or isinstance(dst, tuple):
        for d, s in zip(dst, src):
            copy_to(d, s)
    elif torch.is_tensor(dst):
        dst.copy_(src, non_blocking=True)


# Class to manage cached inputs, outputs and graph for HPU graphs
class CachedParams:
    def __init__(self, graph_inputs, graph_outputs, graph):
        self.graph_inputs = graph_inputs
        self.graph_outputs = graph_outputs
        self.graph = graph
