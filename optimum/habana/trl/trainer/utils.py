# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
import numpy as np

from typing import Optional
from transformers.data.data_collator import DataCollatorMixin


class BaseDataCollatorForLanguageModeling(DataCollatorMixin):
    def __init__(self, pad_token_id: int, return_tensors: str = "pt", buckets: Optional[list[int]] = None):
        self.pad_token_id = pad_token_id
        self.return_tensors = return_tensors
        self.buckets = buckets

    def _get_bucketed_len(self, examples):
        max_sentence_len = max([len(k["input_ids"]) for k in examples])
        if max_sentence_len > self.buckets[-1]:
            self.buckets = np.append(self.buckets, max_sentence_len)
            curr_bucket = max_sentence_len
        else:
            curr_bucket = self.buckets[np.argmin(np.where(max_sentence_len <= self.buckets))]
        return curr_bucket


def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    bucket_size: int = 0
) -> torch.Tensor:
    """
    Copied from pad https://github.com/huggingface/trl/blob/v0.17.0/trl/trainer/utils.py#L418
    The differences are:
        - bucket_size: If bucket_size > 0, all tensors are padded to that fixed length.
            Otherwise, the maximum size in the batch is used
    """
    if len(tensors) == 0:
        return torch.tensor([])

    # Determine the shape to pad to
    max_shape = np.max([t.shape for t in tensors], axis=0)
    target_length = bucket_size if bucket_size > 0 else max_shape[0]
    output_shape = [len(tensors), target_length] + max_shape[1:].tolist()

    # Create the output tensor
    output = torch.full(
        output_shape,
        padding_value,
        dtype=tensors[0].dtype,
        device=tensors[0].device
    )

    for i, t in enumerate(tensors):
        if t.shape[0] > target_length:
            raise ValueError(f"Tensor {i} length ({t.shape[0]}) exceeds bucket size {target_length}")

        if padding_side == "left":
            seq_start = target_length - t.shape[0]
            seq_slice = slice(seq_start, target_length)
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output
