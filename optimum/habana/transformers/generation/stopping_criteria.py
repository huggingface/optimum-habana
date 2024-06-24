# coding=utf-8
# Copyright 2022 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Union

import torch

from optimum.utils import logging


logger = logging.get_logger(__name__)

# Instead of returning a tensor describing status of completeness of each sentence
# we only return a single boolean describing the state of the batch
# only when needs_tensor_output says so, we return array of booleans


def create_return_const_tensor(input_ids, is_done):
    return torch.full((input_ids.shape[0],), 1 if is_done else 0, device=input_ids.device, dtype=torch.uint8)


def gaudi_MaxLengthCriteria_call(
    self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
) -> Union[torch.BoolTensor, bool]:
    token_idx = kwargs.get("token_idx", None)
    if token_idx is not None:
        assert not kwargs["needs_tensor_output"]
        return token_idx >= self.max_length
    else:
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
            logger.warning_once(
                "This is a friendly reminder - the current text generation call will exceed the model's predefined "
                f"maximum length ({self.max_position_embeddings}). Depending on the model, you may observe "
                "exceptions, performance degradation, or nothing at all."
            )
        return create_return_const_tensor(input_ids, is_done)


def gaudi_MaxNewTokensCriteria_call(
    self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
) -> Union[torch.BoolTensor, bool]:
    token_idx = kwargs.get("token_idx", None)
    if token_idx is not None:
        assert not kwargs["needs_tensor_output"]
        return token_idx >= self.max_length
    else:
        is_done = input_ids.shape[-1] >= self.max_length
        return create_return_const_tensor(input_ids, is_done)


def gaudi_MaxTimeCriteria_call(
    self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
) -> Union[torch.BoolTensor, bool]:
    is_done = time.time() - self.initial_timestamp > self.max_time
    if kwargs["needs_tensor_output"]:
        return create_return_const_tensor(input_ids, is_done)
    else:
        return is_done


def gaudi_EosTokenCriteria_call(
    self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
) -> Union[torch.BoolTensor, bool]:
    self.eos_token_id = self.eos_token_id.to(input_ids.device)
    token_idx = kwargs.get("token_idx", None)
    if token_idx is not None:
        assert not kwargs["needs_tensor_output"]
        is_done = torch.isin(input_ids[:, token_idx - 1], self.eos_token_id)
    else:
        is_done = torch.isin(input_ids[:, -1], self.eos_token_id)
    if kwargs["needs_tensor_output"]:
        return is_done.byte()
    else:
        return torch.all(is_done).item()


def needs_tensor_output(token_idx, ignore_eos, eos_token_id) -> bool:
    if token_idx is None:
        return not ignore_eos and eos_token_id is not None
    else:
        # token_idx is present, so we have static shapes, so using single boolean
        return False


def gaudi_StoppingCriteriaList_call(
    self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
) -> Union[torch.BoolTensor, bool]:
    kwargs["needs_tensor_output"] = needs_tensor_output(
        kwargs.get("token_idx", None), kwargs.get("ignore_eos", True), kwargs.get("eos_token_id", None)
    )
    is_done = (
        torch.full((input_ids.shape[0],), 0, device=input_ids.device, dtype=torch.int8)
        if kwargs["needs_tensor_output"]
        else False
    )
    for criteria in self:
        is_done = is_done | criteria(input_ids, scores, **kwargs)
    return is_done
