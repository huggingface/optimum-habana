# coding=utf-8
# Copyright 2024 Tri Dao, Albert Gu, Technological Innovation Institute and HuggingFace Inc. team.
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
"""PyTorch FALCONMAMBA model."""
from typing import Optional

import torch
from transformers.utils import (
    logging,
)

from transformers.cache_utils import MambaCache

logger = logging.get_logger(__name__)

"""
Copys from https://github.com/huggingface/transformers/blob/53fad641cfdb5105e2470bcf3ef17ea8e25cc300/src/transformers/models/falcon_mamba/modeling_falcon_mamba.py#L762
The only differences are:
- Use the torch.index_select function to replace the slicing operation of Line 51
"""
def gaudi_FalconMambaForCausalLM_prepare_inputs_for_generation(
    self,
    input_ids,
    inputs_embeds=None,
    use_cache=None,
    cache_params: Optional[MambaCache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    **kwargs,
):
    if use_cache:
        # `cache_position` should have been initialized in `generate`
        if cache_position is None:
            raise ValueError(
                "`cache_position` should not be None as it should have been initialized in "
                "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
            )
        if cache_position[0] > 0:
            #input_ids = input_ids[:, -1].unsqueeze(-1)
            idx = torch.tensor([input_ids.size(1) - 1])
            input_ids = torch.index_select(input_ids, 1, idx)

            if attention_mask is not None:
                attention_mask = None

        else:
            # we initialize the `cache_position` to full size of `conv_states` at prefill stage
            # considering padding will be applied when input length is shorter, and truncation
            # will be applied when it is longer, so it will be equivalent to always have it match
            # the length of `cache_params.conv_states`, which is `config.conv_kernel`
            cache_position = torch.arange(0, self.config.conv_kernel, device=input_ids.device)

    if inputs_embeds is not None and cache_params is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids.contiguous()}

    model_inputs.update(
        {
            "cache_params": cache_params,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs