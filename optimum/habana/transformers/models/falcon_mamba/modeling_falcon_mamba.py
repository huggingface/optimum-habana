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

from typing import Optional, Tuple, Union

import habana_frameworks.torch.core as htcore
import torch
from transformers.cache_utils import MambaCache
from transformers.models.falcon_mamba.modeling_falcon_mamba import FalconMambaOutput
from transformers.utils import (
    logging,
)


logger = logging.get_logger(__name__)

"""
Copys from https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/falcon_mamba/modeling_falcon_mamba.py#L635
The only differences are:
- Use the mark_step function to reduce the graph compiling time.
"""


def gaudi_FalconMambaModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.LongTensor] = None,
    cache_params: Optional[MambaCache] = None,
    use_cache: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    lazy_mode: Optional[bool] = True,
) -> Union[Tuple, FalconMambaOutput]:
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embeddings(input_ids)

    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False

    if use_cache:
        if cache_params is None:
            cache_params = MambaCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
        elif cache_position is None:
            # cases when we do manual forward instead of using `model.generate` which will initiate
            # `cache_position` and makes sure it is not None, throw error here instead of doing some
            # hack to conjecture the current cache position
            raise ValueError(
                "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                "be initialized for you automatically"
            )
    else:
        cache_params = None
    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    for mixer_block in self.layers:
        if lazy_mode:
            htcore.mark_step()
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                mixer_block.__call__, hidden_states, cache_params, cache_position, attention_mask
            )
        else:
            hidden_states = mixer_block(
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

    hidden_states = self.norm_f(hidden_states)
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

    return FalconMambaOutput(
        last_hidden_state=hidden_states,
        cache_params=cache_params if use_cache else None,
        hidden_states=all_hidden_states,
    )


"""
Copys from https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/falcon_mamba/modeling_falcon_mamba.py#L762
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
            # input_ids = input_ids[:, -1].unsqueeze(-1)
            idx = torch.tensor([input_ids.size(1) - 1], device=input_ids.device)
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
