# coding=utf-8
# Copyright 2022 HuggingFace Inc. team.
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

import transformers
from transformers import BloomForCausalLM

from ...generation_utils import GaudiGenerationMixin
from ...modeling_utils import GaudiMixin, register


def attention_mask_func(attention_scores, attention_mask, causal_mask):
    attention_mask_bool = ~attention_mask.bool()

    query_length, key_length, n_heads = attention_scores.size(2), attention_scores.size(3), attention_scores.size(1)
    padded_causal_mask = torch.logical_or(
        attention_mask_bool[:, None, key_length - query_length : key_length, None].repeat(
            1, 1, 1, attention_mask_bool.shape[-1]
        ),
        ~causal_mask[:, :, key_length - query_length : key_length, :key_length].bool(),
    )
    padded_causal_mask = torch.logical_or(padded_causal_mask, attention_mask_bool[:, None, None, :key_length])
    # Make use of floats
    return (
        attention_scores.masked_fill_(padded_causal_mask.expand(-1, n_heads, -1, -1), -10000.0),
        padded_causal_mask,
    )


transformers.models.bloom.modeling_bloom.attention_mask_func = attention_mask_func


@register(BloomForCausalLM)
class GaudiBloomForCausalLM(BloomForCausalLM, GaudiMixin, GaudiGenerationMixin):
    pass
