# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import math
from typing import Optional, Tuple, Union

import torch


def gaudi_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    # The div has been put inside the matmul because it achieves better performance on HPU.
    attn_weights = torch.matmul(query, key.transpose(-1, -2) / math.sqrt(module.attention_head_size))

    # Normalize the attention scores to probabilities.
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Mask heads if we want to
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def gaudi_vit_self_attention_forward(
    self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    """
    Same method as transformers.models.vit.modeling_vit.ViTSelfAttention.forward with a small tweak:
    the division is performed before the matmul for computing attention scores.
    This gives better performance on HPU.
    """
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    query_layer = self.transpose_for_scores(self.query(hidden_states))

    context_layer, attention_probs = gaudi_eager_attention_forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        head_mask,
        is_causal=self.is_causal,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.dropout_prob,
    )

    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.reshape(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    return outputs
