# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils.import_utils import is_torchdynamo_compiling


@dataclass
class GaudiAttentionMaskConverter(AttentionMaskConverter):
    """
    Adapted from: https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/modeling_attn_mask_utils.py#L21

    Differences:
    - replace `triu` with similar logic here: https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/modeling_attn_mask_utils.py#L169
    """

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        # add lower triangular sliding window mask if necessary
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1

            # Replace tril with below
            row_indices = torch.arange(mask.size(0), device=mask.device).view(-1, 1)  # Reshape to column vector
            col_indices = torch.arange(mask.size(1), device=mask.device)
            context_mask = (col_indices <= row_indices + diagonal).bool().expand_as(mask)  # Expand to match mask shape

            # Recent changes in PyTorch prevent mutations on tensors converted with aten::_to_copy
            # See https://github.com/pytorch/pytorch/issues/127571
            if is_torchdynamo_compiling():
                mask = mask.clone()

            mask.masked_fill_(context_mask, torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        input_shape = (attention_mask_2d.shape[0], query_length)
        device = attention_mask_2d.device

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )
            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )

            # just create a bool tensor with shape [bsz, 1, tgt_seq_len, src_seq_len]
            # OOM problem can be prevent by using bool tensor
            bsz, src_len = attention_mask_2d.size()
            tgt_len = input_shape[-1] if input_shape[-1] is not None else src_len
            bool_mask = attention_mask_2d != 1.0
            expanded_attn_mask = bool_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(device=device)

            return causal_4d_mask.masked_fill(expanded_attn_mask, torch.finfo(dtype).min)
        elif self.sliding_window is not None:
            raise NotImplementedError("Sliding window is currently only implemented for causal masking")

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        return self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(device)


def _gaudi_prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Adapted from: https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/modeling_attn_mask_utils.py#L278

    Differences:
    - replace `AttentionMaskConverter` by `GaudiAttentionMaskConverter`
    """
    attn_mask_converter = GaudiAttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # if the 4D mask has correct shape - invert it and fill with negative infinity
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask
