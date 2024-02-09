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
            diagonal = past_key_values_length - sliding_window + 1

            # Replace triu with below
            row_indices = torch.arange(mask.size(0), device=mask.device).view(-1, 1)  # Reshape to column vector
            col_indices = torch.arange(mask.size(1), device=mask.device)
            context_mask = 1 - (col_indices >= row_indices + diagonal).int().expand_as(
                mask
            )  # Expand to match mask shape

            mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


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


def _gaudi_prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Adapted from: https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/modeling_attn_mask_utils.py#L331

    Differences:
    - No difference with : https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/modeling_attn_mask_utils.py#L331
      keep for potential performance improvement
    """
    attn_mask_converter = GaudiAttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length
    batch_size, query_length = input_shape

    # torch.jit.trace, symbolic_trace and torchdynamo with fullgraph=True are unable to capture the controlflow `is_causal=attention_mask is None and q_len > 1`
    # used as an SDPA argument. We keep compatibility with these tracing tools by always using SDPA's `attn_mask` argument in case we are tracing.
    # TODO: Fix this as well when using torchdynamo with fullgraph=True.
    is_tracing = torch.jit.is_tracing() or isinstance(inputs_embeds, torch.fx.Proxy)

    if attention_mask is not None:
        # 4d mask is passed through
        if len(attention_mask.shape) == 4:
            expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError(
                    f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
                )
            else:
                # if the 4D mask has correct shape - invert it and fill with negative infinity
                inverted_mask = 1.0 - attention_mask.to(inputs_embeds.dtype)
                attention_mask = inverted_mask.masked_fill(
                    inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
                )
                return attention_mask

        elif not is_tracing and torch.all(attention_mask == 1):
            if query_length == 1:
                # For query_length == 1, causal attention and bi-directional attention are the same.
                attention_mask = None
            elif key_value_length == query_length:
                attention_mask = None
            else:
                # Unfortunately, for query_length > 1 and key_value_length != query_length, we cannot generally ignore the attention mask, as SDPA causal mask generation
                # may be wrong. We will set `is_causal=False` in SDPA and rely on Transformers attention_mask instead, hence not setting it to None here.
                # Reference: https://github.com/pytorch/pytorch/issues/108108
                pass
    elif query_length > 1 and key_value_length != query_length:
        # See the comment above (https://github.com/pytorch/pytorch/issues/108108).
        # Ugly: we set it to True here to dispatch in the following controlflow to `to_causal_4d`.
        attention_mask = True
    elif is_tracing:
        raise ValueError(
            'Attention using SDPA can not be traced with torch.jit.trace when no attention_mask is provided. To solve this issue, please either load your model with the argument `attn_implementation="eager"` or pass an attention_mask input when tracing the model.'
        )

    if attention_mask is None:
        expanded_4d_mask = None
    elif attention_mask is True:
        expanded_4d_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
    else:
        expanded_4d_mask = attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            dtype=inputs_embeds.dtype,
            key_value_length=key_value_length,
        )

        # From PyTorch 2.1 onwards, F.scaled_dot_product_attention with the memory-efficient attention backend
        # produces nans if sequences are completely unattended in the attention mask. Details: https://github.com/pytorch/pytorch/issues/110213
        #
        # This fix is not applied in case we are tracing with torch.jit.trace or symbolic_trace, as _unmask_unattended has a data-dependent
        # controlflow that can not be captured properly.
        # TODO: _unmask_unattended does not work either with torch.compile when using fullgraph=True. We should find a way to detect this case.
        if query_length > 1 and not is_tracing:
            expanded_4d_mask = GaudiAttentionMaskConverter._unmask_unattended(
                expanded_4d_mask, attention_mask, unmasked_value=0.0
            )

    return expanded_4d_mask
