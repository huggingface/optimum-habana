# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
# Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
###############################################################################

"""PyTorch ChatGLM model."""

import copy
import json
import math
import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn.utils import skip_init
from transformers.cache_utils import Cache
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import logging

from ...modeling_attn_mask_utils import _gaudi_prepare_4d_causal_attention_mask
from .configuration_chatglm import ChatGLMConfig


"""
Adapted from the following sources:
https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py
https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py
"""

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Cannot import Fused SDPA from Habana Torch")
    FusedSDPA = None

try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV3 as FusedRoPE
except ImportError:
    print("Cannot import Fused Rope from Habana Torch")
    FusedRoPE = None

try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm
except ImportError:
    print("Cannot import Fused RMSNorm from Habana Torch")
    FusedRMSNorm = None


logger = logging.get_logger(__name__)

MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["chatglm"] = "ChatGLMForConditionalGeneration"
_CONFIG_FOR_DOC = "ChatGLMConfig"


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)


#  FusedScaledDotProductAttention
class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        self._hpu_kernel_fsdpa = fusedSDPA

    def forward(self, query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode):
        return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode)


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 198] = 5e4
        return scores


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class Matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class KVCache(torch.nn.Module):
    def __init__(self):
        super(KVCache, self).__init__()
        self.cache = None
        self.inp_seq_len = -1

    def allocate(self, inp_seq_len, dtype, device, shape):
        if self.cache is None or self.cache.shape != shape:
            self.inp_seq_len = inp_seq_len
            # self.cache = torch.zeros(shape, dtype=dtype, device=device)
            self.cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        else:
            assert self.inp_seq_len == inp_seq_len, (
                f"inp_seq_len must be the same. self.inp_seq_len:{self.inp_seq_len} inp_seq_len:{inp_seq_len}"
            )
            self.cache.fill_(0)

    def update(self, prev, cur, dim, idx, inp_seq_len):
        orig_cur = cur
        if prev.shape == cur.shape:
            prev.copy_(cur)
            return orig_cur
        if cur.shape[2] > 1 and cur.shape[2] <= prev.shape[2]:
            # Initialize
            prev[:, :, :inp_seq_len, :].copy_(cur)
            return orig_cur
        assert cur.shape[2] == 1, f"Cannot update kv-cache. Unsupported shapes. prev:{prev.shape} cur:{cur.shape}"
        if idx is not None:
            prev.index_copy_(dim, idx - 1, cur)
            return prev
        else:
            return torch.cat((prev, cur), dim=dim)

    def get_shape(self):
        if self.cache is None:
            return None
        return self.cache.shape

    def forward(self, cur, dim, idx):
        return self.update(self.cache, cur, dim, idx, self.inp_seq_len)


def gaudi_chatglm_repeat_kv(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """
    Refer https://github.com/huggingface/optimum-habana/blob/main/optimum/habana/transformers/models/llama/modeling_llama.py#L109
    Copied from repeat_kv: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
        - Append num_key_value_heads == 1 check as kv states can be broadcasted during matmuls so need to expand and reshape them.
        - Add new args query_states, key_states, value_states and attention_mask and update the logic for expansion.
    The query states go from (batch, num_heads, seqlen, head_dim) to (batch, num_key_value_heads, n_rep, seqlen, head_dim)
    The key/value states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_key_value_heads, 1, seqlen, head_dim)
    """
    batch, num_query_heads, q_len, head_dim = query_layer.shape
    batch, num_key_value_heads, kv_len, head_dim = key_layer.shape
    n_rep = num_query_heads // num_key_value_heads
    if n_rep == 1 or num_key_value_heads == 1:
        return query_layer, key_layer, value_layer, attention_mask

    new_kv_shape = (batch, num_key_value_heads, 1, kv_len, head_dim)
    key_layer = key_layer.reshape(new_kv_shape)
    value_layer = value_layer.reshape(new_kv_shape)

    new_q_shape = (batch, num_key_value_heads, n_rep, q_len, head_dim)
    query_layer = query_layer.reshape(new_q_shape)

    if attention_mask is not None:
        # Add groups dim and set to 1
        attention_mask = attention_mask.unsqueeze(1)

    return query_layer, key_layer, value_layer, attention_mask


def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs


def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    data_dtype = x.dtype
    compute_dtype = rope_cache.dtype

    if x.device.type == "hpu" and FusedRoPE:
        x_out = FusedRoPE.apply(x.to(compute_dtype), rope_cache)
    else:
        x = x.to(compute_dtype)
        # x: [sq, b, np, hn]
        sq, _b, np, _hn = x.size(0), x.size(1), x.size(2), x.size(3)
        rot_dim = rope_cache.shape[-2] * 2
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        # truncate to support variable sizes
        rope_cache = rope_cache[:sq]
        xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
        rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out2 = x_out2.flatten(3)
        x_out = torch.cat((x_out2, x_pass), dim=-1)

    return x_out.to(data_dtype)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
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
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class CoreAttention(torch.nn.Module):
    def __init__(self, config: ChatGLMConfig, layer_number):
        super(CoreAttention, self).__init__()

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.dropout_rate = config.attention_dropout
        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

        self.matmul_qk = Matmul()
        self.matmul_av = Matmul()
        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA) if FusedSDPA else None

        self.q_block_size = 4096

    def gaudi_flash_attn_v1(self, query_layer, key_layer, value_layer, attention_mask, softmax_mode):
        """
        Gaudi version of Flash Attention V1 to support long sequence at prompt phase
        Causal mask is not supported in this optimization
        """
        q_len = query_layer.size(-2)
        q_tiles = (
            (q_len // self.q_block_size) if (q_len % self.q_block_size == 0) else math.ceil(q_len / self.q_block_size)
        )
        q_padding = q_tiles * self.q_block_size - q_len
        query_layer = F.pad(query_layer, (0, 0, 0, q_padding), "constant", 0)
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, 0, 0, q_padding), "constant", torch.finfo(key_layer.dtype).min)

        row_o_list = []
        for i in range(q_tiles):
            s, e = i * self.q_block_size, (i + 1) * self.q_block_size
            row_q = query_layer[:, :, s:e, :]
            row_mask = attention_mask[:, :, s:e, :]
            attn_output_partial = self.fused_scaled_dot_product_attention(
                row_q, key_layer, value_layer, row_mask, self.dropout_rate, False, None, softmax_mode
            )
            row_o_list.append(attn_output_partial)
        attn_output = torch.cat(row_o_list, dim=-2)

        if q_padding != 0:
            attn_output = attn_output[:, :, :-q_padding, :]

        return attn_output

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: Optional[torch.LongTensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
    ):
        bsz, _, q_len, _ = query_layer.shape

        if use_flash_attention and FusedSDPA:
            if not self.training:
                self.dropout_rate = 0.0

            import habana_frameworks.torch.hpu as ht

            softmax_mode = "fast" if flash_attention_fast_softmax else "None"

            if q_len == 1:
                # next token
                use_recompute = True if os.getenv("QUANT_CONFIG", "") else False
                with ht.sdp_kernel(enable_recompute=use_recompute):
                    attn_output = self.fused_scaled_dot_product_attention(
                        query_layer,
                        key_layer,
                        value_layer,
                        attention_mask,
                        self.dropout_rate,
                        False,
                        None,
                        softmax_mode,
                    )
            else:
                # first token
                if flash_attention_causal_mask:
                    # causal masking on first token requires inputs to be of the same length
                    with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                        attn_output = self.fused_scaled_dot_product_attention(
                            query_layer, key_layer, value_layer, None, self.dropout_rate, True, None, softmax_mode
                        )
                else:
                    with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                        if (q_len > 8192 or (q_len >= 6144 and bsz >= 2)) and self.training:
                            attn_output = self.gaudi_flash_attn_v1(
                                query_layer, key_layer, value_layer, attention_mask, softmax_mode
                            )
                            htcore.mark_step()
                        else:
                            attn_output = self.fused_scaled_dot_product_attention(
                                query_layer,
                                key_layer,
                                value_layer,
                                attention_mask,
                                self.dropout_rate,
                                False,
                                None,
                                softmax_mode,
                            )
        else:
            query_layer, key_layer, value_layer, attention_mask = gaudi_chatglm_repeat_kv(
                query_layer, key_layer, value_layer, attention_mask
            )
            attn_weights = self.matmul_qk(query_layer, key_layer.transpose(-2, -1)) / self.norm_factor

            if self.coeff is not None:
                attn_weights = attn_weights * self.coeff

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask
                if cache_position is not None:
                    causal_mask = attention_mask[:, :, cache_position, : key_layer.shape[-2]]
                attn_weights = attn_weights + causal_mask

            if attn_softmax_bf16:
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=query_layer.dtype)
            else:
                # upcast attention to fp32
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                    query_layer.dtype
                )
            if self.training:
                attn_weights = self.attention_dropout(attn_weights)
            attn_output = self.matmul_av(attn_weights, value_layer)
            attn_output = attn_output.reshape(bsz, -1, q_len, self.hidden_size_per_attention_head)

        # =================
        # Output. [sq, b, h]
        # =================

        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()

        context_layer = attn_output.reshape(q_len, bsz, self.hidden_size_per_partition)

        return context_layer


class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)
        self.config = config

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nn.Linear(
            config.hidden_size,
            self.qkv_hidden_size,
            bias=config.add_bias_linear or config.add_qkv_bias,
            device=device,
            **_config_to_kwargs(config),
        )
        self.core_attention = CoreAttention(config, self.layer_number)

        # Output.
        self.dense = nn.Linear(
            self.projection_size,
            config.hidden_size,
            bias=config.add_bias_linear,
            device=device,
            **_config_to_kwargs(config),
        )

        self.k_cache = KVCache()
        self.v_cache = KVCache()
        self.inp_seq_len = -1

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        cache_shape = (
            batch_size,
            self.num_multi_query_groups_per_partition,
            max_seq_len,
            self.hidden_size_per_attention_head,
        )
        device = self.query_key_value.weight.device
        dtype = self.config.torch_dtype
        self.k_cache.allocate(inp_seq_len, dtype, device, cache_shape)
        self.v_cache.allocate(inp_seq_len, dtype, device, cache_shape)

    def reorder(self, tensor, beam_idx, dim_a, dim_b):
        updated = tensor.index_select(0, beam_idx)
        tensor.copy_(updated)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        if self.k_cache.cache is None:
            return (None, None)

        head_dim = self.k_cache.cache.size(-1)
        seq_length = self.k_cache.cache.size(-2)
        self.reorder(self.k_cache.cache, beam_idx, seq_length, head_dim)
        self.reorder(self.v_cache.cache, beam_idx, seq_length, head_dim)
        return (self.k_cache.cache.shape, self.v_cache.cache.shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_encoder: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        cache_idx: int = None,
    ):
        # hidden_states: [sq, b, h]
        q_len, bsz, hiddenSize = hidden_states.size()

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        if prefix_encoder is not None:
            prefix_encoder_key, prefix_encoder_value = prefix_encoder
            if mixed_x_layer.dtype == torch.float8_e4m3fn:
                from habana_frameworks.torch.hpex.kernels.Fp8Ops import cast_to_fp8_v2

                prefix_encoder_key = cast_to_fp8_v2(prefix_encoder_key, None, False, False, mixed_x_layer.dtype)[0]
                prefix_encoder_value = cast_to_fp8_v2(prefix_encoder_value, None, False, False, mixed_x_layer.dtype)[0]
            else:
                prefix_encoder_key = prefix_encoder_key.to(mixed_x_layer.dtype)
                prefix_encoder_value = prefix_encoder_value.to(mixed_x_layer.dtype)

            key_layer = torch.cat((prefix_encoder_key, key_layer), dim=0)
            value_layer = torch.cat((prefix_encoder_value, value_layer), dim=0)

        query_layer = query_layer.permute(1, 2, 0, 3).contiguous()
        key_layer = key_layer.permute(1, 2, 0, 3).contiguous()
        value_layer = value_layer.permute(1, 2, 0, 3).contiguous()

        if use_cache:
            # reuse k, v, self_attention
            if reuse_cache:
                key_layer = self.k_cache(key_layer, 2, token_idx)
                value_layer = self.v_cache(value_layer, 2, token_idx)
                past_key_value = (self.k_cache.get_shape(), self.v_cache.get_shape())
            else:
                if past_key_value is None:
                    past_key = torch.zeros(
                        key_layer.shape, dtype=self.query_key_value.weight.dtype, device=key_layer.device
                    )
                    past_value = torch.zeros(
                        key_layer.shape, dtype=self.query_key_value.weight.dtype, device=key_layer.device
                    )
                    past_key_value = [past_key, past_value]
                key_layer = self.k_cache.update(past_key_value[0], key_layer, 2, token_idx, self.inp_seq_len)
                value_layer = self.v_cache.update(past_key_value[1], value_layer, 2, token_idx, self.inp_seq_len)
                if token_idx is None:
                    past_key_value = (key_layer, value_layer)

            if cache_idx is not None and q_len == 1:
                key_layer = key_layer[:, :, :cache_idx, :]
                value_layer = value_layer[:, :, :cache_idx, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :, :, :cache_idx]
        else:
            past_key_value = None

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            cache_position,
            attn_softmax_bf16,
            use_flash_attention,
            flash_attention_recompute,
            flash_attention_causal_mask,
            flash_attention_fast_softmax,
        )

        output = self.dense(context_layer)

        # No output_attention
        attn_weights = None

        return output, attn_weights, past_key_value


class MLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: ChatGLMConfig, device=None):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config),
        )

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size, config.hidden_size, bias=self.add_bias, device=device, **_config_to_kwargs(config)
        )

    def forward(self, hidden_states):
        # [b, s, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, rope_ratio=1, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl
        self.rope_ratio = rope_ratio

    def forward_impl(self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        base = base * self.rope_ratio
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device)


class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        if hidden_states.device.type == "hpu" and FusedRMSNorm:
            # mixed dtypes are not good for FusedRMSNorm, both inputs need to have same dtype
            if hidden_states.dtype != self.weight.dtype:
                orig_dtype = hidden_states.dtype
                hidden_states = FusedRMSNorm.apply(hidden_states.to(self.weight.dtype), self.weight, self.eps)
                return hidden_states.to(orig_dtype)
            else:
                hidden_states = FusedRMSNorm.apply(hidden_states, self.weight, self.eps)
                return hidden_states
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            return self.weight * hidden_states.to(input_dtype)


class GLMBlock(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(GLMBlock, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(
            config.hidden_size, eps=config.layernorm_epsilon, device=device, dtype=config.torch_dtype
        )

        # Self attention.
        self.self_attention = SelfAttention(config, layer_number, device=device)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(
            config.hidden_size, eps=config.layernorm_epsilon, device=device, dtype=config.torch_dtype
        )

        # MLP
        self.mlp = MLP(config, device=device)

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.self_attention.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.self_attention.reorder_kv_cache(beam_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_encoder: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        cache_idx: int = None,
    ):
        # hidden_states: [b, s, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, self_attn_weights, present_key_value = self.self_attention(
            layernorm_output,
            attention_mask,
            prefix_encoder,
            rotary_pos_emb,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            token_idx,
            attn_softmax_bf16,
            reuse_cache,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_fast_softmax=flash_attention_fast_softmax,
            cache_idx=cache_idx,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        outputs = (output,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class GLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(GLMTransformer, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(
                config.hidden_size, eps=config.layernorm_epsilon, device=device, dtype=config.torch_dtype
            )

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        for layer in self.layers:
            layer.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return tuple(layer.reorder_kv_cache(beam_idx) for layer in self.layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_encoders: Optional[List[torch.FloatTensor]] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        cache_idx: int = None,
        lazy_mode: Optional[bool] = True,
    ):
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        if lazy_mode:
            htcore.mark_step()

        for index in range(self.num_layers):
            if (
                lazy_mode
                and not self.training
                and (torch.distributed.is_initialized() is False or torch.distributed.get_world_size() == 1)
            ):
                htcore.mark_step()

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[index] if past_key_values is not None else None
            prefix_encoder = prefix_encoders[index] if prefix_encoders is not None else None

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(
                            *inputs,
                            past_key_values,
                            output_attentions,
                            use_cache,
                            cache_position,
                            None,
                            attn_softmax_bf16,
                            False,
                            use_flash_attention,
                            flash_attention_recompute,
                            flash_attention_causal_mask,
                            flash_attention_fast_softmax,
                            None,
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    prefix_encoder,
                    rotary_pos_emb,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    prefix_encoder=prefix_encoder,
                    rotary_pos_emb=rotary_pos_emb,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    token_idx=token_idx,
                    attn_softmax_bf16=attn_softmax_bf16,
                    reuse_cache=reuse_cache,
                    use_flash_attention=use_flash_attention,
                    flash_attention_recompute=flash_attention_recompute,
                    flash_attention_fast_softmax=flash_attention_fast_softmax,
                    cache_idx=cache_idx,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        next_cache = next_decoder_cache if use_cache else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, next_cache, all_hidden_states, all_self_attns


class ChatGLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = ChatGLMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        return

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        if past_length:
            full_attention_mask = torch.cat(
                (torch.ones(batch_size, seq_length, past_length, device=input_ids.device), full_attention_mask), dim=-1
            )
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GLMTransformer):
            module.gradient_checkpointing = value


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            kv_size = config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            self.embedding = torch.nn.Embedding(config.pre_seq_len, kv_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(kv_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, kv_size),
            )
        else:
            self.embedding = torch.nn.Embedding(
                config.pre_seq_len, config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            )

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class Embedding(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size, self.hidden_size, dtype=config.torch_dtype, device=device
        )
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class ChatGLMModel(ChatGLMPreTrainedModel):
    def __init__(self, config: ChatGLMConfig, device=None, empty_init=False):
        super().__init__(config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device
        self.embedding = init_method(Embedding, config, **init_kwargs)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        # Rotary positional embeddings
        self.seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        self.rotary_pos_emb = RotaryEmbedding(
            rotary_dim // 2, original_impl=config.original_rope, device=device, dtype=config.torch_dtype
        )
        self.encoder = init_method(GLMTransformer, config, **init_kwargs)
        self.output_layer = init_method(
            nn.Linear,
            config.hidden_size,
            config.padded_vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            **init_kwargs,
        )
        self.pre_seq_len = config.pre_seq_len if config.pre_seq_len is not None else 0
        self.prefix_projection = config.prefix_projection
        if self.pre_seq_len > 0:
            for param in self.parameters():
                param.requires_grad = False
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(config)
            self.dropout = torch.nn.Dropout(0.1)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def get_prompt(self, batch_size, device, dtype=torch.bfloat16):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
        past_key_values = past_key_values.view(
            batch_size, self.pre_seq_len, self.num_layers * 2, self.multi_query_group_num, self.kv_channels
        )
        # seq_len, b, nh, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)

        return past_key_values

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.encoder.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.encoder.reorder_kv_cache(beam_idx)

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length, pre_seq_len
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length + pre_seq_len,
            )
            return combined_attention_mask

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            if pre_seq_len > 0:
                pre_seq_mask = torch.zeros(
                    [input_shape[0], 1, 1, pre_seq_len],
                    dtype=expanded_attn_mask.dtype,
                    device=expanded_attn_mask.device,
                )
                expanded_attn_mask = torch.cat([pre_seq_mask, expanded_attn_mask], dim=-1)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        cache_idx: int = None,
        lazy_mode: Optional[bool] = True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids).to(self.embedding.word_embeddings.weight.dtype)

        past_seen_tokens = 0

        if past_key_values is not None and use_cache:  # kept for BC (cache positions)
            if reuse_cache:
                if isinstance(past_key_values[0][0], torch.Tensor):
                    past_seen_tokens = past_key_values[0][0].shape[2]
                else:
                    past_seen_tokens = past_key_values[0][0][2]
            else:
                past_seen_tokens = past_key_values[0][0].shape[2]

        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens, seq_length + past_seen_tokens, dtype=torch.long, device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0)
        if position_ids.size(-1) < seq_length:
            position_ids = F.pad(position_ids, (0, seq_length - position_ids.size(-1)), "constant", 0)
        cache_position = None

        # embed positions
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        rotary_pos_emb = rotary_pos_emb[position_ids]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        if attention_mask is None:
            mask_len = past_seen_tokens if past_seen_tokens else seq_length
            attention_mask = torch.ones((batch_size, mask_len), dtype=torch.bool, device=inputs_embeds.device)

        prefix_encoders = None
        if self.pre_seq_len > 0:
            if token_idx is not None:
                token_idx = token_idx + self.pre_seq_len
            if past_key_values is None:
                prefix_encoders = self.get_prompt(
                    batch_size=batch_size, device=input_ids.device, dtype=inputs_embeds.dtype
                )
                past_seen_tokens += self.pre_seq_len
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask.new_ones((batch_size, self.pre_seq_len)), attention_mask], dim=-1
                    )

        attention_mask = _gaudi_prepare_4d_causal_attention_mask(
            attention_mask,
            input_ids.shape if input_ids is not None else (batch_size, seq_length),
            inputs_embeds,
            past_seen_tokens,
        )

        # Run encoder.
        hidden_states, next_cache, all_hidden_states, all_self_attns = self.encoder(
            inputs_embeds,
            attention_mask,
            prefix_encoders,
            rotary_pos_emb,
            past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            token_idx=token_idx,
            attn_softmax_bf16=attn_softmax_bf16,
            reuse_cache=reuse_cache,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
            flash_attention_fast_softmax=flash_attention_fast_softmax,
            cache_idx=cache_idx,
            lazy_mode=lazy_mode,
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):
    def __init__(self, config: ChatGLMConfig, empty_init=False, device=None):
        super().__init__(config)

        self.max_sequence_length = config.max_length
        self.transformer = ChatGLMModel(config, empty_init=empty_init, device=device)
        self.config = config

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.transformer.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)
        self.kv_cache_len = max_seq_len

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.transformer.reorder_kv_cache(beam_idx)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask=None,
        inputs_embeds=None,
        token_idx=None,
        **kwargs,
    ):
        reuse_cache = kwargs.get("reuse_cache")
        bucket_internal = kwargs.get("bucket_internal")

        if past_key_values:
            if token_idx is not None:
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)
            else:
                input_ids = input_ids[:, -1:]
        elif (reuse_cache or bucket_internal) and token_idx is not None:
            # With reuse_cache, KV cache is pre allocated hence for the 1st token we can slice the inputs till token idx for the fwd pass
            input_ids = input_ids[:, :token_idx]
            attention_mask = attention_mask[:, :token_idx]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                if token_idx is not None:
                    position_ids = torch.index_select(position_ids, 1, token_idx - 1)
                else:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "token_idx": token_idx,
                "trim_logits": kwargs.get("trim_logits"),
                "attn_softmax_bf16": kwargs.get("attn_softmax_bf16"),
                "reuse_cache": reuse_cache,
                "use_flash_attention": kwargs.get("use_flash_attention"),
                "flash_attention_recompute": kwargs.get("flash_attention_recompute"),
                "flash_attention_causal_mask": kwargs.get("flash_attention_causal_mask"),
                "flash_attention_fast_softmax": kwargs.get("flash_attention_fast_softmax"),
                "cache_idx": kwargs.get("cache_idx"),
                "lazy_mode": kwargs.get("lazy_mode"),
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        trim_logits: Optional[bool] = False,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        cache_idx: int = None,
        lazy_mode: Optional[bool] = True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_idx=token_idx,
            attn_softmax_bf16=attn_softmax_bf16,
            reuse_cache=reuse_cache,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
            flash_attention_fast_softmax=flash_attention_fast_softmax,
            cache_idx=cache_idx,
            lazy_mode=lazy_mode,
        )

        hidden_states = outputs[0].transpose(0, 1).contiguous()
        _, seq_len, _ = hidden_states.shape
        if seq_len > 1 and trim_logits and not self.training:
            if token_idx is not None:
                hidden_states = hidden_states.index_select(1, token_idx - 1)
            else:
                hidden_states = hidden_states[:, -1, :]

        lm_logits = self.transformer.output_layer(hidden_states).float()

        loss = None
        if labels is not None:
            lm_logits = lm_logits

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(1, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )

    def process_response(self, output, history):
        content = ""
        history = copy.deepcopy(history)
        for response in output.split("<|assistant|>"):
            if "\n" in response:
                metadata, content = response.split("\n", maxsplit=1)
            else:
                metadata, content = "", response
            if not metadata.strip():
                content = content.strip()
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                content = content.replace("[[训¤~C¤~W¤¤~W¤]]", "2023年")
            else:
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                if history[0]["role"] == "system" and "tools" in history[0]:
                    parameters = json.loads(content)
                    content = {"name": metadata.strip(), "parameters": parameters}
                else:
                    content = {"name": metadata.strip(), "content": content}
        return content, history

    def build_chat_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        # For chatglm2-6b, we need to use a different method to process the inputs.
        if self.config.name_or_path == "THUDM/chatglm2-6b":
            prompt = tokenizer.build_prompt(query, history=history)
            inputs = tokenizer([prompt], return_tensors="pt")
        else:
            inputs = tokenizer.apply_chat_template(
                history, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
            )

        inputs = inputs.to(self.device)
        return inputs

    def build_stream_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        if history:
            prompt = "\n\n[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = input_ids[1:]
            inputs = tokenizer.batch_encode_plus([(input_ids, None)], return_tensors="pt", add_special_tokens=False)
        else:
            prompt = "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    @torch.inference_mode()
    def chat(
        self,
        tokenizer,
        query: str,
        history: List[Dict] = None,
        role: str = "user",
        num_beams=1,
        do_sample=False,
        top_p=0.8,
        temperature=0.8,
        logits_processor=None,
        **kwargs,
    ):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {
            "num_beams": num_beams,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
            **kwargs,
        }
        history.append({"role": role, "content": query})
        inputs = self.build_chat_inputs(tokenizer, query, history=history)
        eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|user|>"),
            tokenizer.convert_tokens_to_ids("<|observation|>"),
        ]
        outputs = self.generate(**inputs, **gen_kwargs, eos_token_id=eos_token_id, ignore_eos=False)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]) : -1]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response, history = self.process_response(response, history)

        return response, history

    @torch.inference_mode()
    def stream_chat(
        self,
        tokenizer,
        query: str,
        history: List[Tuple[str, str]] = None,
        past_key_values=None,
        max_length: int = 8192,
        do_sample=True,
        top_p=0.8,
        temperature=0.8,
        logits_processor=None,
        return_past_key_values=False,
        **kwargs,
    ):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {
            "max_length": max_length,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
            **kwargs,
        }
        if past_key_values is None and not return_past_key_values:
            inputs = self.build_inputs(tokenizer, query, history=history)
        else:
            inputs = self.build_stream_inputs(tokenizer, query, history=history)
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[0]
            if self.transformer.pre_seq_len is not None:
                past_length -= self.transformer.pre_seq_len
            inputs.position_ids += past_length
            attention_mask = inputs.attention_mask
            attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
            inputs["attention_mask"] = attention_mask
        for outputs in self.stream_generate(
            **inputs, past_key_values=past_key_values, return_past_key_values=return_past_key_values, **gen_kwargs
        ):
            if return_past_key_values:
                outputs, past_key_values = outputs
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]) :]
            response = tokenizer.decode(outputs)
            if response and response[-1] != "�":
                response = self.process_response(response)
                new_history = history + [(query, response)]
                if return_past_key_values:
                    yield response, new_history, past_key_values
                else:
                    yield response, new_history

    @torch.inference_mode()
    def stream_generate(
        self,
        input_ids,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        return_past_key_values=False,
        **kwargs,
    ):
        input_ids_seq_length = input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        model_kwargs["use_cache"] = generation_config.use_cache
        eos_token_id = generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warn(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self._get_logits_warper(generation_config)

        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        scores = None
        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())
            if return_past_key_values:
                yield input_ids, outputs.past_key_values
            else:
                yield input_ids
            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break


class ChatGLMForSequenceClassification(ChatGLMPreTrainedModel):
    def __init__(self, config: ChatGLMConfig, empty_init=False, device=None):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.transformer = ChatGLMModel(config, empty_init=empty_init, device=device)

        self.classifier_head = nn.Linear(config.hidden_size, config.num_labels, bias=True, dtype=torch.bf16)
        if config.classifier_dropout is not None:
            self.dropout = nn.Dropout(config.classifier_dropout)
        else:
            self.dropout = None
        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        cache_idx: int = None,
        lazy_mode: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_idx=token_idx,
            attn_softmax_bf16=attn_softmax_bf16,
            reuse_cache=reuse_cache,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
            flash_attention_fast_softmax=flash_attention_fast_softmax,
            cache_idx=cache_idx,
            lazy_mode=lazy_mode,
        )

        hidden_states = transformer_outputs[0].transpose(0, 1).contiguous()
        pooled_hidden_states = hidden_states[-1]
        if self.dropout is not None:
            pooled_hidden_states = self.dropout(pooled_hidden_states)
        logits = self.classifier_head(pooled_hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze().float(), labels.squeeze())
                else:
                    loss = loss_fct(logits.float(), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels).float(), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.float(), labels.view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
