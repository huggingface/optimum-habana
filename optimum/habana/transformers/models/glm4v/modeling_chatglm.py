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

"""PyTorch GLM-4V model."""

import math
import os
from typing import List, Optional, Tuple, Union

import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn.utils import skip_init
from transformers.generation import GenerationMixin
from transformers.generation.logits_process import LogitsProcessor
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from optimum.habana.transformers.modeling_attn_mask_utils import _gaudi_prepare_4d_causal_attention_mask

from .configuration_chatglm import GLM4VConfig
from .visual import EVA2CLIPModel


"""
Adapted from the following source:
https://huggingface.co/THUDM/glm-4v-9b/blob/main/modeling_chatglm.py
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

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1

_CHECKPOINT_FOR_DOC = "THUDM/GLM4V"
_CONFIG_FOR_DOC = "GLM4VConfig"


#  FusedScaledDotProductAttention
class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        self._hpu_kernel_fsdpa = fusedSDPA

    def forward(self, query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode):
        return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode)


class Matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 198] = 5e4
        return scores


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config: GLM4VConfig):
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


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, rope_ratio=1, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl
        self.rope_ratio = rope_ratio
        self.seq_len_record = -1
        self.cache = None

    def impl(self, seq_length: int, dim: int, device: torch.device, dtype: torch.dtype):
        if self.seq_len_record != seq_length:
            self.seq_len_record = seq_length
            base = 10000 * self.rope_ratio
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
            seq = torch.arange(seq_length, device=inv_freq.device, dtype=torch.float32)
            freqs = torch.outer(seq, inv_freq)
            # first part even vector components, second part odd vector components,
            #  2 * dim in dimension size
            self.cache = torch.cat((freqs, freqs), dim=-1)
        return self.cache

    def forward_impl(self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        if self.seq_len_record != seq_len:
            self.seq_len_record = seq_len
            # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
            base = base * self.rope_ratio
            theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

            # Create position indexes `[0, 1, ..., seq_len - 1]`
            seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

            # Calculate the product of position index and $\theta_i$
            idx_theta = torch.outer(seq_idx, theta).float()

            self.cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

            # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            self.cache = self.cache.bfloat16() if dtype == torch.bfloat16 else self.cache.half()
        return self.cache

    def forward(self, max_seq_len, offset=0):
        if self.original_impl:
            return self.forward_impl(max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        else:
            return self.impl(max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device)


def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    data_dtype = x.dtype
    compute_dtype = rope_cache.dtype

    if x.device.type == "hpu" and FusedRoPE is not None:
        x_out = FusedRoPE.apply(x.to(compute_dtype), rope_cache)
    else:
        x = x.to(compute_dtype)
        # x: [sq, b, np, hn]
        sq, _, np, _ = x.size(0), x.size(1), x.size(2), x.size(3)
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


class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        if hidden_states.device.type == "hpu" and FusedRMSNorm is not None:
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


class KVCache(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cache = None
        self.inp_seq_len = -1

    def allocate(self, inp_seq_len, dtype, device, shape):
        if self.cache is None or self.cache.shape != shape:
            self.inp_seq_len = inp_seq_len
            self.cache = torch.zeros(shape, dtype=dtype, device=device)
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
    def __init__(self, config: GLM4VConfig, layer_number):
        super().__init__()

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
        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA) if FusedSDPA is not None else None

        self.q_block_size = 8192

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
        **kwargs,
    ):
        bsz, _, q_len, _ = query_layer.shape

        if use_flash_attention and FusedSDPA is not None:
            import habana_frameworks.torch.hpu as ht

            softmax_mode = "fast" if flash_attention_fast_softmax else "None"

            dropout_rate = 0.0
            if self.training:
                dropout_rate = self.dropout_rate

            if q_len == 1:
                # next token
                use_recompute = True if os.getenv("QUANT_CONFIG", "") else False
                with ht.sdp_kernel(enable_recompute=use_recompute):
                    attn_output = self.fused_scaled_dot_product_attention(
                        query_layer, key_layer, value_layer, attention_mask, dropout_rate, False, None, softmax_mode
                    )
            else:
                # first token
                if flash_attention_causal_mask:
                    # causal masking on first token requires inputs to be of the same length
                    with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                        attn_output = self.fused_scaled_dot_product_attention(
                            query_layer, key_layer, value_layer, None, dropout_rate, True, None, softmax_mode
                        )
                else:
                    with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                        # WA for long sequence, better perf. than recompute
                        if (q_len > 16384 or (q_len >= 6144 and bsz >= 2)) and self.training:
                            attn_output = self.gaudi_flash_attn_v1(
                                query_layer, key_layer, value_layer, attention_mask, dropout_rate, softmax_mode
                            )
                        else:
                            attn_output = self.fused_scaled_dot_product_attention(
                                query_layer,
                                key_layer,
                                value_layer,
                                attention_mask,
                                dropout_rate,
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


CORE_ATTENTION_CLASSES = {"eager": CoreAttention, "sdpa": CoreAttention, "flash_attention_2": CoreAttention}


class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: GLM4VConfig, layer_number, device=None):
        super().__init__()
        self.config = config
        self.layer_number = max(1, layer_number)

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        self.original_rope = config.original_rope
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = torch.nn.Linear(
            config.hidden_size,
            self.qkv_hidden_size,
            bias=config.add_bias_linear or config.add_qkv_bias,
            device=device,
            **_config_to_kwargs(config),
        )

        self.core_attention = CoreAttention(config, self.layer_number)

        # Output.
        self.dense = torch.nn.Linear(
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
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
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
        **kwargs,
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

        # apply relative positional encoding (rotary embedding)
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
            **kwargs,
        )

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

        # No output_attention
        attn_weights = None

        return output, attn_weights, past_key_value


def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs


class MLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: GLM4VConfig, device=None):
        super().__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = torch.nn.Linear(
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
        self.dense_4h_to_h = torch.nn.Linear(
            config.ffn_hidden_size, config.hidden_size, bias=self.add_bias, device=device, **_config_to_kwargs(config)
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: GLM4VConfig, layer_number, device=None):
        super().__init__()
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
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
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
        **kwargs,
    ):
        # hidden_states: [s, b, h]

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
            **kwargs,
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

    def __init__(self, config: GLM4VConfig, device=None):
        super().__init__()

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
                            None,
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
                    flash_attention_causal_mask=flash_attention_causal_mask,
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


class GLM4VPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = GLM4VConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]

    def _init_weights(self, module: torch.nn.Module):
        """Initialize the weights."""
        return

    def get_masks(self, input_embeds, past_key_values, padding_mask=None):
        batch_size, seq_length, embed_size = input_embeds.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_embeds.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
        if past_length:
            full_attention_mask = torch.cat(
                (torch.ones(batch_size, seq_length, past_length, device=input_embeds.device), full_attention_mask),
                dim=-1,
            )
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    def get_position_ids(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids

    def get_multimodal_position_ids(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids


class Embedding(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, config: GLM4VConfig, device=None):
        super().__init__()

        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = torch.nn.Embedding(
            config.padded_vocab_size, self.hidden_size, dtype=config.torch_dtype, device=device
        )
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


def is_empty(images_list: Optional[List[List[torch.Tensor]]]):
    if images_list is None or len(images_list) == 0:
        return True
    for image_list in images_list:
        if image_list is None:
            raise ValueError("Image list contains some invalid contents (probably None)!")
    return False


class GLM4VModel(GLM4VPreTrainedModel):
    def __init__(self, config: GLM4VConfig, device=None, empty_init=True):
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
            rotary_dim // 2,
            rope_ratio=config.rope_ratio,
            original_impl=config.original_rope,
            device=device,
            dtype=config.torch_dtype,
        )
        self.encoder = init_method(GLMTransformer, config, **init_kwargs)
        self.output_layer = init_method(
            torch.nn.Linear,
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

        self.vision = EVA2CLIPModel(config)

        if hasattr(config, "vision_config"):
            self.image_size: int = self.config.vision_config["image_size"]
            self.patch_size: int = self.config.vision_config["patch_size"]
            self.num_patches = (self.image_size // self.patch_size // 2) ** 2

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value

    def get_prompt(self, batch_size, device, dtype=torch.half):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
        past_key_values = past_key_values.view(
            batch_size, self.pre_seq_len, self.num_layers * 2, self.multi_query_group_num, self.kv_channels
        )
        # seq_len, b, nh, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        return past_key_values

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

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.encoder.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.encoder.reorder_kv_cache(beam_idx)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: torch.Tensor = None,
        images_idx: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
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
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """take care of image_encode, position_ids and (attention_mask = None is fine)"""

        batch_size, seq_length = input_ids.shape

        # generate mode with past_key_values. the image features are already mapped
        if past_key_values is None:
            # not allow for inputs_embeds, because we want to process image feature
            assert input_ids is not None and inputs_embeds is None, f"{input_ids} {inputs_embeds}"
            if not is_empty(images):  # multi-modality
                assert len(input_ids) == len(images), f"{len(input_ids)} {len(images)}"
                # Please make sure to provide position_ids in inputs for Gaudi.
                assert position_ids is not None
                assert images_idx is not None

                inputs_embeds = self.embedding(input_ids)

                images = images.to(dtype=inputs_embeds.dtype)
                images_features = self.vision(images)

                if self.training and self.embedding.word_embeddings.weight.requires_grad:
                    inputs_embeds_list = []
                    for i in range(batch_size):
                        input_embeds_bs = torch.index_copy(inputs_embeds[i], 0, images_idx[i], images_features[i])
                        inputs_embeds_list.append(input_embeds_bs.unsqueeze(0))
                    inputs_embeds = torch.cat(inputs_embeds_list, dim=0)
                else:
                    with torch.no_grad():
                        for i in range(batch_size):
                            inputs_embeds[i].index_copy_(0, images_idx[i], images_features[i])

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        prefix_encoders = None
        if self.pre_seq_len > 0:
            if token_idx is not None:
                token_idx = token_idx + self.pre_seq_len
            if past_key_values is None:
                prefix_encoders = self.get_prompt(
                    batch_size=batch_size, device=input_ids.device, dtype=inputs_embeds.dtype
                )

        # Ptuning for multi-modality? This path is not verified for Gaudi
        """
        if (attention_mask is not None and not attention_mask.all()) or (prefix_encoders and seq_length != 1):
            if self.training:
                for i in range(batch_size):
                    attention_mask[i].index_copy_(0, images_idx[i], torch.ones(self.num_patches, device=attention_mask.device, type=torch.int32))
                    input_ids[i].index_copy_(0, images_idx[i], input_ids[i, -1].repeat(self.num_patches) )

                inputs_embeds = self.embedding(input_ids)
        """

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            if reuse_cache:
                past_key_values_length = past_key_values[0][0][2]
            else:
                past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None and images is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length_with_past, dtype=torch.long, device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0)
        if position_ids.size(-1) < seq_length:
            position_ids = F.pad(position_ids, (0, seq_length - position_ids.size(-1)), "constant", 0)
        cache_position = None

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        rotary_pos_emb = rotary_pos_emb[position_ids]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        if self.pre_seq_len > 0:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, self.pre_seq_len
            )
        else:
            attention_mask = _gaudi_prepare_4d_causal_attention_mask(
                attention_mask,
                input_ids.shape if input_ids is not None else (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
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


def _history_to_prompt(history, query):
    prompt = ""
    flag = False
    for i, (old_query, response) in enumerate(history):
        prompt += ("<|user|>" if flag else "") + old_query + "<|assistant|>" + response + "<|endoftext|>"
        flag = True
    prompt += "{}{}<|assistant|>".format("<|user|>" if flag else "", query)
    return prompt


class GLM4VForConditionalGeneration(GLM4VPreTrainedModel, GenerationMixin):
    def __init__(self, config: GLM4VConfig, empty_init=True, device=None):
        super().__init__(config)

        self.max_sequence_length = config.max_length
        self.transformer = GLM4VModel(config, empty_init=empty_init, device=device)
        self.config = config

        if hasattr(config, "vision_config"):
            self.image_size: int = self.config.vision_config["image_size"]
            self.patch_size: int = self.config.vision_config["patch_size"]
            self.num_patches = (self.image_size // self.patch_size // 2) ** 2

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.transformer.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)
        self.kv_cache_len = max_seq_len

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.transformer.reorder_kv_cache(beam_idx)

    def adjust_multimodal_inputs(self, inputs):
        config = self.config
        assert hasattr(config, "vision_config")
        image_size: int = config.vision_config["image_size"]
        patch_size: int = config.vision_config["patch_size"]
        num_patches = (image_size // patch_size // 2) ** 2

        input_ids = inputs["input_ids"]
        position_ids = inputs["position_ids"]
        attention_mask = inputs["attention_mask"]
        images_idx = []

        batch_size = len(input_ids)

        for i in range(batch_size):
            boi_token_pos, eoi_token_pos = (
                input_ids[i].index(config.boi_token_id),
                input_ids[i].index(config.eoi_token_id),
            )
            assert eoi_token_pos - boi_token_pos == 2

            new_input_ids = (
                input_ids[i][: boi_token_pos + 1] + [input_ids[i][-1]] * num_patches + input_ids[i][eoi_token_pos:]
            )
            new_position_ids = (
                position_ids[i][: boi_token_pos + 1]
                + [position_ids[i][boi_token_pos + 1]] * num_patches
                + position_ids[i][eoi_token_pos:]
            )
            new_attention_mask = (
                attention_mask[i][: boi_token_pos + 1] + [1] * num_patches + attention_mask[i][eoi_token_pos:]
            )
            new_image_idx = list(range(boi_token_pos, boi_token_pos + num_patches + 2))
            input_ids[i] = new_input_ids
            position_ids[i] = new_position_ids
            attention_mask[i] = new_attention_mask
            images_idx.append(new_image_idx)

        inputs.data["images_idx"] = images_idx

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        images: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: torch.Tensor = None,
        inputs_embeds=None,
        token_idx=None,
        **kwargs,
    ) -> dict:
        reuse_cache = kwargs.get("reuse_cache")
        if past_key_values:
            if token_idx is not None:
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)
            else:
                input_ids = input_ids[:, -1:]
        elif reuse_cache and token_idx is not None:
            # With reuse_cache, KV cache is pre allocated hence for the 1st token we can slice the inputs till token idx for the fwd pass
            input_ids = input_ids[:, :token_idx]
            attention_mask = attention_mask[:, :token_idx]

        assert position_ids is not None
        images_idx = kwargs.get("images_idx")

        if past_key_values:
            position_ids = position_ids[..., -1:] + token_idx - position_ids.size(-1) - 1

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "images": images,
                "images_idx": images_idx,
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
        images: torch.Tensor = None,
        images_idx: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids=input_ids,
            images=images,
            images_idx=images_idx,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
            # This part should be done before sending into the model for Gaudi
            """
            new_labels = []
            for i in range(len(input_ids)):
                input_id = input_ids[i].tolist()
                boi_token_pos, eoi_token_pos = input_id.index(self.config.boi_token_id), input_id.index(
                    self.config.eoi_token_id)
                assert eoi_token_pos - boi_token_pos == 2

                new_labels.append(torch.cat(
                    (
                        labels[i, :boi_token_pos + 1],
                        torch.tensor([-100]).to(labels.device).to(labels.dtype).repeat(1600),
                        labels[i, eoi_token_pos:])))

            labels = torch.stack(new_labels, dim=0)
            """
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
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
                layer_past[0].index_select(0, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(0, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )


class GLM4VForSequenceClassification(GLM4VPreTrainedModel):
    def __init__(self, config: GLM4VConfig, empty_init=True, device=None):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.transformer = GLM4VModel(config, empty_init=empty_init, device=device)

        self.classifier_head = torch.nn.Linear(config.hidden_size, config.num_labels, bias=True, dtype=torch.half)
        if config.classifier_dropout is not None:
            self.dropout = torch.nn.Dropout(config.classifier_dropout)
        else:
            self.dropout = None
        self.config = config

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
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

        outputs = self.transformer(
            input_ids=input_ids,
            images=None,
            images_idx=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
