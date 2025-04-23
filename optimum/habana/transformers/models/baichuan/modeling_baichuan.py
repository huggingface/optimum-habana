# Copyright 2023 Baichuan Inc. All Rights Reserved.

# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

"""
Adapted from the following sources:
https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/main/modeling_baichuan.py
https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/blob/main/modeling_baichuan.py
"""

import math
import os
from contextlib import contextmanager
from threading import Thread
from typing import List, Optional, Tuple, Union

import habana_frameworks.torch.core as htcore
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

from ...modeling_attn_mask_utils import _gaudi_prepare_4d_causal_attention_mask
from .configuration_baichuan import BaichuanConfig
from .generation_utils import TextIterStreamer, build_chat_input


logger = logging.get_logger(__name__)

try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE
except ImportError:
    print("Not using HPU fused kernel for apply_rotary_pos_emb")
    FusedRoPE = None

try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm as FusedRMSNorm
except ImportError:
    print("Not using HPU fused kernel for RMSNorm")
    FusedRMSNorm = None

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None


def gaudi_baichuan_build_alibi_tensor(
    attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype
) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`):
            Number of heads.
        dtype (`torch.dtype`):
            Dtype of the output tensor.
        training (`bool`):
            Whether the model is being trained or not.
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(seq_length, device=attention_mask.device).unsqueeze(
        0
    ).unsqueeze(0).expand(num_heads, -1, -1).unsqueeze(0)

    return alibi.to(dtype)


class Matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class KVCache(nn.Module):
    def __init__(self):
        super(KVCache, self).__init__()
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


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if hidden_states.device.type == "hpu" and FusedRMSNorm:
            # mixed dtypes are not good for FusedRMSNorm, both inputs need to have same dtype
            if hidden_states.dtype != self.weight.dtype:
                orig_dtype = hidden_states.dtype
                hidden_states = FusedRMSNorm.apply(
                    hidden_states.to(self.weight.dtype), self.weight, self.variance_epsilon
                )
                return hidden_states.to(orig_dtype)
            else:
                hidden_states = FusedRMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)
                return hidden_states
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    if q.device.type == "hpu" and FusedRoPE:
        # TODO: remove `.clone()` when it is fixed in SynapseAI
        return FusedRoPE.apply(
            q, cos.unsqueeze(0).unsqueeze(0).clone(), sin.unsqueeze(0).unsqueeze(0).clone(), position_ids
        ), FusedRoPE.apply(
            k, cos.unsqueeze(0).unsqueeze(0).clone(), sin.unsqueeze(0).unsqueeze(0).clone(), position_ids
        )
    else:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    # 7B model only
    def pre_mlp_forward(self, x):
        inputs = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        output = self.down_proj(inputs)
        return output

    # 7B model only
    def mlp_all_reduce(self, x):
        if hasattr(self.down_proj, "all_reduce"):
            self.down_proj.all_reduce(x)

    # 7B model only
    def post_mlp_forward(self, x):
        if hasattr(self.down_proj, "post_all_reduce"):
            return self.down_proj.post_all_reduce(x)
        return x

    # 13B model only
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


#  FusedScaledDotProductAttention
class ModuleFusedSDPA(nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        self._hpu_kernel_fsdpa = fusedSDPA

    def forward(self, query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode):
        return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode)


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.config = config
        self.is_7b = hasattr(config, "max_position_embeddings")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings if self.is_7b else config.model_max_length
        self.norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.W_pack = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.matmul_qk = Matmul()
        self.matmul_av = Matmul()
        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA) if FusedSDPA else None

        self.k_cache = KVCache()
        self.v_cache = KVCache()
        self.inp_seq_len = -1

        if self.is_7b:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
            self.block_size = 8192
        else:
            self.q_block_size = 2048

    def update_sincos_cache(self, seq_len):
        # Call rotary emb forward() to update cos/sin cache when infering more than self.max_position_embeddings
        # This helps in avoiding creation of these caches during actual model forward pass and
        # reduce memory consumption and improve performance.
        if seq_len > self.max_position_embeddings:
            self.max_position_embeddings = seq_len
            _, _ = self.rotary_emb(self.W_pack.weight, seq_len=seq_len)

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        cache_shape = (batch_size, self.num_heads, max_seq_len, self.head_dim)
        device = self.W_pack.weight.device
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

    # 7B model only
    def gaudi_flash_attn_v1(self, query_layer, key_layer, value_layer, attention_mask, dropout_rate, softmax_mode):
        """
        Gaudi version of Flash Attention V1 to support long sequence at prompt phase
        Causal mask is not supported in this optimization
        """
        q_len = query_layer.size(-2)
        q_tiles = (q_len // self.block_size) if (q_len % self.block_size == 0) else math.ceil(q_len / self.block_size)
        q_padding = q_tiles * self.block_size - q_len
        query_layer = F.pad(query_layer, (0, 0, 0, q_padding), "constant", 0)
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, 0, 0, q_padding), "constant", torch.finfo(key_layer.dtype).min)

        row_o_list = []
        for i in range(q_tiles):
            s, e = i * self.block_size, (i + 1) * self.block_size
            row_q = query_layer[:, :, s:e, :]
            row_mask = attention_mask[:, :, s:e, :]
            attn_output_partial = self.fused_scaled_dot_product_attention(
                row_q, key_layer, value_layer, row_mask, dropout_rate, False, None, softmax_mode
            )
            row_o_list.append(attn_output_partial)
        attn_output = torch.cat(row_o_list, dim=-2)

        if q_padding != 0:
            attn_output = attn_output[:, :, :-q_padding, :]

        return attn_output

    # 7B model only
    def pre_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        proj = self.W_pack(hidden_states)
        proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
        query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if token_idx is None:
                kv_seq_len += past_key_value[0].shape[-2]
            else:
                if reuse_cache:
                    kv_seq_len = past_key_value[0][-2]
                else:
                    kv_seq_len = past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if use_cache:
            # reuse k, v, self_attention
            if reuse_cache:
                key_states = self.k_cache(key_states, 2, token_idx)
                value_states = self.v_cache(value_states, 2, token_idx)
                past_key_value = (self.k_cache.get_shape(), self.v_cache.get_shape())
            else:
                if past_key_value is None:
                    past_key = torch.zeros(key_states.shape, dtype=self.W_pack.weight.dtype, device=key_states.device)
                    past_value = torch.zeros(
                        key_states.shape, dtype=self.W_pack.weight.dtype, device=key_states.device
                    )
                    past_key_value = [past_key, past_value]
                key_states = self.k_cache.update(past_key_value[0], key_states, 2, token_idx, self.inp_seq_len)
                value_states = self.v_cache.update(past_key_value[1], value_states, 2, token_idx, self.inp_seq_len)
                if token_idx is None:
                    past_key_value = (key_states, value_states)

            if cache_idx is not None and q_len == 1:
                key_states = key_states[:, :, :cache_idx, :]
                value_states = value_states[:, :, :cache_idx, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :, :, :cache_idx]
                kv_seq_len = key_states.shape[-2]
        else:
            past_key_value = None

        if use_flash_attention and FusedSDPA:
            import habana_frameworks.torch.hpu as ht

            softmax_mode = "fast" if flash_attention_fast_softmax else "None"

            if q_len == 1:
                # next token
                use_recompute = True if os.getenv("QUANT_CONFIG", "") else False
                with ht.sdp_kernel(enable_recompute=use_recompute):
                    attn_output = self.fused_scaled_dot_product_attention(
                        query_states, key_states, value_states, attention_mask, 0.0, False, None, softmax_mode
                    )
            elif q_len > 32768:
                # first token case1: Long sequence should go to flash_attn_v1 to avoid OOM issue
                with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                    attn_output = self.gaudi_flash_attn_v1(
                        query_states, key_states, value_states, attention_mask, 0.0, softmax_mode
                    )
                htcore.mark_step()
            else:
                # first token
                if flash_attention_causal_mask:
                    # causal masking on first token requires inputs to be of the same length
                    with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                        attn_output = self.fused_scaled_dot_product_attention(
                            query_states, key_states, value_states, None, 0.0, True, None, softmax_mode
                        )
                else:
                    with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                        attn_output = self.fused_scaled_dot_product_attention(
                            query_states, key_states, value_states, attention_mask, 0.0, False, None, softmax_mode
                        )

        else:
            attn_weights = self.matmul_qk(query_states, key_states.transpose(-2, -1)) * self.norm_factor

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask
                if cache_position is not None:
                    causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            if attn_softmax_bf16:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=query_states.dtype)
            else:
                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = self.matmul_av(attn_weights, value_states)
            attn_output = attn_output.reshape(bsz, -1, q_len, self.head_dim)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    # 7B model only
    def attention_all_reduce(self, attn_output):
        if hasattr(self.o_proj, "all_reduce"):
            self.o_proj.all_reduce(attn_output)

    # 7B model only
    def post_attn_forward(self, attn_output):
        if hasattr(self.o_proj, "post_all_reduce"):
            self.o_proj.post_all_reduce(attn_output)
        return attn_output

    # 13B model only
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        alibi: Optional[torch.Tensor] = None,
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
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        proj = self.W_pack(hidden_states)
        proj = proj.view(bsz, q_len, 3, self.num_heads, self.head_dim)
        query_states, key_states, value_states = proj[..., 0, :, :], proj[..., 1, :, :], proj[..., 2, :, :]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if token_idx is None:
                kv_seq_len += past_key_value[0].shape[-2]
            else:
                if reuse_cache:
                    kv_seq_len = past_key_value[0][-2]
                else:
                    kv_seq_len = past_key_value[0].shape[-2]

        if use_cache:
            # reuse k, v, self_attention
            if reuse_cache:
                key_states = self.k_cache(key_states, 2, token_idx)
                value_states = self.v_cache(value_states, 2, token_idx)
                past_key_value = (self.k_cache.get_shape(), self.v_cache.get_shape())
            else:
                if past_key_value is None:
                    past_key = torch.zeros(key_states.shape, dtype=self.W_pack.weight.dtype, device=key_states.device)
                    past_value = torch.zeros(
                        key_states.shape, dtype=self.W_pack.weight.dtype, device=key_states.device
                    )
                    past_key_value = (past_key, past_value)
                key_states = self.k_cache.update(past_key_value[0], key_states, 2, token_idx, self.inp_seq_len)
                value_states = self.v_cache.update(past_key_value[1], value_states, 2, token_idx, self.inp_seq_len)
                if token_idx is None:
                    past_key_value = (key_states, value_states)

            if cache_idx is not None and q_len == 1:
                key_states = key_states[:, :, :cache_idx, :]
                value_states = value_states[:, :, :cache_idx, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :, :, :cache_idx]
                alibi = alibi[:, :, :, :cache_idx]
                kv_seq_len = key_states.shape[-2]
        else:
            past_key_value = None

        # [batch_size, num_heads, q_length, kv_length]
        if use_flash_attention and FusedSDPA and q_len < 12288:
            import habana_frameworks.torch.hpu as ht

            softmax_mode = "fast" if flash_attention_fast_softmax else "None"

            if q_len == 1:
                # next token
                use_recompute = True if os.getenv("QUANT_CONFIG", "") else False
                with ht.sdp_kernel(enable_recompute=use_recompute):
                    attn_output = self.fused_scaled_dot_product_attention(
                        query_states, key_states, value_states, attention_mask, 0.0, False, None, softmax_mode
                    )
            else:
                # first token
                if flash_attention_causal_mask:
                    raise NotImplementedError(
                        "SDPA with Alibi positional embedding CANNOT use the causal mask! "
                        "Please disable --flash_attention_causal_mask and try again."
                    )
                else:
                    with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                        attn_output = self.fused_scaled_dot_product_attention(
                            query_states, key_states, value_states, attention_mask, 0.0, False, None, softmax_mode
                        )

        elif q_len >= 12288:
            q_tiles = (
                (q_len // self.q_block_size)
                if (q_len % self.q_block_size == 0)
                else math.ceil(q_len / self.q_block_size)
            )
            q_padding = q_tiles * self.q_block_size - q_len
            query_states = F.pad(query_states, (0, 0, 0, q_padding), "constant", 0)
            if attention_mask is not None:
                attention_mask = F.pad(
                    attention_mask, (0, 0, 0, q_padding), "constant", torch.finfo(key_states.dtype).min
                )

            row_o_list = []

            for i in range(q_tiles):
                s, e = i * self.q_block_size, (i + 1) * self.q_block_size
                row_q = query_states[:, :, s:e, :]
                row_mask = attention_mask[:, :, s:e, :]
                attn_weights_partial = self.matmul_qk(row_q, key_states.transpose(-2, -1)) * self.norm_factor
                attn_weights_partial = attn_weights_partial + row_mask
                attn_weights_partial = attn_weights_partial.to(torch.float32) + self.beta * alibi

                if attn_softmax_bf16:
                    attn_weights_partial = attn_weights_partial.to(query_states.dtype)
                    attn_weights_partial = nn.functional.softmax(
                        attn_weights_partial, dim=-1, dtype=query_states.dtype
                    )
                else:
                    # upcast attention to fp32
                    attn_weights_partial = nn.functional.softmax(attn_weights_partial, dim=-1, dtype=torch.float32).to(
                        query_states.dtype
                    )

                attn_output_partial = self.matmul_av(attn_weights_partial, value_states)
                row_o_list.append(attn_output_partial)

            attn_output = torch.cat(row_o_list, dim=-2)
            if q_padding != 0:
                attn_output = attn_output[:, :, :-q_padding, :]
            attn_output = attn_output.reshape(bsz, -1, q_len, self.head_dim)
            htcore.mark_step()

        else:
            # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
            attn_weights = self.matmul_qk(query_states, key_states.transpose(-2, -1)) * self.norm_factor

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask
                if cache_position is not None:
                    causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
                    alibi = alibi[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = attn_weights.to(torch.float32) + self.beta * alibi

            if attn_softmax_bf16:
                attn_weights = attn_weights.to(query_states.dtype)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=query_states.dtype)
            else:
                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = self.matmul_av(attn_weights, value_states)
            attn_output = attn_output.reshape(bsz, -1, q_len, self.head_dim)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class DecoderLayer(nn.Module):
    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config)
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_7b = hasattr(config, "max_position_embeddings")

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.self_attn.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.self_attn.reorder_kv_cache(beam_idx)

    def update_sincos_cache(self, seq_len):
        self.self_attn.update_sincos_cache(seq_len)

    # position_tensor: positional tensor
    # position_tensor is alibi when calling self.forward() for 13B model.
    # position_tensor is position_ids when calling self.forward() for 7B model,
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_tensor: Optional[Union[torch.Tensor, torch.LongTensor]] = None,
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
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        if self.is_7b:
            # Self Attention
            hidden_states, attn_weights, present_key_value = self.pre_attn(
                hidden_states,
                attention_mask,
                position_tensor,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                token_idx,
                attn_softmax_bf16,
                reuse_cache,
                use_flash_attention=use_flash_attention,
                flash_attention_recompute=flash_attention_recompute,
                flash_attention_causal_mask=flash_attention_causal_mask,
                flash_attention_fast_softmax=flash_attention_fast_softmax,
                cache_idx=cache_idx,
                **kwargs,
            )
            self.self_attn.attention_all_reduce(hidden_states)
            hidden_states, residual = self.post_attn_pre_mlp(hidden_states, residual)
            self.mlp.mlp_all_reduce(hidden_states)
            hidden_states = self.post_mlp(hidden_states, residual)
        else:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, attn_weights, present_key_value = self.self_attn(
                hidden_states,
                attention_mask,
                position_tensor,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                token_idx,
                attn_softmax_bf16,
                reuse_cache,
                use_flash_attention=use_flash_attention,
                flash_attention_recompute=flash_attention_recompute,
                flash_attention_causal_mask=flash_attention_causal_mask,
                cache_idx=cache_idx,
                flash_attention_fast_softmax=flash_attention_fast_softmax,
                **kwargs,
            )
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    # 7B model only
    def pre_attn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, attn_weights, present_key_value = self.self_attn.pre_attn_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            token_idx,
            attn_softmax_bf16,
            reuse_cache,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
            flash_attention_fast_softmax=flash_attention_fast_softmax,
            cache_idx=cache_idx,
            **kwargs,
        )
        return hidden_states, attn_weights, present_key_value

    # 7B model only
    def post_attn_pre_mlp(self, hidden_states, residual):
        hidden_states = self.self_attn.post_attn_forward(hidden_states)

        if self.training:
            hidden_states = hidden_states + residual
            residual = hidden_states
        else:
            residual.add_(hidden_states)
            hidden_states = residual

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp.pre_mlp_forward(hidden_states)
        return hidden_states, residual

    # 7B model only
    def post_mlp(self, hidden_states, residual):
        hidden_states = self.mlp.post_mlp_forward(hidden_states)

        if self.training:
            hidden_states = hidden_states + residual
        else:
            residual.add_(hidden_states)
            hidden_states = residual

        return hidden_states


class BaichuanPreTrainedModel(PreTrainedModel):
    config_class = BaichuanConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BaichuanModel):
            module.gradient_checkpointing = value


class BaichuanModel(BaichuanPreTrainedModel):
    def __init__(self, config: BaichuanConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.is_7b = hasattr(config, "max_position_embeddings")

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.is_7b:
            self.gradient_checkpointing = False
        else:
            self.num_heads = config.num_attention_heads
            self.gradient_checkpointing = config.gradient_checkpointing
            self.max_cache_pos = config.model_max_length

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        for layer in self.layers:
            layer.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return tuple(layer.reorder_kv_cache(beam_idx) for layer in self.layers)

    def update_sincos_cache(self, seq_len):
        for layer in self.layers:
            layer.update_sincos_cache(seq_len)

    # position_tensor: positional tensor
    # The passed value of position_tensor is ignored for 13B model, self.forward()
    # will build alibi internally. position_tensor is position_ids for 7B model, it
    # might be None.
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_tensor: Optional[Union[torch.Tensor, torch.LongTensor]] = None,
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
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds simultaneously")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You need to provide input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            if reuse_cache:
                past_key_values_length = past_key_values[0][0][2]
            else:
                past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if self.is_7b:
            # position_tensor is position_ids or None
            if position_tensor is None:
                position_tensor = torch.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                position_tensor = position_tensor.unsqueeze(0)
        else:
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length_with_past), device=inputs_embeds.device)
            else:
                attention_mask = attention_mask.to(inputs_embeds.device)

            # The passed position_tensor is ignored, build position_tensor (alibi) internally
            position_tensor = gaudi_baichuan_build_alibi_tensor(attention_mask, self.num_heads, dtype=torch.float32)

        cache_position = None

        # HPU specific mask generation
        attention_mask = _gaudi_prepare_4d_causal_attention_mask(
            attention_mask,
            input_ids.shape if input_ids is not None else (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        if not self.is_7b and use_flash_attention and FusedSDPA:
            attention_mask = (attention_mask.to(torch.float32) + position_tensor).to(inputs_embeds.dtype)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        if lazy_mode:
            htcore.mark_step()

        for idx, decoder_layer in enumerate(self.layers):
            if (
                lazy_mode
                and not self.training
                and (torch.distributed.is_initialized() is False or torch.distributed.get_world_size() == 1)
            ):
                htcore.mark_step()

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

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
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_tensor,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_tensor=position_tensor,
                    past_key_value=None if past_key_values is None else past_key_values[idx],
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
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class NormHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        norm_weight = nn.functional.normalize(self.weight)
        return nn.functional.linear(hidden_states, norm_weight)


_init_weights = True


@contextmanager
def no_init_weights(_enable=True):
    global _init_weights
    old_init_weights = _init_weights
    if _enable:
        _init_weights = False
    try:
        yield
    finally:
        _init_weights = old_init_weights


class BaichuanForCausalLM(BaichuanPreTrainedModel):
    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config, *model_args, **model_kwargs)
        self.model = BaichuanModel(config)

        self.lm_head = NormHead(config.hidden_size, config.vocab_size, bias=False)
        self.is_7b = hasattr(config, "max_position_embeddings")

        # if hasattr(config, "quantization_config") and config.quantization_config['load_in_4bit']:
        if hasattr(config, "quantization_config"):
            raise RuntimeError(
                "Currently the Habana Optimum does not support INT8/4 quantization for Baichuan2, abort."
            )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.model.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)
        self.kv_cache_len = max_seq_len

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.model.reorder_kv_cache(beam_idx)

    def update_sincos_cache(self, seq_len):
        self.model.update_sincos_cache(seq_len)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, _ = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=False,
                proxies=None,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder="",
                _from_auto=False,
                _from_pipeline=None,
                **kwargs,
            )

        return super(BaichuanForCausalLM, cls).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[Union[torch.Tensor, torch.LongTensor]] = None,
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
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_tensor=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            token_idx=token_idx,
            attn_softmax_bf16=attn_softmax_bf16,
            reuse_cache=reuse_cache,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
            flash_attention_fast_softmax=flash_attention_fast_softmax,
            cache_idx=cache_idx,
            lazy_mode=lazy_mode,
            **kwargs,
        )
        hidden_states = outputs[0]
        _, seq_len, _ = hidden_states.shape
        if seq_len > 1 and trim_logits and not self.training:
            if token_idx is not None:
                hidden_states = hidden_states.index_select(1, token_idx - 1)
            else:
                hidden_states = hidden_states[:, -1, :]

        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            softmax_normalizer = shift_logits.max(-1).values ** 2
            z_loss = self.config.z_loss_weight * softmax_normalizer.mean()
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels) + z_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, token_idx=None, **kwargs
    ):
        past_length = 0

        reuse_cache = kwargs.get("reuse_cache")
        if past_key_values is not None:
            if token_idx is not None:
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)
            else:
                if isinstance(past_key_values, Cache):
                    cache_length = past_key_values.get_seq_length()
                    past_length = past_key_values.seen_tokens
                    max_cache_length = past_key_values.get_max_length()
                else:
                    cache_length = past_length = past_key_values[0][0].shape[2]
                    max_cache_length = None

                # Keep only the unprocessed tokens:
                # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
                # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
                # input)
                if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                    input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
                # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
                # input_ids based on the past_length.
                elif past_length < input_ids.shape[1]:
                    input_ids = input_ids[:, past_length:]
                # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

                # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
                if (
                    max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length
                ):
                    attention_mask = attention_mask[:, -max_cache_length:]
        elif reuse_cache and token_idx is not None:
            # With reuse_cache, KV cache is pre allocated hence for the 1st token we can slice the inputs till token idx for the fwd pass
            input_ids = input_ids[:, :token_idx]
            attention_mask = attention_mask[:, :token_idx]

        if self.is_7b:
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
        else:
            # 13B model uses alibi which is built internally. set position_ids to None here
            position_ids = None
        cache_position = None
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": None if position_ids is None else position_ids.contiguous(),
                "cache_position": cache_position,
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

    def chat(
        self,
        tokenizer,
        messages: List[dict],
        stream=False,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        generation_config = generation_config or self.generation_config
        input_ids = build_chat_input(self, tokenizer, messages, generation_config.max_new_tokens)
        if stream:
            streamer = TextIterStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            Thread(
                target=self.generate,
                inputs=input_ids,
                streamer=streamer,
                generation_config=generation_config,
                lazy_mode=True,
                hpu_graphs=True,
            ).start()
            return streamer
        else:
            outputs = self.generate(
                input_ids, generation_config=generation_config, lazy_mode=True, hpu_graphs=True
            ).cpu()
            response = tokenizer.decode(outputs[0][len(input_ids[0]) :], skip_special_tokens=True)
            return response
