# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Mistral model."""

import math
import os
from typing import List, Optional, Tuple, Union

import habana_frameworks.torch.core as htcore
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralMLP,
    MistralModel,
    MistralRMSNorm,
    apply_rotary_pos_emb,
)
from transformers.utils import logging

from ...modeling_attn_mask_utils import (
    _gaudi_prepare_4d_causal_attention_mask,
)
from ..llama.modeling_llama import (
    GaudiLlamaDynamicNTKScalingRotaryEmbedding,
    GaudiLlamaLinearScalingRotaryEmbedding,
    GaudiLlamaRotaryEmbedding,
)


try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE

    has_fused_rope = True
except ImportError:
    has_fused_rope = False
    print("Not using HPU fused kernel for apply_rotary_pos_emb")

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

logger = logging.get_logger(__name__)


class KVCache(torch.nn.Module):
    def __init__(self):
        super(KVCache, self).__init__()
        self.cache = None
        self.inp_seq_len = -1

    def allocate(self, inp_seq_len, dtype, device, shape):
        if self.cache is None or self.cache.shape != shape:
            self.inp_seq_len = inp_seq_len
            self.cache = torch.zeros(shape, dtype=dtype, device=device)
        else:
            assert (
                self.inp_seq_len == inp_seq_len
            ), f"inp_seq_len must be the same. self.inp_seq_len:{self.inp_seq_len} inp_seq_len:{inp_seq_len}"
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


class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        self._hpu_kernel_fsdpa = fusedSDPA

    def forward(self, query, key, value, attn_mask, dropout_p, is_casual, scale):
        return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_casual, scale)


class Matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


def gaudi_mistral_repeat_kv(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    n_rep: int,
):
    """
    Copied from repeat_kv: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/mistral/modeling_mistral.py
    The only differences are:
        - Append num_key_value_heads == 1 check as kv states can be broadcasted during matmuls so need to expand and reshape them.
        - Add new args query_states, key_states, value_states and attention_mask and update the logic for expansion.
    The query states go from (batch, num_heads, seqlen, head_dim) to (batch, num_key_value_heads, n_rep, seqlen, head_dim)
    The key/value states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_key_value_heads, 1, seqlen, head_dim)
    """
    batch, num_key_value_heads, kv_len, head_dim = key_states.shape
    if n_rep == 1 or num_key_value_heads == 1:
        return query_states, key_states, value_states, attention_mask

    new_kv_shape = (batch, num_key_value_heads, 1, kv_len, head_dim)
    key_states = key_states.reshape(new_kv_shape)
    value_states = value_states.reshape(new_kv_shape)

    batch, _, q_len, head_dim = query_states.shape
    new_q_shape = (batch, num_key_value_heads, n_rep, q_len, head_dim)
    query_states = query_states.reshape(new_q_shape)

    if attention_mask is not None:
        # Add groups dim and set to 1
        attention_mask = attention_mask.unsqueeze(1)

    return query_states, key_states, value_states, attention_mask


def gaudi_mistral_rmsnorm_forward(self, hidden_states):
    """
    Copied from MistralRMSNorm.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py
    The only differences are:
        - override RMSNorm with Habana fused RMSNorm
    """
    if hidden_states.device.type == "hpu" and FusedRMSNorm:
        # mixed dtypes are not good for FusedRMSNorm, both inputs need to have same dtype
        if hidden_states.dtype != self.weight.dtype:
            orig_dtype = hidden_states.dtype
            hidden_states = FusedRMSNorm.apply(hidden_states.to(self.weight.dtype), self.weight, self.variance_epsilon)
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


class GaudiMistralAttention(MistralAttention):
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.k_cache = KVCache()
        self.v_cache = KVCache()
        self.matmul_qk = Matmul()
        self.matmul_av = Matmul()
        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA) if FusedSDPA else None
        self.inp_seq_len = -1
        self._init_rope()
        self.norm_factor = 1.0 / math.sqrt(self.head_dim)

    def _init_rope(self):
        """
        Copied from: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L294
        """
        if self.config.rope_scaling is None:
            self.rotary_emb = GaudiLlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = GaudiLlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = GaudiLlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        cache_shape = (batch_size, self.num_key_value_heads, max_seq_len, self.head_dim)
        device = self.k_proj.weight.device
        dtype = self.config.torch_dtype
        self.k_cache.allocate(inp_seq_len, dtype, device, cache_shape)
        self.v_cache.allocate(inp_seq_len, dtype, device, cache_shape)

    def update_sincos_cache(self, seq_len):
        # Call rotary emb forward() to update cos/sin cache when infering more than self.max_position_embeddings
        # This helps in avoiding creation of these caches during actual model forward pass and
        # reduce memory consumption and improve performance.
        if seq_len > self.max_position_embeddings:
            self.max_position_embeddings = seq_len
            _, _ = self.rotary_emb(self.k_proj.weight, seq_len=seq_len)

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
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        token_idx: Optional[torch.Tensor] = None,
        reuse_cache: Optional[bool] = False,
        cache_idx: Optional[int] = None,
        attn_softmax_bf16: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
         Copied from MistralAttention.forward: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/mistral/modeling_mistral.py
         The only differences are:
         - add new args token_idx
         - add new args reuse_cache
        - add new args cache_idx
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_shape = (
                (past_key_value[0][-2] if reuse_cache else past_key_value[0].shape[-2])
                if isinstance(past_key_value, tuple)
                else past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            )
            if token_idx is not None:
                kv_seq_len = kv_shape
            else:
                kv_seq_len += kv_shape
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_customized_rope(query_states, key_states, cos, sin, position_ids)

        if use_cache:
            # reuse k, v, self_attention
            if reuse_cache:
                key_states = self.k_cache(key_states, 2, token_idx)
                value_states = self.v_cache(value_states, 2, token_idx)
                past_key_value = (self.k_cache.get_shape(), self.v_cache.get_shape())
            else:
                if past_key_value is None:
                    past_key = torch.zeros(key_states.shape, dtype=self.k_proj.weight.dtype, device=key_states.device)
                    past_value = torch.zeros(
                        key_states.shape, dtype=self.k_proj.weight.dtype, device=key_states.device
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
                kv_seq_len = key_states.shape[-2]
        else:
            past_key_value = None

        if use_flash_attention and FusedSDPA:
            import habana_frameworks.torch.hpu as ht

            if q_len == 1:
                # next token
                use_recompute = True if os.getenv("QUANT_CONFIG", "") else False
                with ht.sdp_kernel(enable_recompute=use_recompute):
                    attn_output = self.fused_scaled_dot_product_attention(
                        query_states, key_states, value_states, attention_mask, 0.0, False, None
                    )
            else:
                # first token
                if flash_attention_causal_mask:
                    # causal masking on first token requires inputs to be of the same length
                    with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                        attn_output = self.fused_scaled_dot_product_attention(
                            query_states, key_states, value_states, None, 0.0, True, None
                        )
                else:
                    with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                        attn_output = self.fused_scaled_dot_product_attention(
                            query_states, key_states, value_states, attention_mask, 0.0, False, None
                        )
        else:
            # repeat k/v heads if n_kv_heads < n_heads
            query_states, key_states, value_states, attention_mask = gaudi_mistral_repeat_kv(
                query_states, key_states, value_states, attention_mask, self.num_key_value_groups
            )
            attn_weights = self.matmul_qk(query_states, key_states.transpose(-2, -1)) * self.norm_factor

            if attn_weights.size() not in [
                (bsz, self.num_heads, q_len, kv_seq_len),
                (bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, kv_seq_len),
            ]:
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)} or"
                    f" {(bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() not in [(bsz, 1, q_len, kv_seq_len), (bsz, 1, 1, q_len, kv_seq_len)]:
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)} or {(bsz, 1, 1, q_len, kv_seq_len)},"
                        f" but is {attention_mask.size()}"
                    )

                attn_weights = attn_weights + attention_mask

            if attn_softmax_bf16:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=query_states.dtype)
            else:
                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
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


class GaudiMistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super(MistralDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GaudiMistralAttention(config, layer_idx)

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.self_attn.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.self_attn.reorder_kv_cache(beam_idx)

    def update_sincos_cache(self, seq_len):
        self.self_attn.update_sincos_cache(seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        token_idx: Optional[torch.Tensor] = None,
        reuse_cache: Optional[bool] = False,
        cache_idx: Optional[int] = None,
        attn_softmax_bf16: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Copied from MistralDecoderLayer.forward: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/mistral/modeling_mistral.py
        The only differences are:
        - add new args token_idx
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            token_idx=token_idx,
            reuse_cache=reuse_cache,
            cache_idx=cache_idx,
            attn_softmax_bf16=attn_softmax_bf16,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class GaudiMistralModel(MistralModel):
    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        for layer in self.layers:
            layer.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return tuple(layer.reorder_kv_cache(beam_idx) for layer in self.layers)

    def update_sincos_cache(self, seq_len):
        for layer in self.layers:
            layer.update_sincos_cache(seq_len)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        reuse_cache: Optional[bool] = False,
        cache_idx: Optional[int] = None,
        attn_softmax_bf16: Optional[bool] = False,
        lazy_mode: Optional[bool] = True,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Copied from MistralModel.forward: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/mistral/modeling_mistral.py
        The only differences are:
        - add new args token_idx
        - add new arg lazy_mode
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        use_legacy_cache = True
        use_new_cache = False
        if past_key_values is not None and use_cache and not reuse_cache and use_new_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _gaudi_prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if not use_new_cache else None

        if lazy_mode:
            htcore.mark_step()

        for layer_idx, decoder_layer in enumerate(self.layers):
            if layer_idx == len(self.layers) // 2 or (
                lazy_mode
                and not self.training
                and (torch.distributed.is_initialized() is False or torch.distributed.get_world_size() == 1)
            ):
                htcore.mark_step()
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None if past_key_values is None else past_key_values[layer_idx],
                    output_attentions,
                    use_cache,
                    None,
                    False,
                    cache_idx,
                    attn_softmax_bf16,
                    use_flash_attention,
                    flash_attention_recompute,
                    flash_attention_causal_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None if past_key_values is None else past_key_values[layer_idx],
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    token_idx=token_idx,
                    reuse_cache=reuse_cache,
                    cache_idx=cache_idx,
                    attn_softmax_bf16=attn_softmax_bf16,
                    use_flash_attention=use_flash_attention,
                    flash_attention_recompute=flash_attention_recompute,
                    flash_attention_causal_mask=flash_attention_causal_mask,
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

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache
                if not use_new_cache
                else (next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache)
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GaudiMistralForCausalLM(MistralForCausalLM):
    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.model.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.model.reorder_kv_cache(beam_idx)

    def update_sincos_cache(self, seq_len):
        self.model.update_sincos_cache(seq_len)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        reuse_cache: Optional[bool] = False,
        trim_logits: Optional[bool] = False,
        cache_idx: Optional[int] = None,
        attn_softmax_bf16: Optional[bool] = False,
        lazy_mode: Optional[bool] = True,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Inherits from MistralForCausalLM: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/mistral/modeling_mistral.py
        The only differences are:
        - add new args token_idx
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.generation_config.use_fused_rope is False:
            global has_fused_rope
            has_fused_rope = False

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_idx=token_idx,
            reuse_cache=reuse_cache,
            cache_idx=cache_idx,
            attn_softmax_bf16=attn_softmax_bf16,
            lazy_mode=lazy_mode,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
        )
        hidden_states = outputs[0]
        _, seq_len, _ = hidden_states.shape
        if seq_len > 1 and trim_logits and not self.training:
            if token_idx is not None:
                hidden_states = hidden_states.index_select(1, token_idx - 1)
            else:
                hidden_states = hidden_states[:, -1, :]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Inherits from MistralForCausalLM: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/mistral/modeling_mistral.py
        The only differences are:
        - add new args token_idx
        - add token_idx into model_inputs
        - from step2 when enable KV cache, slice next_input_ids from input_ids base on the token_idx
        - from step2 when enable KV cache, slice next_position_ids from position_ids base on the token_idx
        """
        token_idx = kwargs.get("token_idx", None)

        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if token_idx is None:
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
            else:
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)

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
                "reuse_cache": kwargs.get("reuse_cache"),
                "trim_logits": kwargs.get("trim_logits"),
                "cache_idx": kwargs.get("cache_idx"),
                "attn_softmax_bf16": kwargs.get("attn_softmax_bf16"),
                "lazy_mode": kwargs.get("lazy_mode"),
                "use_flash_attention": kwargs.get("use_flash_attention"),
                "flash_attention_recompute": kwargs.get("flash_attention_recompute"),
                "flash_attention_causal_mask": kwargs.get("flash_attention_causal_mask"),
            }
        )
        return model_inputs


def apply_customized_rope(q, k, cos, sin, position_ids):
    if q.device.type == "hpu" and has_fused_rope:
        # TODO: remove `.clone()` when SynapseAI v1.15 is released
        if k.dtype == torch.bfloat16:
            return FusedRoPE.apply(
                q, cos.unsqueeze(0).unsqueeze(0).clone(), sin.unsqueeze(0).unsqueeze(0).clone(), position_ids
            ), FusedRoPE.apply(
                k,
                cos.unsqueeze(0).unsqueeze(0).clone().to(torch.bfloat16),
                sin.unsqueeze(0).unsqueeze(0).clone().to(torch.bfloat16),
                position_ids,
            )
        return FusedRoPE.apply(
            q, cos.unsqueeze(0).unsqueeze(0).clone(), sin.unsqueeze(0).unsqueeze(0).clone(), position_ids
        ), FusedRoPE.apply(
            k, cos.unsqueeze(0).unsqueeze(0).clone(), sin.unsqueeze(0).unsqueeze(0).clone(), position_ids
        )
    else:
        return apply_rotary_pos_emb(q, k, cos, sin, position_ids)
