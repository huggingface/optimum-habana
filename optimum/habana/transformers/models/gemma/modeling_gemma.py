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
"""PyTorch Gemma model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaConfig,
    GemmaDecoderLayer,
    GemmaForCausalLM,
    GemmaMLP,
    GemmaModel,
    apply_rotary_pos_emb,
)
from transformers.utils import is_torchdynamo_compiling, logging

from ...modeling_attn_mask_utils import (
    _gaudi_prepare_4d_causal_attention_mask,
)
from ...modeling_rope_utils import GaudiRotaryEmbedding


try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None

import habana_frameworks.torch.core as htcore


logger = logging.get_logger(__name__)


def gaudi_gemma_repeat_kv(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    n_rep: int,
):
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


class GaudiGemmaAttention(GemmaAttention):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        self.matmul_qk = Matmul()
        self.matmul_av = Matmul()
        self.k_cache = KVCache()
        self.v_cache = KVCache()
        self.inp_seq_len = -1
        self.norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.block_size = 4096
        self.rotary_emb = GaudiRotaryEmbedding(config=self.config)

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
            self.rotary_emb._set_cos_sin_cache(seq_len, self.k_proj.weight.device, self.k_proj.weight.dtype)

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

    def gaudi_flash_attn_v1(self, query_layer, key_layer, value_layer, attention_mask, dropout_rate, q_block_size):
        """
        Gaudi version of Flash Attention V1 to support long sequence at prompt phase
        Causal mask is not supported in this optimization
        """
        q_len = query_layer.size(-2)
        q_tiles = (q_len // q_block_size) if (q_len % q_block_size == 0) else math.ceil(q_len / q_block_size)
        q_padding = q_tiles * q_block_size - q_len
        query_layer = F.pad(query_layer, (0, 0, 0, q_padding), "constant", 0)
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, 0, 0, q_padding), "constant", -10000.0)

        row_o_list = []
        for i in range(q_tiles):
            s, e = i * q_block_size, (i + 1) * q_block_size
            row_q = query_layer[:, :, s:e, :]
            row_mask = attention_mask[:, :, s:e, :]
            attn_output_partial = FusedSDPA.apply(row_q, key_layer, value_layer, row_mask, dropout_rate, False, None)
            row_o_list.append(attn_output_partial)
        attn_output = torch.cat(row_o_list, dim=-2)

        if q_padding != 0:
            attn_output = attn_output[:, :, :-q_padding, :]

        return attn_output

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
        cache_idx: int = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        The only differences are:
        - add new args token_idx
        - optimize KV cache
        - add new args attn_softmax_bf16
        - add new args reuse_cache
        - add new args use_flash_attention
        - add new arg flash_attention_recompute
        """
        if "padding_mask" in kwargs:
            logger.warning_once(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if token_idx is None:
                if hasattr(past_key_value, "get_usable_length"):
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
                else:
                    kv_seq_len += past_key_value[0].shape[-2]
            else:
                if reuse_cache:
                    kv_seq_len = past_key_value[0][-2]
                else:
                    kv_seq_len = past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos[position_ids], sin[position_ids])

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
                with ht.sdp_kernel(enable_recompute=False):
                    attn_output = FusedSDPA.apply(
                        query_states, key_states, value_states, attention_mask, 0.0, False, None
                    )
            else:
                # first token
                if flash_attention_causal_mask:
                    # causal masking on first token requires inputs to be of the same length
                    with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                        attn_output = FusedSDPA.apply(query_states, key_states, value_states, None, 0.0, True, None)
                else:
                    with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                        if q_len > 16384:
                            attn_output = self.gaudi_flash_attn_v1(
                                query_states, key_states, value_states, attention_mask, 0.0, self.block_size
                            )
                            htcore.mark_step()
                        else:
                            attn_output = FusedSDPA.apply(
                                query_states, key_states, value_states, attention_mask, 0.0, False, None
                            )

        else:
            query_states, key_states, value_states, attention_mask = gaudi_gemma_repeat_kv(
                query_states, key_states, value_states, attention_mask, self.num_key_value_groups
            )

            attn_weights = self.matmul_qk(query_states, key_states.transpose(-2, -1)) * self.norm_factor

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                if cache_position is not None:
                    causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            if attn_softmax_bf16:
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=query_states.dtype)
            else:
                # upcast attention to fp32
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                    query_states.dtype
                )
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = self.matmul_av(attn_weights, value_states)
            attn_output = attn_output.reshape(bsz, -1, q_len, self.head_dim)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def attention_all_reduce(self, attn_output):
        if hasattr(self.o_proj, "all_reduce"):
            self.o_proj.all_reduce(attn_output)

    def post_attn_forward(self, attn_output):
        if hasattr(self.o_proj, "post_all_reduce"):
            self.o_proj.post_all_reduce(attn_output)
        return attn_output


class GaudiGemmaMLP(GemmaMLP):
    def pre_mlp_forward(self, x):
        inputs = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        output = self.down_proj(inputs)
        return output

    def mlp_all_reduce(self, x):
        if hasattr(self.down_proj, "all_reduce"):
            self.down_proj.all_reduce(x)

    def post_mlp_forward(self, x):
        if hasattr(self.down_proj, "post_all_reduce"):
            return self.down_proj.post_all_reduce(x)
        return x


class GaudiGemmaDecoderLayer(GemmaDecoderLayer):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GaudiGemmaAttention(config, layer_idx)
        self.mlp = GaudiGemmaMLP(config)

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.self_attn.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.self_attn.reorder_kv_cache(beam_idx)

    def update_sincos_cache(self, seq_len):
        self.self_attn.update_sincos_cache(seq_len)

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
        cache_idx: int = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states = self.input_layernorm(hidden_states)
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
            use_flash_attention,
            flash_attention_recompute,
            flash_attention_causal_mask,
            cache_idx=cache_idx,
        )
        return hidden_states, attn_weights, present_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
        cache_idx: int = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Copied from GemmaDecoderLayer.forward: https://github.com/huggingface/transformers/blob/v4.38.1/src/transformers/models/gemma/modeling_gemma.py
        The only differences are:
        - add new args token_idx
        """
        residual = hidden_states

        hidden_states, self_attn_weights, present_key_value = self.pre_attn(
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
            cache_idx=cache_idx,
        )

        self.self_attn.attention_all_reduce(hidden_states)

        hidden_states, residual = self.post_attn_pre_mlp(hidden_states, residual)

        self.mlp.mlp_all_reduce(hidden_states)

        hidden_states = self.post_mlp(hidden_states, residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

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

    def post_mlp(self, hidden_states, residual):
        hidden_states = self.mlp.post_mlp_forward(hidden_states)

        if self.training:
            hidden_states = hidden_states + residual
        else:
            residual.add_(hidden_states)
            hidden_states = residual

        return hidden_states


class GaudiGemmaModel(GemmaModel):
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
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        cache_idx: int = None,
        lazy_mode: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Copied from GemmaModel.forward: https://github.com/huggingface/transformers/blob/v4.38.1/src/transformers/models/gemma/modeling_gemma.py
        The only differences are:
        - add new args token_idx
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self._attn_implementation = "eager"

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_new_cache = False  # Ignoring new Cache path for HPU
        past_seen_tokens = 0

        if past_key_values is not None and use_cache:  # kept for BC (cache positions)
            if reuse_cache:
                past_seen_tokens = past_key_values[0][0][2]
            else:
                if use_new_cache:
                    use_legacy_cache = not isinstance(past_key_values, Cache)
                    if use_legacy_cache:
                        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                    past_seen_tokens = past_key_values.get_usable_length(seq_length)
                else:
                    past_seen_tokens = past_key_values[0][0].shape[2]

        cache_position = None

        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens, seq_length + past_seen_tokens, dtype=torch.long, device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0)

        # HPU specific mask generation
        attention_mask = _gaudi_prepare_4d_causal_attention_mask(
            attention_mask,
            input_ids.shape if input_ids is not None else (batch_size, seq_length),
            inputs_embeds,
            past_seen_tokens,
        )
        # embed positions
        hidden_states = inputs_embeds

        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=inputs_embeds.device)
        hidden_states = hidden_states * normalizer

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if not use_new_cache else None

        if lazy_mode:
            htcore.mark_step()

        for layer_idx, decoder_layer in enumerate(self.layers):
            if (
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
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None if past_key_values is None else past_key_values[layer_idx],
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    token_idx=token_idx,
                    attn_softmax_bf16=attn_softmax_bf16,
                    reuse_cache=reuse_cache,
                    use_flash_attention=use_flash_attention,
                    flash_attention_recompute=flash_attention_recompute,
                    flash_attention_causal_mask=flash_attention_causal_mask,
                    cache_idx=cache_idx,
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
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GaudiGemmaForCausalLM(GemmaForCausalLM):
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
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        token_idx: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Inherits from GemmaForCausalLM: https://github.com/huggingface/transformers/blob/v4.38.1/src/transformers/models/gemma/modeling_gemma.py
        The only differences are:
        - add new args token_idx
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            cache_position=cache_position,
            token_idx=token_idx,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
        )

        hidden_states = outputs[0]
        if labels is None and not is_torchdynamo_compiling():
            logger.warning_once(
                "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
            )
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # TODO: remove the float() operation in v4.46
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
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
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=0,
        **kwargs,
    ):
        """
        Inherits from GemmaForCausalLM: https://github.com/huggingface/transformers/blob/v4.38.1/src/transformers/models/gemma/modeling_gemma.py
        The only differences are:
        - add new args token_idx
        - add token_idx into model_inputs
        - from step2 when enable KV cache, slice next_input_ids from input_ids base on the token_idx
        - from step2 when enable KV cache, slice next_position_ids from position_ids base on the token_idx
        """

        token_idx = kwargs.get("token_idx", None)

        if past_key_values is not None:
            if token_idx is None:
                if inputs_embeds is not None:  # Exception 1
                    input_ids = input_ids[:, -cache_position.shape[0] :]
                elif (
                    input_ids.shape[1] != cache_position.shape[0]
                ):  # Default case (the "else", a no op, is Exception 2)
                    input_ids = input_ids[:, cache_position]
            else:
                # past_length += token_idx
                idx = token_idx + kwargs.get("inputs_embeds_offset", 0) - 1
                input_ids = torch.index_select(input_ids, 1, idx)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                if token_idx is not None:
                    position_ids = torch.index_select(position_ids, 1, token_idx - 1)
                else:
                    position_ids = position_ids[:, -input_ids.shape[1] :]
                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        if token_idx is None:
            if past_key_value := getattr(self.model.layers[0].self_attn, "past_key_value", None):
                # generation with static cache
                past_length = past_key_value.get_seq_length()
                input_ids = input_ids[:, past_length:]
                position_ids = position_ids[:, past_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format)}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "num_logits_to_keep": num_logits_to_keep,
                "token_idx": token_idx,
            }
        )
        return model_inputs
