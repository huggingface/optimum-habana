# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Mllama model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.mllama.configuration_mllama import MllamaConfig, MllamaTextConfig
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaForCausalLM,
    MllamaForConditionalGeneration,
    MllamaSelfAttentionDecoderLayer,
    MllamaTextCrossAttention,
    MllamaTextModel,
    MllamaTextSelfAttention,
    _prepare_4d_causal_attention_mask_with_cache_position,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    logging,
)


logger = logging.get_logger(__name__)


def _prepare_cross_attention_mask(
    cross_attention_mask: torch.Tensor,
    num_vision_tokens: int,
    dtype: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Copied from _prepare_cross_attention_mask: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/mllama/modeling_mllama.py#L99
    The only differences are:
        - if there's pading in cross_attention_mask in the right. do not masked it, or else it will impact softmax in crossattention
    """
    # reshape so it can be used by attn module
    batch_size, text_total_length, *_ = cross_attention_mask.shape
    pad_idx = torch.nonzero(cross_attention_mask)[-1][1]
    cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=3)
    cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
    cross_attention_mask = cross_attention_mask.unsqueeze(1)

    # invert the mask
    inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
    cross_attention_mask = inverted_cross_attn_mask.masked_fill(
        inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min
    )

    # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
    # last dimension contains negative infinity values, otherwise it's 1
    negative_inf_value = torch.finfo(dtype).min
    full_text_row_masked_out_mask = (
        (cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None]
    )
    full_text_row_masked_out_mask2 = full_text_row_masked_out_mask.clone()
    full_text_row_masked_out_mask2[:, :, pad_idx + 1 :, :] = 1
    cross_attention_mask *= full_text_row_masked_out_mask2

    return cross_attention_mask, full_text_row_masked_out_mask


class GaudiMllamaTextCrossAttention(MllamaTextCrossAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = None,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Copied from MllamaTextCrossAttention::forward: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/mllama/modeling_mllama.py#L512
        The only differences are:
            - add token_idx support
            - add support if past_key_value is not Cache
        """
        """Input shape: Batch x Time x Channel"""
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)

        if cross_attention_states is not None:
            key_states = self.k_proj(cross_attention_states)
            value_states = self.v_proj(cross_attention_states)
            key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            key_states = self.k_norm(key_states)
            if past_key_value is not None:
                # if we have a new image + new tokens, we only computed key_states on that new image
                # we still update the cross key states, past_image, new_image. And use it!
                if isinstance(past_key_value, Cache):
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                    )
                else:
                    if token_idx is not None:
                        past_key_value[0].index_copy_(2, token_idx - 1, key_states)
                        past_key_value[1].index_copy_(2, token_idx - 1, value_states)
                        key_states = past_key_value[0]
                        value_states = past_key_value[1]
                    else:
                        key_states = torch.cat((past_key_value[0], key_states), dim=2)
                        value_states = torch.cat((past_key_value[1], value_states), dim=2)
            if use_cache and not isinstance(past_key_value, Cache):
                past_key_value = (key_states, value_states)
        elif cache_position[0] != 0:
            if isinstance(past_key_value, Cache):
                key_states, value_states = (
                    past_key_value.key_cache[self.layer_idx],
                    past_key_value.value_cache[self.layer_idx],
                )
            else:
                key_states, value_states = (past_key_value[0], past_key_value[1])
        else:
            raise ValueError(
                "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
            )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class GaudiMllamaTextSelfAttention(MllamaTextSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_value=None,
        cache_position=None,
        token_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Copied from MllamaTextSelfAttention::forward: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/mllama/modeling_mllama.py#L733
        The only differences are:
            - add token_idx support
            - add support if past_key_value is not Cache
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if isinstance(past_key_value, Cache):
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                if token_idx is not None:
                    past_key_value[0].index_copy_(2, token_idx - 1, key_states)
                    past_key_value[1].index_copy_(2, token_idx - 1, value_states)
                    key_states = past_key_value[0]
                    value_states = past_key_value[1]
                else:
                    key_states = torch.cat((past_key_value[0], key_states), dim=2)
                    value_states = torch.cat((past_key_value[1], value_states), dim=2)
        if use_cache and not isinstance(past_key_value, Cache):
            past_key_value = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Modified from transformers.models.llama.modeling_llama.LlamaDecoderLayer
class GaudiMllamaSelfAttentionDecoderLayer(MllamaSelfAttentionDecoderLayer):
    def __init__(self, config: MllamaTextConfig, layer_idx: int) -> None:
        super(GaudiMllamaSelfAttentionDecoderLayer, self).__init__(config, layer_idx)
        self.self_attn = GaudiMllamaTextSelfAttention(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        token_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Copied from MllamaSelfAttentionDecoderLayer::forward: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/mllama/modeling_mllama.py#L904
        The only differences are:
            - add token_idx input
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
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            token_idx=token_idx,
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


class GaudiMllamaCrossAttentionDecoderLayer(MllamaCrossAttentionDecoderLayer):
    def __init__(self, config: MllamaTextConfig, layer_idx: int) -> None:
        super(GaudiMllamaCrossAttentionDecoderLayer, self).__init__(config, layer_idx)
        self.cross_attn = GaudiMllamaTextCrossAttention(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        token_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Copied from MllamaCrossAttentionDecoderLayer::forward: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/mllama/modeling_mllama.py#L989
        The only differences are:
            - add token_idx support
            - pass use_cache to cross_attn
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attn_weights, past_key_value = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            cache_position=cache_position,
            use_cache=use_cache,
            token_idx=token_idx,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if full_text_row_masked_out_mask is not None:
            hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states  # type: ignore
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


class GaudiMllamaTextModel(MllamaTextModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cross_attention_states: Optional[torch.FloatTensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Copied from MllamaTextModel::forward: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/mllama/modeling_mllama.py#L1617
        The only differences are:
            - add token_idx support
            - add support if past_key_value is not Cache
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if cache_position is None:
            if isinstance(past_key_values, Cache):
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            else:
                past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            if token_idx is not None and past_key_values is not None:
                past_seen_tokens = token_idx
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None if isinstance(past_key_values, Cache) else ()

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # For text-only path we should skip cross attention layers.
            # Let's check if the layer is cross attention layer and if we have cross attention states
            # or cached cross attention states.
            is_cross_attention_layer = idx in self.cross_attention_layers
            is_cross_attention_cache_empty = past_key_values is None or (
                past_key_values is not None and past_key_values.get_seq_length(idx) == 0
                if isinstance(past_key_values, Cache)
                else False
            )

            if is_cross_attention_layer and cross_attention_states is None and is_cross_attention_cache_empty:
                continue

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    cross_attention_states,
                    cross_attention_mask,
                    causal_mask,
                    full_text_row_masked_out_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                if isinstance(past_key_values, Cache):
                    past_key_value = past_key_values
                else:
                    past_key_value = None if past_key_values is None else past_key_values[idx]
                layer_outputs = decoder_layer(
                    hidden_states,
                    cross_attention_states=cross_attention_states,
                    cross_attention_mask=cross_attention_mask,
                    attention_mask=causal_mask,
                    full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    token_idx=token_idx,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                if isinstance(past_key_values, Cache):
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                else:
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

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        """
        Copied from MllamaTextModel::_update_causal_mask: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/mllama/modeling_mllama.py#L1768
        The only differences are:
            - add support if past_key_value is not Cache
        """
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        if isinstance(past_key_values, Cache):
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        else:
            past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # TODO: we have only SDPA currently and there's a bug when attn-bias is passed. Need to add eager attn and return the line
        # self.config._attn_implementation == "sdpa" and
        if self.config._attn_implementation == "sdpa" and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class GaudiMllamaForCausalLM(MllamaForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cross_attention_states: Optional[torch.LongTensor] = None,
        cross_attention_mask: Optional[torch.LongTensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Copied from MllamaForCausalLM::forward: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/mllama/modeling_mllama.py#L1871
        The only differences are:
            - add token_idx input
            - add logits handle if token_idx is not None
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            token_idx=token_idx,
        )

        hidden_states = outputs[0]

        if token_idx is None and num_logits_to_keep != 0:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()
        else:
            logits = self.lm_head(hidden_states).float()

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


class GaudiMllamaForConditionalGeneration(MllamaForConditionalGeneration):
    def __init__(self, config: MllamaConfig):
        config._attn_implementation = "eager"
        super(GaudiMllamaForConditionalGeneration, self).__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        token_idx: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Copied from MllamaForConditionalGeneration::forward: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/mllama/modeling_mllama.py#L2077
        The only differences are:
            - add token_idx input
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            cross_attention_states = vision_outputs[0]
            cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                -1, cross_attention_states.shape[-2], self.hidden_size
            )

        if cross_attention_mask is not None:
            cross_attention_mask, full_text_row_masked_out_mask = _prepare_cross_attention_mask(
                cross_attention_mask,
                num_vision_tokens=self.vision_model.num_patches,
                dtype=self.dtype,
            )
        else:
            full_text_row_masked_out_mask = None

        if cross_attention_mask is not None and cache_position is not None:
            cross_attention_mask = cross_attention_mask[:, :, cache_position]
            full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            token_idx=token_idx,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        aspect_ratio_ids=None,
        aspect_ratio_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        """
        Copied from MllamaForConditionalGeneration::forward: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/mllama/modeling_mllama.py#L2208
        The only differences are:
            - add token_idx handling
        """
        token_idx = kwargs.get("token_idx", None)
        if past_key_values is not None:
            if token_idx is not None:
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)
            elif inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        # TODO: we have no attention_mask so this won't work, check if we really won't need attention mask and find another way
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

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        if token_idx is not None and past_key_values:
            cache_position = torch.arange(token_idx - 1, token_idx - 1 + input_ids.shape[1], device=input_ids.device)

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cross_attention_mask": cross_attention_mask,
                "token_idx": token_idx,
            }
        )

        # If we're in pre-fill or cacheless decoding step, then we need pixel_values and aspect ratios
        # to compute image hidden states, otherwise they are cached within each cross attn layer
        if (input_ids == self.config.image_token_index).any():
            model_inputs["pixel_values"] = pixel_values
            model_inputs["aspect_ratio_ids"] = aspect_ratio_ids
            model_inputs["aspect_ratio_mask"] = aspect_ratio_mask

        return model_inputs
