# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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
from typing import Callable, Optional, Union

import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.mllama.configuration_mllama import (
    MllamaConfig,
    MllamaTextConfig,
)
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaForCausalLM,
    MllamaForConditionalGeneration,
    MllamaModel,
    MllamaSelfAttentionDecoderLayer,
    MllamaTextCrossAttention,
    MllamaTextModel,
    MllamaTextRMSNorm,
    MllamaTextSelfAttention,
    MllamaVisionEncoder,
    MllamaVisionModel,
    _prepare_aspect_ratio_attention_mask,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torch_flex_attn_available, logging

from ...modeling_attn_mask_utils import _gaudi_prepare_4d_causal_attention_mask


if is_torch_flex_attn_available():
    from transformers.integrations.flex_attention import make_flex_block_causal_mask


try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm as FusedRMSNorm
except ImportError:
    print("Not using HPU fused kernel for RMSNorm")
    FusedRMSNorm = None


logger = logging.get_logger(__name__)

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None


class GaudiMllamaTextRMSNorm(MllamaTextRMSNorm):
    def forward(self, hidden_states):
        """Copied from MllamaTextRMSNorm::forward https://github.com/huggingface/transformers/blob/v4.55-release/src/transformers/models/mllama/modeling_mllama.py#L372
        The only differences are:
        - Using FusedRMSNorm"""
        input_dtype = hidden_states.dtype
        if FusedRMSNorm is not None:
            hidden_states = FusedRMSNorm.apply(hidden_states.float(), self.weight.float(), self.variance_epsilon)
            return hidden_states.to(input_dtype)
        else:
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)


def _prepare_cross_attention_mask(
    cross_attention_mask: torch.Tensor,
    num_vision_tokens: int,
    dtype: str,
    token_idx: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Copied from _prepare_cross_attention_mask: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/mllama/modeling_mllama.py#L99
    The only differences are:
        - if there's pading in cross_attention_mask in the right. do not masked it, or else it will impact softmax in crossattention
    """
    # reshape so it can be used by attn module
    # Updated cross_attention_mask alignment logic to ensure memory alignment with dtype size (256-byte boundary)
    cross_attention_mask = cross_attention_mask.to(dtype)
    dtype_size = (
        torch.finfo(dtype).bits if torch.is_floating_point(torch.tensor(0, dtype=dtype)) else torch.iinfo(dtype).bits
    )
    alignment = int(256 / (dtype_size / 8))
    aligned_num_vision_tokens = math.ceil(num_vision_tokens / alignment) * alignment
    batch_size, text_total_length, _, original_dim = cross_attention_mask.shape
    cross_attention_mask = cross_attention_mask.repeat_interleave(aligned_num_vision_tokens, dim=3)
    cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
    cross_attention_mask = cross_attention_mask[:, :, : num_vision_tokens * original_dim]
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
    if token_idx is not None:
        cross_attention_mask_2 = cross_attention_mask[:, :, token_idx:, 1]
        cross_attention_mask *= full_text_row_masked_out_mask
        cross_attention_mask[:, :, token_idx:, 1] = cross_attention_mask_2
    else:
        cross_attention_mask *= full_text_row_masked_out_mask

    return cross_attention_mask, full_text_row_masked_out_mask


class GaudiMllamaVisionEncoder(MllamaVisionEncoder):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[tuple, BaseModelOutput]:
        """
        Copied from MllamaVisionEncoder::forward:https://github.com/huggingface/transformers/blob/v4.55-release/src/transformers/models/mllama/modeling_mllama.py#L333
        The only differences are:
            - add mark_step()
            - added patch from PR: https://github.com/huggingface/transformers/pull/40083
        """
        encoder_states = (hidden_states,)
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_state=hidden_states,
                attention_mask=attention_mask,
            )
            encoder_states = encoder_states + (hidden_states,)

            htcore.mark_step()

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)


class GaudiMllamaTextCrossAttention(MllamaTextCrossAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """
        Copied from MllamaTextCrossAttention::forward: https://github.com/huggingface/transformers/blob/v4.55-release/src/transformers/models/mllama/modeling_mllama.py#L410
        The only differences are:
            - add token_idx support
            - add support if past_key_value is not Cache
            - cache position is None
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
                past_key_value = [key_states, value_states]
        elif not isinstance(past_key_value, Cache) and past_key_value is not None:
            key_states, value_states = (past_key_value[0], past_key_value[1])
        elif cache_position is not None and cache_position[0] != 0:
            key_states, value_states = (
                past_key_value.layers[self.layer_idx].keys,
                past_key_value.layers[self.layer_idx].values,
            )
        else:
            raise ValueError(
                "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
            )

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class GaudiMllamaTextSelfAttention(MllamaTextSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        use_cache: bool = False,
        past_key_value=None,
        cache_position=None,
        token_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Copied from MllamaTextSelfAttention::forward: https://github.com/huggingface/transformers/blob/v4.55-release/src/transformers/models/mllama/modeling_mllama.py#L526
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
            past_key_value = [key_states, value_states]

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


# Modified from transformers.models.llama.modeling_llama.LlamaDecoderLayer
class GaudiMllamaSelfAttentionDecoderLayer(MllamaSelfAttentionDecoderLayer):
    def __init__(self, config: MllamaTextConfig, layer_idx: int) -> None:
        super(GaudiMllamaSelfAttentionDecoderLayer, self).__init__(config, layer_idx)
        self.self_attn = GaudiMllamaTextSelfAttention(config, layer_idx=layer_idx)
        self.input_layernorm = GaudiMllamaTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        token_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Copied from MllamaSelfAttentionDecoderLayer::forward: https://github.com/huggingface/transformers/blob/v4.55-release/src/transformers/models/mllama/modeling_mllama.py#L608
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
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            token_idx=token_idx,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

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
        full_text_row_masked_out_mask: tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        """
        Copied from MllamaCrossAttentionDecoderLayer::forward: https://github.com/huggingface/transformers/blob/v4.55-release/src/transformers/models/mllama/modeling_mllama.py#L683
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
            cache_position=cache_position,
            use_cache=use_cache,
            token_idx=token_idx,
            **kwargs,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if full_text_row_masked_out_mask is not None:
            hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states  # type: ignore
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

        outputs = (hidden_states,)

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
        full_text_row_masked_out_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        """
        Copied from MllamaTextModel::forward: https://github.com/huggingface/transformers/blob/v4.55-release/src/transformers/models/mllama/modeling_mllama.py#L1159
        The only differences are:
            - add token_idx support
            - add support if past_key_value is not Cache
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if isinstance(past_key_values, Cache):
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        else:
            past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens,
                inputs_embeds.shape[1] + past_seen_tokens,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            position_ids = position_ids.unsqueeze(0)
        cache_position = None
        causal_mask = _gaudi_prepare_4d_causal_attention_mask(
            attention_mask,
            input_ids.shape,
            inputs_embeds,
            past_seen_tokens,
        )

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        next_decoder_cache = None if isinstance(past_key_values, Cache) else ()

        for idx, decoder_layer in enumerate(self.layers):
            if not self.training and (
                not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1
            ):
                htcore.mark_step()

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
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                token_idx=token_idx,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                if isinstance(past_key_values, Cache):
                    next_decoder_cache = layer_outputs[1]
                else:
                    next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        next_cache = next_decoder_cache if use_cache else None

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        """
        Copied from MllamaTextModel::_update_causal_mask: https://github.com/huggingface/transformers/blob/v4.55-release/src/transformers/models/mllama/modeling_mllama.py#L805
        The only differences are:
            - add support if past_key_value is not Cache
        """
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        if isinstance(past_key_values, Cache):
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False
        else:
            past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            using_compilable_cache = False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class GaudiMllamaForCausalLM(MllamaForCausalLM):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cross_attention_states: Optional[torch.LongTensor] = None,
        cross_attention_mask: Optional[torch.LongTensor] = None,
        full_text_row_masked_out_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        token_idx: Optional[torch.Tensor] = None,
        trim_logits: Optional[bool] = False,
        logits_bf16: Optional[bool] = False,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        """
        Copied from MllamaForCausalLM::forward: https://github.com/huggingface/transformers/blob/v4.55-release/src/transformers/models/mllama/modeling_mllama.py#L1301
        The only differences are:
            - add token_idx input
            - add logits handle if token_idx is not None
            - add trim_logits
            - add logits_bf16
        """
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
            cache_position=cache_position,
            token_idx=token_idx,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        _, seq_len, _ = hidden_states.shape
        if seq_len > 1 and trim_logits and not self.training:
            if token_idx is not None:
                hidden_states = hidden_states.index_select(1, token_idx - 1)
            else:
                hidden_states = hidden_states[:, -1, :]

        if token_idx is None and logits_to_keep != 0:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        else:
            logits = self.lm_head(hidden_states)

        if not logits_bf16:
            logits = logits.float()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GaudiMllamaModel(MllamaModel):
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
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        trim_logits: Optional[bool] = False,
        logits_bf16: Optional[bool] = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        Copied from https://github.com/huggingface/transformers/blob/v4.55-release/src/transformers/models/mllama/modeling_mllama.py#L1433
        The only differences are additional arguments:
        - token_idx
        - trim_logits
        - logits_bf16

        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

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
            )
            cross_attention_states = vision_outputs.last_hidden_state
            cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                -1, cross_attention_states.shape[-2], self.hidden_size
            )

        if cross_attention_mask is not None:
            cross_attention_mask, full_text_row_masked_out_mask = _prepare_cross_attention_mask(
                cross_attention_mask,
                num_vision_tokens=self.vision_model.num_patches,
                dtype=self.dtype,
                token_idx=token_idx,
            )
        else:
            full_text_row_masked_out_mask = None

        if cross_attention_mask is not None:
            if cache_position is not None:
                cross_attention_mask = cross_attention_mask[:, :, cache_position]
                full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]
            elif past_key_values is not None:
                if token_idx is not None:
                    cross_attention_mask = torch.index_select(cross_attention_mask, -2, token_idx - 1)
                    full_text_row_masked_out_mask = torch.index_select(
                        full_text_row_masked_out_mask, -2, token_idx - 1
                    )
                else:
                    cross_attention_mask = cross_attention_mask[:, :, -1:]
                    full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, -1:]

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
            cache_position=cache_position,
            token_idx=token_idx,
            trim_logits=trim_logits,
            logits_bf16=logits_bf16,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GaudiMllamaForConditionalGeneration(MllamaForConditionalGeneration):
    def __init__(self, config: MllamaConfig):
        # Use the user option as the attention for the text model
        config.text_config._attn_implementation = config._attn_implementation or "eager"

        # Use always `sdpa` as it is better for the vision model on HPU
        config.vision_config._attn_implementation = "sdpa"

        super().__init__(config)

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
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        token_idx: Optional[torch.Tensor] = None,
        trim_logits: Optional[bool] = False,
        logits_bf16: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        Copied from: https://github.com/huggingface/transformers/blob/v4.55-release/src/transformers/models/mllama/modeling_mllama.py#L1576
        The only differences are additional arguments:
        - token_idx
        - trim_logits
        - logits_bf16
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            aspect_ratio_mask=aspect_ratio_mask,
            aspect_ratio_ids=aspect_ratio_ids,
            cross_attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            token_idx=token_idx,
            trim_logits=trim_logits,
            logits_bf16=logits_bf16,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.text_config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
        logits_to_keep=None,
        **kwargs,
    ):
        """
        Copied from MllamaForConditionalGeneration::prepare_inputs_for_generation: https://github.com/huggingface/transformers/blob/v4.55-release/src/transformers/models/mllama/modeling_mllama.py#L1687
        The only differences are:
            - add token_idx handling
            - add bucket_internal handling
        """
        token_idx = kwargs.get("token_idx", None)
        bucket_internal = kwargs.get("bucket_internal", None)
        if past_key_values is not None:
            if token_idx is not None:
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)
            elif inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        elif bucket_internal and token_idx is not None:
            # for the 1st token we can slice the inputs till token idx for the fwd pass.
            input_ids = input_ids[:, :token_idx]
            attention_mask = attention_mask[:, :token_idx]
            if cross_attention_mask is not None:
                cross_attention_mask = cross_attention_mask[:, :token_idx, ...]

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

        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep

        # keep cache_position implementation as None for HPU
        cache_position = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cross_attention_mask": cross_attention_mask,
                "token_idx": token_idx,
                "trim_logits": kwargs.get("trim_logits"),
                "logits_bf16": kwargs.get("logits_bf16"),
            }
        )

        # If we're in pre-fill or cacheless decoding step, then we need pixel_values and aspect ratios
        # to compute image hidden states, otherwise they are cached within each cross attn layer
        if (input_ids == self.config.image_token_index).any():
            model_inputs["pixel_values"] = pixel_values
            model_inputs["aspect_ratio_ids"] = aspect_ratio_ids
            model_inputs["aspect_ratio_mask"] = aspect_ratio_mask

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        """
        Copied from MllamaForConditionalGeneration::_update_model_kwargs_for_generation: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/mllama/modeling_mllama.py#L2274
        The only differences are:
            - add token_idx handling
        """
        cross_attention_mask_prev = model_kwargs.get("cross_attention_mask", None)
        model_kwargs = super(MllamaForConditionalGeneration, self)._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

        # add cross-attn mask for new token
        if cross_attention_mask_prev is not None:
            token_idx = model_kwargs.get("token_idx", None)
            token_idx_cpu = model_kwargs.get(
                "token_idx_cpu", None
            )  # returns an integer so following slicing ops happen using int instead of tensor
            if token_idx is not None:
                mask = cross_attention_mask_prev[:, token_idx_cpu - 2 : token_idx_cpu - 1, ...].clone()
                cross_attention_mask_prev.index_copy_(1, token_idx - 1, mask)
                model_kwargs["cross_attention_mask"] = cross_attention_mask_prev
            else:
                model_kwargs["cross_attention_mask"] = torch.cat(
                    [cross_attention_mask_prev, cross_attention_mask_prev[:, -1:, ...]], dim=1
                )
        return model_kwargs


class GaudiMllamaVisionModel(MllamaVisionModel):
    def forward(
        self,
        pixel_values: torch.Tensor,
        aspect_ratio_ids: torch.Tensor,
        aspect_ratio_mask: torch.Tensor,
        **kwargs,
    ) -> Union[BaseModelOutput, tuple[torch.Tensor, ...]]:
        """
        Copied from MllamaVisionModel::forward: https://github.com/huggingface/transformers/blob/v4.55-release/src/transformers/models/mllama/modeling_mllama.py#L990
        The only differences are:
            - Apply patch from PR: https://github.com/huggingface/transformers/pull/40083
        """
        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape

        pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
        aspect_ratio_ids = aspect_ratio_ids.reshape(batch_size * num_concurrent_media, -1)

        # Patch embedding
        target_dtype = self.patch_embedding.weight.dtype
        target_device = self.patch_embedding.weight.device
        patch_embeds = self.patch_embedding(pixel_values.to(target_device, target_dtype))
        hidden_state = patch_embeds.flatten(2).transpose(1, 2)

        # Tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, -1, dim)
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)

        # Add cls token
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches, dim)
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)

        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (0, 0, 0, num_padding_patches)  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = F.pad(hidden_state, padding, mode="constant", value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        # Prepare attention mask
        attention_mask = aspect_ratio_mask.reshape(batch_size * num_concurrent_media, -1)
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2],
            dtype=self.dtype,
        )

        # Apply encoder
        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
        output = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
        )
        hidden_state = output.last_hidden_state

        hidden_state = self.layernorm_post(hidden_state)

        # Apply global encoder
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles * (num_patches + num_padding_patches), dim
        )
        global_output = self.global_transformer(
            hidden_state,
            attention_mask=attention_mask,
        )
        hidden_state = global_output.last_hidden_state

        # Remove padding form hidden state
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim
        )
        hidden_state = hidden_state[:, :, :slice_index]
        hidden_state = hidden_state.reshape(batch_size, num_concurrent_media, num_tiles, num_patches, dim)

        # Collect intermediate layer outputs from encoder output
        all_intermediate_hidden_states = [output.hidden_states[i] for i in self.intermediate_layers_indices]
        intermediate_hidden_states = torch.stack(all_intermediate_hidden_states, dim=-1)

        # Remove padding from intermediate hidden states
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, -1
        )
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, -1
        )

        # Concatenate final hidden state and intermediate hidden states
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states], dim=-1)

        return BaseModelOutput(last_hidden_state=hidden_state)
