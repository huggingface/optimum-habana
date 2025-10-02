from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
from transformers.models.llama4.modeling_llama4 import Llama4TextAttention, eager_attention_forward
from transformers.processing_utils import Unpack

from optimum.utils import logging


logger = logging.get_logger(__name__)


def gaudi_llama4_rotary_embedding_forward(self, x, position_ids):
    """
    Copied from https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/llama4/modeling_llama4.py#L229 forward call of Llama4TextRotaryEmbedding.
    The difference is :
    -   Computes rotary positional embeddings (RoPE) using real-valued sin and cos functions,
        adapted for Gaudi hardware compatibility. Unlike the standard implementation that uses
        `torch.polar` to produce complex tensors(unsupported on Gaudi)

    Args:
        x (torch.Tensor): Input tensor to which the rotary embeddings will be applied.
            Shape [batch_size, seq_len, num_heads, head_dim].
        position_ids (torch.LongTensor): Tensor of position indices for each token in the sequence.
            Shape [batch_size, seq_len].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of (sin, cos) frequency tensors to be
        applied in the rotary embedding step.
    """
    if "dynamic" in self.rope_type:
        self._dynamic_frequency_update(position_ids, device=x.device)
    # Core RoPE block
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
        sin = freqs.sin()
        cos = freqs.cos()
        sin = sin * self.attention_scaling
        cos = cos * self.attention_scaling
        freqs_cis = (sin, cos)
    return freqs_cis


def rotate(x, sin, cos):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.cat([x_even * cos - x_odd * sin, x_odd * cos + x_even * sin], dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sin, cos = freqs_cis
    cos = cos.unsqueeze(-2).to(xq.dtype)
    sin = sin.unsqueeze(-2).to(xq.dtype)
    xq_out = rotate(xq, sin, cos)
    xk_out = rotate(xk, sin, cos)
    return xq_out, xk_out


class GaudiLlama4TextAttention(Llama4TextAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper
    Copied from JointAttnProcessor2_0.forward: https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/llama4/modeling_llama4.py
        The only differences are:
        - Reshaping of attn_scales to match query states.
        - Using custom implementation of rotary positional embeddings.
    """

    def __init__(self, config: Llama4TextConfig, layer_idx):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_rope:  # the 16E model skips rope for long context on certain layers
            query_states, key_states = apply_rotary_emb(query_states, key_states, position_embeddings)

        if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        # Use temperature tuning from https://arxiv.org/abs/2501.19399) to NoROPE layers
        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                torch.log(torch.floor((cache_position.float() + 1.0) / self.floor_scale) + 1.0) * self.attn_scale + 1.0
            )
            # Reshape attn_scales from [seq_len] to [batch_size, seq_len, 1, 1] so it can be broadcasted during elementwise scaling of query_states.
            # Previous reshape assumed enough samples, causing size mismatch.
            attn_scales = attn_scales.view(1, input_shape[1], 1, 1).expand(input_shape[0], input_shape[1], 1, 1)
            query_states = (query_states * attn_scales).to(query_states.dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def gaudi_llama4_text_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> Union[Tuple, BaseModelOutputWithPast]:
    """
    Copied from https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/llama4/modeling_llama4.py#L624 forward call of Llama4TextModel
    The difference is :
    -   Ensure positional arguments are correctly aligned for the decoder layer call
        when using gradient checkpointing, since checkpointing does not support keyword arguments.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask, chunk_causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    freq_cis = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                chunk_causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                False,
                use_cache,
                cache_position,
                freq_cis,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                chunk_causal_mask=chunk_causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=freq_cis,
                **flash_attn_kwargs,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    output = BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    return output if return_dict else output.to_tuple()
