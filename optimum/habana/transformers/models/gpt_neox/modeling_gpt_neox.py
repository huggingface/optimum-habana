from typing import Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    GPTNeoXForCausalLM,
    GPTNeoXLayer,
    GPTNeoXMLP,
    GPTNeoXModel,
    KwargsForCausalLM,
    apply_rotary_pos_emb,
    logger,
)
from transformers.processing_utils import Unpack

from ...modeling_attn_mask_utils import _gaudi_prepare_4d_causal_attention_mask
from ...modeling_rope_utils import GaudiRotaryEmbedding


try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE
except ImportError:
    print("Not using HPU fused kernel for apply_rotary_pos_emb")
    FusedRoPE = None

from ..modeling_all_models import apply_customized_rope_module


def gaudi_eager_attention_forward(
    query, key, value, attention_mask, head_mask, norm_factor, attention_dropout, training, **_kwargs
):
    """
    Copied from: https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L98
    Changes:
    - transposition at the end is commented
    """
    # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
    batch_size, num_attention_heads, query_length, attn_head_size = query.size()
    key_length = key.size(-2)

    query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
    key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
    attn_scores = torch.zeros(
        batch_size * num_attention_heads,
        query_length,
        key_length,
        dtype=query.dtype,
        device=key.device,
    )
    attn_scores = torch.baddbmm(
        attn_scores,
        query,
        key.transpose(1, 2),
        beta=1.0,
        alpha=norm_factor,
    )
    attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_scores = attn_scores + causal_mask

    attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
    attn_weights = attn_weights.to(value.dtype)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_weights = torch.nn.functional.dropout(attn_weights, p=attention_dropout, training=training)
    attn_output = torch.matmul(attn_weights, value)

    # # Reshape outputs
    # attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class GaudiGPTNeoXAttention(GPTNeoXAttention):
    def __init__(self, config: GPTNeoXConfig, layer_idx=None):
        super().__init__(config, layer_idx)
        self.rotary_emb = GaudiRotaryEmbedding(config=self.config)
        self.num_attention_heads = config.num_attention_heads

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        padding_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        token_idx: Optional[torch.Tensor] = None,
    ):
        """
        Copied from GPTNeoXAttention.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py
        The only differences are:
        - add new args token_idx
        - optimize KV cache
        """
        bsz, seq_len, _ = hidden_states.shape
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_customized_rope(query_rot, key_rot, cos, sin, position_ids, training=self.training)
        query = torch.cat((query, query_pass), dim=-1).contiguous()
        key = torch.cat((key, key_pass), dim=-1).contiguous()
        value = value.contiguous()

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            if token_idx is not None:
                past_key.index_copy_(2, token_idx - 1, key)
                past_value.index_copy_(2, token_idx - 1, value)
                key = past_key
                value = past_value
            else:
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights = gaudi_eager_attention_forward(
            query,
            key,
            value,
            attention_mask=attention_mask,
            head_mask=head_mask,
            norm_factor=self.scaling,
            attention_dropout=self.config.attention_dropout,
            training=self.training,
        )

        # Reshape outputs and final projection
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor


class GaudiGPTNeoXLayer(GPTNeoXLayer):
    def __init__(self, config, layer_idx):
        super(GPTNeoXLayer, self).__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_dropout = torch.nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = torch.nn.Dropout(config.hidden_dropout)
        self.attention = GaudiGPTNeoXAttention(config, layer_idx)
        self.mlp = GPTNeoXMLP(config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
    ):
        """
        Copied from GPTNeoxLayer.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py
        The only differences are:
        - add new args token_idx
        """
        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            token_idx=token_idx,
        )
        attn_output = attention_layer_outputs[0]  # output_attn: attn_output, present, (attn_weights)
        attn_output = self.post_attention_dropout(attn_output)
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output

        if use_cache:
            outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs


def gaudi_gpt_neox_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.FloatTensor]]]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    token_idx: Optional[torch.Tensor] = None,
    **kwargs,
) -> BaseModelOutputWithPast:
    """
    Copied from GPTNeoxModel.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py
    The only differences are:
    - add new args token_idx
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    elif input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    batch_size, seq_length = input_shape

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * self.config.num_hidden_layers)
    else:
        past_length = past_key_values[0][0].size(-2)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_in(input_ids)

    # Attention mask.
    attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
    attention_mask = _gaudi_prepare_4d_causal_attention_mask(
        attention_mask=attention_mask,
        input_shape=(batch_size, seq_length),
        inputs_embeds=inputs_embeds,
        past_key_values_length=past_length,
    )

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    hidden_states = self.emb_dropout(inputs_embeds)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    presents = () if use_cache else None
    all_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(
                layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                head_mask[i],
                use_cache,
                None,
                output_attentions,
                cache_position,
                None,
            )
        else:
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
                token_idx=token_idx,
            )
        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)
        if output_attentions:
            all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

    hidden_states = self.final_layer_norm(hidden_states)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
    )


class GaudiGPTNeoXForCausalLM(GPTNeoXForCausalLM):
    """
    Inherits from GPTNeoXForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py
    The only differences are:
    - add new args token_idx
    - add token_idx into model_inputs
    - from step2 when enable KV cache, slice next_input_ids from input_ids base on the token_idx
    - from step2 when enable KV cache, slice next_position_ids from position_ids base on the token_idx
    """

    def __init__(self, config):
        super(GPTNeoXForCausalLM, self).__init__(config)

        config._attn_implementation = "eager"
        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.FloatTensor]]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs: BaseModelOutputWithPast = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            token_idx=token_idx,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.embed_out(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

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
        token_idx=None,
        **kwargs,
    ):
        input_shape = input_ids.shape

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            if token_idx is not None:
                idx = token_idx + kwargs.get("inputs_embeds_offset", 0) - 1
                input_ids = torch.index_select(input_ids, 1, idx)
            else:
                past_length = past_key_values[0][0].shape[2]

                # Some generation methods already pass only the last input ID
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # Default to old behavior: keep only final ID
                    remove_prefix_length = input_ids.shape[1] - 1

                input_ids = input_ids[:, remove_prefix_length:]

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

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

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
                "token_idx": token_idx,
            }
        )

        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past


def apply_customized_rope(q, k, cos, sin, position_ids, training=True):
    if q.device.type == "hpu" and FusedRoPE is not None:
        if training:
            return apply_customized_rope_module(q.to(torch.float), k.to(torch.float), cos, sin, position_ids, training)
        else:
            return apply_customized_rope_module(q, k, cos, sin, position_ids, training)
    else:
        return apply_rotary_pos_emb(q.to(torch.float), k.to(torch.float), cos[position_ids], sin[position_ids])
