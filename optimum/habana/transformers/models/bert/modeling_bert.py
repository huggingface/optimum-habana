from typing import Optional, Union

import torch
import torch.utils.checkpoint
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from optimum.utils import logging
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.models.bert.modeling_bert import BaseModelOutputWithPoolingAndCrossAttentions, BertSdpaSelfAttention


logger = logging.get_logger(__name__)


def gaudi_BertModel_forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.Tensor] = None,
) -> Union[tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
    r"""
    Copied from https://github.com/huggingface/transformers/blob/15c74a28294fe9082b81b24efe58df16fed79a9e/src/transformers/models/bert/modeling_bert.py
    Changes:
      - Added dtype to allow for bf16 autocast support on HPU
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if self.config.is_decoder:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
    else:
        use_cache = False

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    batch_size, seq_length = input_shape
    device = input_ids.device if input_ids is not None else inputs_embeds.device

    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = (
            past_key_values[0][0].shape[-2]
            if not isinstance(past_key_values, Cache)
            else past_key_values.get_seq_length()
        )

    if token_type_ids is None:
        if hasattr(self.embeddings, "token_type_ids"):
            buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    embedding_output = self.embeddings(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
        inputs_embeds=inputs_embeds,
        past_key_values_length=past_key_values_length,
    )

    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, dtype=self.dtype)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    encoder_outputs = self.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

    if not return_dict:
        return (sequence_output, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        past_key_values=encoder_outputs.past_key_values,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
        cross_attentions=encoder_outputs.cross_attentions,
    )


def gaudi_Bert_Sdpa_SelfAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    past_key_value: Optional[tuple[tuple[torch.FloatTensor]]] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor]:
    r"""
    Copied from https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py
    Changes:
        - Use HPU's FusedSDPA(fast mode for softmax) to replace `torch.nn.functional.scaled_dot_product_attention`
    """
    if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
        logger.warning_once(
            "BertSdpaSelfAttention is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
            "non-absolute `position_embedding_type` or `output_attentions=True` or `head_mask`. Falling back to "
            "the manual attention implementation, but specifying the manual implementation will be required from "
            "Transformers version v5.0.0 onwards. This warning can be removed using the argument "
            '`attn_implementation="eager"` when loading the model.'
        )
        return BertSdpaSelfAttention.forward(
            self,
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            cache_position,
        )

    bsz, tgt_len, _ = hidden_states.size()

    query_layer = (
        self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
    )

    is_cross_attention = encoder_hidden_states is not None
    current_states = encoder_hidden_states if is_cross_attention else hidden_states
    if past_key_value is not None:
        if isinstance(past_key_value, EncoderDecoderCache):
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache
        else:
            curr_past_key_value = past_key_value

    current_states = encoder_hidden_states if is_cross_attention else hidden_states
    if is_cross_attention and past_key_value is not None and is_updated:
        # reuse k,v, cross_attentions
        key_layer = curr_past_key_value.layers[self.layer_idx].keys
        value_layer = curr_past_key_value.layers[self.layer_idx].values
    else:
        key_layer = (
            self.key(current_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        )
        value_layer = (
            self.value(current_states)
            .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

    if past_key_value is not None:
        # save all key/value_layer to cache to be re-used for fast auto-regressive generation
        cache_position = cache_position if not is_cross_attention else None
        key_layer, value_layer = curr_past_key_value.update(
            key_layer, value_layer, self.layer_idx, {"cache_position": cache_position}
        )
        # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
        if is_cross_attention:
            past_key_value.is_updated[self.layer_idx] = True

    is_causal = self.is_decoder and not is_cross_attention and attention_mask is None and tgt_len > 1

    attention_mask = attention_mask.to(query_layer.dtype)
    softmax_algo = "None"
    if not self.training:
        softmax_algo = "fast"
    attn_output = FusedSDPA.apply(
        query_layer, key_layer, value_layer, attention_mask, 0.0, is_causal, None, softmax_algo, False
    )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)

    return attn_output, None
