from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.models.seamless_m4t.modeling_seamless_m4t import (
    SeamlessM4TForTextToSpeech,
    SeamlessM4TGenerationOutput,
    _compute_new_attention_mask,
    format_speech_generation_kwargs,
    shift_tokens_right,
)
from transformers.utils import logging

from ...modeling_attn_mask_utils import (
    _gaudi_prepare_4d_causal_attention_mask,
)


logger = logging.get_logger(__name__)


def gaudi_SeamlessM4TAttention_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    token_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Copied from SeamlessM4TAttention.forward: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
    The only differences are:
    - add token_idx args
    """

    # if encoder_hidden_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = encoder_hidden_states is not None

    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scaling
    # get key, value proj
    # `past_key_value[0].shape[2] == encoder_hidden_states.shape[1]`
    # is checking that the `sequence_length` of the `past_key_value` is the same as
    # the provided `encoder_hidden_states` to support prefix tuning
    if (
        is_cross_attention
        and past_key_value is not None
        and past_key_value[0].shape[2] == encoder_hidden_states.shape[1]
    ):
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = self._shape(self.k_proj(encoder_hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(encoder_hidden_states), -1, bsz)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if token_idx is None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            past_key_value[0].index_copy_(2, token_idx - 1, key_states)
            past_key_value[1].index_copy_(2, token_idx - 1, value_states)
            key_states = past_key_value[0]
            value_states = past_key_value[1]
    else:
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.reshape(*proj_shape)
    value_states = value_states.reshape(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if output_attentions:
        # this operation is a bit awkward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to be reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned across GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_reshaped, past_key_value


def gaudi_SeamlessM4TDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = True,
    token_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Copied from SeamlessM4TDecoderLayer.forward: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
    The only differences are:
    - add token_idx args
    """
    residual = hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)

    # Self Attention
    # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
    self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    # add present self-attn cache to positions 1,2 of present_key_value tuple
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        past_key_value=self_attn_past_key_value,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        token_idx=token_idx,
    )
    hidden_states = self.attn_dropout(hidden_states)
    hidden_states = residual + hidden_states

    # Cross-Attention Block
    cross_attn_present_key_value = None
    cross_attn_weights = None
    if encoder_hidden_states is not None:
        residual = hidden_states
        hidden_states = self.cross_attention_layer_norm(hidden_states)

        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

        hidden_states, cross_attn_weights, cross_attn_present_key_value = self.cross_attention(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            past_key_value=cross_attn_past_key_value,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states

        # add cross-attn to positions 3,4 of present_key_value tuple
        present_key_value += cross_attn_present_key_value

    # Fully Connected
    residual = hidden_states

    hidden_states = self.ffn_layer_norm(hidden_states)

    hidden_states = self.ffn(hidden_states)
    hidden_states = self.ffn_dropout(hidden_states)

    hidden_states = residual + hidden_states

    outputs = (hidden_states, present_key_value)

    if output_attentions:
        outputs += (self_attn_weights, cross_attn_weights)

    return outputs


def gaudi_SeamlessM4TDecoder_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    token_idx: Optional[torch.Tensor] = None,
) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
    """
    Copied from SeamlessM4TDecoder.forward: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
    The only differences are:
    - add token_idx args
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
        input = input_ids
        input_shape = input.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        input = inputs_embeds[:, :, -1]
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    # past_key_values_length
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    attention_mask = _gaudi_prepare_4d_causal_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    )

    # expand encoder attention mask
    if encoder_hidden_states is not None and encoder_attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        encoder_attention_mask = _prepare_4d_attention_mask(
            encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        )

    # embed positions
    if past_key_values_length != 0 and token_idx is not None:
        past_key_values_length = token_idx - 1
    positions = self.embed_positions(input, past_key_values_length=past_key_values_length)

    hidden_states = inputs_embeds + positions.to(inputs_embeds.device)

    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:
                continue

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                None,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                token_idx=token_idx,
            )
        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[1],)

        if output_attentions:
            all_self_attns += (layer_outputs[2],)

            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs[3],)

    hidden_states = self.layer_norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        cross_attentions=all_cross_attentions,
    )


def gaudi_SeamlessM4TTextToUnitModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.LongTensor] = None,
    encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    token_idx: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
    """
    Copied from SeamlessM4TTextToUnitModel.forward: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
    The only differences are:
    - add token_idx args
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if encoder_outputs is None:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )

    # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_outputs[0],
        encoder_attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        token_idx=token_idx,
    )

    if not return_dict:
        return decoder_outputs + encoder_outputs

    return Seq2SeqModelOutput(
        last_hidden_state=decoder_outputs.last_hidden_state,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )


def gaudi_SeamlessM4TTextToUnitForConditionalGeneration_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.LongTensor] = None,
    encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    token_idx: Optional[torch.Tensor] = None,
) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
    """
    Copied from SeamlessM4TTextToUnitForConditionalGeneration.forward: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
    The only differences are:
    - add token_idx args
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if labels is not None:
        if use_cache:
            logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
        use_cache = False
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.t2u_pad_token_id, self.config.t2u_decoder_start_token_id
            )

    outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs,
        decoder_attention_mask=decoder_attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        decoder_inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        token_idx=token_idx,
    )
    lm_logits = self.lm_head(outputs[0])

    masked_lm_loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        labels = labels.to(lm_logits.device)
        masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

    if not return_dict:
        output = (lm_logits,) + outputs[1:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    return Seq2SeqLMOutput(
        loss=masked_lm_loss,
        logits=lm_logits,
        past_key_values=outputs.past_key_values,
        decoder_hidden_states=outputs.decoder_hidden_states,
        decoder_attentions=outputs.decoder_attentions,
        cross_attentions=outputs.cross_attentions,
        encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        encoder_hidden_states=outputs.encoder_hidden_states,
        encoder_attentions=outputs.encoder_attentions,
    )


def gaudi_SeamlessM4TTextToUnitForConditionalGeneration_prepare_inputs_for_generation(
    self,
    decoder_input_ids,
    past_key_values=None,
    attention_mask=None,
    use_cache=None,
    encoder_outputs=None,
    **kwargs,
):
    """
    Copied from SeamlessM4TTextToUnitForConditionalGeneration.prepare_inputs_for_generation: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
    The only differences are:
    - add token_idx args
    """
    token_idx = kwargs.get("token_idx", None)
    # cut decoder_input_ids if past is used
    if past_key_values is not None:
        if token_idx is None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        else:
            decoder_input_ids = torch.index_select(decoder_input_ids, 1, token_idx - 1)

    return {
        "input_ids": None,  # encoder_outputs is defined. input_ids not needed
        "encoder_outputs": encoder_outputs,
        "past_key_values": past_key_values,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "use_cache": use_cache,
        "token_idx": token_idx,
        "decoder_attention_mask": kwargs.get("decoder_attention_mask", None),
    }


def gaudi_SeamlessM4TCodeHifiGan_get_output_hifigan_lengths(self, input_lengths: Union[torch.LongTensor, int]):
    """
    Copied from SeamlessM4TCodeHifiGan._get_output_hifigan_lengths: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
    The only differences are:
    - fix torch.div issue
    """

    def _conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        return (
            torch.div(input_length.item() + 2 * pad - dilation * (kernel_size - 1) - 1, stride, rounding_mode="floor")
            + 1
        )

    def _transpose_conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
        return (input_length - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + 1

    # conv_pre
    input_lengths = _conv_out_length(input_lengths, 7, 1, 3)

    # upsampler
    for i, (upsample_rate, kernel_size) in enumerate(
        zip(self.config.upsample_rates, self.config.upsample_kernel_sizes)
    ):
        input_lengths = _transpose_conv_out_length(
            input_lengths, kernel_size, upsample_rate, (kernel_size - upsample_rate) // 2
        )

    # resblock
    for i in range(len(self.config.upsample_rates)):
        for kernel_size, dilation in zip(self.config.resblock_kernel_sizes, self.config.resblock_dilation_sizes):
            for dil in dilation:
                input_lengths = _conv_out_length(
                    input_lengths, kernel_size, 1, (kernel_size - 1) * dil // 2, dilation=dil
                )

            for dil in dilation:
                input_lengths = _conv_out_length(input_lengths, kernel_size, 1, (kernel_size - 1) // 2, dilation=1)

    # conv_post
    input_lengths = _conv_out_length(input_lengths, 7, 1, 3)

    return input_lengths


def gaudi_SeamlessM4TForTextToSpeech_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.LongTensor] = None,
    encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    token_idx: Optional[torch.Tensor] = None,
) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
    """
    Copied from SeamlessM4TForTextToSpeech.forward: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
    The only differences are:
    - add token_idx args
    """
    if labels is not None:
        if use_cache:
            logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
        use_cache = False
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if encoder_outputs is None:
        # if encoder_outputs is not None, it's probably used within a .generate method so no need to warn
        logger.warning(
            "This is the same forward method as `SeamlessM4TForTextToText`."
            "It doesn't use the text-to-unit model `SeamlessM4TTextToUnitForConditionalGeneration`."
            "If you want to generate speech, use the `.generate` method."
        )
        encoder_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )

    encoder_attention_mask = attention_mask

    # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
    decoder_outputs = self.text_decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_outputs[0],
        encoder_attention_mask=encoder_attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        token_idx=token_idx,
    )

    lm_logits = self.lm_head(decoder_outputs[0])

    masked_lm_loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        labels = labels.to(lm_logits.device)
        masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

    if not return_dict:
        outputs = decoder_outputs + encoder_outputs
        output = (lm_logits,) + outputs[1:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    return Seq2SeqLMOutput(
        loss=masked_lm_loss,
        logits=lm_logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )


@torch.no_grad()
def gaudi_SeamlessM4TForTextToSpeech_generate(
    self,
    input_ids: Optional[torch.Tensor] = None,
    return_intermediate_token_ids: Optional[bool] = None,
    tgt_lang: Optional[str] = None,
    spkr_id: Optional[int] = 0,
    **kwargs,
) -> Union[torch.Tensor, SeamlessM4TGenerationOutput]:
    """
    Copied from SeamlessM4TForTextToSpeech.generate: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
    The only differences are:
    - delete pad id for unit_ids output
    """
    batch_size = len(input_ids) if input_ids is not None else len(kwargs.get("inputs_embeds"))

    if tgt_lang is None:
        raise ValueError("You must specify a `tgt_lang` to generate translated speech.")
    else:
        # also accept __xxx__
        tgt_lang = tgt_lang.replace("__", "")
        for key in ["text_decoder_lang_to_code_id", "t2u_lang_code_to_id", "vocoder_lang_code_to_id"]:
            lang_code_to_id = getattr(self.generation_config, key, None)
            if lang_code_to_id is None:
                raise ValueError(
                    f"""This model generation config doesn't have a `{key}` key which maps the target language
                    to the right token id. Make sure to load the right generation config."""
                )
            elif tgt_lang not in lang_code_to_id:
                raise ValueError(
                    f"""`tgt_lang={tgt_lang}` is not supported by this model.
                Please specify a `tgt_lang` in {",".join(lang_code_to_id.keys())}. Note that SeamlessM4T supports
                more languages for text translation than for speech synthesis."""
                )
    if kwargs.get("hpu_graphs", True):
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        if not hasattr(self, "clear_cache"):
            self = wrap_in_hpu_graph(self)
        if not hasattr(self.t2u_model, "clear_cache"):
            self.t2u_model = wrap_in_hpu_graph(self.t2u_model)
        if not hasattr(self.vocoder, "clear_cache"):
            self.vocoder = wrap_in_hpu_graph(self.vocoder)

    kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)
    kwargs_text["output_hidden_states"] = True
    kwargs_text["return_dict_in_generate"] = True
    kwargs_text["output_scores"] = True

    text_decoder_input_ids = kwargs_text.get("decoder_input_ids")

    # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
    text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
    text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size, device=self.device)

    kwargs_text["decoder_input_ids"] = text_decoder_input_ids

    # first generation
    text_generation_output = super(SeamlessM4TForTextToSpeech, self).generate(input_ids, **kwargs_text)
    sequences = text_generation_output.sequences

    # prepare second generation
    num_return_sequences = len(sequences) // batch_size
    attention_mask = kwargs_speech.get("attention_mask", kwargs_text.get("attention_mask", None))

    encoder_hidden_states = text_generation_output.encoder_hidden_states[-1]

    # take care of num_return_sequences
    # take most probable hidden states per batch of return_sequences
    # (batch_size*num_return_sequences, ...) -> (batch_size,...)
    if num_return_sequences > 1:
        idx_most_probable_sequences_per_batch = text_generation_output.sequences_scores.view(batch_size, -1)
        idx_most_probable_sequences_per_batch = idx_most_probable_sequences_per_batch.argmax(-1)
        idx_most_probable_sequences_per_batch = (
            idx_most_probable_sequences_per_batch + torch.arange(batch_size, device=self.device) * num_return_sequences
        )
        sequences = sequences[idx_most_probable_sequences_per_batch]

    # get decoder last hidden state - must do a pass through the text decoder
    t2u_input_embeds = self.text_decoder(
        input_ids=sequences,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=attention_mask,
    ).last_hidden_state

    pad_token_id = self.generation_config.pad_token_id

    # Compute new attention mask
    seq_lens = (sequences != pad_token_id).int().sum(1)
    t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
    kwargs_speech["attention_mask"] = t2u_model_attention_mask

    # Compute t2u decoder_input_ids
    t2u_decoder_input_ids = kwargs_speech.get("decoder_input_ids")
    t2u_tgt_lang_id = self.generation_config.t2u_lang_code_to_id.get(tgt_lang)
    t2u_decoder_input_ids = torch.tensor(
        [[self.config.t2u_eos_token_id, t2u_tgt_lang_id]] * batch_size, device=self.device
    )
    kwargs_speech["decoder_input_ids"] = t2u_decoder_input_ids

    # second generation
    unit_ids = self.t2u_model.generate(inputs_embeds=t2u_input_embeds, **kwargs_speech)
    seq_lens = (unit_ids != self.config.t2u_pad_token_id).int().sum(1)
    unit_ids = unit_ids[:, 0:seq_lens]
    output_unit_ids = unit_ids.detach().clone()

    # get rid of t2u_decoder_input_ids
    unit_ids = unit_ids[:, kwargs_speech["decoder_input_ids"].shape[1] :]
    # replace eos per pad
    unit_ids[unit_ids == self.config.t2u_eos_token_id] = self.config.t2u_pad_token_id
    # offset of control symbols
    unit_ids = torch.where(unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset)

    vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
    vocoder_tgt_lang_id = torch.tensor([[vocoder_tgt_lang_id]] * len(unit_ids), device=self.device)

    spkr_id = torch.tensor([[spkr_id]] * len(unit_ids), device=self.device)

    waveform, waveform_lengths = self.vocoder(input_ids=unit_ids, spkr_id=spkr_id, lang_id=vocoder_tgt_lang_id)

    if return_intermediate_token_ids:
        return SeamlessM4TGenerationOutput(
            waveform=waveform,
            waveform_lengths=waveform_lengths,
            sequences=sequences,
            unit_sequences=output_unit_ids,
        )

    return waveform, waveform_lengths


def gaudi_SeamlessM4TForTextToSpeech_prepare_inputs_for_generation(
    self,
    decoder_input_ids,
    past_key_values=None,
    attention_mask=None,
    use_cache=None,
    encoder_outputs=None,
    **kwargs,
):
    """
    Copied from SeamlessM4TForTextToSpeech.prepare_inputs_for_generation: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
    The only differences are:
    - add token_idx
    """
    token_idx = kwargs.get("token_idx", None)
    # cut decoder_input_ids if past is used
    if past_key_values is not None:
        if token_idx is None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        else:
            decoder_input_ids = torch.index_select(decoder_input_ids, 1, token_idx - 1)

    return {
        "input_ids": None,  # encoder_outputs is defined. input_ids not needed
        "encoder_outputs": encoder_outputs,
        "past_key_values": past_key_values,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "use_cache": use_cache,
        "token_idx": token_idx,
        "decoder_attention_mask": kwargs.get("decoder_attention_mask", None),
    }
