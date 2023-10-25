import torch


__package__ = "transformers.models.t5"

import warnings
from typing import Optional, Tuple, Union

import habana_frameworks.torch.core as htcore
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from transformers.models.t5.modeling_t5 import (
    T5DenseActDense,
    T5DenseGatedActDense,
    T5LayerCrossAttention,
    T5LayerFF,
    T5LayerSelfAttention,
    T5Stack,
)

from ...utils import (
    logging,
)


logger = logging.get_logger(__name__)
"""
Copied from T5Attention: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L452
https://github.com/huggingface/transformers/commit/04ab5605fbb4ef207b10bf2772d88c53fc242e83
There are following differences:
wrap `nn.functional.dropout` with `mark_step` for numerical improvement.
introduce static shapes to improve eval/pred performance on HPU
"""

try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm as FusedRMSNorm
except ImportError:
    print("Not using HPU fused kernel for RMSNorm")
    FusedRMSNorm = None


def gaudi_t5_layernorm_forward(self, hidden_states):
    """
    Copied from T5LayerNorm.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
    The only differences are:
        - override RMSNorm with Habana fused RMSNorm
    """
    if hidden_states.device.type == "hpu" and FusedRMSNorm:
        orig_dtype = hidden_states.dtype
        hidden_states = FusedRMSNorm.apply(hidden_states.float(), self.weight.float(), self.variance_epsilon)
        return hidden_states.to(orig_dtype)
    else:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


def gaudi_T5Attention_forward(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
    token_idx=None,
    max_output_length=0,
):
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]
    real_seq_length = seq_length

    if past_key_value is not None:
        assert (
            len(past_key_value) == 2
        ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
        if token_idx is None:
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length
        else:
            real_seq_length = max_output_length

    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states):
        """projection"""
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(hidden_states))
        elif past_key_value is None:
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(key_value_states))

        if past_key_value is not None:
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, key_length, dim_per_head)
                if token_idx is None:
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    hidden_states = past_key_value.index_copy_(-2, token_idx - 1, hidden_states)
            elif past_key_value.shape[2] != key_value_states.shape[1]:
                # checking that the `sequence_length` of the `past_key_value` is the same as
                # the provided `key_value_states` to support prefix tuning
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))
            else:
                # cross-attn
                hidden_states = past_key_value
        return hidden_states

    # get query states
    query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
    )
    value_states = project(
        hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
    )

    # compute scores
    scores = torch.matmul(
        query_states, key_states.transpose(3, 2)
    )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

    if position_bias is None:
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value is not None:
            if token_idx is None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]
            else:
                position_bias = position_bias.index_select(-2, token_idx - 1)

        if mask is not None:
            position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

    if self.pruned_heads:
        mask = torch.ones(position_bias.shape[1])
        mask[list(self.pruned_heads)] = 0
        position_bias_masked = position_bias[:, mask.bool()]
    else:
        position_bias_masked = position_bias

    scores += position_bias_masked
    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        scores
    )  # (batch_size, n_heads, seq_length, key_length)
    if self.training:
        htcore.mark_step()
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.dropout, training=self.training
    )  # (batch_size, n_heads, seq_length, key_length)
    if self.training:
        htcore.mark_step()

    # Mask heads if we want to
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs


class GaudiDropout(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        """
        Avoids dropout kernel fusion with others to stablize training numerical stability.
        """
        if self.dropout.training:
            htcore.mark_step()
        out = self.dropout(x)
        if self.dropout.training:
            htcore.mark_step()
        return out


class GaudiT5DenseActDense(T5DenseActDense):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = GaudiDropout(self.dropout)


class GaudiT5DenseGatedActDense(T5DenseGatedActDense):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = GaudiDropout(self.dropout)


class GaudiT5LayerFF(T5LayerFF):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = GaudiDropout(self.dropout)


class GaudiT5LayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.dropout = GaudiDropout(self.dropout)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        token_idx=None,
        max_output_length=0,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        # print(normed_hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            token_idx=token_idx,
            max_output_length=max_output_length,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class GaudiT5LayerCrossAttention(T5LayerCrossAttention):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = GaudiDropout(self.dropout)


class GaudiT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        self.dropout = GaudiDropout(self.dropout)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        token_idx=None,
        max_output_length=0,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        if token_idx is not None:
            mask_seq_length = max_output_length if past_key_values is not None else seq_length
        else:
            mask_seq_length = (
                past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
            )

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                # print(hidden_states)
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    token_idx=token_idx,
                    max_output_length=max_output_length,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


def gaudi_T5Block_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_bias=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    encoder_decoder_position_bias=None,
    layer_head_mask=None,
    cross_attn_layer_head_mask=None,
    past_key_value=None,
    use_cache=False,
    output_attentions=False,
    return_dict=True,
    token_idx=None,
    max_output_length=0,
):
    if past_key_value is not None:
        if not self.is_decoder:
            logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
        expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

        if len(past_key_value) != expected_num_past_key_values:
            raise ValueError(
                f"There should be {expected_num_past_key_values} past states. "
                f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                f"Got {len(past_key_value)} past key / value states"
            )

        self_attn_past_key_value = past_key_value[:2]
        cross_attn_past_key_value = past_key_value[2:]
    else:
        self_attn_past_key_value, cross_attn_past_key_value = None, None
    # print(hidden_states)
    self_attention_outputs = self.layer[0](
        hidden_states,
        attention_mask=attention_mask,
        position_bias=position_bias,
        layer_head_mask=layer_head_mask,
        past_key_value=self_attn_past_key_value,
        use_cache=use_cache,
        output_attentions=output_attentions,
        token_idx=token_idx,
        max_output_length=max_output_length,
    )
    hidden_states, present_key_value_state = self_attention_outputs[:2]
    attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

    # clamp inf values to enable fp16 training
    if hidden_states.dtype == torch.float16:
        clamp_value = torch.where(
            torch.isinf(hidden_states).any(),
            torch.finfo(hidden_states.dtype).max - 1000,
            torch.finfo(hidden_states.dtype).max,
        )
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    do_cross_attention = self.is_decoder and encoder_hidden_states is not None
    if do_cross_attention:
        # the actual query length is unknown for cross attention
        # if using past key value states. Need to inject it here
        if present_key_value_state is not None:
            query_length = present_key_value_state[0].shape[2]
        else:
            query_length = None

        cross_attention_outputs = self.layer[1](
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            position_bias=encoder_decoder_position_bias,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value,
            query_length=query_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = cross_attention_outputs[0]

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # Combine self attn and cross attn key value states
        if present_key_value_state is not None:
            present_key_value_state = present_key_value_state + cross_attention_outputs[1]

        # Keep cross-attention outputs and relative position weights
        attention_outputs = attention_outputs + cross_attention_outputs[2:]

    # Apply Feed Forward layer
    hidden_states = self.layer[-1](hidden_states)

    # clamp inf values to enable fp16 training
    if hidden_states.dtype == torch.float16:
        clamp_value = torch.where(
            torch.isinf(hidden_states).any(),
            torch.finfo(hidden_states.dtype).max - 1000,
            torch.finfo(hidden_states.dtype).max,
        )
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    outputs = (hidden_states,)
    if use_cache:
        outputs = outputs + (present_key_value_state,) + attention_outputs
    else:
        outputs = outputs + attention_outputs

    return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


def gaudi_T5ForConditionalGeneration_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.BoolTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    decoder_head_mask: Optional[torch.FloatTensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    token_idx: Optional[torch.LongTensor] = None,
    max_output_length=0,
) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
        config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
        labels in `[0, ..., config.vocab_size]`

    Returns:

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
    >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

    >>> # training
    >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
    >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
    >>> outputs = model(input_ids=input_ids, labels=labels)
    >>> loss = outputs.loss
    >>> logits = outputs.logits

    >>> # inference
    >>> input_ids = tokenizer(
    ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
    ... ).input_ids  # Batch size 1
    >>> outputs = model.generate(input_ids)
    >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    >>> # studies have shown that owning a dog is good for you.
    ```"""
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
    if head_mask is not None and decoder_head_mask is None:
        if self.config.num_layers == self.config.num_decoder_layers:
            warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
            decoder_head_mask = head_mask

    # Encode if needed (training, first prediction pass)
    if encoder_outputs is None:
        # Convert encoder inputs in embeddings if needed
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )

    hidden_states = encoder_outputs[0]

    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)

    if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
        # get decoder inputs from shifting lm labels to the right
        decoder_input_ids = self._shift_right(labels)

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)
        hidden_states = hidden_states.to(self.decoder.first_device)
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.decoder.first_device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        past_key_values=past_key_values,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=attention_mask,
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        token_idx=token_idx,
        max_output_length=max_output_length,
    )

    sequence_output = decoder_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.encoder.first_device)
        self.lm_head = self.lm_head.to(self.encoder.first_device)
        sequence_output = sequence_output.to(self.lm_head.weight.device)

    if self.config.tie_word_embeddings:
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim**-0.5)

    lm_logits = self.lm_head(sequence_output)

    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        # move labels to correct device to enable PP
        labels = labels.to(lm_logits.device)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

    if not return_dict:
        output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        return ((loss,) + output) if loss is not None else output

    return Seq2SeqLMOutput(
        loss=loss,
        logits=lm_logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )


def gaudi_T5ForConditionalGeneration_prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
    use_cache=None,
    encoder_outputs=None,
    token_idx=None,
    decoder_attention_mask=None,
    max_output_length=0,
    **kwargs,
):
    # cut decoder_input_ids if past is used
    if past_key_values is not None:
        if token_idx is not None:
            input_ids = torch.index_select(input_ids, 1, token_idx - 1)
        else:
            input_ids = input_ids[:, -1:].unsqueeze(-1)

    return {
        "decoder_input_ids": input_ids,
        "past_key_values": past_key_values,
        "encoder_outputs": encoder_outputs,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
        "use_cache": use_cache,
        "token_idx": token_idx,
        "max_output_length": max_output_length,
    }


def gaudi_T5ForConditionalGeneration_reorder_cache(self, past_key_values, beam_idx):
    # if decoder past is not included in output
    # speedy decoding is disabled and no need to reorder
    if past_key_values is None:
        logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
        return past_key_values
    reordered_decoder_past = ()
    for layer_past_states in past_key_values:
        # get the correct batch idx from layer past batch dim
        # batch dim of `past` is at 2nd position
        reordered_layer_past_states = ()
        for layer_past_state in layer_past_states:
            # need to set correct `past` for each of the four key / value states
            reordered_layer_past_states = reordered_layer_past_states + (
                layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
            )

        assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
        assert len(reordered_layer_past_states) == len(layer_past_states)
        reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
    return reordered_decoder_past


def _gaudi_get_resized_embeddings(
    self, old_embeddings: torch.nn.Embedding, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None
) -> torch.nn.Embedding:
    """
    Copied from: https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/modeling_utils.py#L1424
    """
    if new_num_tokens is None:
        return old_embeddings

    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    else:
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

    if old_num_tokens == new_num_tokens:
        return old_embeddings

    if not isinstance(old_embeddings, torch.nn.Embedding):
        raise TypeError(
            f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {torch.nn.Embedding}. You"
            " should either use a different resize function or make sure that `old_embeddings` are an instance of"
            f" {torch.nn.Embedding}."
        )

    # Build new embeddings
    new_embeddings = torch.nn.Embedding(new_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

    # initialize all new embeddings (in particular added tokens)
    self._init_weights(new_embeddings)

    # Copy token embeddings from the previous weights

    # numbers of tokens to copy
    n = min(old_num_tokens, new_num_tokens)
    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
    else:
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

    return new_embeddings


def _gaudi_get_resized_lm_head(
    self, old_lm_head: torch.nn.Linear, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False
) -> torch.nn.Linear:
    """
    Copied from: https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/modeling_utils.py#L1488
    """
    if new_num_tokens is None:
        return old_lm_head

    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(old_lm_head.weight, modifier_rank=None):
            old_num_tokens, old_lm_head_dim = (
                old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
            )
    else:
        old_num_tokens, old_lm_head_dim = (
            old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
        )

    if old_num_tokens == new_num_tokens:
        return old_lm_head

    if not isinstance(old_lm_head, torch.nn.Linear):
        raise TypeError(
            f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {torch.nn.Linear}. You"
            " should either use a different resize function or make sure that `old_lm_head` are an instance of"
            f" {torch.nn.Linear}."
        )

    # Build new lm head
    new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
    has_new_lm_head_bias = old_lm_head.bias is not None
    new_lm_head = torch.nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias)
    new_lm_head = new_lm_head.to(old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)

    # initialize new lm head (in particular added tokens)
    self._init_weights(new_lm_head)

    num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

    # XXX: put the long block of code in a wrapper
    if is_deepspeed_zero3_enabled():
        import deepspeed

        params = [old_lm_head.weight, old_lm_head.bias, new_lm_head.weight, new_lm_head.bias]
        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                # Copy old lm head weights to new lm head
                if not transposed:
                    new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
                else:
                    new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

                # Copy bias weights to new lm head
                if has_new_lm_head_bias:
                    new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]
    else:
        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

    return new_lm_head