import math
from typing import Optional, Tuple, Union

import torch

from ....utils import get_device_name


try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused kernel for scaled_dot_product_attention")
    FusedSDPA = None

# TODO: remove this workaround when FusedRoPE properly works on Gaudi
if get_device_name() == "gaudi2":
    try:
        from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV1 as FusedRoPE
    except ImportError:
        print("Not using HPU fused kernel for apply_rotary_pos_emb")
        FusedRoPE = None
else:
    FusedRoPE = None

import habana_frameworks.torch.core as htcore
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.falcon.modeling_falcon import (
    FalconForCausalLM,
    FalconModel,
    build_alibi_tensor,
    dropout_add,
    rotate_half,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)


def gaudi_falcon_rotary_embedding_forward(self, query, key, seq_len, position_ids, past_key_values_length=0):
    """
    Copied from FalconRotaryEmbedding.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
    The only differences are:
    - add new args position_ids
    - use Habana optimized RotaryPosEmbedding op
    """
    cos, sin = self.cos_sin(seq_len, past_key_values_length, position_ids, query.device, query.dtype)

    if FusedRoPE:
        return FusedRoPE.apply(query, cos, sin, 0), FusedRoPE.apply(key, cos, sin, 0)
    else:
        return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


def gaudi_falcon_attention_forward(
    self,
    hidden_states: torch.Tensor,
    alibi: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    token_idx: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
):
    """
    Copied from FalconAttention.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
    The only differences are:
    - add new args token_idx and position_ids
    - replace F.scaled_dot_product_attention with Habana torch's version
    """
    fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

    batch_size, query_length, _, _ = query_layer.shape

    query_layer = query_layer.transpose(1, 2).reshape(-1, query_length, self.head_dim)
    key_layer = key_layer.transpose(1, 2).reshape(-1, query_length, self.head_dim)
    value_layer = value_layer.transpose(1, 2).reshape(-1, query_length, self.head_dim)

    past_kv_length = 0
    seq_len = query_layer.shape[1]
    if layer_past is not None:
        if token_idx is not None:
            # When token_idx is used,
            # past_kv_length = 0
            # static seq len = (input token len + max output token len)
            seq_len = layer_past[0].shape[1]
        else:
            past_kv_length = layer_past[0].shape[1]

    query_layer, key_layer = self.maybe_rotary(query_layer, key_layer, seq_len, position_ids, past_kv_length)

    if layer_past is not None:
        past_key, past_value = layer_past
        if token_idx is not None:
            past_key.index_copy_(1, token_idx - 1, key_layer)
            past_value.index_copy_(1, token_idx - 1, value_layer)
            key_layer = past_key
            value_layer = past_value
        else:
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, kv_length, head_dim]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)

    _, kv_length, _ = key_layer.shape
    if use_cache:
        present = (key_layer, value_layer)
    else:
        present = None

    attention_mask_float = (attention_mask * 1.0).masked_fill(attention_mask, float("-1e9")).to(query_layer.dtype)

    query_layer_ = query_layer.reshape(batch_size, -1, query_length, self.head_dim)
    key_layer_ = key_layer.reshape(batch_size, -1, seq_len, self.head_dim)
    value_layer_ = value_layer.reshape(batch_size, -1, seq_len, self.head_dim)

    if alibi is None:
        if output_attentions:
            attention_scores = query_layer_ @ key_layer_.transpose(-1, -2)
            attention_scores /= math.sqrt(self.head_dim)

            attention_scores = F.softmax(attention_scores + attention_mask_float, dim=-1, dtype=hidden_states.dtype)
            attn_output = attention_scores @ value_layer_
        else:
            if FusedSDPA:
                attn_output = FusedSDPA.apply(query_layer_, key_layer_, value_layer_, attention_mask_float, 0.0, False)
            else:
                # Workaround util scaled_dot_product_attention support broadcast.
                if self.training is True and query_layer_.shape != key_layer_.shape:
                    key_layer_ = torch.broadcast_to(key_layer_, query_layer_.shape)
                    value_layer_ = torch.broadcast_to(value_layer_, query_layer_.shape)
                attn_output = F.scaled_dot_product_attention(
                    query_layer_, key_layer_, value_layer_, attention_mask_float, 0.0, is_causal=False
                )
            # Performance improvement for HPU
            if self.training is True and htcore:
                htcore.mark_step()
            attention_scores = None

        attn_output = attn_output.view(batch_size, -1, query_length, self.head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, query_length, -1)

        output_tensor = self.dense(attn_output)

        if output_attentions:
            return output_tensor, present, attention_scores
        else:
            return output_tensor, present

    else:
        matmul_result = query_layer_ @ key_layer_.transpose(-1, -2)

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(batch_size, self.num_heads, query_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
            attention_scores = attention_scores.to(torch.float32)
        # Matt (HF) note: We could possibly use F.scaled_dot_product_attention here too, by
        # adding (alibi * self.inv_norm_factor) to attention_mask_float. I think this would be mathematically
        # equivalent and more performant, but there might be a numerical difference. If you're reading this
        # and you'd like to experiment and maybe file a PR, feel free!
        attention_logits = attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)
        attention_logits *= self.inv_norm_factor
        attention_probs = F.softmax(attention_logits + attention_mask_float, dim=-1, dtype=hidden_states.dtype)
        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size, num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size, self.num_heads, query_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = (attention_probs_reshaped @ value_layer_).flatten(0, 1)

        # change view [batch_size, q_length, num_heads * head_dim]
        context_layer = self._merge_heads(context_layer)

        output_tensor = self.dense(context_layer)

        if output_attentions:
            return output_tensor, present, attention_probs
        else:
            return output_tensor, present


def gaudi_falcon_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    alibi: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    token_idx: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
):
    """
    Copied from FalconDecoderLayer.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
    The only differences are:
    - add new args token_idx and position_ids
    - add token_idx and position_ids into attention inputs
    """
    residual = hidden_states

    if self.config.new_decoder_architecture:
        attention_layernorm_out = self.ln_attn(hidden_states)
        mlp_layernorm_out = self.ln_mlp(hidden_states)
    else:
        attention_layernorm_out = self.input_layernorm(hidden_states)

    # Self attention.
    attn_outputs = self.self_attention(
        attention_layernorm_out,
        layer_past=layer_past,
        attention_mask=attention_mask,
        alibi=alibi,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        token_idx=token_idx,
        position_ids=position_ids,
    )

    attention_output = attn_outputs[0]

    if not self.config.new_decoder_architecture:
        if self.config.parallel_attn:
            mlp_layernorm_out = attention_layernorm_out
        else:
            residual = dropout_add(attention_output, residual, self.config.attention_dropout, training=self.training)
            mlp_layernorm_out = self.post_attention_layernorm(residual)

    outputs = attn_outputs[1:]

    # MLP.
    mlp_output = self.mlp(mlp_layernorm_out)

    if self.config.new_decoder_architecture or self.config.parallel_attn:
        mlp_output += attention_output

    output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)

    if use_cache:
        outputs = (output,) + outputs
    else:
        outputs = (output,) + outputs[1:]

    return outputs  # hidden_states, present, attentions


def gaudi_falcon_attention_split_heads(
    self, fused_qkv: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Copied from FalconAttention._split_heads https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/falcon/modeling_falcon.py
    Changing index operation of qkv[:::] to use torch.index_select to work around gradient accuracy issue and improve performance.
    """
    if self.new_decoder_architecture:
        batch, seq_len, _ = fused_qkv.shape
        qkv = fused_qkv.view(batch, seq_len, self.num_kv_heads, -1, self.head_dim)
        # query = qkv[:, :, :, :-2]
        # key = qkv[:, :, :, [-2]]
        # value = qkv[:, :, :, [-1]]
        d3 = qkv.shape[3] - 2
        query = torch.index_select(qkv, 3, index=torch.arange(d3, device=qkv.device))
        key = torch.index_select(qkv, 3, index=torch.tensor([d3], device=qkv.device))
        value = torch.index_select(qkv, 3, index=torch.tensor([d3 + 1], device=qkv.device))

        key = torch.broadcast_to(key, query.shape)
        value = torch.broadcast_to(value, query.shape)

        query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
        return query, key, value
    elif not self.multi_query:
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        # TODO : Need to be fixed to use index_select()
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
    else:
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads + 2, self.head_dim)
        # return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]
        d2 = fused_qkv.shape[2] - 2
        query = torch.index_select(fused_qkv, 2, index=torch.arange(d2, device=fused_qkv.device))
        key = torch.index_select(fused_qkv, 2, index=torch.tensor([d2], device=fused_qkv.device))
        value = torch.index_select(fused_qkv, 2, index=torch.tensor([d2 + 1], device=fused_qkv.device))
        return query, key, value


def _expand_mask(mask: torch.Tensor, past_key_values_length: int, tgt_len: int) -> torch.BoolTensor:
    """
    Copied from transformers.models.falcon.modeling_falcon._expand_mask
    Expands attention_mask from `[batch_size, seq_length]` to `[batch_size, 1, seq_length, seq_length + past_length]`
    when past_key_values_length is not 0 or to `[batch_size, 1, seq_length, tgt_len] when past_key_values_length is 0.`
    """
    batch_size, total_length = mask.shape
    if tgt_len > 0:
        seq_length = tgt_len
    else:
        seq_length = total_length - past_key_values_length if past_key_values_length is not None else total_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, seq_length, total_length)


def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    batch_size, target_length = input_ids_shape

    mask = torch.empty((target_length, target_length + past_key_values_length), dtype=torch.bool, device=device)

    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


class GaudiFalconModel(FalconModel):
    """
    Inherits from FalconModel: https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
    The only differences are:
    - add new args token_idx and position_ids
    - add token_idx and position_ids into decoder inputs
    - set past_key_values_length=0 when token_idx is used (with static input shape)
    - add new arg tgt_len to _expand_mask because past_key_values_length is no longer valid with token_idx
    - use old version of _make_causal_mask to workaround toch.triu that is not supported in Synapse
    """

    def _prepare_attn_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
        # Create a causal mask
        # The attention mask we receive as input should cover the whole extended sequence, including any past
        # cache, so its shape should be [batch_size, seq_length + past_key_values_length]
        # The output shape will be [batch_size, 1, seq_length, seq_length + past_key_values_length]
        if past_key_values_length > 0:
            if input_shape[1] + past_key_values_length != attention_mask.shape[1]:
                raise ValueError(
                    "Attention mask shape should be (batch_size, seq_length + past_key_values_length)"
                    f" but is {attention_mask.shape} with input_ids shape {input_shape} and past length"
                    f" {past_key_values_length}."
                )

        combined_attention_mask = None
        device = attention_mask.device
        _, seq_length = input_shape

        if seq_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )

        # [batch_size, seq_length + past_key_values_length] -> [batch_size, 1, seq_length, seq_length + past_key_values_length]
        expanded_attn_mask = _expand_mask(
            attention_mask, past_key_values_length=past_key_values_length, tgt_len=seq_length
        )

        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))
        else:
            past_key_values = self._convert_to_rw_cache(past_key_values)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = inputs_embeds

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        past_key_values_length = 0
        if past_key_values[0] is not None and token_idx is None:
            past_key_values_length = past_key_values[0][0].shape[1]  # 1 because RW-cache, not standard format

        if position_ids is None:
            if token_idx is not None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        if self.use_alibi:
            alibi = build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)
        else:
            alibi = None

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    alibi,
                    causal_mask,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                    token_idx=token_idx,
                    position_ids=position_ids,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if presents is not None:
            presents = self._convert_cache_to_standard_format(presents, batch_size)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class GaudiFalconForCausalLM(FalconForCausalLM):
    """
    Inherits from FalconForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
    The only differences are:
    - add new args token_idx and position_ids
    - add token_idx and position_ids into model inputs
    - from step2 when enable KV cache, slice next_input_ids from input_ids base on the token_idx
    - from step2 when enable KV cache, slice next_position_ids from position_ids base on the token_idx
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_idx=token_idx,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        if past_key_values is not None:
            if token_idx is not None:
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)
            else:
                input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None and token_idx is not None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                if token_idx is not None:
                    position_ids = torch.index_select(position_ids, 1, token_idx - 1)
                else:
                    position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "token_idx": token_idx,
            "position_ids": position_ids,
        }
