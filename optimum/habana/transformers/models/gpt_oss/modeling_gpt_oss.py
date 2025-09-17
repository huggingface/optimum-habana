from functools import partial
from typing import Callable, Optional, Union

import habana_frameworks.torch.core as htcore
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssExperts,
    GptOssForCausalLM,
    GptOssModel,
    apply_rotary_pos_emb,
    load_balancing_loss_func,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from ...modeling_attn_mask_utils import (
    _gaudi_prepare_4d_causal_attention_mask,
)
from ...modeling_rope_utils import GaudiRotaryEmbedding
from ..modeling_all_models import KVCache, Matmul, apply_customized_rope_module


try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE  # noqa

    has_fused_rope = True
except ImportError:
    has_fused_rope = False
    print("Not using HPU fused kernel for apply_rotary_pos_emb")

try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm
except ImportError:
    print("Not using HPU fused kernel for RMSNorm")
    FusedRMSNorm = None


class GaudiGptOssRotaryEmbedding(GaudiRotaryEmbedding):
    def __init__(self, config: GptOssConfig):
        config.rope_scaling = config.rope_scaling if hasattr(config, "rope_scaling") else None
        super().__init__(config=config)


def gaudi_gpt_oss_rmsnorm_forward(self, hidden_states):
    if hidden_states.device.type == "hpu" and FusedRMSNorm is not None:
        hidden_states = FusedRMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)
        return hidden_states
    else:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def gaudi_repeat_kv(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    n_rep: int,
) -> torch.Tensor:
    """
    Copied from gaudi_llama_repeat_kv: https://github.com/huggingface/optimum-habana/blob/2e8f7724a1974af32a42baf091f82ac4ae88a4bf/optimum/habana/transformers/models/llama/modeling_llama.py#L240
    """
    batch, num_key_value_heads, slen, head_dim = key_states.shape
    if n_rep == 1 or num_key_value_heads == 1:
        return query_states, key_states, value_states, attention_mask

    new_kv_shape = (batch, num_key_value_heads, 1, slen, head_dim)
    key_states = key_states.reshape(new_kv_shape)
    value_states = value_states.reshape(new_kv_shape)

    batch, _, q_len, head_dim = query_states.shape
    new_q_shape = (batch, num_key_value_heads, n_rep, q_len, head_dim)
    query_states = query_states.reshape(new_q_shape)

    if attention_mask is not None:
        # Add groups dim and set to 1
        attention_mask = attention_mask.unsqueeze(1)

    return query_states, key_states, value_states, attention_mask


def apply_customized_rope(q, k, cos, sin, position_ids, training=True):
    if q.device.type == "hpu" and has_fused_rope:
        return apply_customized_rope_module(q, k, cos, sin, position_ids, training)
    else:
        return apply_rotary_pos_emb(q, k, cos[position_ids], sin[position_ids])


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    token_idx: Optional[torch.Tensor] = None,
    s_aux: Optional[torch.Tensor] = None,
    **kwargs,
):
    query_states, key_states, value_states, attention_mask = gaudi_repeat_kv(
        query, key, value, attention_mask, module.num_key_value_groups
    )
    attn_weights = module.matmul_qk(query_states, key_states.transpose(-2, -1)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    sinks = s_aux.reshape(1, query_states.shape[1], query_states.shape[2], 1, 1).expand(
        query_states.shape[0], -1, -1, query_states.shape[-2], -1
    )

    if token_idx is not None:
        combined_logits = attn_weights.clone()
        combined_logits = combined_logits.index_copy_(-1, token_idx, sinks)
    else:
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)

    if token_idx is not None:
        # index_copy_() was used to avoid dynamicity in probs[..., token_idx]
        zeros = torch.zeros(probs.shape[:-1] + (1,), dtype=probs.dtype, device=probs.device)
        probs.index_copy_(-1, token_idx, zeros)
        scores = probs
    else:
        scores = probs[..., :-1]

    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)

    attn_output = module.matmul_av(attn_weights, value_states)
    return attn_output, attn_weights


class GaudiGptOssExperts(GptOssExperts):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.matmul_gate_up = Matmul()
        self.matmul_down = Matmul()

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None):
        batch_size = hidden_states.shape[0]

        # TODO: find a way to split parameters in experts; DeepSpeed currently can't split BMMs.
        # The original hidden_size is used here, since DeepSpeed updates it to hidden_size/num_worker even without splitting.
        self.hidden_size = self.config.hidden_size

        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]
        if hidden_states.device.type == "cpu" or self.training:
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                # we sum on the top_k and on the sequence length to get which experts
                # are hit this time around
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit[:]:
                # expert_idx only have 1 element, so we can use scale for fast indexing
                expert_idx = expert_idx[0]
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu
                out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
                weighted_output = out * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(num_experts, 1)
            hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
            gate_up = self.matmul_gate_up(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            next_states = self.matmul_down(((up + 1) * glu), self.down_proj)
            next_states = next_states + self.down_proj_bias[..., None, :]
            next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
            next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
            next_states = next_states.sum(dim=0)
        return next_states


class GaudiGptOssAttention(GptOssAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.matmul_qk = Matmul()
        self.matmul_av = Matmul()
        self.k_cache = KVCache()
        self.v_cache = KVCache()
        self.inp_seq_len = -1

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        q_len = input_shape[1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        query_states, key_states = apply_customized_rope(
            query_states, key_states, cos, sin, position_ids, self.training
        )

        if use_cache:
            if past_key_values is None:
                past_key = torch.zeros(key_states.shape, dtype=self.k_proj.weight.dtype, device=key_states.device)
                past_value = torch.zeros(key_states.shape, dtype=self.k_proj.weight.dtype, device=key_states.device)
                past_key_values = (past_key, past_value)
                key_states = self.k_cache.update(past_key_values[0], key_states, 2, token_idx, key_states.shape[-2])
                value_states = self.v_cache.update(
                    past_key_values[1], value_states, 2, token_idx, value_states.shape[-2]
                )
            else:
                key_states = self.k_cache.update(past_key_values[0], key_states, 2, token_idx, self.inp_seq_len)
                value_states = self.v_cache.update(past_key_values[1], value_states, 2, token_idx, self.inp_seq_len)
            if token_idx is None:
                past_key_values = (key_states, value_states)

        else:
            past_key_values = None

        # TODO: Habana fused SDPA with sink is enabeld in 1.23.0. Update attention after 1.23.0 release
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.world_size > 1:
            local_sink = torch.chunk(self.sinks, self.world_size)[self.rank]
        else:
            local_sink = self.sinks
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=local_sink,
            token_idx=token_idx,
            **kwargs,
        )

        attn_output = attn_output.reshape(input_shape[0], -1, q_len, self.head_dim)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(*input_shape, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_values


def gaudi_gpt_oss_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    token_idx: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    hidden_states, _, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
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
    hidden_states, _ = self.mlp(hidden_states)  # diff with llama: router scores
    hidden_states = residual + hidden_states

    if use_cache:
        return (hidden_states, present_key_value)
    else:
        return hidden_states


class GaudiGptOssModel(GptOssModel):
    def __init__(self, config: GptOssConfig):
        super().__init__(config)
        self.rotary_emb = GaudiGptOssRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        lazy_mode: Optional[bool] = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if past_key_values is not None and use_cache:  # kept for BC (cache positions)
            past_seen_tokens = past_key_values[0][0].shape[2]

        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens, seq_length + past_seen_tokens, dtype=torch.long, device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0)
        cache_position = None

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            causal_mask_mapping = {
                "full_attention": _gaudi_prepare_4d_causal_attention_mask(
                    attention_mask,
                    input_ids.shape if input_ids is not None else (batch_size, seq_length),
                    inputs_embeds,
                    past_seen_tokens,
                ),
                "sliding_attention": _gaudi_prepare_4d_causal_attention_mask(
                    attention_mask,
                    input_ids.shape if input_ids is not None else (batch_size, seq_length),
                    inputs_embeds,
                    past_seen_tokens,
                    self.config.sliding_window,
                ),
            }

        hidden_states = inputs_embeds
        kv_seq_len = hidden_states.shape[-2]
        if past_key_values is not None:
            if token_idx is not None:
                kv_seq_len = past_key_values[0][0].shape[-2]
            else:
                kv_seq_len += past_key_values[0][0].shape[-2]

        position_embeddings = self.rotary_emb(hidden_states, seq_len=kv_seq_len)

        if use_cache:
            next_decoder_cache = ()
        else:
            next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if (
                lazy_mode
                and not self.training
                and (torch.distributed.is_initialized() is False or torch.distributed.get_world_size() == 1)
            ):
                htcore.mark_step()

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **kwargs),
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    token_idx=token_idx,
                    **kwargs,
                )
            else:
                past_key_value = None if past_key_values is None else past_key_values[layer_idx]
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_value,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    token_idx=token_idx,
                    **kwargs,
                )

            # layer_outputs is a tuple of (hidden_states, past_key_value)
            if use_cache:
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,  # kv cache for all layers
        )


class GaudiGptOssForCausalLM(GptOssForCausalLM):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        token_idx: Optional[torch.Tensor] = None,
        lazy_mode: Optional[bool] = True,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            token_idx=token_idx,
            lazy_mode=lazy_mode,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        output_router_logits=False,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        token_idx=None,
        **kwargs,
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if token_idx is not None:
                idx = token_idx + kwargs.get("inputs_embeds_offset", 0) - 1
                input_ids = torch.index_select(input_ids, 1, idx)
            else:
                if inputs_embeds is not None:  # Exception 1
                    input_ids = input_ids[:, -cache_position.shape[0] :]
                elif (
                    input_ids.shape[1] != cache_position.shape[0]
                ):  # Default case (the "else", a no op, is Exception 2)
                    input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                if token_idx is not None:
                    position_ids = torch.index_select(position_ids, 1, token_idx - 1)
                else:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

        cache_position = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": None,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "output_router_logits": output_router_logits,
                "token_idx": token_idx,
                "lazy_mode": kwargs.get("lazy_mode"),
            }
        )
        return model_inputs
