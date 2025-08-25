from functools import wraps
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE  # noqa
    has_fused_rope = True
except ImportError:
    has_fused_rope = False
    print("Not using HPU fused kernel for apply_rotary_pos_emb")

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused kernel for scaled_dot_product_attention")
    FusedSDPA = None

import habana_frameworks.torch.core as htcore
from ..modeling_all_models import Matmul, apply_customized_rope_module
from .configuration_gpt_oss import GptOssConfig

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations.hub_kernels import use_kernel_forward_from_hub
from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
#from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack

from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from typing import Any, Callable, ContextManager, Optional, TypedDict
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssForCausalLM,
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssMLP,
    GptOssModel,
    GptOssExperts,
    GptOssRMSNorm,
    apply_rotary_pos_emb,
)
from ..modeling_all_models import KVCache
from ...modeling_attn_mask_utils import (
    _gaudi_prepare_4d_causal_attention_mask,
)
from ...modeling_rope_utils import GaudiRotaryEmbedding

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

def gaudi_repeat_kv(query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, attention_mask: torch.Tensor, n_rep: int) -> torch.Tensor:
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
    **kwargs,
):

    #Q 8,64,283,64 [bs, #head, seq len, head dim]
    #K 8,64,283,64 [bs, #head, seq len, head dim]
    #key_states = repeat_kv(key, module.num_key_value_groups)
    #value_states = repeat_kv(value, module.num_key_value_groups)
    query_states, key_states, value_states, attention_mask = gaudi_repeat_kv(query, key, value, attention_mask, module.num_key_value_groups)
    #attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    #query_states (bs, # kv head, # group, seq_len, head_dim)
    #key_states (bs, # kv head, 1, slen, head_dim)
    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scaling
    #batch_size, num_kv_heads, num_kv_groups, q_len, kv len
    
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    
    #sinks = module.sinks.reshape(1, -1, 1, 1).expand(query_states.shape[0], -1, query.shape[.-2], -1)
    sinks = module.sinks.reshape(1, query_states.shape[1], query_states.shape[2], 1, 1).expand(query_states.shape[0], -1, -1, query_states.shape[-2], -1)
    
    #if token_idx == attn_weights.shape[-1]-1:
    #    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    #else:
    if token_idx is not None:
        combined_logits = attn_weights.clone()
        combined_logits = combined_logits.index_copy_(-1, token_idx, sinks)#+1, sinks)
    else:
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    
    #scores = probs[..., :-1]  # we drop the sink here 
    ####how do we handle the sink drop with padded inputs??????
    ####drops the sink at token_idx+1
    
    #if token_idx == attn_weights.shape[-1]-1:
    #    scores = probs[..., :-1]
    #else:
    if token_idx is not None:
        probs[..., token_idx]=0#+1] = 0
        scores = probs
    else:
        scores = probs[..., :-1]
    
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value_states)
    #attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class GaudiGptOssAttention(GptOssAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.k_cache = KVCache()
        self.v_cache = KVCache()
        self.inp_seq_len = -1

    #def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
    #    cache_shape = (batch_size, self.num_key_value_groups, max_seq_len, self.head_dim)
    #    device = self.k_proj.weight.device
    #    dtype = self.config.torch_dtype
    #    self.k_cache.allocate(inp_seq_len, dtype, device, cache_shape)
    #    self.v_cache.allocate(inp_seq_len, dtype, device, cache_shape)

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
        #reuse_cache: Optional[bool] = False,
        #cache_idx: int = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        q_len = input_shape[1]
        hidden_shape = (*input_shape, -1, self.head_dim)#(8,283, -1, 64) 

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        query_states, key_states = apply_customized_rope(
            query_states, key_states, cos, sin, position_ids, self.training)
        #query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        """
        kv_seq_len = key_states.shape[-2]
        if past_key_values is not None:
            #cache_kwargs = {"cache_position": cache_position}
            #key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            if token_idx is None:
                if hasattr(past_key_values, "get_usable_length"):
                    kv_seq_len += past_key_values.get_usable_length(kv_seq_len, self.layer_idx)
                else:
                    kv_seq_len += past_key_values[0].shape[-2]
            else:
                if reuse_cache:
                    kv_seq_len = past_key_values[0][-2]
                else:
                    kv_seq_len = past_key_values[0].shape[-2]
        """
        if use_cache:
            #if reuse_cache:
            #    key_states = self.k_cache(key_states, 2, token_idx)
            #    value_states = self.v_cache(value_states, 2, token_idx)
            #    past_key_values = (self.k_cache.get_shape(), self.v_cache.get_shape())
            #else:
            if past_key_values is None:
                past_key = torch.zeros(key_states.shape, dtype=self.k_proj.weight.dtype, device=key_states.device)
                past_value = torch.zeros(
                    key_states.shape, dtype=self.k_proj.weight.dtype, device=key_states.device
                )
                past_key_values = (past_key, past_value)
                key_states = self.k_cache.update(past_key_values[0], key_states, 2, token_idx, key_states.shape[-2])
                value_states = self.v_cache.update(past_key_values[1], value_states, 2, token_idx, value_states.shape[-2])
            else:
                key_states = self.k_cache.update(past_key_values[0], key_states, 2, token_idx, self.inp_seq_len)
                value_states = self.v_cache.update(past_key_values[1], value_states, 2, token_idx, self.inp_seq_len)
            if token_idx is None:
                past_key_values = (key_states, value_states)

            #if cache_idx is not None and q_len == 1:
            #    key_states = key_states[:, :, :cache_idx, :]
            #    value_states = value_states[:, :, :cache_idx, :]
            #    if attention_mask is not None:
            #        attention_mask = attention_mask[:, :, :, :cache_idx]
            #    kv_seq_len = key_states.shape[-2]
        else:
            past_key_values = None

        #if past_key_values is not None:
        #    cache_kwargs = {"cache_position": cache_position}
        #    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states, #8,64,1,64
            key_states, #8,8,19,64
            value_states,
            attention_mask, #8,1,1,19?
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,  # diff with Llama
            token_idx=token_idx,
            **kwargs,
        )

        attn_output = attn_output.reshape(input_shape[0], -1, q_len, self.head_dim)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(*input_shape, -1)
        #attn_output = attn_output.reshape(*input_shape, -1, self.head_dim).contiguous()
        
        attn_output = self.o_proj(attn_output)
        #if use_cache:
        return attn_output, attn_weights, past_key_values
        #else:
        #    return attn_output, attn_weights


def gaudi_gpt_oss_decoder_layer_forward(
    #def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    token_idx: Optional[torch.Tensor] = None,
    #reuse_cache: Optional[bool] = False,
    #cache_idx: int = None,
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
        #reuse_cache=reuse_cache,
        #cache_idx=cache_idx,
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

    #def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
    #    for layer in self.layers:
    #        layer.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    #def update_sincos_cache(self, seq_len):
    #    for layer in self.h:
    #        layer.update_sincos_cache(seq_len)


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
        #reuse_cache: Optional[bool] = False,
        #cache_idx: int = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        #if use_cache and past_key_values is None:
        #    past_key_values = DynamicCache()

        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        """
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        """

        #ignore_cache_position = True
        past_seen_tokens = 0
        if past_key_values is not None and use_cache:  # kept for BC (cache positions)
            #if reuse_cache:
            #    if isinstance(past_key_values[0][0], torch.Tensor):
            #        past_seen_tokens = past_key_values[0][0].shape[2]
            #    else:
            #        past_seen_tokens = past_key_values[0][0][2]
            #else:
            past_seen_tokens = past_key_values[0][0].shape[2]
        print("****pst",past_seen_tokens)
        
        #if not ignore_cache_position: ####cache position is arange(input+output_len) #ignore for hpu
        #    if cache_position is None:
        #        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #        cache_position = torch.arange(
        #            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        #        )
        #    if position_ids is None and cache_position:
        #        position_ids = cache_position.unsqueeze(0)
        #else:###for HPU
        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens, seq_length + past_seen_tokens, dtype=torch.long, device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0)
        cache_position = None
        #cache position needed for create_causal_mask()
        #cache_position = torch.arange(
        #        past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device
        #    )


        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            #mask_kwargs = {
            #    "config": self.config,
            #    "input_embeds": inputs_embeds,
            #    "attention_mask": attention_mask,
            #    "cache_position": cache_position,
            #    "past_key_values": past_key_values,
            #}

            causal_mask_mapping = {
                #"full_attention": create_causal_mask(**mask_kwargs),
                "full_attention": _gaudi_prepare_4d_causal_attention_mask(
                                    attention_mask,
                                    input_ids.shape if input_ids is not None else (batch_size, seq_length),
                                    inputs_embeds,
                                    past_seen_tokens,
                                ), #(8,283) -> (8,1,283,283)
                #"sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
                "sliding_attention": _gaudi_prepare_4d_causal_attention_mask(
                                    attention_mask,
                                    input_ids.shape if input_ids is not None else (batch_size, seq_length),
                                    inputs_embeds,
                                    past_seen_tokens,
                                    self.config.sliding_window,
                                ), #(8,283) -> (8,1,283,283)
            }


        hidden_states = inputs_embeds
        kv_seq_len = hidden_states.shape[-2]
        if past_key_values is not None:
            if token_idx is not None:
                kv_seq_len = past_key_values[0][0].shape[-2]
            else:
                kv_seq_len += past_key_values[0][0].shape[-2]

        position_embeddings = self.rotary_emb(hidden_states, seq_len=kv_seq_len)#position_ids) #cos, sin up to seq_len
        #position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if use_cache:
            next_decoder_cache = ()
        else:
            next_decoder_cache = None

        if (
            lazy_mode and
            not self.training
            and (torch.distributed.is_initialized() is False or torch.distributed.get_world_size() == 1)
        ):
            htcore.mark_step()
        
        for layer_idx, decoder_layer in enumerate(self.layers):
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
                #reuse_cache=reuse_cache,
                **kwargs,
            )

            #layer_outputs is a tuple of (hidden_states, past_key_value)
            if use_cache:
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache, #kv cache for all layers
            #past_key_values=past_key_values,
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
        #reuse_cache: Optional[bool] = None,
        lazy_mode: Optional[bool] = True,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GptOssForCausalLM

        >>> model = GptOssForCausalLM.from_pretrained("mistralai/GptOss-8x7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/GptOss-8x7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
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
            token_idx=token_idx, ##sc
            #reuse_cache=reuse_cache, ##sc
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
        #reuse_cache = kwargs.get("reuse_cache")

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
        #elif reuse_cache and token_idx is not None:
        #    # With reuse_cache, KV cache is pre allocated hence for the 1st token we can slice the inputs till token idx for the fwd pass
        #    input_ids = input_ids[:, :token_idx]
        #    attention_mask = attention_mask[:, :token_idx]

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
                "cache_position": None, ####for hpu#cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "output_router_logits": output_router_logits,
                "token_idx": token_idx,
                #"reuse_cache": reuse_cache,
                #"cache_idx": kwargs.get("cache_idx"),
                "lazy_mode": kwargs.get("lazy_mode"),
            }
        )
        return model_inputs

