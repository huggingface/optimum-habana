import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    apply_rotary_pos_emb,
    logger,
)


try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE
except ImportError:
    print("Not using HPU fused kernel for apply_rotary_pos_emb")
    FusedRoPE = None

try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm as FusedRMSNorm
except ImportError:
    print("Not using HPU fused kernel for RMSNorm")
    FusedRMSNorm = None

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None


def update(prev, cur, dim, idx, inp_seq_len):
    orig_cur = cur
    if prev.dtype == torch.float8_e4m3fn:
        from habana_frameworks.torch.hpex.kernels.Fp8Ops import cast_to_fp8_v2

        cur = cast_to_fp8_v2(cur, None, False, False, prev.dtype)[0]
    if cur.shape[2] > 1 and cur.shape[2] <= prev.shape[2]:
        # Initialize
        prev[:, :, :inp_seq_len, :].copy_(cur)
        return orig_cur
    assert cur.shape[2] == 1, f"Cannot update kv-cache. Unsupported shapes. prev:{prev.shape} cur:{cur.shape}"
    if idx is not None:
        prev.index_copy_(dim, idx - 1, cur)
        prev_cast = prev.to(orig_cur.dtype)
        return prev_cast
    else:
        return torch.cat((prev, cur), dim=dim)


def gaudi_llama_rmsnorm_forward(self, hidden_states):
    """
    Copied from LlamaRMSNorm.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
        - override RMSNorm with Habana fused RMSNorm
    """
    if hidden_states.device.type == "hpu" and FusedRMSNorm:
        # mixed dtypes are not good for FusedRMSNorm, both inputs need to have same dtype
        if hidden_states.dtype != self.weight.dtype:
            orig_dtype = hidden_states.dtype
            hidden_states = FusedRMSNorm.apply(hidden_states.to(self.weight.dtype), self.weight, self.variance_epsilon)
            return hidden_states.to(orig_dtype)
        else:
            hidden_states = FusedRMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)
            return hidden_states
    else:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def gaudi_llama_repeat_kv(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    n_rep: int,
):
    """
    Copied from repeat_kv: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
        - Append num_key_value_heads == 1 check as kv states can be broadcasted during matmuls so need to expand and reshape them.
        - Add new args query_states, key_states, value_states and attention_mask and update the logic for expansion.
    The query states go from (batch, num_heads, seqlen, head_dim) to (batch, num_key_value_heads, n_rep, seqlen, head_dim)
    The key/value states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_key_value_heads, 1, seqlen, head_dim)
    """
    batch, num_key_value_heads, kv_len, head_dim = key_states.shape
    if n_rep == 1 or num_key_value_heads == 1:
        return query_states, key_states, value_states, attention_mask

    new_kv_shape = (batch, num_key_value_heads, 1, kv_len, head_dim)
    key_states = key_states.reshape(new_kv_shape)
    value_states = value_states.reshape(new_kv_shape)

    batch, _, q_len, head_dim = query_states.shape
    new_q_shape = (batch, num_key_value_heads, n_rep, q_len, head_dim)
    query_states = query_states.reshape(new_q_shape)

    if attention_mask is not None:
        # Add groups dim and set to 1
        attention_mask = attention_mask.unsqueeze(1)

    return query_states, key_states, value_states, attention_mask


class Matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class KVCache(torch.nn.Module):
    def __init__(self):
        super(KVCache, self).__init__()
        self.cache = None
        self.inp_seq_len = -1

    def allocate(self, inp_seq_len, kv_cache_fp8, dtype, device, shape):
        if self.cache is None or self.cache.shape != shape:
            self.inp_seq_len = inp_seq_len
            if kv_cache_fp8:
                dtype = torch.float8_e4m3fn
            self.cache = torch.zeros(shape, dtype=dtype, device=device)
        else:
            assert (
                self.inp_seq_len == inp_seq_len
            ), f"inp_seq_len must be the same. self.inp_seq_len:{self.inp_seq_len} inp_seq_len:{inp_seq_len}"
            self.cache.fill_(0)

    def get_shape(self):
        if self.cache is None:
            return None
        return self.cache.shape

    def forward(self, cur, dim, idx):
        return update(self.cache, cur, dim, idx, self.inp_seq_len)


class GaudiLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.matmul_qk = Matmul()
        self.matmul_av = Matmul()
        self.k_cache = KVCache()
        self.v_cache = KVCache()
        self.inp_seq_len = -1
        self.norm_factor = 1.0 / math.sqrt(self.head_dim)

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len, kv_cache_fp8):
        cache_shape = (batch_size, self.num_key_value_heads, max_seq_len, self.head_dim)
        device = self.k_proj.weight.device
        dtype = self.config.torch_dtype
        self.k_cache.allocate(inp_seq_len, kv_cache_fp8, dtype, device, cache_shape)
        self.v_cache.allocate(inp_seq_len, kv_cache_fp8, dtype, device, cache_shape)

    def update_sincos_cache(self, seq_len):
        # Call rotary emb forward() to update cos/sin cache when infering more than self.max_position_embeddings
        # This helps in avoiding creation of these caches during actual model forward pass and
        # reduce memory consumption and improve performance.
        if seq_len > self.max_position_embeddings:
            self.max_position_embeddings = seq_len
            _, _ = self.rotary_emb(self.k_proj.weight, seq_len=seq_len)

    def reorder(self, tensor, beam_idx, dim_a, dim_b):
        updated = tensor.index_select(0, beam_idx)
        tensor.copy_(updated)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        if self.k_cache.cache is None:
            return (None, None)

        head_dim = self.k_cache.cache.size(-1)
        seq_length = self.k_cache.cache.size(-2)
        self.reorder(self.k_cache.cache, beam_idx, seq_length, head_dim)
        self.reorder(self.v_cache.cache, beam_idx, seq_length, head_dim)
        return (self.k_cache.cache.shape, self.v_cache.cache.shape)

    def pre_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Copied from LlamaAttention.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
        The only differences are:
        - add new args token_idx
        - optimize KV cache
        - add new args attn_softmax_bf16
        - add new args reuse_cache
        - add new args use_flash_attention
        - add new arg flash_attention_recompute
        """
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # TODO: update when auto mp params is enabled in DeepSpeed (cf. https://github.com/HabanaAI/DeepSpeed/blob/94309c7b5dfc1a69858f5c9f25737b2f81a332a5/deepspeed/module_inject/replace_module.py#L440)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if token_idx is None:
                kv_seq_len += past_key_value[0].shape[-2]
            else:
                if reuse_cache:
                    kv_seq_len = past_key_value[0][-2]
                else:
                    kv_seq_len = past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_customized_rope(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None or reuse_cache:
            # reuse k, v, self_attention
            if reuse_cache:
                key_states = self.k_cache(key_states, 2, token_idx)
                value_states = self.v_cache(value_states, 2, token_idx)
            else:
                key_states = update(past_key_value[0], key_states, 2, token_idx, self.inp_seq_len)
                value_states = update(past_key_value[1], value_states, 2, token_idx, self.inp_seq_len)

        if use_cache:
            if reuse_cache:
                past_key_value = (self.k_cache.get_shape(), self.v_cache.get_shape())
            else:
                past_key_value = (key_states.contiguous(), value_states.contiguous())
        else:
            past_key_value = None

        if use_flash_attention and FusedSDPA:
            import habana_frameworks.torch.hpu as ht

            if q_len == 1:
                # next token
                with ht.sdp_kernel(enable_recompute=False):
                    attn_output = FusedSDPA.apply(
                        query_states, key_states, value_states, attention_mask, 0.0, False, None
                    )
            else:
                # first token
                with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                    attn_output = FusedSDPA.apply(
                        query_states, key_states, value_states, attention_mask, 0.0, False, None
                    )

        else:
            query_states, key_states, value_states, attention_mask = gaudi_llama_repeat_kv(
                query_states, key_states, value_states, attention_mask, self.num_key_value_groups
            )

            attn_weights = self.matmul_qk(query_states, key_states.transpose(-2, -1)) * self.norm_factor

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len) and attn_weights.size() != (
                bsz,
                self.num_key_value_heads,
                self.num_key_value_groups,
                q_len,
                kv_seq_len,
            ):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)} or"
                    f" {(bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len) and attention_mask.size() != (
                    bsz,
                    1,
                    1,
                    q_len,
                    kv_seq_len,
                ):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)} or {(bsz, 1, 1, q_len, kv_seq_len)},"
                        f" but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask

            if attn_softmax_bf16:
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=query_states.dtype)
            else:
                # upcast attention to fp32
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                    query_states.dtype
                )

            attn_output = self.matmul_av(attn_weights, value_states)
            attn_output = attn_output.reshape(bsz, -1, q_len, self.head_dim)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def attention_all_reduce(self, attn_output):
        if hasattr(self.o_proj, "all_reduce"):
            self.o_proj.all_reduce(attn_output)

    def post_attn_forward(self, attn_output):
        if hasattr(self.o_proj, "post_all_reduce"):
            self.o_proj.post_all_reduce(attn_output)
        return attn_output


class GaudiLlamaMLP(LlamaMLP):
    def pre_mlp_forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            output = sum(down_proj)
        else:
            input = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            output = self.down_proj(input)
        return output

    def mlp_all_reduce(self, x):
        if hasattr(self.down_proj, "all_reduce"):
            self.down_proj.all_reduce(x)

    def post_mlp_forward(self, x):
        if self.config.pretraining_tp > 1:
            return x
        if hasattr(self.down_proj, "post_all_reduce"):
            return self.down_proj.post_all_reduce(x)
        return x


class GaudiLlamaDecoderLayer(LlamaDecoderLayer):
    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len, kv_cache_fp8):
        self.self_attn.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len, kv_cache_fp8)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.self_attn.reorder_kv_cache(beam_idx)

    def update_sincos_cache(self, seq_len):
        self.self_attn.update_sincos_cache(seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Copied from LlamaDecoderLayer.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
        The only differences are:
        - add new args token_idx
        - add new args attn_softmax_bf16
        - add new args reuse_cache
        - add new args use_flash_attention
        - add new arg flash_attention_recompute
        """
        residual = hidden_states
        output_pre_attn, self_attn_weights, present_key_value = self.pre_attn(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            token_idx,
            attn_softmax_bf16,
            reuse_cache,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
        )
        self.self_attn.attention_all_reduce(output_pre_attn)
        output_post_attn_pre_mlp, residual_mlp = self.post_attn_pre_mlp(output_pre_attn, residual)
        self.mlp.mlp_all_reduce(output_post_attn_pre_mlp)
        output_post_mlp = self.post_mlp(output_post_attn_pre_mlp, residual_mlp)

        outputs = (output_post_mlp,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def pre_attn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states = self.input_layernorm(hidden_states)
        output_attn, attn_weights, present_key_value = self.self_attn.pre_attn_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            token_idx,
            attn_softmax_bf16,
            reuse_cache,
            use_flash_attention,
            flash_attention_recompute,
        )
        return output_attn, attn_weights, present_key_value

    def post_attn_pre_mlp(self, input, residual):
        output_post_attn = self.self_attn.post_attn_forward(input)

        hidden_states = residual + output_post_attn
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp.pre_mlp_forward(hidden_states)
        return hidden_states, residual

    def post_mlp(self, input, residual):
        output_post_mlp = self.mlp.post_mlp_forward(input)
        output = output_post_mlp + residual
        return output


class GaudiLlamaModel(LlamaModel):
    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len, kv_cache_fp8):
        for layer in self.layers:
            layer.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len, kv_cache_fp8)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return tuple(layer.reorder_kv_cache(beam_idx) for layer in self.layers)

    def update_sincos_cache(self, seq_len):
        for layer in self.layers:
            layer.update_sincos_cache(seq_len)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Copied from LlamaModel.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
        The only differences are:
        - add new args token_idx
        - add new args attn_softmax_bf16
        - add new args reuse_cache
        - add new args use_flash_attention
        - add new arg flash_attention_recompute
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            if reuse_cache:
                past_key_values_length = past_key_values[0][0][2]
            else:
                past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(
                            *inputs,
                            past_key_value,
                            output_attentions,
                            attn_softmax_bf16=attn_softmax_bf16,
                            use_flash_attention=use_flash_attention,
                            flash_attention_recompute=flash_attention_recompute,
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    token_idx=token_idx,
                    attn_softmax_bf16=attn_softmax_bf16,
                    reuse_cache=reuse_cache,
                    use_flash_attention=use_flash_attention,
                    flash_attention_recompute=flash_attention_recompute,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
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


class GaudiLlamaForCausalLM(LlamaForCausalLM):
    """
    Inherits from LlamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
    - add new args token_idx
    - add token_idx into model_inputs
    - from step2 when enable KV cache, slice next_input_ids from input_ids base on the token_idx
    - from step2 when enable KV cache, slice next_position_ids from position_ids base on the token_idx
    - add new args attn_softmax_bf16
    - add new args reuse_cache
    """

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len, kv_cache_fp8):
        self.model.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len, kv_cache_fp8)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.model.reorder_kv_cache(beam_idx)

    def update_sincos_cache(self, seq_len):
        self.model.update_sincos_cache(seq_len)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        trim_logits: Optional[bool] = False,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_idx=token_idx,
            attn_softmax_bf16=attn_softmax_bf16,
            reuse_cache=reuse_cache,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
        )
        hidden_states = outputs[0]
        _, seq_len, _ = hidden_states.shape
        if seq_len > 1 and trim_logits and not self.training:
            if token_idx is not None:
                hidden_states = hidden_states.index_select(1, token_idx - 1)
            else:
                hidden_states = hidden_states[:, -1, :]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, token_idx=None, **kwargs
    ):
        reuse_cache = kwargs.get("reuse_cache")
        if past_key_values:
            if token_idx is not None:
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)
            else:
                input_ids = input_ids[:, -1:]
        elif reuse_cache and token_idx is not None:
            # With reuse_cache, KV cache is pre allocated hence for the 1st token we can slice the inputs till token idx for the fwd pass
            input_ids = input_ids[:, :token_idx]
            attention_mask = attention_mask[:, :token_idx]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                if token_idx is not None:
                    position_ids = torch.index_select(position_ids, 1, token_idx - 1)
                else:
                    position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "token_idx": token_idx,
                "trim_logits": kwargs.get("trim_logits"),
                "attn_softmax_bf16": kwargs.get("attn_softmax_bf16"),
                "reuse_cache": reuse_cache,
                "use_flash_attention": kwargs.get("use_flash_attention"),
                "flash_attention_recompute": kwargs.get("flash_attention_recompute"),
            }
        )
        return model_inputs


def apply_customized_rope(q, k, cos, sin, position_ids):
    if q.device.type == "hpu" and FusedRoPE:
        # TODO: remove `.clone()` when SynapseAI v1.15 is released
        return FusedRoPE.apply(q, cos.clone(), sin.clone(), position_ids), FusedRoPE.apply(
            k, cos.clone(), sin.clone(), position_ids
        )
    else:
        return apply_rotary_pos_emb(q, k, cos, sin, position_ids)
