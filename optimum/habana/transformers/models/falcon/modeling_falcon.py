import contextlib
import math
import os
from typing import Optional, Tuple, Union

import torch


try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused kernel for scaled_dot_product_attention")
    FusedSDPA = None

try:
    from habana_frameworks.torch.hpu import sdp_kernel

    SDPContext = True
except ImportError:
    SDPContext = False

try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE
except ImportError:
    print("Not using HPU fused kernel for apply_rotary_pos_emb")
    FusedRoPE = None


import habana_frameworks.torch.core as htcore
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.falcon.configuration_falcon import FalconConfig
from transformers.models.falcon.modeling_falcon import (
    FalconAttention,
    FalconDecoderLayer,
    FalconForCausalLM,
    FalconMLP,
    FalconModel,
    apply_rotary_pos_emb,
    build_alibi_tensor,
)
from transformers.utils import logging

from ...modeling_attn_mask_utils import (
    GaudiAttentionMaskConverter,
    _gaudi_prepare_4d_causal_attention_mask,
)


logger = logging.get_logger(__name__)


def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """
    Copied from transformers.models.falcon.modeling_falcon/dropout_add
    https://github.com/huggingface/transformers/blob/b338a6c3b8eda29610d4d472cad8cd87cbfdaaed/src/transformers/models/falcon/modeling_falcon.py#L248
    """
    out = F.dropout(x, p=prob, training=training)
    if training:
        out = residual + out
        return out
    else:
        residual.add_(out)
        return residual


def apply_customized_rope(q, k, cos, sin, position_ids):
    if q.device.type == "hpu" and FusedRoPE:
        # TODO: remove `.clone()` when it is fixed in SynapseAI
        return FusedRoPE.apply(
            q, cos.unsqueeze(0).unsqueeze(0).clone(), sin.unsqueeze(0).unsqueeze(0).clone(), position_ids
        ), FusedRoPE.apply(
            k, cos.unsqueeze(0).unsqueeze(0).clone(), sin.unsqueeze(0).unsqueeze(0).clone(), position_ids
        )
    else:
        return apply_rotary_pos_emb(q, k, cos, sin, position_ids)


def gaudi_falcon_linear_forward(self, input: torch.Tensor) -> torch.Tensor:
    hidden_states = F.linear(input, self.weight, bias=self.bias)
    return hidden_states


#  FusedScaledDotProductAttention
class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        self._hpu_kernel_fsdpa = fusedSDPA

    def forward(self, query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode):
        return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode)


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim=None, invAttnHead=None):
        return torch.ops.hpu.softmax_fp8(x, dim, None, None, invAttnHead)


class Matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.matmul(*args, **kwargs)


# ScaledDotProductAttention is based on torch.nn.functional.scaled_dot_product_attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, config: FalconConfig):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.bmm1 = Matmul()
        self.bmm2 = Matmul()
        self.softmax = Softmax()
        self.num_key_value_groups = config.num_attention_heads // config.num_kv_heads

    def repeat_kv(
        self,
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

    def forward(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(self.head_dim)
        invAttnHead = torch.tensor(scale_factor, dtype=torch.float32).to("hpu")

        if is_causal:
            assert attn_mask is None
            attn_bias = torch.zeros(L, S, dtype=query.dtype)
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))

        query, key, value, attn_mask = self.repeat_kv(query, key, value, attn_mask, self.num_key_value_groups)

        attn_weight = self.bmm1(query, key.transpose(-2, -1))
        attn_weight += attn_mask
        attn_weight = self.softmax(attn_weight, dim=-1, invAttnHead=invAttnHead)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        attn_output = self.bmm2(attn_weight, value)
        return attn_output


def update(prev, cur, dim, idx, inp_seq_len):
    orig_cur = cur
    cur = cur.to(dtype=prev.dtype)

    if prev.shape == cur.shape:
        prev.copy_(cur)
        return orig_cur

    if cur.shape[-2] > 1 and cur.shape[-2] <= prev.shape[-2]:
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


class KVCache(torch.nn.Module):
    def __init__(self):
        super(KVCache, self).__init__()
        self.cache = None
        self.inp_seq_len = -1

    def allocate(self, inp_seq_len, dtype, device, shape):
        if self.cache is None or self.cache.shape != shape:
            self.inp_seq_len = inp_seq_len
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
        return self.update(self.cache, cur, dim, idx, self.inp_seq_len)

    @staticmethod
    def update(prev, cur, dim, idx, inp_seq_len):
        return update(prev, cur, dim, idx, inp_seq_len)


class GaudiFalconAttention(FalconAttention):
    """
    Inherits from FalconAttention: https://github.com/huggingface/transformers/blob/838b87abe231fd70be5132088d0dee72a7bb8d62/src/transformers/models/falcon/modeling_falcon.py#L267
    The only differences are:
    - add new args token_idx and position_ids
    - replace F.scaled_dot_product_attention with Habana torch's version for BF16
    - use ScaledDotProductAttention for FP8 quantization
    - add new arg reuse_cache
    - add new args use_flash_attention
    - add new arg flash_attention_recompute
    - add new arg flash_attention_causal_mask
    Choice of SDPA:
        There are these variables: use_flash_attention and datatype (bf16/fp8)
        datatype is determined by presence of QUANT_CONFIG env var, presence of which indicates fp8
        1. use_flash_attention, fp8: use ModuleFusedSDPA. most optimal
        2. use_flash_attention, bf16: use FusedSDPA
        3. not use_flash_attention, fp8: Use ScaledDotProductAttention, along with QUANT_CONFIG. This is the case before this PR
        4. not use_flash_attention, bf16: F.scaled_dot_product_attention. Slowest option
    """

    def __init__(self, config: FalconConfig):
        super().__init__(config)

        self.is_fp8 = os.getenv("QUANT_CONFIG", "") != ""

        # In the constructor we do not know which one we will need later in the forward, so creating both
        # TODO, Does this affect memory usage?
        if self.is_fp8:
            self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA)
        self.unfused_scaled_dot_product_attention = ScaledDotProductAttention(config)

        self.k_cache = KVCache()
        self.v_cache = KVCache()
        self.inp_seq_len = -1
        self.max_position_embeddings = config.max_position_embeddings

    def _split_heads(
        self, fused_qkv: torch.Tensor, broadcast: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.new_decoder_architecture:
            batch, seq_len, _ = fused_qkv.shape

            if self.config.num_attention_heads != self.num_heads:  # When DS divides heads for TP
                num_heads = self.config.num_attention_heads
                num_kv_heads = self.config.num_kv_heads
            else:  # When DS not in use
                num_heads = self.num_heads
                num_kv_heads = self.num_kv_heads

            qkv = fused_qkv.view(batch, seq_len, -1, num_heads // num_kv_heads + 2, self.head_dim)
            # query = qkv[:, :, :, :-2]
            # key = qkv[:, :, :, [-2]]
            # value = qkv[:, :, :, [-1]]
            d3 = qkv.shape[3] - 2
            query = torch.index_select(qkv, 3, index=torch.arange(d3, device=qkv.device))
            key = torch.index_select(qkv, 3, index=torch.tensor([d3], device=qkv.device))
            value = torch.index_select(qkv, 3, index=torch.tensor([d3 + 1], device=qkv.device))
            if broadcast:
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

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        if self.config.new_decoder_architecture:
            cache_shape = (batch_size, self.num_kv_heads, max_seq_len, self.head_dim)
        else:
            cache_shape = (batch_size, 1, max_seq_len, self.head_dim)
        device = self.query_key_value.weight.device
        dtype = self.config.torch_dtype
        self.k_cache.allocate(inp_seq_len, dtype, device, cache_shape)
        self.v_cache.allocate(inp_seq_len, dtype, device, cache_shape)

    def update_sincos_cache(self, seq_len):
        # Call rotary emb forward() to update cos/sin cache when infering more than self.max_position_embeddings
        # This helps in avoiding creation of these caches during actual model forward pass and
        # reduce memory consumption and improve performance.
        if seq_len > self.max_position_embeddings:
            self.max_position_embeddings = seq_len
            self.rotary_emb._set_cos_sin_cache(
                seq_len, self.query_key_value.weight.device, self.query_key_value.weight.dtype
            )

    def pre_attn_forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        token_idx: Optional[torch.Tensor] = None,
        reuse_cache: Optional[bool] = False,
        cache_idx: int = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        # 3 x [batch_size, seq_length, num_heads, head_dim]

        train_with_flash_attention = self.training and self._use_sdpa and not output_attentions and head_mask is None
        (query_layer, key_layer, value_layer) = self._split_heads(
            fused_qkv, not use_flash_attention and not self.is_fp8 and not train_with_flash_attention
        )

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size, -1, query_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(batch_size, -1, query_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size, -1, query_length, self.head_dim)

        kv_seq_len = key_layer.shape[-2]
        if layer_past is not None:
            if token_idx is not None:
                if reuse_cache:
                    kv_seq_len = layer_past[0][-2]
                else:
                    kv_seq_len = layer_past[0].shape[-2]
            else:
                kv_seq_len += layer_past[0].shape[-2]

        if alibi is None:
            cos, sin = self.rotary_emb(value_layer, seq_len=kv_seq_len)
            query_layer, key_layer = apply_customized_rope(query_layer, key_layer, cos, sin, position_ids)

        if use_cache:
            if self.training:
                present = None
            else:
                if reuse_cache:
                    key_layer = self.k_cache(key_layer, -2, token_idx)
                    value_layer = self.v_cache(value_layer, -2, token_idx)
                    present = (self.k_cache.get_shape(), self.v_cache.get_shape())
                else:
                    if layer_past is None:
                        past_key = torch.zeros(
                            key_layer.shape,
                            dtype=self.query_key_value.weight.dtype,
                            device=self.query_key_value.weight.device,
                        )
                        past_value = torch.zeros(
                            key_layer.shape,
                            dtype=self.query_key_value.weight.dtype,
                            device=self.query_key_value.weight.device,
                        )
                        layer_past = [past_key, past_value]
                    key_layer = self.k_cache.update(
                        layer_past[0], key_layer, -2, token_idx, self.inp_seq_len
                    )  # k_layer bs*1, q_len, head_dim
                    value_layer = self.v_cache.update(layer_past[1], value_layer, -2, token_idx, self.inp_seq_len)
                    if token_idx is None:
                        layer_past = (key_layer, value_layer)
                    present = layer_past

                if cache_idx is not None and query_length == 1:
                    key_layer = key_layer[:, :, :cache_idx, :]
                    value_layer = value_layer[:, :, :cache_idx, :]
                    attention_mask = attention_mask[:, :, :, :cache_idx]
        else:
            present = None

        if self.training or present is None:
            kv_length = key_layer.shape[-2]
        else:
            kv_length = present[0][-2] if reuse_cache else present[0].shape[-2]

        if (not reuse_cache) and (token_idx is not None) and (cache_idx is not None) and (query_length == 1):
            # Return only past key value shapes and not the tensors during decode phase (q len is 1)
            # to avoid making past key values as persistent output tensors of HPU graphs.
            present = (present[0].shape, present[1].shape)

        if alibi is None:  # both train/inference
            if output_attentions:
                attention_scores = query_layer @ key_layer.transpose(-1, -2)
                attention_scores /= math.sqrt(self.head_dim)

                attention_scores = F.softmax(attention_scores + attention_mask, dim=-1, dtype=hidden_states.dtype)
                # It is unclear why neither dropout nor head_mask is applied here (while it is with alibi).
                attn_output = attention_scores @ value_layer
            else:
                if use_flash_attention or train_with_flash_attention:
                    is_causal = self.is_causal and query_length > 1 and flash_attention_causal_mask
                    if self.is_fp8:
                        attn_mask = None if is_causal else attention_mask
                        flash_attention_fast_softmax = True  # TODO pass this along
                        softmax_mode = "fast" if flash_attention_fast_softmax else "None"
                        enable_recompute = self.is_fp8 if query_length == 1 else flash_attention_recompute
                        with sdp_kernel(enable_recompute=enable_recompute):
                            attn_output = self.fused_scaled_dot_product_attention(
                                query_layer, key_layer, value_layer, attn_mask, 0.0, is_causal, None, softmax_mode
                            )
                    else:
                        # TODO very similar to the fp8 case above, could be merged.
                        with sdp_kernel(
                            enable_recompute=flash_attention_recompute
                        ) if SDPContext else contextlib.nullcontext():
                            attn_output = FusedSDPA.apply(
                                query_layer,
                                key_layer,
                                value_layer,
                                attention_mask,
                                0.0,
                                # The query_length > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case query_length == 1.
                                is_causal and attention_mask is None,
                            )
                else:
                    if self.is_fp8:
                        attn_output = self.unfused_scaled_dot_product_attention(
                            query_layer, key_layer, value_layer, attention_mask, 0.0, is_causal=False
                        )
                    else:
                        # Workaround util scaled_dot_product_attention support broadcast.
                        if self.training is True and query_layer.shape != key_layer.shape:
                            key_layer = torch.broadcast_to(key_layer, query_layer.shape)
                            value_layer = torch.broadcast_to(value_layer, query_layer.shape)
                        attn_output = F.scaled_dot_product_attention(
                            query_layer,
                            key_layer,
                            value_layer,
                            attention_mask,
                            0.0,
                            # The query_length > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case query_length == 1.
                            is_causal=self.is_causal and attention_mask is None and query_length > 1,
                        )

                # Performance improvement for HPU
                if self.training is True and htcore:
                    htcore.mark_step()
                attention_scores = None

            attn_output = attn_output.view(batch_size, -1, query_length, self.head_dim)
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(batch_size, query_length, -1)

            attn_output = self.dense(attn_output)

            if output_attentions:
                return attn_output, present, attention_scores
            else:
                return attn_output, present, _

        else:
            if train_with_flash_attention:
                if FusedSDPA:
                    # TODO needs to be turned into a module for quantization
                    with sdp_kernel(enable_recompute=False) if SDPContext else contextlib.nullcontext():
                        attn_output = FusedSDPA.apply(
                            query_layer,
                            key_layer,
                            value_layer,
                            attention_mask,
                            self.attention_dropout.p if self.training else 0.0,
                            self.is_causal and attention_mask is None and query_length > 1,
                        )
                else:
                    attn_output = F.scaled_dot_product_attention(
                        query_layer,
                        key_layer,
                        value_layer,
                        attn_mask=attention_mask,
                        dropout_p=self.attention_dropout.p if self.training else 0.0,
                        is_causal=self.is_causal and attention_mask is None and query_length > 1,
                    )
                attn_output = attn_output.transpose(1, 2)
                attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

                attn_output = self.dense(attn_output)
            else:
                matmul_result = query_layer @ key_layer.transpose(-1, -2)

                # change view to [batch_size, num_heads, q_length, kv_length]
                attention_scores = matmul_result.view(batch_size, self.num_heads, query_length, kv_length)

                # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
                input_dtype = attention_scores.dtype
                # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
                if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                    attention_scores = attention_scores.to(torch.float32)

                attention_logits = attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)
                attention_logits *= self.inv_norm_factor
                attention_probs = F.softmax(attention_logits + attention_mask, dim=-1, dtype=hidden_states.dtype)
                # [batch_size, num_heads, q_length, kv_length]
                attention_probs = self.attention_dropout(attention_probs)

                if head_mask is not None:
                    attention_probs = attention_probs * head_mask

                # change view [batch_size, num_heads, q_length, kv_length]
                attention_probs_reshaped = attention_probs.view(batch_size, self.num_heads, query_length, kv_length)

                # matmul: [batch_size * num_heads, q_length, head_dim]
                attn_output = (attention_probs_reshaped @ value_layer).flatten(0, 1)

                # change view [batch_size, q_length, num_heads * head_dim]
                attn_output = self._merge_heads(attn_output)

                attn_output = self.dense(attn_output)

            if output_attentions:
                return attn_output, present, attention_probs
            else:
                return attn_output, present, _

    def attention_all_reduce(self, attn_output):
        if hasattr(self.dense, "all_reduce"):
            self.dense.all_reduce(attn_output)

    def post_attn_forward(self, attn_output):
        if hasattr(self.dense, "all_reduce"):
            self.dense.post_all_reduce(attn_output)
        return attn_output


class GaudiFalconMLP(FalconMLP):
    """
    Inherits from FalconMLP: https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
    """

    def pre_mlp_forward(self, x):
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x

    def mlp_all_reduce(self, x):
        if hasattr(self.dense_4h_to_h, "all_reduce"):
            self.dense_4h_to_h.all_reduce(x)

    def post_mlp_forward(self, x):
        if hasattr(self.dense_4h_to_h, "all_reduce"):
            self.dense_4h_to_h.post_all_reduce(x)
        return x


class GaudiFalconDecoderLayer(FalconDecoderLayer):
    """
    Inherits from FalconDecoderLayer: https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
    The only differences are:
    - add new args token_idx and position_ids
    - add token_idx and position_ids into attention inputs
    - add new args reuse_cache
    - add new args use_flash_attention
    - add new arg flash_attention_recompute
    - add new arg flash_attention_causal_mask
    """

    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.self_attention = GaudiFalconAttention(config)

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.self_attention.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def update_sincos_cache(self, seq_len):
        self.self_attention.update_sincos_cache(seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        token_idx: Optional[torch.Tensor] = None,
        reuse_cache: Optional[bool] = False,
        cache_idx: int = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        **kwargs,
    ):
        residual = hidden_states

        (
            hidden_states,
            present,
            attn_scores,
            attention_layernorm_out,
            mlp_layernorm_out,
        ) = self.pre_attn(  # layernorm + attention before AllReduce
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            token_idx=token_idx,
            reuse_cache=reuse_cache,
            cache_idx=cache_idx,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
        )

        self.self_attention.attention_all_reduce(hidden_states)
        hidden_states = self.self_attention.post_attn_forward(hidden_states)

        attention_output = hidden_states

        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(
                    attention_output, residual, self.config.attention_dropout, training=self.training
                )
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        if (
            self.config.new_decoder_architecture
            and self.config.parallel_attn
            and self.config.num_ln_in_parallel_attn == 1
        ):
            mlp_layernorm_out = attention_layernorm_out

        outputs = (present, attn_scores)

        hidden_states = self.mlp.pre_mlp_forward(mlp_layernorm_out)
        self.mlp.mlp_all_reduce(hidden_states)
        hidden_states = self.mlp.post_mlp_forward(hidden_states)

        if self.config.new_decoder_architecture or self.config.parallel_attn:
            hidden_states += attention_output

        output = dropout_add(hidden_states, residual, self.config.hidden_dropout, training=self.training)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions

    def pre_attn(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        token_idx: Optional[torch.Tensor] = None,
        reuse_cache: Optional[bool] = False,
        cache_idx: int = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
    ):
        if self.config.new_decoder_architecture and self.config.num_ln_in_parallel_attn == 2:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)
            mlp_layernorm_out = None

        # Self attention.
        attn_outputs, present, attn_scores = self.self_attention.pre_attn_forward(
            attention_layernorm_out,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            token_idx=token_idx,
            reuse_cache=reuse_cache,
            cache_idx=cache_idx,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
        )

        return attn_outputs, present, attn_scores, attention_layernorm_out, mlp_layernorm_out


class GaudiFalconModel(FalconModel):
    """
    Inherits from FalconModel: https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
    The only differences are:
    - add new args token_idx and position_ids
    - add token_idx and position_ids into decoder inputs
    - add new arg reuse_cache
    - add new args use_flash_attention
    - add new arg flash_attention_recompute
    - add new arg flash_attention_causal_mask
    """

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        for layer in self.h:
            layer.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def update_sincos_cache(self, seq_len):
        for layer in self.h:
            layer.update_sincos_cache(seq_len)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        reuse_cache: Optional[bool] = False,
        cache_idx: int = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
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

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        past_key_values_length = 0
        if past_key_values[0] is not None and token_idx is None:
            if reuse_cache:
                past_key_values_length = past_key_values[0][0][-2]
            else:
                past_key_values_length = past_key_values[0][0].shape[-2]

        if self.use_alibi:
            mask = (
                torch.ones(
                    (batch_size, seq_length + past_key_values_length), device=inputs_embeds.device, dtype=torch.long
                )
                if attention_mask is None
                else attention_mask
            )
            alibi = build_alibi_tensor(mask, self.num_heads, dtype=hidden_states.dtype)
        else:
            alibi = None
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

        # TODO: Due to perf degradation, disable spda_attn_mask
        use_sdpa_attn_mask = False

        if self._use_sdpa and not output_attentions and use_sdpa_attn_mask:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            if alibi is None:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            elif head_mask is None:
                alibi = alibi.reshape(batch_size, -1, *alibi.shape[1:])

                # We don't call _prepare_4d_causal_attention_mask_for_sdpa as we need to mask alibi using the 4D attention_mask untouched.
                attention_mask = _gaudi_prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )

                # We take care to integrate alibi bias in the attention_mask here.
                min_dtype = torch.finfo(alibi.dtype).min
                attention_mask = torch.masked_fill(
                    alibi / math.sqrt(self.config.hidden_size // self.num_heads),
                    attention_mask < -1,
                    min_dtype,
                )

                # From PyTorch 2.1 onwards, F.scaled_dot_product_attention with the memory-efficient attention backend
                # produces nans if sequences are completely unattended in the attention mask. Details: https://github.com/pytorch/pytorch/issues/110213
                if seq_length > 1:
                    attention_mask = GaudiAttentionMaskConverter._unmask_unattended(
                        attention_mask, min_dtype=min_dtype
                    )
            else:
                # PyTorch SDPA does not support head_mask, we fall back on the eager implementation in this case.
                attention_mask = _gaudi_prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )

        else:
            # 4d mask is passed through the layers
            attention_mask = _gaudi_prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    alibi,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                    layer_past,
                    use_cache,
                    output_attentions,
                    None,
                    use_flash_attention,
                    flash_attention_recompute,
                    flash_attention_causal_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                    token_idx=token_idx,
                    reuse_cache=reuse_cache,
                    cache_idx=cache_idx,
                    use_flash_attention=use_flash_attention,
                    flash_attention_recompute=flash_attention_recompute,
                    flash_attention_causal_mask=flash_attention_causal_mask,
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
    - add new args reuse_cache
    - add use_flash_attention
    - add flash_attention_recompute
    - add flash_attention_causal_mask
    """

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.transformer.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)
        self.kv_cache_len = max_seq_len

    def update_sincos_cache(self, seq_len):
        self.transformer.update_sincos_cache(seq_len)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        reuse_cache = kwargs.get("reuse_cache")
        bucket_internal = kwargs.get("bucket_internal")
        if past_key_values is not None:
            if token_idx is not None:
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)
            else:
                past_length = past_key_values[0][0].shape[2]

                # Some generation methods already pass only the last input ID
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # Default to old behavior: keep only final ID
                    remove_prefix_length = input_ids.shape[1] - 1

                input_ids = input_ids[:, remove_prefix_length:]
        elif (reuse_cache or bucket_internal) and token_idx is not None:
            # KV cache is pre allocated with reuse cache or will be padded with bucket internal
            # hence for the 1st token we can slice the inputs till token idx for the fwd pass.
            input_ids = input_ids[:, :token_idx]
            attention_mask = attention_mask[:, :token_idx]

        # Note: versions of Falcon with alibi do not use position_ids. It is used with RoPE.
        if (
            not self.transformer.use_alibi
            and attention_mask is not None
            and position_ids is None
            and token_idx is not None
        ):
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                if token_idx is not None:
                    position_ids = torch.index_select(position_ids, 1, token_idx - 1)
                else:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

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
                "reuse_cache": reuse_cache,
                "cache_idx": kwargs.get("cache_idx"),
                "use_flash_attention": kwargs.get("use_flash_attention"),
                "flash_attention_recompute": kwargs.get("flash_attention_recompute"),
                "flash_attention_causal_mask": kwargs.get("flash_attention_causal_mask"),
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        reuse_cache: Optional[bool] = False,
        trim_logits: Optional[bool] = False,
        cache_idx: int = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if use_flash_attention:
            assert FusedSDPA, "`use_flash_attention` is True, but cannot find FusedSDPA. Please import it as `from habana_frameworks.torch.hpex.kernels import FusedSDPA` or set use_flash_attention to False (at the expense of a possible performance degradation)."
        if flash_attention_recompute:
            assert use_flash_attention, "flash_attention_recompute is set, but use_flash_attention is not"
        if flash_attention_causal_mask:
            assert use_flash_attention, "flash_attention_causal_mask is set, but use_flash_attention is not"

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_idx=token_idx,
            reuse_cache=reuse_cache,
            cache_idx=cache_idx,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
        )
        hidden_states = transformer_outputs[0]

        _, seq_len, _ = hidden_states.shape
        if seq_len > 1 and trim_logits and not self.training:
            if token_idx is not None:
                hidden_states = hidden_states.index_select(1, token_idx - 1)
            else:
                hidden_states = hidden_states[:, -1:, :]

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
