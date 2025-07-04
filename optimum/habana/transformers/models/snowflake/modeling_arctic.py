# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Arctic model. Adapted from https://huggingface.co/Snowflake/snowflake-arctic-instruct/tree/be318cae5aba5291208f27d30991a5150500887d.

Changes made:
- Use HPU FusedRoPE implementation
- Use HPU FusedRMSNorm implementation
- Added mark steps
"""

import copy
import inspect
import math
import re
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import GenerationMixin
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations.deepspeed import is_deepspeed_available
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available

from ..llama.modeling_llama import (
    GaudiLlamaDynamicNTKScalingRotaryEmbedding,
    GaudiLlamaLinearScalingRotaryEmbedding,
    GaudiLlamaRotaryEmbedding,
)
from .configuration_arctic import ArcticConfig
from ..modeling_all_models import KVCache, apply_customized_rope_module

import habana_frameworks.torch.core as htcore

try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE
except ImportError:
    print("Not using HPU fused kernel for apply_rotary_pos_emb")
    FusedRoPE = None

try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm
except ImportError:
    print("Not using HPU fused kernel for RMSNorm")
    FusedRMSNorm = None

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None


if is_deepspeed_available():
    from deepspeed.moe.layer import MoE

    # Note that below will crash if there is an available deepspeed that does not have ds_linear.
    try:
        import deepspeed.linear as ds_linear
    except Exception:
        pass
else:
    MoE = None

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "ArcticConfig"
USE_DEEPSPEED_MOE_ARG = "use_deepspeed_moe_implementation"
MOE_EXPERT_PARALLEL_SIZE_ARG = "moe_expert_parallel_size"
DEEPSPEED_QUANTIZATION_CONFIG = "deepspeed_quantization"
DEEPSPEED_LORA_CONFIG = "deepspeed_lora"
QUANTIZATION_CONFIG = "ds_quantization_config"

# REQUIRED_DEEPSPEED_VERSION = "deepspeed>0.14.5"
# def is_deepspeed_valid_and_available(raise_error=False, error_msg=""):
#     available_and_valid = True
#     if not is_deepspeed_available():
#         available_and_valid = False
#         if raise_error:
#             raise ValueError(f"DeepSpeed is required for this feature, {error_msg}")
#     else:

#     return available_and_valid


def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=4, attention_mask: Optional[torch.Tensor] = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, 2, num_experts))
            .reshape(-1, 2, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Arctic
class ArcticRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        ArcticRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Modified from original ArcticRMS implementation:
        - Use Habana fused RMSNorm

        Modifications copied from ../llama/modeling_llama.py:gaudi_llama_rmsnorm_forward()
        """
        if hidden_states.device.type == "hpu" and FusedRMSNorm:
            # mixed dtypes are not good for FusedRMSNorm, both inputs need to have same dtype
            if hidden_states.dtype != self.weight.dtype:
                orig_dtype = hidden_states.dtype
                hidden_states = FusedRMSNorm.apply(
                    hidden_states.to(self.weight.dtype), self.weight, self.variance_epsilon
                )
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


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Arctic
class ArcticRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from ../llama/modeling_llama.py gaudi_llama_repeat_kv()
def repeat_kv(
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


# Copied from transformers.models.mistral.modeling_mistral.MistralAttention with Mistral->Arctic
class ArcticAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: ArcticConfig, layer_idx: Optional[int] = None, **kwargs):
        super().__init__()
        config.rope_scaling = getattr(config, "rope_scaling", None)
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.use_deepspeed_implementation = USE_DEEPSPEED_MOE_ARG in kwargs and kwargs[USE_DEEPSPEED_MOE_ARG]
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        deepspeed_lora_config = kwargs.get(DEEPSPEED_LORA_CONFIG)
        quantization_config = kwargs.get(QUANTIZATION_CONFIG, None)

        self.q_proj = get_arctic_linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            use_deepspeed_implementation=self.use_deepspeed_implementation,
            ds_optimized_lora_config=deepspeed_lora_config,
            ds_optimized_quantization_config=quantization_config,
            ds_optimized_base_weight_sharding=True,
            dtype=torch.bfloat16,
        )
        self.k_proj = get_arctic_linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            use_deepspeed_implementation=self.use_deepspeed_implementation,
            ds_optimized_lora_config=deepspeed_lora_config,
            ds_optimized_quantization_config=quantization_config,
            ds_optimized_base_weight_sharding=True,
            dtype=torch.bfloat16,
        )
        self.v_proj = get_arctic_linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            use_deepspeed_implementation=self.use_deepspeed_implementation,
            ds_optimized_lora_config=deepspeed_lora_config,
            ds_optimized_quantization_config=quantization_config,
            ds_optimized_base_weight_sharding=True,
            dtype=torch.bfloat16,
        )
        self.o_proj = get_arctic_linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            use_deepspeed_implementation=self.use_deepspeed_implementation,
            ds_optimized_lora_config=deepspeed_lora_config,
            ds_optimized_quantization_config=quantization_config,
            ds_optimized_base_weight_sharding=True,
            dtype=torch.bfloat16,
        )

        self._init_rope()

    def _init_rope(self):
        """
        Copied from: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L294
        """
        if self.config.rope_scaling is None:
            self.rotary_emb = GaudiLlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = GaudiLlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = GaudiLlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        """
        Allocate KV cache. Copied from ../mixtral/modeling_mixtral.py GaudiMixtralAttention.allocate_kv_cache
        """
        cache_shape = (batch_size, self.num_key_value_heads, max_seq_len, self.head_dim)
        device = self.k_proj.weight.device
        dtype = self.config.torch_dtype
        self.k_cache.allocate(inp_seq_len, dtype, device, cache_shape)
        self.v_cache.allocate(inp_seq_len, dtype, device, cache_shape)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        token_idx: Optional[torch.Tensor] = None,
        reuse_cache: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        cache_idx: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            if token_idx is None:
                if hasattr(past_key_value, "get_usable_length"):
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
                else:
                    kv_seq_len += past_key_value[0].shape[-2]
            else:
                if reuse_cache:
                    kv_seq_len = past_key_value[0][-2]
                else:
                    kv_seq_len = past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_customized_rope(query_states, key_states, cos, sin, position_ids, self.training)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        query_states, key_states, value_states, attention_mask = repeat_kv(
            query_states, key_states, value_states, attention_mask, self.num_key_value_groups
        )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions or FusedSDPA:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def get_arctic_linear(
    input_dim,
    output_dim,
    bias=False,
    use_deepspeed_implementation=False,
    ds_optimized_lora_config=None,
    ds_optimized_quantization_config=None,
    ds_optimized_base_weight_sharding=False,
    dtype=torch.bfloat16,
):
    """Can return deepspeed optimized linear if available.
    Args:
        input_dim, output_dim, bias, dtype: self explanatory (same as from nn.Linear)
        ds_optimized_lora_config: config of type ds_linear.LoRAConfig that contains lora specific parameter if we want to add lora to this layer.
        ds_optimized_quantization_config: config of type ds_linear.QuantizationConfig.
        ds_optimized_base_weight_sharding: bool. If true, the base weight for lora (provided ds_optimized_lora_config is not None) will be sharded across all available gpus
        in a tensor parallel way.
    """
    if is_deepspeed_available():
        if ds_optimized_lora_config is not None:
            ds_optimized_lora_config: ds_linear.LoRAConfig = copy.deepcopy(ds_optimized_lora_config)
            ds_optimized_lora_config.base_weight_sharding = (
                torch.distributed.get_world_size() if ds_optimized_base_weight_sharding else 1
            )
        return ds_linear.OptimizedLinear(
            input_dim, output_dim, bias, ds_optimized_lora_config, ds_optimized_quantization_config, dtype=dtype
        )
    return nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)


class ArcticMLP(nn.Module):
    def __init__(
        self,
        config: ArcticConfig,
        use_deepspeed_implementation=False,
        ds_optimized_lora_config=None,
        ds_optimized_quantization_config=None,
        shard_base_weights_if_doing_lora=False,
        is_residual_mlp=False,
    ):
        """MLP class for Arctic supporting vanilla linear layers as well as some deepspeed optimizations.

        ds_optimized_lora_config: config of type ds_linear.LoRAConfig that contains lora specific parameter if we want to add lora to this layer.
        ds_optimized_quantization_config: config of type ds_linear.QuantizationConfig.
        ds_optimized_base_weight_sharding: bool. If true, the base weight for lora (provided ds_optimized_lora_config is not None) will be sharded across all available gpus
        in a tensor parallel way.
        is_residual_mlp: bool. If true, this is MLP inside arctic residual layer which has ffn_dim the same as full intermediate_size.
        """
        super(ArcticMLP, self).__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size if not is_residual_mlp else self.hidden_dim
        self.w1 = get_arctic_linear(
            self.hidden_dim,
            self.ffn_dim,
            False,
            use_deepspeed_implementation=use_deepspeed_implementation,
            ds_optimized_lora_config=ds_optimized_lora_config,
            ds_optimized_quantization_config=ds_optimized_quantization_config,
            ds_optimized_base_weight_sharding=shard_base_weights_if_doing_lora,
            dtype=torch.bfloat16,
        )
        self.w2 = get_arctic_linear(
            self.ffn_dim,
            self.hidden_dim,
            False,
            use_deepspeed_implementation=use_deepspeed_implementation,
            ds_optimized_lora_config=ds_optimized_lora_config,
            ds_optimized_quantization_config=ds_optimized_quantization_config,
            ds_optimized_base_weight_sharding=shard_base_weights_if_doing_lora,
            dtype=torch.bfloat16,
        )
        self.w3 = get_arctic_linear(
            self.hidden_dim,
            self.ffn_dim,
            False,
            use_deepspeed_implementation=use_deepspeed_implementation,
            ds_optimized_lora_config=ds_optimized_lora_config,
            ds_optimized_quantization_config=ds_optimized_quantization_config,
            ds_optimized_base_weight_sharding=shard_base_weights_if_doing_lora,
            dtype=torch.bfloat16,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class ArcticMoE(nn.Module):
    def __init__(self, config: ArcticConfig, layer_id: int, **kwargs):
        super(ArcticMoE, self).__init__()

        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.layer_id = layer_id
        self.top_k = config.num_experts_per_tok
        self.is_moe_layer = (layer_id + 1) % config.moe_layer_frequency == 0

        self.use_deepspeed_implementation = USE_DEEPSPEED_MOE_ARG in kwargs and kwargs[USE_DEEPSPEED_MOE_ARG]
        if self.use_deepspeed_implementation and MoE is None:
            raise ValueError("Deepspeed is not installed")
        quantization_config = kwargs.get(QUANTIZATION_CONFIG, None)
        deepspeed_lora = kwargs.get(DEEPSPEED_LORA_CONFIG)
        if not self.is_moe_layer:  # dense, not MoE
            self.mlp = ArcticMLP(
                config,
                use_deepspeed_implementation=self.use_deepspeed_implementation,
                ds_optimized_quantization_config=quantization_config,
                ds_optimized_lora_config=deepspeed_lora,
                shard_base_weights_if_doing_lora=True,
            )
        else:
            if self.use_deepspeed_implementation:  # DeepSpeed's MoE
                moe_expert_parallel_size = kwargs.get(MOE_EXPERT_PARALLEL_SIZE_ARG, 1)
                self.mlp = MoE(
                    self.hidden_dim,
                    # base weight sharding false for all deepspeed moe calls because it is already sharded
                    ArcticMLP(
                        config,
                        use_deepspeed_implementation=True,
                        ds_optimized_quantization_config=quantization_config,
                        ds_optimized_lora_config=deepspeed_lora,
                        shard_base_weights_if_doing_lora=False,
                    ),
                    num_experts=config.num_local_experts,
                    ep_size=moe_expert_parallel_size,
                    k=config.num_experts_per_tok,
                    use_residual=False,
                    capacity_factor=config.moe_train_capacity_factor,
                    eval_capacity_factor=config.moe_eval_capacity_factor,
                    enable_expert_tensor_parallelism=config.enable_expert_tensor_parallelism,
                    min_capacity=config.moe_min_capacity,
                    drop_tokens=config.moe_token_dropping,
                )
            else:
                # "local" MoE implementation
                self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
                self.experts = nn.ModuleList(
                    [
                        ArcticMLP(
                            config,
                            use_deepspeed_implementation=self.use_deepspeed_implementation,
                            ds_optimized_quantization_config=quantization_config,
                            ds_optimized_lora_config=deepspeed_lora,
                            shard_base_weights_if_doing_lora=True,
                        )
                        for i in range(self.num_experts)
                    ]
                )

        # if torch.distributed.get_rank() == 0:
        #     deepspeed.runtime.utils.see_memory_usage("", force=True)

    # Similar in behavior to transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock.forward but more efficient.
    def _moe_foreward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size, sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        padded_weights = torch.zeros(
            (batch_size * sequence_length, self.num_experts), dtype=hidden_states.dtype, device=hidden_states.device
        )
        padded_weights.scatter_(-1, selected_experts, routing_weights)
        padded_weights = padded_weights.reshape(-1, sequence_length, self.num_experts)
        padded_weights = padded_weights.permute(2, 0, 1).unsqueeze(-1)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            padded_weight = padded_weights[expert_idx]
            current_state_static = hidden_states.reshape(-1, hidden_dim)
            current_hidden_states_static = (
                expert_layer(current_state_static).reshape(-1, sequence_length, hidden_dim) * padded_weight
            )
            final_hidden_states += current_hidden_states_static
            # support long sequences exceeding 8192
            if not self.training and sequence_length > 8192:
                htcore.mark_step()
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, load_balancing_loss_func(
            (router_logits,), self.num_experts, self.top_k
        )  # ZY: let's directly output the loss to align what we have in ds

    def forward(self, hidden_states: torch.Tensor):
        if self.is_moe_layer:
            if self.use_deepspeed_implementation:
                # deepspeed returns a tuple including output, gate loss, and expert count.
                hidden_states, moe_loss, _ = self.mlp(hidden_states)
                return hidden_states, moe_loss
            else:
                return self._moe_foreward(hidden_states)
        else:
            return self.mlp(hidden_states), torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)


class ArcticDecoderLayer(nn.Module):
    def __init__(self, config: ArcticConfig, layer_idx: int, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = ArcticAttention(config, layer_idx, **kwargs)
        self.block_sparse_moe = ArcticMoE(config, layer_id=layer_idx, **kwargs)
        self.input_layernorm = ArcticRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ArcticRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_deepspeed_implementation = USE_DEEPSPEED_MOE_ARG in kwargs and kwargs[USE_DEEPSPEED_MOE_ARG]

        self.parallel_attn_mlp_res = (
            config.parallel_attn_mlp_res and self.block_sparse_moe.is_moe_layer
        )  # add residual only when it is moe layer
        deepspeed_quantization = kwargs.get(DEEPSPEED_QUANTIZATION_CONFIG)
        deepspeed_lora = kwargs.get(DEEPSPEED_LORA_CONFIG)
        if self.parallel_attn_mlp_res:
            self.residual_layernorm = ArcticRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.residual_mlp = ArcticMLP(
                config,
                use_deepspeed_implementation=self.use_deepspeed_implementation,
                is_residual_mlp=True,
                ds_optimized_quantization_config=deepspeed_quantization,
                ds_optimized_lora_config=deepspeed_lora,
                shard_base_weights_if_doing_lora=True,
            )  # for the residual layer. always shard the base weight if doing deepspeed lora.

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.self_attn.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """

        residual_input = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual_input + hidden_states

        residual_attn = hidden_states

        if self.parallel_attn_mlp_res:
            # Note the architecture here is that the MOE layers reads the **pre-attention** input while there is a "normal" transformer residual part.
            # This is to achieve better parallelization.

            # residual mlp part

            hidden_states = self.residual_layernorm(hidden_states)
            hidden_states = self.residual_mlp(hidden_states)
            residual_residual = residual_attn + hidden_states
            # parallel mlp moe part
            hidden_states = self.post_attention_layernorm(residual_input)  # parallel attn mlp has the same input
            hidden_states, gate_loss = self.block_sparse_moe(hidden_states)
            hidden_states = residual_residual + hidden_states
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, gate_loss = self.block_sparse_moe(hidden_states)
            hidden_states = residual_attn + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (gate_loss,)

        return outputs


ARCTIC_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ArcticConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Arctic Model outputting raw hidden-states without any specific head on top.",
    ARCTIC_START_DOCSTRING,
)
# Copied from transformers.models.mistral.modeling_mistral.MistralPreTrainedModel with Mistral->Arctic
class ArcticPreTrainedModel(PreTrainedModel):
    config_class = ArcticConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ArcticDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        # if is_deepspeed_available():
        #     # TODO(rajhans): remove this once ds has init for quantizedlinear.
        #     try:
        #         from deepspeed.linear.quantization import QuantizedLinear, QuantizedParameter
        #     if isinstance(module, QuantizedLinear):
        #         weights = module.weight.dequantized()
        #         weights.normal_(mean=0.0, std=std)
        #         if module.bias is not None:
        #             module.bias.data.zero_()
        #         module.weight = QuantizedParameter(weights)
        #         module.weight.to(dtype=torch.bfloat16, device=weights.device)
        # el
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


MIXTRAL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Arctic Model outputting raw hidden-states without any specific head on top.",
    ARCTIC_START_DOCSTRING,
)
# Copied from transformers.models.mistral.modeling_mistral.MistralModel with MISTRAL->MIXTRAL,Mistral->Arctic
class ArcticModel(ArcticPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ArcticDecoderLayer`]

    Args:
        config: ArcticConfig
    """

    def __init__(self, config: ArcticConfig, **kwargs):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ArcticDecoderLayer(config, layer_idx, **kwargs) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = ArcticRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = True
        # Initialize weights and apply final processing
        self.post_init()

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        for layer in self.layers:
            layer.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Ignore copy
    @add_start_docstrings_to_model_forward(MIXTRAL_INPUTS_DOCSTRING)
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
    ) -> Union[Tuple, MoeModelOutputWithPast]:
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
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

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

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Arctic. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_losses = ()
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                if hasattr(layer_outputs[2 if output_attentions else 1], "to_legacy_cache"):
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                else:
                    if next_decoder_cache is None:
                        next_decoder_cache = [layer_outputs[2 if output_attentions else 1]]
                    else:
                        next_decoder_cache.append(layer_outputs[2 if output_attentions else 1])

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            all_router_losses += (layer_outputs[-1],)
            htcore.mark_step()
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache and hasattr(next_decoder_cache, "to_legacy_cache")
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_losses]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_losses,
        )


class ArcticForCausalLM(ArcticPreTrainedModel, GenerationMixin):
    # TODO(jeffra): update _keys_to_ignore_on_load_unexpected with expert keys not relevant for this rank
    _keys_to_ignore_on_load_unexpected = [
        r"model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.w\d+\.weight"
        r"model\.layers\.\d+\.block_sparse_moe\.gate\.weight"
    ]
    _keys_to_ignore_on_load_missing = [
        r"model\.layers\.\d+\.block_sparse_moe\.mlp\.deepspeed_moe\.experts\.deepspeed_experts\.\d+\.w\d+\.weight",
        r"model\.layers\.\d+\.block_sparse_moe\.mlp\.deepspeed_moe\.gate\.wg\.weight",
    ]
    _tied_weights_keys = []  # ["lm_head.weight"]

    def __init__(self, config: ArcticConfig, **kwargs):
        super().__init__(config)
        self.model = ArcticModel(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.use_deepspeed_moe = kwargs.get(USE_DEEPSPEED_MOE_ARG, False)
        self.moe_expert_parallel_size = kwargs.get(MOE_EXPERT_PARALLEL_SIZE_ARG, 1)
        self.is_deepspeed_lora = kwargs.get(DEEPSPEED_LORA_CONFIG) is not None
        self.gradient_checkpointing = True
        # self.shard_base_weights_if_doing_lora = kwargs.get("shard_base_weights_if_doing_lora", False)
        # Initialize weights and apply final processing
        self.post_init()

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.model.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)
        self.kv_cache_len = max_seq_len

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _expert_number_from_param_name(self, param_name):
        # example param_name: model.layers.1.block_sparse_moe.experts.10.w1.weight
        pattern = r"experts\.(\d+)\."
        m = re.search(pattern, param_name)
        if m:
            return int(m[1])
        else:
            return None

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)

        if not self.use_deepspeed_moe:
            return state_dict

        # when trying to construct the deepspeed checkpoint we don't want to gather everything
        if not getattr(self, "_gather_expert_params", False):
            return state_dict

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        # non-lora experts
        pattern = (
            r"model\.layers\.\d+\.block_sparse_moe\.mlp\.deepspeed_moe\.experts\.deepspeed_experts\.\d+\.w\d+\.weight"
        )
        expert_params = [s for s in state_dict.keys() if re.search(pattern, s)]

        for param_name in expert_params:
            param_tensor = state_dict[param_name].to("cuda")
            output = [torch.zeros_like(param_tensor) for _ in range(world_size)]
            torch.distributed.gather(param_tensor, gather_list=output if rank == 0 else None, dst=0, group=None)
            # rename from local rank to global rank
            for gather_rank, gather_param in enumerate(output):
                experts_per_rank = self.num_experts // self.moe_expert_parallel_size
                new_expert_number = gather_rank * experts_per_rank + self._expert_number_from_param_name(param_name)
                new_param_name = re.sub(r"(experts\.)(\d+)(\.)", rf"\g<1>{new_expert_number}\3", param_name)
                state_dict[new_param_name] = gather_param
                if rank == 0:
                    print(f"adding to state_dict and renaming: {param_name} -> {new_param_name}")

        # Handle custom LoRA implementation
        # TODO(rajhans): the part below is untested and shows up when doing lora training. Should not affect inference.
        if self.is_deepspeed_lora:
            for param_name in list(
                state_dict.keys()
            ):  # Use list to avoid RuntimeError due to changing size during iteration
                if param_name.endswith("base_weight"):
                    base_weight = state_dict[param_name].to("cuda")

                    # If the base weight is sharded, gather weights from multiple ranks and concatenate
                    # except if the weights are from deespeed_moe which is not sharded (due to EP).
                    if (
                        self.shard_base_weights_if_doing_lora
                        and "deepspeed_moe.experts.deepspeed_experts" not in param_name
                    ):
                        gathered_weights = [
                            torch.zeros_like(base_weight, device=base_weight.device, dtype=base_weight.dtype)
                            for _ in range(world_size)
                        ]
                        torch.distributed.gather(
                            base_weight, gather_list=gathered_weights if rank == 0 else None, dst=0, group=None
                        )
                        base_weight = torch.cat(gathered_weights, dim=1)

                    ## The part below is useful if we want to output HF transformer path weights, but commenting it for now
                    # Merge the LoRA weights into the base weights
                    # lora_weight_1 = state_dict.get(param_name.replace("base_weight", "lora_weight_1.weight"))
                    # lora_weight_2 = state_dict.get(param_name.replace("base_weight", "lora_weight_2.weight"))
                    # if lora_weight_1 is not None and lora_weight_2 is not None:
                    #     lora_weights = torch.matmul(lora_weight_2, lora_weight_1)
                    #     base_weight += lora_weights
                    # else:
                    #     raise ValueError

                    # # Rename the base weight to weight
                    # new_param_name = param_name.replace("base_weight", "weight")
                    # state_dict[new_param_name] = base_weight

                    # Remove the base weight from the state dict
                    # del state_dict[param_name]
        return state_dict

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if not self.use_deepspeed_moe:
            return super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        # TODO(jeffra): currently assumes fine-tuning only on one node, fix for world_size != ep size
        if self.moe_expert_parallel_size > 1:
            assert (
                self.moe_expert_parallel_size == world_size
            ), f"currently only support expert parallel size equal to world size but {self.moe_expert_parallel_size=} and {world_size=}"

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        num_local_experts = self.num_experts // self.moe_expert_parallel_size
        local_expert_range = range(num_local_experts * rank, num_local_experts * rank + num_local_experts)

        # no deepspeed
        #   model.layers.1.block_sparse_moe.experts.10.w1.weight
        #   model.layers.1.block_sparse_moe.gate.weight
        # w. deepspeed
        #   model.layers.1.block_sparse_moe.mlp.deepspeed_moe.gate.wg.weight
        #   model.layers.1.block_sparse_moe.mlp.deepspeed_moe.experts.deepspeed_experts.10.w1.weight

        gate_pattern = r"model\.layers\.\d+\.block_sparse_moe\.gate\.weight"

        expert_params_to_keep = []
        expert_params_to_remove = []
        gate_params = []
        for param_name in state_dict.keys():
            expert_number = self._expert_number_from_param_name(param_name)
            if expert_number is not None:
                if expert_number in local_expert_range:
                    expert_params_to_keep.append(param_name)
                else:
                    expert_params_to_remove.append(param_name)
            elif re.search(gate_pattern, param_name):
                gate_params.append(param_name)

        # drop all experts in the state_dict that we don't need locally
        for param_name in expert_params_to_remove:
            print(f"{rank=} dropping {param_name}")
            del state_dict[param_name]

        # rename remaining experts to align with the local config
        for param_name in expert_params_to_keep:
            # adjust expert number wrt expert parallelism
            new_expert_number = self._expert_number_from_param_name(param_name) % num_local_experts
            new_param_name = re.sub(r"(experts\.)(\d+)(\.)", rf"\g<1>{new_expert_number}\3", param_name)

            # use deepspeed moe param path
            split_param_name = new_param_name.split(".")
            idx = split_param_name.index("experts")
            ds_moe_path = "mlp.deepspeed_moe.experts.deepspeed_experts".split(".")
            new_param_name = split_param_name[0:idx] + ds_moe_path + split_param_name[idx + 1 :]
            new_param_name = ".".join(new_param_name)

            print(f"Deepspeed {rank=}, renaming {param_name} -> {new_param_name}")
            state_dict[new_param_name] = state_dict.pop(param_name)

        # rename gate params
        ds_suffix = "mlp.deepspeed_moe.gate.wg.weight".split(".")
        for param_name in gate_params:
            new_param_name = ".".join(param_name.split(".")[:4] + ds_suffix)
            print(f"Gating: {rank=}, renaming {param_name} -> {new_param_name}")
            state_dict[new_param_name] = state_dict.pop(param_name)

        # If deepspeed lora is enabled, then we need to rename weight to base_weight.
        # Furthermore, if the base_weight is sharded, we need to shard each weight and select the slice of local rank.
        if self.is_deepspeed_lora:
            local_state_dict = self.state_dict()
            for param_name in local_state_dict:
                if not param_name.endswith("base_weight"):
                    continue

                incoming_param_name = param_name.replace("base_weight", "weight")
                if incoming_param_name not in state_dict:
                    continue

                incoming_param = state_dict[incoming_param_name]

                shape_local = local_state_dict[param_name].shape
                shape_incoming = incoming_param.shape
                if "deepspeed_moe" in incoming_param_name:
                    assert shape_local == shape_incoming, "deepspeed moe weights are never sharded"
                else:
                    assert (
                        shape_incoming[1] == shape_local[1] * world_size
                    ), "weights should be sharded equally across world size"
                    incoming_param = incoming_param[:, rank * shape_local[1] : (rank + 1) * shape_local[1]]
                print(f"Deepspeed lora: {rank=}, renaming {incoming_param_name} -> {param_name}")
                state_dict[param_name] = incoming_param
                del state_dict[incoming_param_name]

        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @add_start_docstrings_to_model_forward(MIXTRAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # Ignore copy
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
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, ArcticForCausalLM

        >>> model = ArcticForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

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
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # Move to same device for model parallelism.
        aux_loss = sum([out.to(logits.device) for out in outputs[-1]])
        if labels is not None:
            loss += self.router_aux_loss_coef * aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            # torch.distributed.barrier()
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

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
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The Arctic Model transformer with a sequence classification head on top (linear layer).

    [`ArcticForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    ARCTIC_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with Llama->Arctic, LLAMA->MIXTRAL
class ArcticForSequenceClassification(ArcticPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = ArcticModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(MIXTRAL_INPUTS_DOCSTRING)
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
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

# Copied from optimum.habana.transformers.models.llama.modeling_llama:apply_customized_rope()
def apply_customized_rope(q, k, cos, sin, position_ids, training=True):
    if q.device.type == "hpu" and FusedRoPE:
        return apply_customized_rope_module(q, k, cos, sin, position_ids, training)
    else:
        # keep the same implementation as Transformers v4.37.2
        return apply_rotary_pos_emb(q, k, cos[position_ids], sin[position_ids], position_ids)
