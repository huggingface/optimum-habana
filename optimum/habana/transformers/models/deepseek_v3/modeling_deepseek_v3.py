# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
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
"""
PyTorch DeepSeekV3 model. Adapted from https://huggingface.co/deepseek-ai/DeepSeek-R1/resolve/main/modeling_deepseek.py

The main differences are:
- Use Gaudi Flash Attention
- Optimized KV cache with support for static shapes
- Use fused Gaudi MoE, RoPE, and RMSNorm operators
- Enable expert parallelism
"""

import math
import warnings
from typing import List, Optional, Tuple, Union

import habana_frameworks.torch.core as htcore
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.integrations.deepspeed import is_deepspeed_available
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from ....distributed.tensorparallel import _all_reduce
from ...modeling_attn_mask_utils import _gaudi_prepare_4d_causal_attention_mask
from ..modeling_all_models import apply_customized_rope_module
from .configuration_deepseek_v3 import DeepseekV3Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeepseekV3Config"

# Maximum number of experts supported by dynamic MoE op (mixture_of_experts)
SLICE_MAX_EXPERT = 80

# import hpu fused ops
try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE

    print("Using HPU fused kernel for apply_rotary_pos_emb")
except ImportError:
    print("Not using HPU fused kernel for apply_rotary_pos_emb")
    FusedRoPE = None

try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm

    print("Using HPU fused kernel for RMSNorm")
except ImportError:
    print("Not using HPU fused kernel for RMSNorm")
    FusedRMSNorm = None

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None


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


class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if hidden_states.device.type == "hpu" and FusedRMSNorm:
            # use hpu fused rmsnorm
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


ALL_LAYERNORM_LAYERS.append(DeepseekV3RMSNorm)


class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.

        # make it static (max_position_embeddings) instead of updating depending on
        # longest seq_len seen till now: seq_len > self.max_seq_len_cached
        self.max_seq_len_cached = max_position_embeddings
        self._set_cos_sin_cache(
            seq_len=self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->DeepseekV3
class DeepseekV3LinearScalingRotaryEmbedding(DeepseekV3RotaryEmbedding):
    """DeepseekV3RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->DeepseekV3
class DeepseekV3DynamicNTKScalingRotaryEmbedding(DeepseekV3RotaryEmbedding):
    """DeepseekV3RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


# Find dim range bounds based on rotations
def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV3YarnRotaryEmbedding(DeepseekV3RotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False)


def apply_customized_rope(q, k, cos, sin, position_ids, training=True):
    if q.device.type == "hpu" and FusedRoPE:  # use fused hpu op
        return apply_customized_rope_module(q, k, cos, sin, position_ids, training)
    else:
        return apply_rotary_pos_emb(q, k, cos, sin, position_ids)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q: torch.Tensor, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
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

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    if q.device.type == "hpu" and FusedRoPE:
        return FusedRoPE.apply(
            q, cos.unsqueeze(0).unsqueeze(0).clone(), sin.unsqueeze(0).unsqueeze(0).clone(), position_ids
        )
    else:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        return q_embed


class DeepseekV3MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(torch.empty((self.n_routed_experts)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        ### select top-k experts
        if self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
            )  # [n, n_group]
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(f"insupportable TopK function for MoE gating: {self.topk_method}")

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor  # must multiply the scaling factor

        return topk_idx, topk_weight


class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        if hasattr(config, "ep_size") and config.ep_size > 1:
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.ModuleList(
                [
                    (
                        DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                        if i >= self.ep_rank * self.experts_per_rank and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            self.experts = nn.ModuleList(
                [
                    DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                    for i in range(config.n_routed_experts)
                ]
            )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV3MLP(config=config, intermediate_size=intermediate_size)

        # Slice experts for max experts supported by fused dynamic mixture_of_experts op
        self.expert_slice = math.ceil(self.experts_per_rank / SLICE_MAX_EXPERT)
        self.expert_chunk = math.ceil(self.experts_per_rank / self.expert_slice)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        # we cast back to the input dtype
        topk_weight = topk_weight.to(hidden_states.dtype)
        batch = orig_shape[0]
        sequence_length = orig_shape[1]
        hidden_dim = orig_shape[2]
        # changes for expert parallelism -- replacement for moe_infer()
        if self.training:
            padded_weights = torch.zeros(
                (batch * sequence_length, self.config.n_routed_experts),
                dtype=topk_weight.dtype,
                device=topk_weight.device,
            )
            padded_weights.scatter_(-1, topk_idx, topk_weight)
            padded_weights = padded_weights.reshape(-1, sequence_length, self.config.n_routed_experts)
            padded_weights = padded_weights.permute(2, 0, 1).unsqueeze(-1)

            final_hidden_states = torch.zeros(
                (batch, sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
            for i, expert in enumerate(self.experts):
                current_hidden_state = expert(hidden_states)
                current_padded_weight = padded_weights[i]
                final_hidden_states = (
                    final_hidden_states
                    + current_hidden_state.reshape(-1, sequence_length, hidden_dim) * current_padded_weight
                )
            final_hidden_states = final_hidden_states.type(hidden_states.dtype)
            final_hidden_states = final_hidden_states.view(*orig_shape)
            # final_hidden_states = AddAuxiliaryLoss.apply(final_hidden_states, aux_loss)
        else:
            final_hidden_states = torch.zeros(
                (batch * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
            # changes to support hpu fused dynamic MoE op -- replacement for moe_infer()
            # loop through expert slices due to limits on max. experts supported by mixture_of_experts op
            for idx in range(self.expert_slice):
                experts_min = (self.ep_rank * self.experts_per_rank) + (self.expert_chunk * idx)
                experts_max = min((experts_min + self.expert_chunk), (self.ep_rank + 1) * self.experts_per_rank)
                experts_range = range(experts_min, experts_max)
                gate_proj_list = [self.experts[i].gate_proj.weight.squeeze() for i in experts_range]
                down_proj_list = [self.experts[i].down_proj.weight.squeeze() for i in experts_range]
                up_proj_list = [self.experts[i].up_proj.weight.squeeze() for i in experts_range]

                hidden_states_slice = torch.ops.hpu.mixture_of_experts(
                    hidden_states=hidden_states,
                    expert_routing_table=topk_idx,
                    router_weights=topk_weight,
                    w1=gate_proj_list,
                    w2=up_proj_list,
                    w3=down_proj_list,
                    permuted_weights=True,
                    activation="silu",
                    experts_min=experts_min,
                    experts_max=experts_max - 1,
                )
                final_hidden_states = final_hidden_states + hidden_states_slice
                htcore.mark_step()

            if self.ep_size > 1:
                final_hidden_states = _all_reduce(final_hidden_states)
            elif is_deepspeed_available():
                from deepspeed import comm as dist

                if dist.is_initialized():
                    dist.all_reduce(final_hidden_states, op=dist.ReduceOp.SUM)

            final_hidden_states = final_hidden_states.type(hidden_states.dtype)
            final_hidden_states = final_hidden_states.reshape(-1, sequence_length, hidden_dim)

        if self.config.n_shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(identity)

        return final_hidden_states


# Functional apis need to be wrapped in classes for quantization on hpu
class Matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


def gaudi_deepseekv3_repeat_kv(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    n_rep: int,
):
    """
    Copied from repeat_kv: https://github.com/huggingface/transformers/blob/v4.37.0/src/transformers/models/mixtral/modeling_mixtral.py
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

    batch, q_heads, q_len, head_dim = query_states.shape
    new_q_shape = (batch, num_key_value_heads, n_rep, q_len, head_dim)
    query_states = query_states.reshape(new_q_shape)

    if attention_mask is not None:
        # Add groups dim and set to 1
        attention_mask = attention_mask.unsqueeze(1)

    return query_states, key_states, value_states, attention_mask


# hpu specific. kv cache handling. similar to optimum-habana deepseek_v2
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
            assert self.inp_seq_len == inp_seq_len, (
                f"inp_seq_len must be the same. self.inp_seq_len:{self.inp_seq_len} inp_seq_len:{inp_seq_len}"
            )
            self.cache.fill_(0)

    def update(self, prev, cur, dim, idx, inp_seq_len):
        orig_cur = cur
        if prev.shape == cur.shape:
            prev.copy_(cur)
            return orig_cur
        if cur.shape[1] > 1 and cur.shape[1] <= prev.shape[1]:
            # Initialize
            prev[:, :inp_seq_len, :].copy_(cur)
            return orig_cur
        assert cur.shape[1] == 1, f"Cannot update kv-cache. Unsupported shapes. prev:{prev.shape} cur:{cur.shape}"

        if idx is not None:
            prev.index_copy_(dim, idx - 1, cur)
            return prev
        else:
            return torch.cat((prev, cur), dim=dim)

    def get_shape(self):
        if self.cache is None:
            return None
        return self.cache.shape

    def forward(self, cur, dim, idx):
        return self.update(self.cache, cur, dim, idx, self.inp_seq_len)


# hpu specific fused op. wrapped in a class as functional apis not supported for quantization
class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA, scale, attention_dropout, enable_recompute, flash_attention_fp8):
        super().__init__()
        self._hpu_kernel_fsdpa = fusedSDPA
        self.scale = scale
        self.attention_dropout = attention_dropout
        self.enable_recompute = enable_recompute
        self.flash_attention_fp8 = flash_attention_fp8

    def forward(
        self,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_casual,
        scale,
        softmax_mode,
        recompute_mode,
        valid_sequence_lengths,
        padding_side="left",
    ):
        return self._hpu_kernel_fsdpa.apply(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_casual,
            scale,
            softmax_mode,
            recompute_mode,
            valid_sequence_lengths,
            padding_side,
        )


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV3
class DeepseekV3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.num_key_value_groups = self.num_heads // config.num_key_value_heads
        # hpu specific wrapping functional api into nn.module classes for quantization
        self.matmul_qk = Matmul()
        self.matmul_av = Matmul()
        self.k_cache = KVCache()
        self.v_cache = KVCache()
        self.inp_seq_len = -1

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.norm_factor = self.softmax_scale
        # hpu specific warpping functional api into nn.module classes for quantization
        self.fused_scaled_dot_product_attention = (
            ModuleFusedSDPA(
                FusedSDPA,
                scale=self.norm_factor,
                attention_dropout=self.attention_dropout,
                enable_recompute=False,
                flash_attention_fp8=getattr(config, "flash_attention_fp8", False),
            )
            if FusedSDPA
            else None
        )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV3RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV3LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV3DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    # hpu-specific, similar to other model files in OH
    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        compressed_kv_cache_shape = (batch_size, max_seq_len, self.kv_lora_rank)
        k_pe_cache_shape = (batch_size, max_seq_len, self.qk_rope_head_dim)
        device = self.kv_a_proj_with_mqa.weight.device
        dtype = self.config.torch_dtype

        self.k_cache.allocate(inp_seq_len, dtype, device, compressed_kv_cache_shape)
        self.v_cache.allocate(inp_seq_len, dtype, device, k_pe_cache_shape)

    def update_sincos_cache(self, seq_len):
        # Call rotary emb forward() to update cos/sin cache when inferring more than self.max_position_embeddings
        # This helps in avoiding creation of these caches during actual model forward pass and
        # reduce memory consumption and improve performance.
        if seq_len > self.max_position_embeddings:
            self.max_position_embeddings = seq_len
            _, _ = self.rotary_emb(self.k_b_proj.weight, seq_len=seq_len)

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

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2).contiguous()

    def split_kv_b_proj(self):
        kv_b_proj_weight = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
        self.q_absorb = kv_b_proj_weight[:, : self.qk_nope_head_dim, :].unsqueeze(0).transpose(0, 1)
        self.out_absorb = kv_b_proj_weight[:, self.qk_nope_head_dim :, :].unsqueeze(0)

    def compress_kv(
        self,
        hidden_states_kv: torch.Tensor,
        kv_position_ids: torch.LongTensor,
        past_key_value: Optional[Cache] = None,
    ) -> torch.Tensor:
        # return the RoPE'ed & compressed kv
        bsz, kv_seq_len, _ = hidden_states_kv.size()
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states_kv)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, kv_seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb.cos_cached, self.rotary_emb.sin_cached
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, kv_position_ids).view(bsz, kv_seq_len, self.qk_rope_head_dim)
        return compressed_kv, k_pe

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
        cache_idx: int = None,
        cache_position: Optional[torch.LongTensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        valid_sequence_lengths: Optional[torch.Tensor] = None,
        num_virtual_tokens: int = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Attention masks and past cache are removed.
        Input:
        - hidden_states: [bsz, q_len, hidden_size]
        - position_ids: [bsz, q_len]
        """

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        if self.training:
            if "padding_mask" in kwargs:
                warnings.warn(
                    "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
                )
            bsz, q_len, _ = hidden_states.size()
            if self.q_lora_rank is None:
                q = self.q_proj(hidden_states)
            else:
                q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
            q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
            q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

            compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
            compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
            kv = (
                self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
                .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
                .transpose(1, 2)
            )

            k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            kv_seq_len = value_states.shape[-2]
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
                    if num_virtual_tokens is not None and num_virtual_tokens == past_key_value[0].shape[-2]:
                        kv_seq_len = past_key_value[0].shape[-2] + kv_seq_len
                    else:
                        kv_seq_len = past_key_value[0].shape[-2]

            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            q_pe, k_pe = apply_customized_rope(q_pe, k_pe, cos, sin, position_ids, self.training)

            query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
            query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
            query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

            key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
            key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
            key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            # hpu specific optimization, similar to other modeling files in optimum-habana
            if use_flash_attention and FusedSDPA is not None:
                if q_len == 1:
                    # next token
                    attn_output = self.fused_scaled_dot_product_attention(
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        0.0,
                        False,
                        None,
                        "None",
                        False,
                        None,
                        "None",
                    )
                else:
                    # first token
                    softmax_mode = "fast" if flash_attention_fast_softmax else "None"
                    if flash_attention_causal_mask:
                        attn_output = self.fused_scaled_dot_product_attention(
                            query_states,
                            key_states,
                            value_states,
                            None,
                            0.0,
                            True,
                            None,
                            softmax_mode,
                            flash_attention_recompute,
                            valid_sequence_lengths,
                            "left",
                        )
                    else:
                        attn_output = self.fused_scaled_dot_product_attention(
                            query_states,
                            key_states,
                            value_states,
                            attention_mask,
                            0.0,
                            False,
                            None,
                            softmax_mode,
                            flash_attention_recompute,
                            None,
                            "None",
                        )

            else:
                query_states, key_states, value_states, attention_mask = gaudi_deepseekv3_repeat_kv(
                    query_states, key_states, value_states, attention_mask, self.num_key_value_groups
                )

                attn_weights = self.matmul_qk(query_states, key_states.transpose(-2, -1)) * self.softmax_scale
                htcore.mark_step()

                if attention_mask is not None:  # no matter the length, we just slice it
                    causal_mask = attention_mask
                    if cache_position is not None:
                        causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
                    attn_weights = attn_weights + causal_mask.float()

                if attn_softmax_bf16:
                    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=query_states.dtype)
                else:
                    # upcast attention to fp32
                    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                        query_states.dtype
                    )
                attn_weights = torch.nn.functional.dropout(
                    attn_weights, p=self.attention_dropout, training=self.training
                )
                attn_output = self.matmul_av(attn_weights, value_states)
        else:
            # inference
            hidden_states_q = hidden_states
            hidden_states_kv = hidden_states
            self.split_kv_b_proj()
            q_position_ids = position_ids
            kv_position_ids = position_ids
            bsz, q_len, _ = hidden_states_q.size()

            if self.q_lora_rank is None:
                q = self.q_proj(hidden_states_q)
            else:
                q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states_q)))

            q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

            q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

            kv_seq_len = q_pe.shape[-2]

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

            cos, sin = self.rotary_emb(q_pe, seq_len=kv_seq_len)
            q_pe = apply_rotary_pos_emb(q_pe, cos, sin, q_position_ids)
            q_nope = torch.matmul(q_nope.transpose(0, 1), self.q_absorb).transpose(0, 1)
            compressed_kv, k_pe = self.compress_kv(hidden_states_kv, kv_position_ids)

            # update & get all compressed_kv, k_pe
            if use_cache:
                if reuse_cache:
                    if past_key_value is not None and isinstance(past_key_value[0], torch.Tensor):
                        # prefix tuning case. attach past_key_value to generate first token.
                        compressed_kv = torch.cat((past_key_value[0], compressed_kv), -2)
                        k_pe = torch.cat((past_key_value[1], k_pe), -2)

                    compressed_kv = self.k_cache(compressed_kv, 1, token_idx)

                    k_pe = self.v_cache(k_pe, 1, token_idx)
                    past_key_value = (self.k_cache.get_shape(), self.v_cache.get_shape())

                else:
                    if past_key_value is None:
                        dtype_1 = hidden_states.dtype
                        device_1 = hidden_states.device
                        past_key = torch.zeros(compressed_kv.shape, dtype=dtype_1, device=device_1)
                        past_value = torch.zeros(k_pe.shape, dtype=dtype_1, device=device_1)
                        past_key_value = (past_key, past_value)
                    compressed_kv = self.k_cache.update(
                        past_key_value[0], compressed_kv, 1, token_idx, self.inp_seq_len
                    )
                    k_pe = self.v_cache.update(past_key_value[1], k_pe, 1, token_idx, self.inp_seq_len)

                    if token_idx is None:
                        past_key_value = (compressed_kv, k_pe)

                if cache_idx is not None and q_len == 1:
                    compressed_kv = compressed_kv[:, :cache_idx, :]

                    k_pe = k_pe[:, :cache_idx, :]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, :, :, :cache_idx]

                    kv_seq_len = compressed_kv.shape[-2]
            else:
                past_key_value = None

            kv_seq_len = compressed_kv.size(1)

            k_pe = k_pe.view(bsz, 1, kv_seq_len, self.qk_rope_head_dim)

            attn_weights = (
                torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT)
            ) * self.softmax_scale

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )
            # Commenting below line as MMLU tasks are failing with this assertion
            # assert attention_mask is not None
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_nope.dtype)

            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.einsum("bhql,blc->bhqc", attn_weights, compressed_kv)

            attn_output = torch.matmul(attn_output.permute(2, 1, 0, 3), self.out_absorb.mT).permute(2, 1, 0, 3)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekV3Attention(config=config, layer_idx=layer_idx)

        self.mlp = (
            DeepseekV3MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV3MLP(config)
        )
        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.self_attn.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

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
        reuse_cache: Optional[bool] = False,
        cache_idx: int = None,
        cache_position: Optional[torch.LongTensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        valid_sequence_lengths: Optional[torch.Tensor] = None,
        num_virtual_tokens: int = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            token_idx=token_idx,
            reuse_cache=reuse_cache,
            cache_idx=cache_idx,
            cache_position=cache_position,
            attn_softmax_bf16=attn_softmax_bf16,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
            flash_attention_fast_softmax=flash_attention_fast_softmax,
            valid_sequence_lengths=valid_sequence_lengths,
            num_virtual_tokens=num_virtual_tokens,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


DeepseekV3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DeepseekV3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare DeepseekV3 Model outputting raw hidden-states without any specific head on top.",
    DeepseekV3_START_DOCSTRING,
)
class DeepseekV3PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


DeepseekV3_INPUTS_DOCSTRING = r"""
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

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
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
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
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
    "The bare DeepseekV3 Model outputting raw hidden-states without any specific head on top.",
    DeepseekV3_START_DOCSTRING,
)
class DeepseekV3Model(DeepseekV3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV3DecoderLayer`]

    Args:
        config: DeepseekV3Config
    """

    def __init__(self, config: DeepseekV3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DeepseekV3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = "eager"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        for layer in self.layers:
            layer.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return tuple(layer.reorder_kv_cache(beam_idx) for layer in self.layers)

    def update_sincos_cache(self, seq_len):
        for layer in self.layers:
            layer.update_sincos_cache(seq_len)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(DeepseekV3_INPUTS_DOCSTRING)
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
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        cache_idx: int = None,
        lazy_mode: Optional[bool] = True,
        valid_sequence_lengths: Optional[torch.Tensor] = None,
        num_virtual_tokens: int = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
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
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 4d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        if attention_mask is not None:
            attention_mask = _gaudi_prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        if lazy_mode:
            htcore.mark_step()

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if (
                lazy_mode
                and not self.training
                and (torch.distributed.is_initialized() is False or torch.distributed.get_world_size() == 1)
            ):
                htcore.mark_step()
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                token_idx=token_idx,
            )
            if (
                lazy_mode
                and not self.training
                and (torch.distributed.is_initialized() is False or torch.distributed.get_world_size() == 1)
            ):
                htcore.mark_step()
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


class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

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

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.model.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)
        self.kv_cache_len = max_seq_len

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.model.reorder_kv_cache(beam_idx)

    def update_sincos_cache(self, seq_len):
        self.model.update_sincos_cache(seq_len)

    @add_start_docstrings_to_model_forward(DeepseekV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV3ForCausalLM

        >>> model = DeepseekV3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
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
            token_idx=token_idx,
        )

        hidden_states = outputs[0]
        _, seq_len, _ = hidden_states.shape
        if seq_len > 1 and trim_logits and not self.training:
            if token_idx is not None:
                hidden_states = hidden_states.index_select(1, token_idx - 1)
            else:
                hidden_states = hidden_states[:, -1, :]
        logits = self.lm_head(hidden_states)

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
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        token_idx = kwargs.get("token_idx")
        past_length = 0
        max_cache_length = None
        if past_key_values is not None:
            if token_idx is not None:
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)
            else:
                if isinstance(past_key_values, Cache):
                    cache_length = past_key_values.get_seq_length()
                    past_length = past_key_values.seen_tokens
                    max_cache_length = past_key_values.get_max_length()
                else:
                    cache_length = past_length = past_key_values[0][0].shape[2]
                    max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
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
                if token_idx is not None:
                    position_ids = torch.index_select(position_ids, 1, token_idx - 1)
                else:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "token_idx": token_idx,
                "trim_logits": kwargs.get("trim_logits"),
            }
        )
        return model_inputs


@add_start_docstrings(
    """
    The DeepseekV3 Model transformer with a sequence classification head on top (linear layer).

    [`DeepseekV3ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    DeepseekV3_START_DOCSTRING,
)
class DeepseekV3ForSequenceClassification(DeepseekV3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = DeepseekV3Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(DeepseekV3_INPUTS_DOCSTRING)
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
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, transformers.,
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
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
                    logits.device
                )
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
