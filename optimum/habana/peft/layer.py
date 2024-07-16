import inspect
import math
from typing import Any

import torch
import torch.nn.functional as F
from peft.tuners.adaption_prompt.config import TRANSFORMERS_MODEL_CONFIG
from peft.tuners.adaption_prompt.utils import llama_apply_rotary_pos_emb, llama_rotate_half


def GaudiAdaloraLayerSVDLinearForward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """
    Copied from SVDLinear.forward: https://github.com/huggingface/peft/blob/v0.9.0/src/peft/tuners/adalora/layer.py#L158
    The only differences are:
    - fix batch_gemm failure for BF16 case
    """
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            lora_E = self.lora_E[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            ranknum = self.ranknum[active_adapter] + 1e-5

            x = x.to(lora_A.dtype)
            result += (dropout(x) @ (lora_A * lora_E).T @ lora_B.T) * (scaling / ranknum)

    return result


def GaudiPolyLayerLinearForward(
    self, x: torch.Tensor, *args: Any, task_ids: torch.Tensor = None, **kwargs: Any
) -> torch.Tensor:
    """
    Copied from Linear.forward: https://github.com/huggingface/peft/blob/v0.10.0/src/peft/tuners/poly/layer.py#L135
    The only differences are:
    - /r equal to *(1.0/r). /r makes batch_gemm BF16 failure
    """
    previous_dtype = x.dtype
    if self.disable_adapters:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.poly_lora_A.keys():
                continue

            r = self.r[active_adapter]
            poly_router = self.poly_router[active_adapter]
            poly_lora_A = self.poly_lora_A[active_adapter]
            poly_lora_B = self.poly_lora_B[active_adapter]

            # Combine the output of LoRAs
            # https://github.com/microsoft/mttl/blob/ce4ca51dbca73be656feb9b3e5233633e3c5dec7/mttl/models/poly.py#L293
            mixing_weights = poly_router(task_ids=task_ids, input_ids=x)
            bs, n_splits, n_skills = mixing_weights.size()

            # A is    n_splits, n_skills, D // n_splits, rank
            # we want bs,       n_splits, D // n_splits, rank
            A = torch.einsum("bqs,qsdr->bqdr", (mixing_weights, poly_lora_A))
            B = torch.einsum("bqs,qsrd->bqrd", (mixing_weights, poly_lora_B))

            A = A.reshape(bs, self.in_features, r)
            B = B.transpose(1, 2).reshape(bs, r, self.out_features)

            x = x.to(A.dtype)
            result += x.bmm(A).bmm(B) * (1.0 / r)

    result = result.to(previous_dtype)
    return result


def compute_query_states(model: torch.nn.Module, **kwargs) -> torch.Tensor:
    """
    Copied from https://github.com/huggingface/peft/blob/v0.10.0/src/peft/tuners/adaption_prompt/utils.py#L60
    The only differences are:
    -add reuse cache support.
    -add past key value list support
    """
    hidden_states = kwargs.get("hidden_states")
    position_ids = kwargs.get("position_ids")
    past_key_value = kwargs.get("past_key_value")
    bsz, q_len, _ = hidden_states.size()
    query_states = model.q_proj(hidden_states).view(bsz, q_len, model.num_heads, model.head_dim).transpose(1, 2)

    factor = model.k_proj.in_features // model.k_proj.out_features
    value_states = (
        model.v_proj(hidden_states).view(bsz, q_len, (model.num_heads // factor), model.head_dim).transpose(1, 2)
    )

    seq_len = q_len

    if past_key_value is not None:
        if kwargs.get("reuse_cache", False):
            seq_len += past_key_value[0][-2]
        elif isinstance(past_key_value, tuple) or isinstance(past_key_value, list):
            # for transformers <= 4.35
            seq_len += past_key_value[0].shape[-2]
        else:
            # since transformers 4.36, this is a DynamicCache instance
            seq_len += past_key_value.get_seq_length(model.layer_idx)

    # For transformers > 4.37.2 `position_ids` became a required arguments in the rotary embedding's forward pass.
    if "position_ids" not in inspect.signature(model.rotary_emb.forward).parameters:
        # TODO we assume that position_ids is not None here, not sure if that is safe but the old code also did that
        cos, sin = model.rotary_emb(value_states, seq_len=seq_len)
        return llama_apply_rotary_pos_emb(query_states, cos, sin, position_ids)

    past_seen_tokens = 0
    if position_ids is None:
        # Compute position_ids, since they are required for transformers > 4.37.2
        if past_key_value is None:
            new_cache_positions = torch.arange(q_len, q_len + q_len, device=value_states.device)
        else:
            past_seen_tokens = past_key_value.get_usable_length(q_len, model.layer_idx)
            new_cache_positions = torch.arange(past_seen_tokens, past_seen_tokens + q_len, device=value_states.device)
        position_ids = new_cache_positions.unsqueeze(0)

    rotary_emb_kwargs = {"position_ids": position_ids}
    # The `seq_len` argument has been officially removed in transformers >= 4.39.0
    if "seq_len" in inspect.signature(model.rotary_emb.forward).parameters:
        rotary_emb_kwargs["seq_len"] = q_len + past_seen_tokens

    cos, sin = model.rotary_emb(value_states, **rotary_emb_kwargs)

    # For batched inference unsqueeze it on the correct dim
    # since: https://github.com/huggingface/transformers/pull/29109
    if len(cos.shape) == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    return (query_states * cos) + (llama_rotate_half(query_states) * sin)


def GaudiAdaptedAttentionPreAttnForward(self, *args, **kwargs):
    """
    Copied from AdaptedAttention.forward: https://github.com/huggingface/peft/blob/v0.10.0/src/peft/tuners/adaption_prompt/layer.py#L57
    The only differences are:
    - replace self.model() with self.model.pre_attn_forward()
    """
    if kwargs.get("output_attention", False):
        raise NotImplementedError("output_attention is not currently supported.")

    output, _, past_key_value = self.model.pre_attn_forward(*args, **kwargs)
    bsz = output.shape[0]
    q_len = output.shape[1]
    embed_dim = output.shape[2]
    k_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].k_proj_layer
    v_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].v_proj_layer
    o_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].o_proj_layer
    factor = (
        self.model.k_proj.in_features // self.model.k_proj.out_features
    )  # Mistral has different input and output dimension for k_proj and v_proj layers

    if k_proj_layer == v_proj_layer:
        _, key, value = getattr(self.model, k_proj_layer)(self.adaption_prompt).split(embed_dim, dim=2)
    else:
        key = getattr(self.model, k_proj_layer)(self.adaption_prompt)
        value = getattr(self.model, v_proj_layer)(self.adaption_prompt)

    # (bsz, num_key_value_heads, adapter_len, head_dim)
    adapter_k = (
        key.view(1, self.adapter_len, (self.model.num_heads // factor), self.model.head_dim)
        .repeat(bsz, 1, 1, 1)
        .transpose(1, 2)
    )
    adapter_v = (
        value.view(1, self.adapter_len, (self.model.num_heads // factor), self.model.head_dim)
        .repeat(bsz, 1, 1, 1)
        .transpose(1, 2)
    )
    # Below is taken from https://github.com/huggingface/transformers/blob/e547458c43dfdbbb8f6a7757237e234c44e20a8f/src/transformers/models/mistral/modeling_mistral.py#L181
    # (bsz, num_heads, adapter_len, head_dim)
    adapter_k = torch.repeat_interleave(adapter_k, repeats=factor, dim=1)
    adapter_v = torch.repeat_interleave(adapter_v, repeats=factor, dim=1)
    # Recompute query states.
    # (bsz, num_heads, q_len, head_dim)
    query_states = compute_query_states(model=self.model, **kwargs)

    previous_dtype = query_states.dtype

    # (bsz, num_heads, q_len, adapter_len)
    scores = torch.matmul(query_states, adapter_k.transpose(2, 3).to(previous_dtype)) / math.sqrt(self.model.head_dim)
    # Upcast attention to fp32
    # (bsz, num_heads, q_len, adapter_len)
    scores = self.adaption_gate * F.softmax(scores, dim=-1, dtype=torch.float32).to(previous_dtype)
    # (bsz, q_len, num_heads * head_dim)
    adapter_output = torch.matmul(scores, adapter_v).transpose(1, 2).reshape(bsz, q_len, -1)

    # (bsz, q_len, hidden_size)
    if o_proj_layer is not None:
        adapter_output = getattr(self.model, o_proj_layer)(adapter_output)

    # Add adaption prompt output to original output.
    output = output + adapter_output

    # Restore original dtype.
    output = output.to(previous_dtype)
    return output, None, past_key_value


def GaudiAdaptedAttention_getattr(self, name: str):
    """Forward missing attributes to the wrapped module."""
    try:
        return super(self.__class__, self).__getattr__(name)
    except AttributeError:
        # This is necessary as e.g. causal models have various methods that we
        # don't want to re-implement here.
        return getattr(self.model, name)
