import math
from typing import Any

import torch
import torch.nn.functional as F
from peft.tuners.adaption_prompt.config import TRANSFORMERS_MODEL_CONFIG
from peft.utils.other import transpose


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
    compute_query_states = TRANSFORMERS_MODEL_CONFIG[self.model_type].compute_query_states
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


def GaudiAdaptedAttentionAttentionAllReduce(self, attn_output):
    if hasattr(self.model.o_proj, "all_reduce"):
        self.model.o_proj.all_reduce(attn_output)


def GaudiAdaptedAttentionPostAttnForward(self, attn_output):
    if hasattr(self.model.o_proj, "post_all_reduce"):
        self.model.o_proj.post_all_reduce(attn_output)
    return attn_output


class LoRALinear:
    def __init__(self, module):
        has_bias = module.bias is not None
        self.module = module
        import habana_frameworks.torch.hpex.experimental.transformer_engine as te

        self.module.te_linear = te.Linear(
            module.in_features,
            module.out_features,
            bias=has_bias,
            params_dtype=module.weight.dtype,
            skip_weight_param_allocation=True,
        )

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        # TODO: to check if bias is removed from lora linear
        if hasattr(self.module, "bias"):
            return self.module.te_linear(
                input, transpose(self.module.weight, self.module.fan_in_fan_out), bias=self.module.bias
            )
        else:
            return self.module.te_linear(input, transpose(self.module.weight, self.module.fan_in_fan_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.module.disable_adapters:
            if self.module.merged:
                self.module.unmerge()
            result = self._linear(x)
        elif self.module.merged:
            result = self._linear(x)
        else:
            result = self._linear(x)
            for active_adapter in self.module.active_adapters:
                if active_adapter not in self.module.lora_A.keys():
                    continue
                lora_A = self.module.lora_A[active_adapter]
                lora_B = self.module.lora_B[active_adapter]
                dropout = self.module.lora_dropout[active_adapter]
                scaling = self.module.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                result = result.clone() + lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        return result

    @staticmethod
    def replace_forward(module):
        lora_linear = LoRALinear(module)
        module.forward = lora_linear.forward
