import torch


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
