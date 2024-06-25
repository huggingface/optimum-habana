from typing import Any

import torch
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
