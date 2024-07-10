from typing import Any

import torch


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
