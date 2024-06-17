from typing import Any, Dict, Optional

import torch
from transformers.models.mamba.modeling_mamba import (
    MambaCache,
)
from transformers.utils import (
    ModelOutput,
    logging,
)


logger = logging.get_logger(__name__)

def gaudi_MambaForCausalLM_update_model_kwargs_for_generation(
    self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
) -> Dict[str, Any]:
    model_kwargs["cache_params"] = outputs.get("cache_params", None)
    token_idx = model_kwargs.get("token_idx", None)
    if token_idx is not None:
        token_idx.add_(1)
        if "token_idx_cpu" in model_kwargs:
            model_kwargs["token_idx_cpu"] += 1
    return model_kwargs


def gaudi_MambaForCausalLM_prepare_inputs_for_generation(
    self, input_ids, cache_params: Optional[MambaCache] = None, inputs_embeds=None, attention_mask=None, **kwargs
):
    token_idx = kwargs.get("token_idx", None)
    token_idx_cpu = kwargs.get("token_idx_cpu", None)
    if cache_params is not None:
        if token_idx is None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        else:
            input_ids = torch.index_select(input_ids, 1, token_idx - 1)
    else:
        if token_idx is not None:
            input_ids = torch.index_select(input_ids, 1, torch.arange(token_idx_cpu, device=input_ids.device))
    if inputs_embeds is not None and cache_params is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}
    model_inputs["cache_params"] = cache_params
    return model_inputs
