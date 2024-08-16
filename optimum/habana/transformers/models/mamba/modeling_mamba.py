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
    self, outputs: ModelOutput, model_kwargs: Dict[str, Any], num_new_tokens: int = 1, **kwargs
) -> Dict[str, Any]:
    model_kwargs["cache_params"] = outputs.get("cache_params", None)
    token_idx = model_kwargs.get("token_idx", None)
    if (
        model_kwargs.get("use_cache", True)
        and "cache_position" in model_kwargs
        and model_kwargs["cache_position"] is not None
    ):
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
    if token_idx is not None:
        token_idx.add_(1)
        if "token_idx_cpu" in model_kwargs:
            model_kwargs["token_idx_cpu"] += 1
    return model_kwargs


def gaudi_MambaForCausalLM_prepare_inputs_for_generation(
    self,
    input_ids,
    inputs_embeds=None,
    use_cache=None,
    cache_params: Optional[MambaCache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask=None,
    **kwargs,
):
    token_idx = kwargs.get("token_idx", None)
    token_idx_cpu = kwargs.get("token_idx_cpu", None)
    if use_cache:
        if token_idx is None:
            # `cache_position` should have been initialized in `generate`
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )
            if cache_position[0] > 0:
                input_ids = input_ids[:, -1].unsqueeze(-1)
            else:
                # we initialize the `cache_position` to full size of `conv_states` at prefill stage
                # considering padding will be applied when input length is shorter, and truncation
                # will be applied when it is longer, so it will be equivalent to always have it match
                # the length of `cache_params.conv_states`, which is `config.conv_kernel`
                cache_position = torch.arange(0, self.config.conv_kernel, device=input_ids.device)
        else:
            input_ids = torch.index_select(input_ids, 1, token_idx - 1)
    else:
        if token_idx is not None:
            input_ids = torch.index_select(input_ids, 1, torch.arange(token_idx_cpu, device=input_ids.device))
    if inputs_embeds is not None and cache_params is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids.contiguous()}

    model_inputs.update(
        {
            "cache_params": cache_params,
            "use_cache": use_cache,
            "cache_position": cache_position,
        }
    )
    return model_inputs
