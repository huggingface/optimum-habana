import inspect
from typing import TYPE_CHECKING, Dict, Optional

import torch
from transformers.generation.candidate_generator import (
    AssistedCandidateGenerator,
)


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transfromers.generation.logits_process import LogitsProcessorList

    from .configuration_utils import GaudiGenerationConfig


class GaudiAssistedCandidateGenerator(AssistedCandidateGenerator):
    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        generation_config: "GaudiGenerationConfig",
        model_kwargs: Dict,
        inputs_tensor: Optional[torch.Tensor] = None,
        logits_processor: "LogitsProcessorList" = None,
    ):
        super().__init__(
            input_ids,
            assistant_model,
            generation_config,
            model_kwargs,
            inputs_tensor,
            logits_processor,
        )

        # Remove model kwargs that are specific to optimized models
        # E.g. token_idx, use_flash_attention, etc...
        # Otherwise it will trigger an error in GenerationMixin._validate_model_kwargs
        # TODO: may need to complete this for encoder-decoders: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/generation/utils.py#L1133
        model_args = set(inspect.signature(assistant_model.prepare_inputs_for_generation).parameters)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(assistant_model.forward).parameters)
        for key, value in list(self.assistant_kwargs.items()):
            if value is not None and key not in model_args:
                del self.assistant_kwargs[key]
