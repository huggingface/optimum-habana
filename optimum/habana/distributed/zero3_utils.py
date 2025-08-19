# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team.
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

import logging

import torch


logger = logging.getLogger(__name__)


def apply_zero3_leaf_promotion(model: torch.nn.Module) -> None:
    """
    Promote registered modules to ZeRO-3 leafs.

    Parameters
    ----------
    model : torch.nn.Module
        The model (or PEFT wrapper) on which `set_z3_leaf_modules` will be called.
    """
    # Caller guarantees DeepSpeed is available; safe to import
    from deepspeed.utils import set_z3_leaf_modules

    config = getattr(model, "config", model)
    model_type = getattr(config, "model_type", None)

    if model_type == "llama":
        from optimum.habana.transformers.models.llama.modeling_llama import GaudiLlamaDecoderLayer

        set_z3_leaf_modules(model, [GaudiLlamaDecoderLayer])

    elif model_type == "mixtral":
        from optimum.habana.transformers.models.mixtral.modeling_mixtral import GaudiMixtralSparseMoeBlock

        set_z3_leaf_modules(model, [GaudiMixtralSparseMoeBlock])

    elif model_type == "qwen3_moe":
        from optimum.habana.transformers.models.qwen3_moe.modeling_qwen3_moe import GaudiQwen3MoeSparseMoeBlock

        set_z3_leaf_modules(model, [GaudiQwen3MoeSparseMoeBlock])

    else:
        logger.debug(f"Model type '{model_type}' is not registered for ZeRO-3 leaf promotion.")
        return

    logger.debug(f"Model type '{model_type}' is registered for ZeRO-3 leaf promotion.")
