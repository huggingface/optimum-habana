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

import importlib
import logging
from typing import Dict, Tuple, Type

import torch


logger = logging.getLogger(__name__)

_MODEL_Z3_LEAF_REGISTRY: Dict[str, Tuple[str, str]] = {
    "llama": ("optimum.habana.transformers.models.llama.modeling_llama", "GaudiLlamaDecoderLayer"),
    "mixtral": ("optimum.habana.transformers.models.mixtral.modeling_mixtral", "GaudiMixtralSparseMoeBlock"),
    "qwen3_moe": ("optimum.habana.transformers.models.qwen3_moe.modeling_qwen3_moe", "GaudiQwen3MoeSparseMoeBlock"),
    # Add more as needed
}


def _import_class(module_path: str, class_name: str) -> Type:
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def apply_zero3_leaf_promotion(
    model: torch.nn.Module,
    *,
    is_deepspeed_zero3_enabled: bool = False,
    use_zero3_leaf_promotion: bool = False,
) -> None:
    """
    Promote registered modules to ZeRO-3 leafs **only if the caller
    explicitly confirms** that DeepSpeed ZeRO-3 is enabled and the
    user-toggle `use_zero3_leaf_promotion` is True.

    Parameters
    ----------
    model : torch.nn.Module
        The model (or PEFT wrapper) on which `set_z3_leaf_modules` will be called.
    is_deepspeed_zero3_enabled : bool
        Must be True when the caller has already verified that DeepSpeed
        ZeRO-3 is currently active.  DeepSpeed is imported only if this
        flag is True.
    use_zero3_leaf_promotion : bool
        User-level opt-in flag that must also be True for the patch to run.
    """
    if not is_deepspeed_zero3_enabled or not use_zero3_leaf_promotion:
        return

    # Caller guarantees DeepSpeed is available; safe to import
    from deepspeed.utils import set_z3_leaf_modules

    config = getattr(model, "config", model)
    model_type = getattr(config, "model_type", None)
    if model_type not in _MODEL_Z3_LEAF_REGISTRY:
        logger.debug(f"Model type '{model_type}' is not registered for ZeRO-3 leaf promotion.")
        return

    module_path, class_name = _MODEL_Z3_LEAF_REGISTRY[model_type]
    leaf_cls = _import_class(module_path, class_name)
    set_z3_leaf_modules(model, [leaf_cls])
    logger.debug(f"Registered {class_name} as ZeRO-3 leaf for {model_type}.")
