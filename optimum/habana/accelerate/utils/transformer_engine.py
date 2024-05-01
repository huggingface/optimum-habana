# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import torch
from peft.tuners.lora.layer import Linear as PEFTLinear

from optimum.habana.peft.layer import LoRALinear


try:
    import habana_frameworks.torch.hpex.experimental.transformer_engine as te

    has_transformer_engine = True
except ImportError:
    has_transformer_engine = False


def is_fp8_available():
    return has_transformer_engine


def convert_model(model, to_transformer_engine=True, _convert_linear=True):
    """
    Recursively converts the linear and layernorm layers of a model to their `transformers_engine` counterpart.
    """
    if not has_transformer_engine:
        raise ImportError("Using `convert_model` requires transformer_engine to be installed.")

    for name, module in model.named_children():
        if type(module) == PEFTLinear and to_transformer_engine and _convert_linear:
            LoRALinear.replace_forward(module)
        if (
            isinstance(module, torch.nn.Linear)
            and not type(module) == PEFTLinear
            and to_transformer_engine
            and _convert_linear
        ):
            has_bias = module.bias is not None
            te_module = te.Linear(
                module.in_features,
                module.out_features,
                bias=has_bias,
                params_dtype=module.weight.dtype,
                skip_weight_param_allocation=True,
            )
            te_module.weight = module.weight

            if has_bias:
                te_module.bias = module.bias

            setattr(model, name, te_module)

        elif isinstance(module, te.Linear) and not to_transformer_engine and _convert_linear:
            has_bias = module.bias is not None
            new_module = torch.nn.Linear(
                module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
            )
            new_module.weight.copy_(module.weight)
            if has_bias:
                new_module.bias.copy_(module.bias)

            setattr(model, name, new_module)
        else:
            convert_model(module, to_transformer_engine=to_transformer_engine, _convert_linear=_convert_linear)


def has_transformer_engine_layers(model):
    """
    Returns whether a given model has some `transformer_engine` layer or not.
    """
    if not is_fp8_available():
        raise ImportError("Using `has_transformer_engine_layers` requires transformer_engine to be installed.")
    for m in model.modules():
        if isinstance(m, (te.Linear)):
            return True
    return False
