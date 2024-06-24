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


te = None


class SwitchableForwardMaker:
    def __init__(self, module, fp8_recipe_handler):
        self.original_forward = module.forward
        self.fp8_forward = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe_handler)(module.forward)
        self.module = module
        module.forward = self.forward

    def forward(self, *args, **kwargs):
        if self.module.training:
            return self.fp8_forward(*args, **kwargs)
        else:
            return self.original_forward(*args, **kwargs)

    @staticmethod
    def convert(module, fp8_recipe_handler):
        SwitchableForwardMaker(module, fp8_recipe_handler)


def get_te():
    global te
    if te is None:
        try:
            import habana_frameworks.torch.hpex.experimental.transformer_engine as te

            te = te
        except ImportError:
            te = None


def convert_model(model, to_transformer_engine=True, _convert_linear=True):
    """
    Recursively converts the linear and layernorm layers of a model to their `transformers_engine` counterpart.
    """
    if te is None:
        raise ImportError("Using `convert_model` requires transformer_engine to be installed.")
    from peft.tuners.lora.layer import Linear as PEFTLinear

    from optimum.habana.peft.layer import LoRALinear

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
    if te is None:
        raise ImportError("Using `has_transformer_engine_layers` requires transformer_engine to be installed.")
    for m in model.modules():
        if isinstance(m, (te.Linear)):
            return True
    return False


def te_setup_fp8_recipe_handler(fp8_recipe_format):
    get_te()
    fp8_format = te.recipe.Format.E5M2
    if fp8_recipe_format == "E4M3":
        fp8_format = te.recipe.Format.E4M3
    elif fp8_recipe_format == "HYBRID":
        fp8_format = te.recipe.Format.HYBRID
    fp8_recipe_handler = te.recipe.DelayedScaling(
        fp8_format=fp8_format,
        margin=0,
        interval=16,
        amax_history_len=1,
        amax_compute_algo="most_recent",
        reduce_amax=False,
    )
    fp8_recipe_handler.backend = "TE"
    return fp8_recipe_handler


def te_wrap_fp8(model):
    if not has_transformer_engine_layers(model):
        with torch.no_grad():
            convert_model(model)
        model._converted_to_transformer_engine = True
    return model


def te_wrap_fp8_forward_convert(model, fp8_recipe_handler):
    model = te_wrap_fp8(model)
    SwitchableForwardMaker.convert(model, fp8_recipe_handler)
    return model


def te_forward_convert(model, fp8_recipe_handler):
    SwitchableForwardMaker.convert(model, fp8_recipe_handler)
