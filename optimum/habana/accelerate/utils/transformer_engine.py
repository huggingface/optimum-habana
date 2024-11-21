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

import functools

import torch


has_transformer_engine = False


def import_te():
    global te, has_transformer_engine
    try:
        import habana_frameworks.torch.hpex.experimental.transformer_engine as te

        has_transformer_engine = True

    except ImportError:
        has_transformer_engine = False


def is_fp8_available():
    if not has_transformer_engine:
        import_te()
    return has_transformer_engine


def _convert_model(model, to_transformer_engine=True, _convert_linear=True):
    """
    Recursively converts the linear layer of a model to their `transformers_engine` counterpart.
    """
    from optimum.habana.transformers.models.llama.modeling_llama import ModuleFusedSDPA

    if not is_fp8_available():
        raise ImportError("Using `convert_model` requires transformer_engine to be installed.")
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear) and to_transformer_engine and _convert_linear:
            has_bias = module.bias is not None
            # Initializing TE linear without weights and biases and shallow copying them from the original module.
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
                module.in_features,
                module.out_features,
                bias=has_bias,
                dtype=module.weight.dtype,
                device=module.weight.device,
            )
            new_module.weight.copy_(module.weight)
            if has_bias:
                new_module.bias.copy_(module.bias)

            setattr(model, name, new_module)
        elif isinstance(module, ModuleFusedSDPA) and module.flash_attention_fp8 and to_transformer_engine:
            from habana_frameworks.torch.hpex.experimental.transformer_engine import (
                FusedAttention as TE_FusedAttention,
            )

            class TE_ModuleFusedSDPA(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self._hpu_kernel_fsdpa = TE_FusedAttention(
                        scale=module.scale,
                        attention_dropout=module.attention_dropout,
                        enable_recompute=module.enable_recompute,
                    )

                def forward(self, query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode):
                    return self._hpu_kernel_fsdpa(query, key, value, attn_mask, is_causal, softmax_mode)

            setattr(model, name, TE_ModuleFusedSDPA())
        else:
            _convert_model(module, to_transformer_engine=to_transformer_engine, _convert_linear=_convert_linear)


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


def convert_model(model):
    """
    Converts torch.nn.Linear modules to `transformers_engine` Linear modules.
    Adapted from: https://github.com/huggingface/accelerate/blob/v0.27.2/src/accelerate/accelerator.py#L1303
    """
    if not has_transformer_engine_layers(model):
        with torch.no_grad():
            _convert_model(model)
        model._converted_to_transformer_engine = True
    return model


def get_fp8_recipe(fp8_recipe_handler):
    """
    Creates transformer engine FP8 recipe object.
    Adapted from: https://github.com/huggingface/accelerate/blob/v0.27.2/src/accelerate/accelerator.py#L1309
    """
    if not is_fp8_available():
        raise ImportError("Using `get_fp8_recipe` requires transformer_engine to be installed.")
    kwargs = fp8_recipe_handler.to_dict() if fp8_recipe_handler is not None else {}
    if "fp8_format" in kwargs:
        kwargs["fp8_format"] = getattr(te.recipe.Format, kwargs["fp8_format"])
    fp8_recipe_handler = te.recipe.DelayedScaling(**kwargs)
    fp8_recipe_handler.backend = "TE"
    return fp8_recipe_handler


class FP8ContextWrapper:
    """
    Helper class for FP8 context related operations.
    """

    def __init__(self, ctx, fp8_recipe):
        self.ctx = ctx
        self.fp8_ctx = self.create_fp8_context(fp8_recipe)

    def __enter__(self):
        self.ctx.__enter__()
        self.fp8_ctx.__enter__()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.fp8_ctx.__exit__(exc_type, exc_value, exc_traceback)
        self.ctx.__exit__(exc_type, exc_value, exc_traceback)

    @staticmethod
    def create_fp8_context(fp8_recipe):
        return te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)

    @staticmethod
    def _gradient_checkpointing_wrap(func, *args, **kwargs):
        """
        `_gradient_checkpointing_func` always takes the function to be recomputed as the first argument. The function
        below wraps this first argument with `transformer_engine`'s `activation_checkpointing` context.
        """
        _args = list(args)
        _args[0] = te.distributed.activation_checkpointing()(_args[0])
        args = tuple(_args)

        return func(*args, **kwargs)

    @staticmethod
    def gradient_checkpointing_wrap(model):
        """
        Wrap `_gradient_checkpointing_func` in the model with `transformer_engine`'s `activation_checkpointing` context.
        This context is used to signal the `transformer_engine` modules whether they have been called with activation checkpointing enabled or not.
        """
        if hasattr(model, "gradient_checkpointing") and model.gradient_checkpointing:
            model._gradient_checkpointing_func = functools.partial(
                FP8ContextWrapper._gradient_checkpointing_wrap, model._gradient_checkpointing_func
            )
            return

        for module in model.modules():
            if hasattr(module, "gradient_checkpointing") and module.gradient_checkpointing:
                module._gradient_checkpointing_func = functools.partial(
                    FP8ContextWrapper._gradient_checkpointing_wrap, module._gradient_checkpointing_func
                )
