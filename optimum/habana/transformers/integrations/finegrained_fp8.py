import os
from enum import Enum
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from optimum.utils import logging
from torch.nn import functional as F


logger = logging.get_logger(__name__)


class FP8Method(str, Enum):
    FP8_DYNAMIC = "fp8_dynamic"
    FP8_INC = "fp8_inc"
    NONE = "none"


HPU_MODULES_TO_NOT_CONVERT = ["lm_head", "k_b_proj", "kv_b_proj"]


def get_fp8_method(config):
    if os.getenv("QUANT_CONFIG") is not None:
        # FP8 quantization with Intel Neural Compressor (INC)
        return FP8Method.FP8_INC

    if hasattr(config, "quantization_config"):
        quantization_config = config.quantization_config
        # Dynamic FP8 quantization without INC
        if hasattr(quantization_config, "quant_method") and quantization_config.quant_method == "fp8":
            activation_scheme = quantization_config.activation_scheme
            if activation_scheme != "dynamic":
                raise ValueError(f"Unsupported fp8 activation scheme: {activation_scheme}")
            return FP8Method.FP8_DYNAMIC
    return FP8Method.NONE


class GaudiFP8Linear(nn.Module):
    """
    Finegrained FP8 quantization using Intel Neural Compressor (INC)
    Adapted from: https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/integrations/finegrained_fp8.py#L294
    The main changes are:
    - Support for runtime FP8 dequantization with INC
    - Extends nn.Module instead of F8Linear to avoid triton import errors
    """

    dtype = torch.float8_e4m3fn

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None,
        block_size: Optional[Tuple[int, int]] = None,
        device=None,
        activation_scheme="dynamic",
        high_precision=torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_precision = high_precision

        self.register_buffer(
            "weight", torch.empty(out_features, in_features, dtype=GaudiFP8Linear.dtype, device=device)
        )

        scale_out_features = (out_features + block_size[0] - 1) // block_size[0]
        scale_in_features = (in_features + block_size[1] - 1) // block_size[1]
        self.register_buffer(
            "weight_scale_inv", torch.empty(scale_out_features, scale_in_features, dtype=torch.float32, device=device)
        )

        self.block_size = block_size
        self.custom_name = "gaudi_fp8_linear"

        self.activation_scheme = activation_scheme

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

        if hasattr(torch.ops.hpu, "cast_to_fp8_just_in_time"):
            self.fp8_linear_func = self.fp8_linear_blockwise
        else:
            self.fp8_linear_func = self.fp8_linear_dequant

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.fp8_linear_func(input)

    def fp8_linear_blockwise(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype == self.high_precision:
            output = F.linear(input, self.weight, self.bias)
        else:
            input_fp8, input_scale = torch.ops.hpu.cast_to_fp8_just_in_time(
                input, [1, self.block_size[1]], out_dtype=GaudiFP8Linear.dtype, scale_dtype=torch.bfloat16
            )
            output = torch.ops.hpu.fp8_gemm_v2(
                A=input_fp8,
                trans_A=False,
                B=self.weight,
                trans_B=True,
                D=None,
                out_dtype=torch.bfloat16,
                A_scale_inv=input_scale,
                B_scale_inv=self.weight_scale_inv.t(),
                bias=None,
                accumulate=False,
            )
        return output

    def fp8_linear_dequant(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.get_dequant_weight(), self.bias)

    def pad_weight_naive(self):
        # Pad weight to block dimensions
        weight, orig_M, orig_N = pad_block_fp8_weight_naive(self.weight, self.weight_scale_inv, self.block_size)
        self.weight = weight
        self.orig_M = orig_M
        self.orig_N = orig_N

    def get_dequant_weight(self):
        if self.weight.dtype == self.high_precision:
            return self.weight

        return dequant_block_fp8_weight_naive(
            self.weight,
            self.weight_scale_inv,
            block_size=self.block_size,
            dtype=self.high_precision,
            original_M=self.orig_M,
            original_N=self.orig_N,
            do_unpad=True,
        )

    def dequant_block_fp8_weight(self, layer) -> torch.Tensor:
        """
        This function is called by INC during either the measurement or quantization phase.
        - In the quantization phase, INC requantizes the BF16 weight to FP8 and updates the weight.
        - In the measurement phase, INC only measures the BF16 weight without updating it.
        Tracking the BF16 weight can lead to Out of Memory (OoM) issues, so we avoid storing it.
        If the weight has already been updated, we return it directly.

        Adapted from: https://github.com/HabanaAI/vllm-fork/blob/deepseek_r1/vllm/model_executor/layers/quantization/fp8.py#L282
        """
        if hasattr(layer, "updated_fp8_weight") and layer.updated_fp8_weight:
            return layer.weight
        dequant_weight = layer.get_dequant_weight()
        return dequant_weight

    def get_dequant_weights_func(
        self,
    ) -> Optional[Callable[[torch.nn.Module], torch.Tensor]]:
        return self.dequant_block_fp8_weight


def _replace_with_fp8_linear(
    model,
    tp_plan=None,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
):
    """Replace Linear layers with GaudiFP8Linear."""
    if current_key_name is None:
        current_key_name = []

    for name, module in model.named_children():
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and name not in (modules_to_not_convert or []):
            current_key_name_str = ".".join(current_key_name)
            if not any(key in current_key_name_str for key in (modules_to_not_convert or [])):
                with init_empty_weights():
                    model._modules[name] = GaudiFP8Linear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                        activation_scheme=quantization_config.activation_scheme,
                        block_size=quantization_config.weight_block_size,
                    )
                    has_been_replaced = True

        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_fp8_linear(
                module,
                tp_plan,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )

        current_key_name.pop(-1)

    return model, has_been_replaced


def replace_with_fp8_linear(
    model,
    modules_to_not_convert=None,
    quantization_config=None,
):
    """Helper function to replace model layers with FP8 versions."""
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert

    if quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))
    model, has_been_replaced = _replace_with_fp8_linear(
        model,
        tp_plan=model._tp_plan,
        modules_to_not_convert=modules_to_not_convert,
        quantization_config=quantization_config,
    )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using fp8 but no linear modules were found in your model."
            " Please double check your model architecture."
        )

    return model


def pad_weight(weight, block_size):
    """Pads a matrix to make its dimensions multiples of block_size."""
    M, N = weight.shape[-2:]
    block_size_m, block_size_n = block_size
    pad_M = (block_size_m - M % block_size_m) % block_size_m
    pad_N = (block_size_n - N % block_size_n) % block_size_n

    if pad_M == 0 and pad_N == 0:
        return weight, M, N  # No padding needed
    padded_weight = torch.nn.functional.pad(weight, (0, pad_N, 0, pad_M), mode="constant", value=0)
    return padded_weight, M, N  # Return original dimensions for unpadding


def pad_block_fp8_weight_naive(weight, weight_scale, block_size):
    assert len(block_size) == 2
    block_size_m, block_size_n = block_size
    weight_scale_m, weight_scale_n = weight_scale.shape[-2:]

    weight, orig_M, orig_N = pad_weight(weight, block_size)
    M, N = weight.shape[-2:]

    assert weight_scale_m == M // block_size_m
    assert weight_scale_n == N // block_size_n
    return weight, orig_M, orig_N


def unpad_weight(weight, original_M, original_N, keep_first_dim=False):
    """
    Removes padding from the matrix to restore its original shape.
    Adapted from: https://github.com/HabanaAI/vllm-fork/blob/deepseek_r1/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L142
    """
    if (weight.shape[-2] == original_M) and (weight.shape[-1] == original_N):
        return weight

    if keep_first_dim:
        return weight[:, :original_M, :original_N]
    else:
        return weight[:original_M, :original_N]


def dequant_block_fp8_weight_naive(
    weight, weight_scale, block_size, dtype=torch.bfloat16, original_M=None, original_N=None, do_unpad=False
):
    """Adapted from: https://github.com/HabanaAI/vllm-fork/blob/deepseek_r1/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L175"""
    if weight_scale is None:
        return weight
    assert len(block_size) == 2

    weight_shape_len = len(weight.shape)

    block_size_m, block_size_n = block_size

    # mul scale
    if weight_shape_len == 2:
        weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(weight_scale_m * block_size_m, weight_scale_n * block_size_n)
        keep_first_dim = False
    elif weight_shape_len == 3:
        fd, weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(fd, weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(fd, weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(fd, weight_scale_m * block_size_m, weight_scale_n * block_size_n)
        keep_first_dim = True
    else:
        raise ValueError("Only support original weight shape is either 2 or 3")

    if do_unpad:
        dequant_weight = unpad_weight(dequant_weight, original_M, original_N, keep_first_dim=keep_first_dim)

    return dequant_weight
