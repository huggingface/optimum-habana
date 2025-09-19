import importlib
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import habana_frameworks.torch.utils.experimental as htexp
import torch
from packaging import version
from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
from transformers.quantizers.quantizers_utils import get_module_from_name
from transformers.utils import is_torch_available, logging


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

if htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi2:
    ON_GAUDI2 = True
    FP8_MIN = torch.finfo(torch.float8_e4m3fnuz).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fnuz).max
else:
    ON_GAUDI2 = False
    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

logger = logging.get_logger(__name__)


class GaudiFineGrainedFP8HfQuantizer(FineGrainedFP8HfQuantizer):
    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        """
        Adapted from: https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/quantizers/quantizer_finegrained_fp8.py
        The main changes are:
        - Overrides all methods to bypass import errors from triton package
        - Removes check for cuda and CPU devices
        """
        if not is_torch_available() or version.parse(importlib.metadata.version("torch")) < version.parse("2.6.0"):
            raise ImportError(
                "Using fp8 quantization requires torch >= 2.6.0"
                "Please install the latest version of torch ( pip install --upgrade torch )"
            )

        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Converting into FP8 weights from tf/flax weights is currently not supported, "
                "please make sure the weights are in PyTorch format."
            )

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
    ):
        """
        Quantizes weights to FP8 format using Block-wise quantization
        """
        from accelerate.utils import set_module_tensor_to_device

        set_module_tensor_to_device(model, param_name, target_device, param_value)

        module, tensor_name = get_module_from_name(model, param_name)

        block_size_m, block_size_n = self.quantization_config.weight_block_size

        rows, cols = param_value.shape[-2:]

        if rows % block_size_m != 0 or cols % block_size_n != 0:
            raise ValueError(
                f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_size_m}, {block_size_n})"
            )
        param_value_orig_shape = param_value.shape

        param_value = param_value.reshape(
            -1, rows // block_size_m, block_size_m, cols // block_size_n, block_size_n
        ).permute(0, 1, 3, 2, 4)

        # Calculate scaling factor for each block
        max_abs = torch.amax(torch.abs(param_value), dim=(-1, -2))
        scale = FP8_MAX / max_abs
        scale_orig_shape = scale.shape
        scale = scale.unsqueeze(-1).unsqueeze(-1)

        # Quantize the weights
        quantized_param = torch.clamp(param_value * scale, min=FP8_MIN, max=FP8_MAX).to(torch.float8_e4m3fn)

        quantized_param = quantized_param.permute(0, 1, 3, 2, 4)
        # Reshape back to matrix shape
        quantized_param = quantized_param.reshape(param_value_orig_shape)

        # Reshape scale to match the number of blocks
        scale = scale.reshape(scale_orig_shape).squeeze().reciprocal()

        module._buffers[tensor_name] = quantized_param.to(target_device)
        module._buffers["weight_scale_inv"] = scale.to(target_device)

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ):
        from optimum.habana.transformers.integrations.finegrained_fp8 import GaudiFP8Linear

        module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, GaudiFP8Linear):
            if self.pre_quantized or tensor_name == "bias":
                if tensor_name == "weight" and param_value.dtype != torch.float8_e4m3fn:
                    raise ValueError("Expect quantized weights but got an unquantized weight")
                return False
            else:
                if tensor_name == "weight_scale_inv":
                    raise ValueError("Expect unquantized weights but got a quantized weight_scale")
                return True
        return False

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        modules_to_not_convert: List[str] = [],
        **kwargs,
    ):
        from optimum.habana.transformers.integrations.finegrained_fp8 import (
            replace_with_fp8_linear,
        )

        self.modules_to_not_convert = ["lm_head"] + modules_to_not_convert

        if self.quantization_config.modules_to_not_convert:
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)

        model = replace_with_fp8_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
        )

        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model, **kwargs):
        return _gaudi_rescale_pad_fp8_weights(model)

    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        from optimum.habana.transformers.integrations.finegrained_fp8 import GaudiFP8Linear

        not_missing_keys = []
        for name, module in model.named_modules():
            if isinstance(module, GaudiFP8Linear):
                for missing in missing_keys:
                    if (
                        (name in missing or name in f"{prefix}.{missing}")
                        and not missing.endswith(".weight")
                        and not missing.endswith(".bias")
                    ):
                        not_missing_keys.append(missing)
        return [k for k in missing_keys if k not in not_missing_keys]


def _gaudi_rescale_pad_fp8_weights(model, current_key_name=None):
    """
    Rescale FP8 weights to float8_e4m3fnuz on Gaudi2 after loading
    and pads weights on Gaudi2/Gaudi3 to match block dimensions.

    Also dequantizes layers in INC blocklist
    """

    from neural_compressor.torch.algorithms.fp8_quant._core.utils import is_re_match
    from neural_compressor.torch.quantization import FP8Config

    from optimum.habana.transformers.integrations.finegrained_fp8 import HPU_MODULES_TO_NOT_CONVERT, GaudiFP8Linear

    quant_config = os.getenv("QUANT_CONFIG")
    blocklist = []
    if quant_config is not None:
        fp8_config = FP8Config.from_json_file(quant_config)
        if "names" in fp8_config.blocklist:
            blocklist = fp8_config.blocklist["names"]
    else:
        blocklist = HPU_MODULES_TO_NOT_CONVERT

    rescale_factor = torch.finfo(torch.float8_e4m3fnuz).max / torch.finfo(torch.float8_e4m3fn).max
    rescale_factor_inv = 1.0 / rescale_factor

    for name, module in model.named_modules():
        if isinstance(module, GaudiFP8Linear):
            # module = model._modules[name]
            if ON_GAUDI2:  # rescale scale and weight for Gaudi2
                weight = module.weight.to(module.high_precision) * rescale_factor
                scale = module.weight_scale_inv * rescale_factor_inv
                module.weight = weight.to(module.weight.dtype)
                module.weight_scale_inv = scale.to(module.weight_scale_inv.dtype)

            module.pad_weight_naive()

            if len(blocklist) > 0:
                if is_re_match(blocklist, name):
                    module.weight = module.get_dequant_weight()
                else:
                    module.weight_scale_inv = module.weight_scale_inv

    return model
