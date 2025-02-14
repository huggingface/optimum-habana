import importlib
from enum import Enum

import torch.nn as nn
from packaging import version
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import is_accelerate_available, is_auto_awq_available
from transformers.utils.quantization_config import (
    AwqBackendPackingMethod,
)

from optimum.utils import logging


logger = logging.get_logger(__name__)


class GaudiAWQLinearVersion(str, Enum):
    GEMM = "gemm"
    GEMV = "gemv"
    EXLLAMA = "exllama"
    HPU = "hpu"

    @staticmethod
    def from_str(version: str):
        version = version.lower()
        if version == "gemm":
            return GaudiAWQLinearVersion.GEMM
        elif version == "gemv":
            return GaudiAWQLinearVersion.GEMV
        elif version == "exllama":
            return GaudiAWQLinearVersion.EXLLAMA
        elif version == "hpu":
            return GaudiAWQLinearVersion.HPU
        else:
            raise ValueError(f"Unknown GaudiAWQLinearVersion {version}")


# override post_init in AwqConfig
def gaudi_awq_config_post_init(self):
    """
    Adapted from: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/utils/quantization_config.py#L818
    - support HPU.
    """
    if self.backend not in [AwqBackendPackingMethod.AUTOAWQ]:
        raise ValueError(
            f"Only supported quantization backends in {AwqBackendPackingMethod.AUTOAWQ} - not recognized backend {self.backend}"
        )

    self.version = GaudiAWQLinearVersion.from_str(self.version)
    if self.version not in [
        GaudiAWQLinearVersion.HPU,
        GaudiAWQLinearVersion.GEMM,
    ]:
        raise ValueError(
            f"Only supported versions are in [GaudiAWQLinearVersion.HPU, GaudiAWQLinearVersion.GEMM] - not recognized version {self.version}"
        )

    if self.do_fuse and self.fuse_max_seq_len is None:
        raise ValueError(
            "You cannot enable fused modules without specifying a `fuse_max_seq_len`, make sure to pass a valid `fuse_max_seq_len` for your usecase"
        )

    if self.do_fuse:
        awq_version_supports_fusing = False
        MIN_AWQ_VERSION = "0.1.7"
        if is_auto_awq_available():
            awq_version_supports_fusing = version.parse(importlib.metadata.version("autoawq")) >= version.parse(
                MIN_AWQ_VERSION
            )

        if not awq_version_supports_fusing:
            raise ValueError(
                f"You current version of `autoawq` does not support module fusing, please upgrade `autoawq` package to at least {MIN_AWQ_VERSION}."
            )

    if self.modules_to_not_convert is not None:
        awq_version_supports_non_conversion = False
        MIN_AWQ_VERSION = "0.1.8"
        if is_auto_awq_available():
            awq_version_supports_non_conversion = version.parse(
                importlib.metadata.version("autoawq")
            ) >= version.parse(MIN_AWQ_VERSION)

        if not awq_version_supports_non_conversion:
            raise ValueError(
                f"You current version of `autoawq` does not support module quantization skipping, please upgrade `autoawq` package to at least {MIN_AWQ_VERSION}."
            )

    if self.do_fuse and self.modules_to_fuse is not None:
        raise ValueError("You current implementation of `autoawq` does not support do_fuse and modules_to_fuse.")


def gaudi_replace_with_awq_linear(
    model,
    modules_to_not_convert=None,
    quantization_config=None,
    current_key_name=None,
    has_been_replaced=False,
) -> bool:
    """
    Adapted from: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/integrations/awq.py#L90
    - support HPU.
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    assert quantization_config is not None
    backend = quantization_config.backend

    if not is_auto_awq_available():
        raise ValueError(
            "AWQ (either `autoawq` or `llmawq`) is not available. Please install it with `pip install autoawq` or check out the installation guide in https://github.com/mit-han-lab/llm-awq"
        )

    if backend == AwqBackendPackingMethod.AUTOAWQ and quantization_config.version == GaudiAWQLinearVersion.HPU:
        from ...AutoAWQ.gemm_hpu import WQLinear_HPU

        target_cls = WQLinear_HPU
    else:
        raise ValueError(f"Unrecognized AWQ version: {quantization_config.version} and backend {backend}")

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                in_features = module.in_features
                out_features = module.out_features

                model._modules[name] = target_cls(
                    w_bit=quantization_config.bits,
                    group_size=quantization_config.group_size,
                    in_features=in_features,
                    out_features=out_features,
                    bias=module.bias is not None,
                    dev=module.weight.device,
                )
                has_been_replaced = True

                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = gaudi_replace_with_awq_linear(
                module,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                quantization_config=quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def post_init_awq_gemm_hpu_modules(model):
    """
    Runs post init for gemm hpu layers which performs:
        - Weights unpacking, reordering and repacking
    """
    from ...AutoAWQ.gemm_hpu import hpu_post_init

    model = hpu_post_init(model)

    return model


def gaudi_awq_quantizer_process_model_after_weight_loading(self, model, **kwargs):
    if self.quantization_config.version == GaudiAWQLinearVersion.HPU:
        model = post_init_awq_gemm_hpu_modules(model)
    else:
        raise ValueError(f"Unrecognized AWQ version: {self.quantization_config.version}, only hpu is supported")


def gaudi_awq_quantizer_validate_environment(self, device_map, **kwargs):
    if not is_auto_awq_available():
        raise ImportError("Loading an AWQ quantized model requires auto-awq library (`pip install autoawq`)")

    if not is_accelerate_available():
        raise ImportError("Loading an AWQ quantized model requires accelerate (`pip install accelerate`)")

    if device_map is None:
        logger.warning_once(
            "You have loaded an AWQ model on CPU and have a CUDA device available, make sure to set "
            "your model on a GPU device in order to run your model."
        )
    elif device_map is not None:
        if isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
            raise ValueError(
                "You are attempting to load an AWQ model with a device_map that contains a CPU or disk device."
                " This is not supported. Please remove the CPU or disk device from the device_map."
            )


def gaudi_awq_quantizer_process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
    from transformers.integrations import get_keys_to_not_convert, replace_quantization_scales

    self.modules_to_not_convert = get_keys_to_not_convert(model)

    if self.quantization_config.modules_to_not_convert is not None:
        self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)

    model, has_been_replaced = gaudi_replace_with_awq_linear(
        model, quantization_config=self.quantization_config, modules_to_not_convert=self.modules_to_not_convert
    )

    model = replace_quantization_scales(model, model.config.model_type)

    if not has_been_replaced:
        logger.warning(
            "You are loading an AWQ model but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is a bug."
        )
