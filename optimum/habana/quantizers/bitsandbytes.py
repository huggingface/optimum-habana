from functools import lru_cache
from typing import Any, Dict, List, Optional

from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from transformers.quantizers.quantizers_utils import get_module_from_name
from transformers.utils import (
    ACCELERATE_MIN_VERSION,
    get_available_devices,
    is_accelerate_available,
    is_bitsandbytes_multi_backend_available,
    is_ipex_available,
    is_torch_available,
    logging,
)
from transformers.utils.import_utils import _is_package_available


if is_torch_available():
    import torch

_bitsandbytes_available = _is_package_available("bitsandbytes")
logger = logging.get_logger(__name__)


def gaudi_bitsandbytesconfig_post_init(self):
    r"""
    Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
    Copied from https://github.com/huggingface/transformers/blob/53fad641cfdb5105e2470bcf3ef17ea8e25cc300/src/transformers/utils/quantization_config.py#L430
    Only difference is removed check on bitsandbytes version
    """
    if not isinstance(self.load_in_4bit, bool):
        raise TypeError("load_in_4bit must be a boolean")

    if not isinstance(self.load_in_8bit, bool):
        raise TypeError("load_in_8bit must be a boolean")

    if not isinstance(self.llm_int8_threshold, float):
        raise TypeError("llm_int8_threshold must be a float")

    if self.llm_int8_skip_modules is not None and not isinstance(self.llm_int8_skip_modules, list):
        raise TypeError("llm_int8_skip_modules must be a list of strings")
    if not isinstance(self.llm_int8_enable_fp32_cpu_offload, bool):
        raise TypeError("llm_int8_enable_fp32_cpu_offload must be a boolean")

    if not isinstance(self.llm_int8_has_fp16_weight, bool):
        raise TypeError("llm_int8_has_fp16_weight must be a boolean")

    if self.bnb_4bit_compute_dtype is not None and not isinstance(self.bnb_4bit_compute_dtype, torch.dtype):
        raise TypeError("bnb_4bit_compute_dtype must be torch.dtype")

    if not isinstance(self.bnb_4bit_quant_type, str):
        raise TypeError("bnb_4bit_quant_type must be a string")

    if not isinstance(self.bnb_4bit_use_double_quant, bool):
        raise TypeError("bnb_4bit_use_double_quant must be a boolean")


@lru_cache()
def gaudi_is_bitsandbytes_available():
    """
    Copied from https://github.com/huggingface/transformers/blob/5523e38b553ff6c46b04d2376870fcd842feeecc/src/transformers/utils/import_utils.py#L871
    Only difference is that CUDA related checks are removed.
    """
    if not is_torch_available() or not _bitsandbytes_available:
        return False

    # Newer versions of `bitsandbytes` can be imported on systems without CUDA.
    return True


def gaudi_validate_bnb_backend_availability(raise_exception=False):
    """
    Validates if the available devices are supported by bitsandbytes, optionally raising an exception if not.
    Copied from https://github.com/huggingface/transformers/blob/5523e38b553ff6c46b04d2376870fcd842feeecc/src/transformers/integrations/bitsandbytes.py#L545
    Only difference is that CUDA related functions calls are deleted.
    """
    if is_bitsandbytes_multi_backend_available():
        return _gaudi_validate_bnb_multi_backend_availability(raise_exception)


def _gaudi_validate_bnb_multi_backend_availability(raise_exception):
    """
    Copied https://github.com/huggingface/transformers/blob/5523e38b553ff6c46b04d2376870fcd842feeecc/src/transformers/integrations/bitsandbytes.py#L484
    Only difference is addition of check for HPU
    """
    import bitsandbytes as bnb

    bnb_supported_devices = getattr(bnb, "supported_torch_devices", set())
    available_devices = get_available_devices()

    if "hpu" in bnb_supported_devices:
        logger.debug("Multi-backend validation successful.")
        return True

    if available_devices == {"cpu"} and not is_ipex_available():
        from importlib.util import find_spec

        if find_spec("intel_extension_for_pytorch"):
            logger.warning(
                "You have Intel IPEX installed but if you're intending to use it for CPU, it might not have the right version. Be sure to double check that your PyTorch and IPEX installs are compatible."
            )

        available_devices.discard("cpu")  # Only Intel CPU is supported by BNB at the moment

    if not available_devices.intersection(bnb_supported_devices):
        if raise_exception:
            bnb_supported_devices_with_info = set(  # noqa: C401
                '"cpu" (needs an Intel CPU and intel_extension_for_pytorch installed and compatible with the PyTorch version)'
                if device == "cpu"
                else device
                for device in bnb_supported_devices
            )
            err_msg = (
                f"None of the available devices `available_devices = {available_devices or None}` are supported by the bitsandbytes version you have installed: `bnb_supported_devices = {bnb_supported_devices_with_info}`. "
                "Please check the docs to see if the backend you intend to use is available and how to install it: https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend"
            )

            logger.error(err_msg)
            raise RuntimeError(err_msg)

        logger.warning("No supported devices found for bitsandbytes multi-backend.")
        return False

    logger.debug("Multi-backend validation successful.")
    return True


def gaudi_validate_environment(self, *args, **kwargs):
    """
    Copied from https://github.com/huggingface/transformers/blob/5523e38b553ff6c46b04d2376870fcd842feeecc/src/transformers/quantizers/quantizer_bnb_4bit.py#L68
    Only difference is deletion of bitsandbytes version checks
    """
    if not is_accelerate_available():
        raise ImportError(
            f"Using `bitsandbytes` 4-bit quantization requires Accelerate: `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
        )
    if not gaudi_is_bitsandbytes_available():
        raise ImportError(
            "Using `bitsandbytes` 4-bit quantization requires the latest version of bitsandbytes: `pip install -U bitsandbytes`"
        )

    bnb_multibackend_is_enabled = is_bitsandbytes_multi_backend_available()
    gaudi_validate_bnb_backend_availability(raise_exception=True)

    if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
        raise ValueError(
            "Converting into 4-bit or 8-bit weights from tf/flax weights is currently not supported, please make"
            " sure the weights are in PyTorch format."
        )

    device_map = kwargs.get("device_map", None)
    if (
        device_map is not None
        and isinstance(device_map, dict)
        and not self.quantization_config.llm_int8_enable_fp32_cpu_offload
    ):
        device_map_without_lm_head = {
            key: device_map[key] for key in device_map.keys() if key not in self.modules_to_not_convert
        }
        if set(device_map.values()) == {"cpu"} and bnb_multibackend_is_enabled:
            pass
        elif "cpu" in device_map_without_lm_head.values() or "disk" in device_map_without_lm_head.values():
            raise ValueError(
                "Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the "
                "quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules "
                "in 32-bit, you need to set `load_in_8bit_fp32_cpu_offload=True` and pass a custom `device_map` to "
                "`from_pretrained`. Check "
                "https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu "
                "for more details. "
            )


def gaudi_create_quantized_param(
    self,
    model: "PreTrainedModel",
    param_value: "torch.Tensor",
    param_name: str,
    target_device: "torch.device",
    state_dict: Dict[str, Any],
    unexpected_keys: Optional[List[str]] = None,
):
    """
    Copied from https://github.com/huggingface/transformers/blob/62c60a30181a65e1a3a7f19c3055a240a6a21335/src/transformers/quantizers/quantizer_bnb_4bit.py#L138
    only diiference is addition of HPU device
    """
    import bitsandbytes as bnb

    module, tensor_name = get_module_from_name(model, param_name)

    if tensor_name not in module._parameters:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")

    old_value = getattr(module, tensor_name)

    if tensor_name == "bias":
        if param_value is None:
            new_value = old_value.to(target_device)
        else:
            new_value = param_value.to(target_device)

        new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad)
        module._parameters[tensor_name] = new_value
        return

    if not isinstance(module._parameters[tensor_name], bnb.nn.Params4bit):
        raise ValueError("this function only loads `Linear4bit components`")
    if (
        old_value.device == torch.device("meta")
        and target_device not in ["meta", torch.device("meta")]
        and param_value is None
    ):
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {target_device}.")

    # construct `new_value` for the module._parameters[tensor_name]:
    if self.pre_quantized:
        # 4bit loading. Collecting components for restoring quantized weight
        # This can be expanded to make a universal call for any quantized weight loading

        if not self.is_serializable:
            raise ValueError(
                "Detected int4 weights but the version of bitsandbytes is not compatible with int4 serialization. "
                "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
            )

        if (param_name + ".quant_state.bitsandbytes__fp4" not in state_dict) and (
            param_name + ".quant_state.bitsandbytes__nf4" not in state_dict
        ):
            raise ValueError(
                f"Supplied state dict for {param_name} does not contain `bitsandbytes__*` and possibly other `quantized_stats` components."
            )

        quantized_stats = {}
        for k, v in state_dict.items():
            if param_name + "." in k:
                quantized_stats[k] = v
                if unexpected_keys is not None and k in unexpected_keys:
                    unexpected_keys.remove(k)

        param_kwargs = {}
        if self.is_bnb_supports_quant_storage_module:
            param_kwargs["module"] = module

        new_value = bnb.nn.Params4bit.from_prequantized(
            data=param_value,
            quantized_stats=quantized_stats,
            requires_grad=False,
            device=target_device,
            **param_kwargs,
        )
    else:
        if target_device == "hpu":
            new_value = param_value.to("hpu")
        else:
            new_value = param_value.to("cpu")

        # Support models using `Conv1D` in place of `nn.Linear` (e.g. openai-community/gpt2) by transposing the weight matrix prior to quantization.
        # Since weights are saved in the correct "orientation", we skip transposing when loading.
        if issubclass(module.source_cls, Conv1D):
            new_value = new_value.T

        kwargs = old_value.__dict__
        new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(target_device)

    module._parameters[tensor_name] = new_value
