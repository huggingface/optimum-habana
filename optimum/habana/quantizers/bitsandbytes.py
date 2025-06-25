from transformers.utils import (
    get_available_devices,
    is_bitsandbytes_multi_backend_available,
    is_ipex_available,
    logging,
)


logger = logging.get_logger(__name__)


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
