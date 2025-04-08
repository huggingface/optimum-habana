import os

from optimum.utils import logging


logger = logging.get_logger(__name__)


def get_hw():
    """
    Determines the type of Habana device being used.

    Returns:
        str: The type of Habana device ('gaudi', 'gaudi2', 'gaudi3') or None if unknown.
    """
    import habana_frameworks.torch.utils.experimental as htexp

    device_type = htexp._get_device_type()
    match device_type:
        case htexp.synDeviceType.synDeviceGaudi2:
            return "gaudi2"
        case htexp.synDeviceType.synDeviceGaudi3:
            return "gaudi3"
    logger.warning(f"Unknown device type: {device_type}")
    return None


def get_build():
    """
    Retrieves the version of the 'habana-torch-plugin' package installed.

    Returns:
        str: The version of the 'habana-torch-plugin' package or None if not found.
    """
    import re
    import subprocess
    import sys

    output = subprocess.run(
        f"{sys.executable} pip show habana-torch-plugin",
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    version_re = re.compile(r"Version:\s*(?P<version>.*)")
    match = version_re.search(output.stdout)
    if output.returncode == 0 and match:
        return match.group("version")
    else:
        logger.warning("Unknown software version")
        return None


runtime_params = {}


def get_model_config():
    """
    Retrieves the current model configuration stored in runtime parameters.

    Returns:
        dict: The model configuration dictionary or None if not set.
    """
    return runtime_params.get("model_config", {})


def set_model_config(config):
    """
    Sets the model configuration in runtime parameters.

    Args:
        config: The configuration object or dictionary to be stored.
    """
    global runtime_params
    config_dict = vars(config) if hasattr(config, "__dict__") else config
    runtime_params["model_config"] = config_dict


def get_environment_variables():
    """
    Retrieves the current environment variables.

    Returns:
        dict: A dictionary of environment variables.
    """
    return dict(os.environ)


def get_environment(**overrides):
    """
    Constructs a dictionary of environment information including build version, hardware type, model configuration, and environment variables.

    Args:
        **overrides: Optional overrides for specific environment information.

    Returns:
        dict: A dictionary containing environment information.
    """
    overrides = {k: lambda: v for k, v in overrides.items()}
    getters = {
        "build": get_build,
        "hw": get_hw,
        "model_config": get_model_config,
        "environment_variables": get_environment_variables,
    }
    return {k: g() for k, g in (getters | overrides).items()}
