import os

from optimum.utils import logging

from ..utils import get_device_name


logger = logging.get_logger(__name__)


def get_hw():
    """
    Determines the type of Habana device being used.

    Returns:
        str: The type of Habana device ('gaudi', 'gaudi2', 'gaudi3') or None if unknown.
    """
    try:
        return get_device_name()
    except ValueError as e:
        logger.warning(f"Unknown device type: {e}")
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


def get_disabled_kernels():
    """
    Retrieves the current disabled kernels stored in runtime parameters.

    Returns:
        dict: The disabled kernels dictionary or None if not set.
    """
    return runtime_params.get("disabled_kernels", {})


def set_kernel_availability(kernel_name, enabled):
    """
    Enables or disables a specific kernel at runtime.

    Args:
        kernel_name (str): The name of the kernel to enable or disable.
        enabled (bool): True to enable the kernel, False to disable it.
    """
    global runtime_params
    if "disabled_kernels" not in runtime_params:
        runtime_params["disabled_kernels"] = set()
    if enabled:
        runtime_params["disabled_kernels"].discard(kernel_name)
    else:
        runtime_params["disabled_kernels"].add(kernel_name)


def disable_kernel(kernel_name):
    """
    Disables a specific kernel at runtime.

    Args:
        kernel_name (str): The name of the kernel to disable.
    """
    global runtime_params
    if "disabled_kernels" not in runtime_params:
        runtime_params["disabled_kernels"] = set()
    runtime_params["disabled_kernels"].add(kernel_name)


def get_environment_variables():
    """
    Retrieves the current environment variables.

    Returns:
        dict: A dictionary of environment variables.
    """
    return dict(os.environ)


def get_environment(**overrides):
    """
    Constructs a dictionary of environment information including build version, hardware type, and environment variables.

    Args:
        **overrides: Optional overrides for specific environment information.

    Returns:
        dict: A dictionary containing environment information.
    """
    overrides = {k: lambda: v for k, v in overrides.items()}
    getters = {
        "build": get_build,
        "hw": get_hw,
        "environment_variables": get_environment_variables,
        "disabled_kernels": get_disabled_kernels,
    }
    return {k: g() for k, g in (getters | overrides).items()}
