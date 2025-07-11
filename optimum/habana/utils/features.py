from functools import cache

from optimum.habana.utils.feature_detection_utils import (
    EnvVariable,
    Feature,
    Hardware,
    IsKernelExplicitlyDisabled,
    Kernel,
    Not,
    SynapseVersionRange,
)


class IsGaudi1Available(Feature):
    """
    Represents the availability of Gaudi1 hardware.
    """

    def __init__(self):
        super().__init__(Hardware("gaudi"))


class IsGaudi2Available(Feature):
    """
    Represents the availability of Gaudi2 hardware.
    """

    def __init__(self):
        super().__init__(Hardware("gaudi2"))


class IsGaudi3Available(Feature):
    """
    Represents the availability of Gaudi3 hardware.
    """

    def __init__(self):
        super().__init__(Hardware("gaudi3"))


class IsGaudiAvailable(Feature):
    """
    Represents the availability of any Gaudi hardware (Gaudi1, Gaudi2, or Gaudi3).
    """

    def __init__(self):
        super().__init__(IsGaudi1Available() or IsGaudi2Available() or IsGaudi3Available())


class IsFusedRMSNormAvailable(Feature):
    """
    Represents the availability of the FusedRMSNorm kernel.
    """

    def __init__(self):
        super().__init__(Kernel("habana_frameworks.torch.hpex.normalization", "FusedRMSNorm"))


class IsLazyMode(Feature):
    """
    Represents whether lazy mode is enabled via environment variable.
    """

    def __init__(self):
        super().__init__(EnvVariable("PT_HPU_LAZY_MODE", "1"))


class IsSynapsePublicVersion(Feature):
    """
    Represents whether the Synapse version is within the public release range.
    """

    def __init__(self):
        super().__init__(SynapseVersionRange(">=1.20.0", "<1.21.0"))


class IsSynapseUnreleasedVersion(Feature):
    """
    Represents whether the Synapse version is an unreleased version.
    """

    def __init__(self):
        super().__init__(SynapseVersionRange(">=1.21.0"))


@cache
def import_usable_component(module_path, component):
    """
    Imports a specific component from a module and checks if it can be used.

    Args:
        module_path (str): The path to the module.
        component (str): The component to import.

    Returns:
        object: The imported component, or None if import fails or is not usable.
    """
    try:
        module = __import__(module_path, fromlist=[component])
        imported_component = getattr(module, component)
        if is_component_usable(imported_component):
            return imported_component, True
        return None, False
    except (ImportError, AttributeError):
        return None, False


@cache
def get_conditional_components():
    """
    Initializes the conditional components and their conditions.

    Returns:
        dict: A dictionary mapping component names to their conditions.
    """
    print("Initializing conditional components...")
    return {
        "FusedRMSNorm": Not(IsKernelExplicitlyDisabled("FusedRMSNorm")),
    }


@cache
def is_component_usable(component):
    """
    Determines if a specific component can be used based on precomputed conditions.

    Args:
        component (object): The component to check (imported Python object or None).

    Returns:
        bool: True if the component can be used, False otherwise.
    """
    component_name = component.__name__
    conditional_components = get_conditional_components()
    return conditional_components.get(component_name, True)
