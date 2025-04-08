from contextlib import contextmanager
from functools import cache

from optimum.habana.feature_detection_utils import (
    EnvVariable,
    Feature,
    Hardware,
    Kernel,
    Not,
    OptionalModelConfig,
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


class IsFusedRMSNormDisabledInConfig(Feature):
    """
    Represents whether the FusedRMSNorm is disabled in the model configuration.
    """

    def __init__(self):
        super().__init__(OptionalModelConfig("use_fused_rms_norm", value_expected=False))


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
def import_component(module_path, component):
    """
    Imports a specific component from a module.

    Args:
        module_path (str): The path to the module.
        component (str): The component to import.

    Returns:
        object: The imported component, or None if import fails.
    """
    try:
        module = __import__(module_path, fromlist=[component])
        return getattr(module, component)
    except (ImportError, AttributeError):
        return None


@contextmanager
def import_hpex(module_path, component):
    """
    Context manager for conditionally importing components from the hpex module.

    Args:
        module_path (str): The path to the module.
        component (str): The component to import.

    Yields:
        object: The imported component, or None if conditions are not met.
    """
    conditional_components = {"FusedRMSNorm": Not(IsFusedRMSNormDisabledInConfig())}
    if component in conditional_components and not conditional_components[component]:
        yield None
    yield import_component(module_path, component)
