from functools import cache

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from optimum.habana.utils.environment import get_environment
from optimum.utils import logging


logger = logging.get_logger(__name__)
environment = None


class Feature:
    """
    Represents a feature with a specific check and required parameters.

    Attributes:
        _check (callable): The function to check the feature.
        _required_params (set): The set of required parameters for the check.
        _enabled (bool): The cached result of the feature check.
        _assert_missing (bool): Whether to assert if required parameters are missing.
    """

    @classmethod
    def get_environment(cls):
        """
        Retrieves the current environment information.

        Returns:
            dict: The environment information.
        """
        global environment
        if environment is None:
            environment = get_environment()
            logger.info(f"Detected environment: {environment}")
        return environment

    @classmethod
    def reset_environment(cls):
        """
        Resets the cached environment information.
        """
        global environment
        environment = None

    def __init__(self, check, *required_params, assert_missing=True):
        self._check = check
        self._required_params = set(required_params)
        if isinstance(check, Feature):
            self._required_params.update(check._required_params)
        self._enabled = None
        self._assert_missing = assert_missing

    def __call__(self, **kwargs):
        """
        Executes the feature check with the provided parameters.

        Args:
            **kwargs: The parameters for the feature check.

        Returns:
            bool: The result of the feature check.
        """
        missing_params = self._required_params - kwargs.keys()
        if self._assert_missing:
            assert len(missing_params) == 0, f"Missing keys: {missing_params}!"
        params = {k: v for k, v in kwargs.items() if k in self._required_params}
        missing_values = {k for k, v in params.items() if v is None}
        if self._assert_missing:
            assert len(missing_values) == 0, f"Missing values for parameters: {missing_values}!"
        return self.check(**params)

    def check(self, **kwargs):
        """
        Performs the feature check.

        Args:
            **kwargs: The parameters for the feature check.

        Returns:
            bool: The result of the feature check.
        """
        return self._check(**kwargs)

    def __bool__(self):
        """
        Evaluates the feature as a boolean.

        Returns:
            bool: The cached result of the feature check.
        """
        if self._enabled is None:
            environment = self.get_environment()
            self._enabled = self.check(**environment)
        return self._enabled

    def __and__(self, rhs):
        """
        Combines this feature with another using logical AND.

        Args:
            rhs (Feature): The other feature to combine.

        Returns:
            And: The combined feature.
        """
        return And(self, rhs)

    def __or__(self, rhs):
        """
        Combines this feature with another using logical OR.

        Args:
            rhs (Feature): The other feature to combine.

        Returns:
            Or: The combined feature.
        """
        return Or(self, rhs)


class Not(Feature):
    """
    Represents the negation of a feature.

    Attributes:
        child (Feature): The feature to negate.
    """

    def __init__(self, child):
        super().__init__(self.check, *child._required_params)
        self.child = child

    def check(self, **kwargs):
        """
        Performs the negation check.

        Args:
            **kwargs: The parameters for the feature check.

        Returns:
            bool: The negated result of the feature check.
        """
        return not self.child.check(**kwargs)


class And(Feature):
    """
    Represents the logical AND of two features.

    Attributes:
        lhs (Feature): The left-hand side feature.
        rhs (Feature): The right-hand side feature.
    """

    def __init__(self, lhs, rhs):
        super().__init__(self.check, *lhs._required_params, *rhs._required_params)
        self.lhs = lhs
        self.rhs = rhs

    def check(self, **kwargs):
        """
        Performs the logical AND check.

        Args:
            **kwargs: The parameters for the feature check.

        Returns:
            bool: The result of the logical AND check.
        """
        return self.lhs(**kwargs) and self.rhs(**kwargs)


class Or(Feature):
    """
    Represents the logical OR of two features.

    Attributes:
        lhs (Feature): The left-hand side feature.
        rhs (Feature): The right-hand side feature.
    """

    def __init__(self, lhs, rhs):
        super().__init__(self.check, *lhs._required_params, *rhs._required_params)
        self.lhs = lhs
        self.rhs = rhs

    def check(self, **kwargs):
        """
        Performs the logical OR check.

        Args:
            **kwargs: The parameters for the feature check.

        Returns:
            bool: The result of the logical OR check.
        """
        return self.lhs(**kwargs) or self.rhs(**kwargs)


class Value(Feature):
    """
    Represents a feature that checks for a specific value.

    Attributes:
        key (str): The key to check.
        value: The expected value.
    """

    def __init__(self, key, value):
        super().__init__(self.check, key)
        self.key = key
        self.value = value

    def check(self, **kwargs):
        """
        Performs the value check.

        Args:
            **kwargs: The parameters for the feature check.

        Returns:
            bool: The result of the value check.
        """
        return kwargs[self.key] == self.value


class Hardware(Value):
    """
    Represents a feature that checks for a specific hardware type.

    Attributes:
        target_hw (str): The target hardware type.
    """

    def __init__(self, target_hw):
        super().__init__("hw", target_hw)


class EnvVariable(Feature):
    """
    Represents a feature that checks for a specific environment variable.

    Attributes:
        key (str): The key of the environment variable.
        value_expected: The expected value of the environment variable.
    """

    def __init__(self, key, value_expected):
        super().__init__(self.check, "environment_variables")
        self.key = key
        self.value_expected = value_expected

    def check(self, **kwargs):
        """
        Performs the environment variable check.

        Args:
            **kwargs: The parameters for the feature check.

        Returns:
            bool: The result of the environment variable check.
        """
        return (
            kwargs["environment_variables"][self.key] == self.value_expected
            if self.key in kwargs["environment_variables"]
            else False
        )


class Kernel(Feature):
    """
    Represents a feature that checks for the presence of a specific kernel in a module.

    Attributes:
        module_path (str): The path to the module.
        kernel_name (str): The name of the kernel.
    """

    def __init__(self, module_path, kernel_name):
        super().__init__(self.check)
        self.module_path = module_path
        self.kernel_name = kernel_name

    @cache
    def check(self):
        """
        Performs the kernel presence check.

        Returns:
            bool: True if the kernel is present, False otherwise.
        """
        try:
            module = __import__(self.module_path, fromlist=[self.kernel_name])
            return hasattr(module, self.kernel_name)
        except (ImportError, AttributeError):
            return False


class IsKernelExplicitlyDisabled(Feature):
    """
    Represents whether a specific kernel is explicitly disabled via runtime configuration.
    """

    def __init__(self, kernel_name):
        super().__init__(self.check, "disabled_kernels")
        self.kernel_name = kernel_name

    def check(self, **kwargs):
        """
        Checks if the kernel is explicitly disabled via runtime configuration.

        Args:
            **kwargs: The parameters for the feature check.

        Returns:
            bool: True if the kernel is explicitly disabled, False otherwise.
        """
        disabled_kernels = kwargs.get("disabled_kernels", set())
        return self.kernel_name in disabled_kernels


class SynapseVersionRange(Feature):
    """
    Represents a feature that checks if the build version is within a specified range.

    Attributes:
        specifiers (list): A list of version specifiers.
    """

    def __init__(self, *specifiers):
        super().__init__(self.check, "build")
        self.specifiers = [SpecifierSet(s) for s in specifiers]

    def check(self, build):
        """
        Performs the version range check.

        Args:
            build (str): The build version to check.

        Returns:
            bool: True if the build version is within the specified range, False otherwise.
        """
        version = Version(build)
        return all(version in s for s in self.specifiers)
