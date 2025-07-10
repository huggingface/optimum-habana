import os
import unittest
from unittest.mock import patch

# Assuming the feature detection code is in a module named feature_detection
from optimum.habana.utils.features import (
    Feature,
    IsFusedRMSNormAvailable,
    IsGaudi1Available,
    IsGaudi2Available,
    IsGaudi3Available,
    IsGaudiAvailable,
    IsLazyMode,
    IsSynapsePublicVersion,
    IsSynapseUnreleasedVersion,
    Not,
)


class TestFeatureDetection(unittest.TestCase):
    """
    Unit tests for feature detection functionality.
    """

    def setUp(self):
        """
        Sets up the test environment by initializing necessary environment variables.
        """
        os.environ["PT_HPU_LAZY_MODE"] = "1"
        Feature.reset_environment()

    @patch("optimum.habana.feature_detection_utils.get_environment")
    def test_gaudi1_enabled(self, mock_get_environment):
        """
        Tests the detection of Gaudi1 hardware.
        """
        mock_get_environment.return_value = {"hw": "gaudi"}

        self.assertTrue(IsGaudiAvailable())
        self.assertTrue(IsGaudi1Available())
        self.assertFalse(IsGaudi2Available())
        self.assertFalse(IsGaudi3Available())
        self.assertTrue(IsGaudiAvailable())

    @patch("optimum.habana.feature_detection_utils.get_environment")
    def test_gaudi2_enabled(self, mock_get_environment):
        """
        Tests the detection of Gaudi2 hardware.
        """
        mock_get_environment.return_value = {"hw": "gaudi2"}

        self.assertFalse(IsGaudi1Available())
        self.assertTrue(IsGaudi2Available())
        self.assertFalse(IsGaudi3Available())
        self.assertTrue(IsGaudiAvailable())

    @patch("optimum.habana.feature_detection_utils.get_environment")
    def test_gaudi3_enabled(self, mock_get_environment):
        """
        Tests the detection of Gaudi3 hardware.
        """
        mock_get_environment.return_value = {"hw": "gaudi3"}

        self.assertFalse(IsGaudi1Available())
        self.assertFalse(IsGaudi2Available())
        self.assertTrue(IsGaudiAvailable())
        self.assertTrue(IsGaudi3Available())

    @patch("optimum.habana.feature_detection_utils.get_environment")
    def test_not_operator(self, mock_get_environment):
        """
        Tests the logical NOT operator for feature detection.
        """
        mock_get_environment.return_value = {"hw": "gaudi"}
        self.assertTrue(IsGaudi1Available())
        self.assertFalse(Not(IsGaudi1Available()))
        self.assertTrue(Not(Not(IsGaudi1Available())))
        self.assertFalse(IsGaudi2Available())

    @patch("optimum.habana.feature_detection_utils.get_environment")
    def test_or_operator(self, mock_get_environment):
        """
        Tests the logical OR operator for feature detection.
        """
        mock_get_environment.return_value = {"hw": "gaudi"}
        self.assertTrue(IsGaudi1Available() or IsGaudi2Available())
        self.assertTrue(IsGaudi2Available() or IsGaudi1Available())
        self.assertFalse(IsGaudi2Available() or IsGaudi3Available())

    @patch("optimum.habana.feature_detection_utils.get_environment")
    def test_and_operator(self, mock_get_environment):
        """
        Tests the logical AND operator for feature detection.
        """
        mock_get_environment.return_value = {"hw": "gaudi"}
        self.assertTrue(IsGaudi1Available() and Not(IsGaudi2Available()))
        self.assertFalse(IsGaudi1Available() and IsGaudi2Available())
        self.assertFalse(IsGaudi2Available() and IsGaudi3Available())

    @patch("optimum.habana.environment.get_hw")
    def test_env_flag(self, mock_get_hw):
        """
        Tests the detection of lazy mode via environment variable.
        """
        mock_get_hw.return_value = None
        self.assertTrue(IsLazyMode())

    @patch("optimum.habana.environment.get_hw")
    def test_version(self, mock_get_hw):
        """
        Tests the detection of Synapse version ranges.
        """
        mock_get_hw.return_value = None
        self.assertFalse(IsSynapsePublicVersion())
        self.assertTrue(IsSynapseUnreleasedVersion())

    @patch("optimum.habana.environment.get_hw")
    def test_kernels(self, mock_get_hw):
        """
        Tests the detection of kernel availability.
        """
        mock_get_hw.return_value = None
        self.assertTrue(IsFusedRMSNormAvailable())


if __name__ == "__main__":
    unittest.main()
