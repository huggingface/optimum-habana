# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run
import doctest
import warnings

import pytest
import _pytest

from transformers.testing_utils import HfDoctestModule, HfDocTestParser


NOT_DEVICE_TESTS = {
    "test_tokenization",
    "test_processor",
    "test_processing",
    "test_beam_constraints",
    "test_configuration_utils",
    "test_data_collator",
    "test_trainer_callback",
    "test_trainer_utils",
    "test_feature_extraction",
    "test_image_processing",
    "test_image_processor",
    "test_image_transforms",
    "test_optimization",
    "test_retrieval",
    "test_config",
    "test_from_pretrained_no_checkpoint",
    "test_keep_in_fp32_modules",
    "test_gradient_checkpointing_backward_compatibility",
    "test_gradient_checkpointing_enable_disable",
    "test_save_load_fast_init_from_base",
    "test_fast_init_context_manager",
    "test_fast_init_tied_embeddings",
    "test_save_load_fast_init_to_base",
    "test_torch_save_load",
    "test_initialization",
    "test_forward_signature",
    "test_model_get_set_embeddings",
    "test_model_main_input_name",
    "test_correct_missing_keys",
    "test_tie_model_weights",
    "test_can_use_safetensors",
    "test_load_save_without_tied_weights",
    "test_tied_weights_keys",
    "test_model_weights_reload_no_missing_tied_weights",
    "test_pt_tf_model_equivalence",
    "test_mismatched_shapes_have_properly_initialized_weights",
    "test_matched_shapes_have_loaded_weights_when_some_mismatched_shapes_exist",
    "test_model_is_small",
    "test_tf_from_pt_safetensors",
    "test_flax_from_pt_safetensors",
    "ModelTest::test_pipeline_",  # None of the pipeline tests from PipelineTesterMixin (of which XxxModelTest inherits from) are running on device
    "ModelTester::test_pipeline_",
    "/repo_utils/",
    "/utils/",
    "/agents/",
}

# silence FutureWarning warnings in tests since often we can't act on them until
# they become normal warnings - i.e. the tests still need to test the current functionality
warnings.simplefilter(action="ignore", category=FutureWarning)


class Secret:
    """
    Taken from: https://stackoverflow.com/a/67393351
    """

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "Secret(********)"

    def __str___(self):
        return "*******"

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "is_pt_tf_cross_test: mark test to run only when PT and TF interactions are tested"
    )
    config.addinivalue_line(
        "markers", "is_pt_flax_cross_test: mark test to run only when PT and FLAX interactions are tested"
    )
    config.addinivalue_line("markers", "is_pipeline_test: mark test to run only when pipelines are tested")
    config.addinivalue_line("markers", "is_staging_test: mark test to run only in the staging environment")
    config.addinivalue_line("markers", "accelerate_tests: mark test that require accelerate")
    config.addinivalue_line("markers", "agent_tests: mark the agent tests that are run on their specific schedule")
    config.addinivalue_line("markers", "not_device_test: mark the tests always running on cpu")

def pytest_collection_modifyitems(items):
    for item in items:
        if any(test_name in item.nodeid for test_name in NOT_DEVICE_TESTS):
            item.add_marker(pytest.mark.not_device_test)

def pytest_addoption(parser):
    from transformers.testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)

def pytest_terminal_summary(terminalreporter):
    from transformers.testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)

def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = Secret(metafunc.config.option.token)
    if "token" in metafunc.fixturenames:
        metafunc.parametrize("token", [option_value])

# Doctest custom flag to ignore output.
IGNORE_RESULT = doctest.register_optionflag("IGNORE_RESULT")

OutputChecker = doctest.OutputChecker


class CustomOutputChecker(OutputChecker):
    def check_output(self, want, got, optionflags):
        if IGNORE_RESULT & optionflags:
            return True
        return OutputChecker.check_output(self, want, got, optionflags)


doctest.OutputChecker = CustomOutputChecker
_pytest.doctest.DoctestModule = HfDoctestModule
doctest.DocTestParser = HfDocTestParser
