# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the Transformers security patches. These do not require a Gaudi device."""

import pytest
import transformers
from transformers import LlamaConfig

from optimum.habana.transformers.security_patches import (
    BLOCKED_CONFIG_KWARGS,
    apply_transformers_security_patches,
)


MALICIOUS_REPO_ID = "attacker/malicious-kernel-repo"


@pytest.fixture(autouse=True)
def _patched():
    apply_transformers_security_patches()


@pytest.mark.parametrize("blocked_key", BLOCKED_CONFIG_KWARGS)
def test_internal_implementation_field_is_dropped_from_config_dict(blocked_key):
    """CVE-2026-4372: a config file must not be able to set the internal attn-dispatch fields."""
    config = LlamaConfig.from_dict({"model_type": "llama", blocked_key: MALICIOUS_REPO_ID})

    assert getattr(config, blocked_key, None) != MALICIOUS_REPO_ID
    assert config._attn_implementation != MALICIOUS_REPO_ID


def test_internal_implementation_field_is_dropped_from_kwargs():
    config = LlamaConfig(_attn_implementation_internal=MALICIOUS_REPO_ID)

    assert config._attn_implementation != MALICIOUS_REPO_ID


def test_public_attn_implementation_argument_still_works():
    """The documented way of selecting an attention implementation must keep working."""
    assert LlamaConfig(attn_implementation="eager")._attn_implementation == "eager"
    assert LlamaConfig.from_dict({"model_type": "llama"}, attn_implementation="sdpa")._attn_implementation == "sdpa"


def test_unknown_config_keys_are_still_accepted():
    """Only the blocked keys are filtered; arbitrary custom fields must survive."""
    assert LlamaConfig.from_dict({"model_type": "llama", "some_custom_field": 42}).some_custom_field == 42


def test_patch_is_idempotent():
    """`adapt_transformers_to_gaudi()` may run more than once; re-patching must not recurse."""
    patched_init = transformers.PretrainedConfig.__init__

    for _ in range(3):
        apply_transformers_security_patches()

    assert transformers.PretrainedConfig.__init__ is patched_init
    assert LlamaConfig(_attn_implementation_internal=MALICIOUS_REPO_ID)._attn_implementation != MALICIOUS_REPO_ID


def test_patched_symbols_still_exist_upstream():
    """Fails loudly if a Transformers upgrade moves what the patch targets, instead of silently no-op'ing."""
    assert hasattr(transformers, "PretrainedConfig")
    assert hasattr(transformers.configuration_utils, "PretrainedConfig")
    assert transformers.PretrainedConfig is transformers.configuration_utils.PretrainedConfig
    assert hasattr(transformers.PretrainedConfig, "_attn_implementation")
