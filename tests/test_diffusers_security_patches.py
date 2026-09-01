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
"""Tests for the Diffusers security patches. These do not require a Gaudi device."""

import json
import threading

import pytest
from diffusers.utils import dynamic_modules_utils

from optimum.habana.diffusers.security_patches import apply_diffusers_security_patches


CUSTOM_PIPELINE_SOURCE = """
from pathlib import Path

Path(r"{marker}").write_text("executed")


class EvilPipeline:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()
"""


@pytest.fixture(autouse=True)
def _patched():
    apply_diffusers_security_patches()


@pytest.fixture
def evil_repo(tmp_path, request):
    """A local 'repo' whose custom pipeline module writes a marker file when imported."""
    repo = tmp_path / "evil_repo"
    repo.mkdir()
    marker = tmp_path / "PWNED"
    # Diffusers derives the `sys.modules` key from the module path, so a name reused across tests
    # would return the cached module and skip re-execution, masking the result.
    module_file = f"evil_pipeline_{request.node.name}.py"
    (repo / module_file).write_text(CUSTOM_PIPELINE_SOURCE.format(marker=marker))
    (repo / "model_index.json").write_text(
        json.dumps({"_class_name": [module_file[:-3], "EvilPipeline"], "_diffusers_version": "0.35.1"})
    )
    return repo, marker, module_file


def test_local_custom_module_is_refused(evil_repo):
    """CVE-2026-44513: a local path must not bypass the trust gate."""
    repo, marker, module_file = evil_repo

    with pytest.raises(ValueError, match="trust_remote_code"):
        dynamic_modules_utils.get_class_from_dynamic_module(
            str(repo), module_file=module_file, class_name="EvilPipeline"
        )

    # The payload runs at import time, so an exception alone would not prove it was blocked.
    assert not marker.exists()


def test_custom_module_is_allowed_when_trusted(evil_repo):
    """The feature must keep working for callers that explicitly opt in."""
    repo, marker, module_file = evil_repo

    from optimum.habana.diffusers.security_patches import _trust_remote_code

    token = _trust_remote_code.set(True)
    try:
        dynamic_modules_utils.get_class_from_dynamic_module(
            str(repo), module_file=module_file, class_name="EvilPipeline"
        )
    finally:
        _trust_remote_code.reset(token)

    assert marker.exists()


@pytest.mark.parametrize("truthy_non_true", ["False", "0", 1, "true"])
def test_only_literal_true_enables_trust(evil_repo, truthy_non_true):
    """Only the literal `True` may enable remote code - any other truthy value must be refused."""
    repo, marker, module_file = evil_repo

    from diffusers import DiffusionPipeline

    with pytest.raises(Exception):  # noqa: PT011 - repo does not exist; the guard must fire first
        DiffusionPipeline.from_pretrained(str(repo), trust_remote_code=truthy_non_true)

    assert not marker.exists()


def test_trust_scope_does_not_leak_across_threads(evil_repo):
    """A ContextVar (not a module global) is required for concurrent pipeline loads."""
    _repo, marker, _module_file = evil_repo
    from optimum.habana.diffusers.security_patches import _trust_remote_code

    seen = {}

    def worker():
        seen["value"] = _trust_remote_code.get()

    token = _trust_remote_code.set(True)
    try:
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    finally:
        _trust_remote_code.reset(token)

    assert seen["value"] is False
    assert not marker.exists()


def test_trust_scope_is_restored_after_failure():
    """A failed load must not leave remote code enabled for subsequent loads."""
    from diffusers import DiffusionPipeline

    from optimum.habana.diffusers.security_patches import _trust_remote_code

    with pytest.raises(Exception):
        DiffusionPipeline.from_pretrained("this/repo-does-not-exist-xyz", trust_remote_code=True)

    assert _trust_remote_code.get() is False


def test_patch_is_idempotent():
    patched = dynamic_modules_utils.get_cached_module_file

    for _ in range(3):
        apply_diffusers_security_patches()

    assert dynamic_modules_utils.get_cached_module_file is patched


def test_patched_symbols_still_exist_upstream():
    """Fails loudly if a Diffusers upgrade moves what the patch targets, instead of silently no-op'ing."""
    from diffusers import DiffusionPipeline

    assert hasattr(dynamic_modules_utils, "get_cached_module_file")
    assert hasattr(dynamic_modules_utils, "get_class_from_dynamic_module")
    assert callable(getattr(DiffusionPipeline, "from_pretrained", None))
