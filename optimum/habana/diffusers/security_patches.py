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
"""Security patches applied to Diffusers when `optimum.habana.diffusers` is imported.

This module deliberately imports nothing from `habana_frameworks`, so the patches can be
unit-tested on any machine without a Gaudi device.
"""

import contextvars
import functools

from diffusers import DiffusionPipeline
from diffusers.utils import dynamic_modules_utils
from optimum.utils import logging


logger = logging.get_logger(__name__)


# CVE-2026-44513 / CVE-2026-45804: in diffusers <= 0.37 the `trust_remote_code` gate lives in
# `DiffusionPipeline.download()`, which is skipped for local paths and only inspects the main repo,
# and which resolves the revision separately from the download that actually runs the code. The gate
# is moved here, to the single point where any dynamic module is resolved, so local paths, local
# snapshots and Hub repos are all covered and there is no resolve-then-fetch window left to race.
# The caller's choice is carried in a ContextVar because the intermediate diffusers functions have
# no `trust_remote_code` parameter to thread it through.
_trust_remote_code = contextvars.ContextVar("gaudi_diffusers_trust_remote_code", default=False)

_original_get_cached_module_file = dynamic_modules_utils.get_cached_module_file


@functools.wraps(_original_get_cached_module_file)
def gaudi_get_cached_module_file(pretrained_model_name_or_path, module_file, *args, **kwargs):
    if not _trust_remote_code.get():
        raise ValueError(
            f"Loading {module_file} from {pretrained_model_name_or_path} would execute custom code "
            "that is not part of Diffusers. Pass `trust_remote_code=True` to `from_pretrained()` if "
            "you have read that code and trust it."
        )
    logger.warning(f"`trust_remote_code` is enabled: executing custom code from {pretrained_model_name_or_path}.")
    return _original_get_cached_module_file(pretrained_model_name_or_path, module_file, *args, **kwargs)


gaudi_get_cached_module_file._is_gaudi_security_patch = True


def _with_trust_remote_code_scope(original):
    """Publishes the caller's `trust_remote_code` value for the duration of the wrapped call."""

    @functools.wraps(original)
    def wrapper(cls, *args, **kwargs):
        # Fail closed on anything but the literal `True` (e.g. `"False"` or `1` must not enable it).
        token = _trust_remote_code.set(kwargs.get("trust_remote_code", False) is True)
        try:
            return original(cls, *args, **kwargs)
        finally:
            _trust_remote_code.reset(token)

    wrapper._is_gaudi_security_patch = True
    return classmethod(wrapper)


def apply_diffusers_security_patches():
    """Applies the Diffusers security patches. Safe to call more than once."""
    if getattr(dynamic_modules_utils.get_cached_module_file, "_is_gaudi_security_patch", False):
        return

    # `get_class_from_dynamic_module` resolves this by module global at call time, so patching the
    # module attribute is enough to cover every caller.
    dynamic_modules_utils.get_cached_module_file = gaudi_get_cached_module_file

    DiffusionPipeline.from_pretrained = _with_trust_remote_code_scope(DiffusionPipeline.from_pretrained.__func__)

    # Modular pipelines have their own (correct) gate, but they resolve modules through the same
    # chokepoint, so they must publish their decision too or they would be blocked outright.
    try:
        from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks
    except ImportError:
        return
    ModularPipelineBlocks.from_pretrained = _with_trust_remote_code_scope(
        ModularPipelineBlocks.from_pretrained.__func__
    )
