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
"""Security patches applied to Transformers by `adapt_transformers_to_gaudi()`.

This module deliberately imports nothing from `habana_frameworks`, so the patches can be
unit-tested on any machine without a Gaudi device.
"""

import transformers
from optimum.utils import logging


logger = logging.get_logger(__name__)


# CVE-2026-4372: `_attn_implementation_internal` is a computed, private field. Transformers resolves
# it as a kernel repo id and imports code from the Hub, without consulting `trust_remote_code`, so a
# crafted config.json that sets it directly is an arbitrary code execution vector. Callers must go
# through the public `attn_implementation` argument instead.
# `_experts_implementation_internal` does not exist in transformers 4.55.x; it is listed for
# forward-compatibility with the upstream fix.
BLOCKED_CONFIG_KWARGS = ("_attn_implementation_internal", "_experts_implementation_internal")

_original_pretrained_config_init = transformers.PretrainedConfig.__init__


def gaudi_pretrained_config_init(self, **kwargs):
    """
    Wraps `transformers.PretrainedConfig.__init__` to drop internal attention/experts
    implementation fields before they reach its `setattr` loop (CVE-2026-4372).
    """
    for blocked_key in BLOCKED_CONFIG_KWARGS:
        if kwargs.pop(blocked_key, None) is not None:
            logger.warning(
                f"Ignoring `{blocked_key}` found in a model configuration: it is an internal field "
                "and cannot be set from a config file. Use the `attn_implementation` argument of "
                "`from_pretrained()` instead."
            )
    _original_pretrained_config_init(self, **kwargs)


# Set on the patched function rather than on the module, so a re-import cannot mask an applied patch.
gaudi_pretrained_config_init._is_gaudi_security_patch = True


def apply_transformers_security_patches():
    """Applies the Transformers security patches. Safe to call more than once."""
    if getattr(transformers.PretrainedConfig.__init__, "_is_gaudi_security_patch", False):
        return
    transformers.PretrainedConfig.__init__ = gaudi_pretrained_config_init
    transformers.configuration_utils.PretrainedConfig.__init__ = gaudi_pretrained_config_init
