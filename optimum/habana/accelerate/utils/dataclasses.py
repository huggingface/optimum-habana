# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from enum import Enum


class GaudiDistributedType(str, Enum):
    """
    Represents a type of distributed environment.
    Adapted from: https://github.com/huggingface/accelerate/blob/8514c35192ac9762920f1ab052e5cea4c0e46eeb/src/accelerate/utils/dataclasses.py#L176

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **MULTI_HPU** -- Distributed on multiple HPUs.
        - **DEEPSPEED** -- Using DeepSpeed.
    """

    # Subclassing str as well as Enum allows the `GaudiDistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    MULTI_HPU = "MULTI_HPU"
    DEEPSPEED = "DEEPSPEED"
