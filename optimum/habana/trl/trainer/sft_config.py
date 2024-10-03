# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Dict, Optional

from ... import GaudiTrainingArguments


@dataclass
class GaudiSFTConfig(GaudiTrainingArguments):
    r"""
    Initialize GaudiSFTConfig.
        Adapted from https://github.com/huggingface/trl/blob/v0.9.6/trl/trainer/sft_config.py#L21
        - inherit from GaudiTrainingArguments
    """

    dataset_text_field: Optional[str] = None
    packing: Optional[bool] = True
    max_seq_length: Optional[int] = 1024
    pad_max: Optional[bool] = True
    dataset_num_proc: Optional[int] = None
    dataset_batch_size: int = 1000
    neftune_noise_alpha: Optional[float] = None
    model_init_kwargs: Optional[Dict] = None
    dataset_kwargs: Optional[Dict] = None
    eval_packing: Optional[bool] = None
    num_of_sequences: Optional[int] = 1024
    chars_per_token: Optional[float] = 3.6
