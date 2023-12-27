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

import numpy as np
from trl import PPOConfig, is_wandb_available
from trl.trainer.utils import exact_div


@dataclass
class GaudiPPOConfig(PPOConfig):
    """
    Configuration class for GaudiPPOTrainer
    """

    use_habana: bool = False
    """Use habana. Only applicable if use_habana is True"""
    pad_for_acceleration: bool = False
    """Use pad_for_acceleration. Only applicable if pad_for_acceleration is True"""
    pad_max_len: int = 0
    """Use pad_for_acceleration. Only applicable if pad_for_acceleration is True"""
    pad_max_input_len: int = 0

    def __post_init__(self):
        self.backward_batch_size = self.mini_batch_size * self.gradient_accumulation_steps
        exact_div(
            self.batch_size,
            self.backward_batch_size,
            "`batch_size`",
            "`mini_batch_size * gradient_accumulation_steps`",
            "`batch_size` must be a multiple of `mini_batch_size * gradient_accumulation_steps`",
        )
        self.total_ppo_epochs = int(np.ceil(self.steps / self.batch_size))

        # check if wandb is installed
        if self.log_with == "wandb":
            # raise error if wandb is not installed
            if not is_wandb_available():
                raise ImportError(
                    "Please install wandb to use wandb logging. You can do this by running `pip install wandb`."
                )

        if self.pad_for_acceleration:
            if self.pad_max_input_len == 0:
                raise AssertionError("pad_max_input_len ({self.pad_max_input_len}) must be set for pad input ")
            if self.pad_max_input_len >= self.pad_max_len:
                raise AssertionError(
                    "pad_max_input_len ({self.pad_max_input_len}) must be smaller "
                    " then pad_max_len ({self.pad_max_len})"
                )

        if self.use_habana:
            from optimum.habana.transformers.modeling_utils import (  # pylint: disable=E0611, E0401
                adapt_transformers_to_gaudi,
            )

            adapt_transformers_to_gaudi()

        assert self.kl_penalty in ["kl", "abs", "mse", "full"]
