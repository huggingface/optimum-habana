# coding=utf-8
# Copyright 2022-present the HuggingFace Inc. team.
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
"""
Callbacks to use with the GaudiTrainer class to customize the training loop.
"""
import numpy as np
from tqdm.auto import tqdm

from habana_frameworks.torch.hpu import memory_stats
from transformers.trainer_callback import ProgressCallback


class GaudiProgressCallback(ProgressCallback):
    """
    Built on top of [`ProgressCallback`] to display live memory consumption on Gaudi.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar = tqdm(total=state.max_steps)
            self.training_bar.postfix = self.hpu_memory_stats_as_string()
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step
            self.training_bar.postfix = self.hpu_memory_stats_as_string()

    @staticmethod
    def to_gb_rounded(mem: float):
        """
        Rounds and converts to GB.
        """
        return np.round(mem / (1024 * 1024 * 1024), 2)

    @staticmethod
    def hpu_memory_stats_as_string():
        """
        Returns memory stats of HPU as a string to be printed.
        """
        mem_stats = memory_stats()

        current_memory_consumption = GaudiProgressCallback.to_gb_rounded(mem_stats["InUse"])
        max_memory_consumption = GaudiProgressCallback.to_gb_rounded(mem_stats["MaxInUse"])
        total_memory = GaudiProgressCallback.to_gb_rounded(mem_stats["Limit"])

        return f"mem: {current_memory_consumption} GB (max: {max_memory_consumption} GB, total: {total_memory} GB)"
