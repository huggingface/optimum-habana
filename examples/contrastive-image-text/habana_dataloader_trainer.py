# coding=utf-8
# Copyright 2023 The HuggingFace Team All rights reserved.
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
A subclass of `GaudiTrainer` specific to habana dataloader
"""

from typing import Optional

import datasets
import torch
from clip_mediapipe_dataloader import MediaApiDataLoader
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSampler,
    DistributedSamplerWithLoop,
    LengthGroupedSampler,
    ShardSampler,
)
from transformers.trainer_utils import has_length, seed_worker
from transformers.utils import is_datasets_available

from optimum.habana import GaudiTrainer


class HabanaDataloaderTrainer(GaudiTrainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training Habana Media Dataloader.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return MediaApiDataLoader(train_dataset, **dataloader_params)

    def get_eval_dataloader(self, eval_dataset: Optional[datasets.Dataset] = None) -> DataLoader:
        """
        Returns the eval Habana Media Dataloader.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = True

        return MediaApiDataLoader(eval_dataset, **dataloader_params)

    def get_test_dataloader(self, test_dataset: datasets.Dataset) -> DataLoader:
        """
        Returns the test Habana Media Dataloader.
        """
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="test")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset)
            dataloader_params["drop_last"] = True

        # We use the same batch_size as for eval.
        return MediaApiDataLoader(test_dataset, **dataloader_params)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Copied from: https://github.com/huggingface/optimum-habana/blob/v1.6.1/optimum/habana/transformers/trainer.py#L257
        `_get_train_sampler` from Transformers v4.31 does not work with distributed runs using the media pipe.
        Probably because a `DistributedSampler` is not used there.
        """
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                num_samples = len(self.train_dataset)
                if (
                    self.args.use_lazy_mode
                    and not self.args.dataloader_drop_last
                    and len(self.train_dataset) % self.args.per_device_train_batch_size != 0
                ):
                    # Make the total number of samples divisible by the batch size in lazy mode if needed
                    num_samples += (
                        self.args.per_device_train_batch_size
                        - len(self.train_dataset) % self.args.per_device_train_batch_size
                    )
                return RandomSampler(self.train_dataset, num_samples=num_samples, generator=generator)
            else:
                if self.args.use_lazy_mode and not self.args.dataloader_drop_last:
                    # Use a loop for HPUs when drop_last is False to have all batches have the same size
                    return DistributedSamplerWithLoop(
                        self.train_dataset,
                        batch_size=self.args.per_device_train_batch_size,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                    )
                else:
                    return DistributedSampler(
                        self.train_dataset,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                    )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        """
        Copied from; https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer.py#L918
        `_get_eval_sampler` from Transformers v4.31 may return `None` which breaks the media pipe.
        """
        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return ShardSampler(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_processes=self.args.world_size,
                process_index=self.args.process_index,
            )
