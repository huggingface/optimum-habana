# coding=utf-8
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
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers
from transformers.training_args import ParallelMode

from ..transformers import GaudiTrainingArguments


logger = logging.getLogger(__name__)


@dataclass
class SentenceTransformerGaudiTrainingArguments(GaudiTrainingArguments):
    """
    SentenceTransformerGaudiTrainingArguments extends :class:`~transformers.TrainingArguments` with additional arguments
    specific to Sentence Transformers. See :class:`~transformers.TrainingArguments` for the complete list of
    available arguments.

    It inherits from GaudiTrainingArguments and adapted from:
        https://github.com/UKPLab/sentence-transformers/blob/v3.3.1/sentence_transformers/training_args.py

    Args:
        output_dir (`str`):
            The output directory where the model checkpoints will be written.
        prompts (`Union[Dict[str, Dict[str, str]], Dict[str, str], str]`, *optional*):
            The prompts to use for each column in the training, evaluation and test datasets. Four formats are accepted:

            1. `str`: A single prompt to use for all columns in the datasets, regardless of whether the training/evaluation/test
               datasets are :class:`datasets.Dataset` or a :class:`datasets.DatasetDict`.
            2. `Dict[str, str]`: A dictionary mapping column names to prompts, regardless of whether the training/evaluation/test
               datasets are :class:`datasets.Dataset` or a :class:`datasets.DatasetDict`.
            3. `Dict[str, str]`: A dictionary mapping dataset names to prompts. This should only be used if your training/evaluation/test
               datasets are a :class:`datasets.DatasetDict` or a dictionary of :class:`datasets.Dataset`.
            4. `Dict[str, Dict[str, str]]`: A dictionary mapping dataset names to dictionaries mapping column names to
               prompts. This should only be used if your training/evaluation/test datasets are a
               :class:`datasets.DatasetDict` or a dictionary of :class:`datasets.Dataset`.

        batch_sampler (Union[:class:`~sentence_transformers.training_args.BatchSamplers`, `str`], *optional*):
            The batch sampler to use. See :class:`~sentence_transformers.training_args.BatchSamplers` for valid options.
            Defaults to ``BatchSamplers.BATCH_SAMPLER``.
        multi_dataset_batch_sampler (Union[:class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers`, `str`], *optional*):
            The multi-dataset batch sampler to use. See :class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers`
            for valid options. Defaults to ``MultiDatasetBatchSamplers.PROPORTIONAL``.
    """

    prompts: Optional[Union[Dict[str, Dict[str, str]], Dict[str, str], str]] = None
    batch_sampler: Union[BatchSamplers, str] = field(
        default=BatchSamplers.BATCH_SAMPLER, metadata={"help": "The batch sampler to use."}
    )
    multi_dataset_batch_sampler: Union[MultiDatasetBatchSamplers, str] = field(
        default=MultiDatasetBatchSamplers.PROPORTIONAL, metadata={"help": "The multi-dataset batch sampler to use."}
    )

    def __post_init__(self):
        super().__post_init__()

        self.batch_sampler = BatchSamplers(self.batch_sampler)
        self.multi_dataset_batch_sampler = MultiDatasetBatchSamplers(self.multi_dataset_batch_sampler)

        # The `compute_loss` method in `SentenceTransformerTrainer` is overridden to only compute the prediction loss,
        # so we set `prediction_loss_only` to `True` here to avoid
        self.prediction_loss_only = True

        # Disable broadcasting of buffers to avoid `RuntimeError: one of the variables needed for gradient computation
        # has been modified by an inplace operation.` when training with DDP & a BertModel-based model.
        self.ddp_broadcast_buffers = False

        if self.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
            # If output_dir is "unused", then this instance is created to compare training arguments vs the defaults,
            # so we don't have to warn.
            if self.output_dir != "unused":
                logger.warning(
                    "Currently using DataParallel (DP) for multi-gpu training, while DistributedDataParallel (DDP) is recommended for faster training. "
                    "See https://sbert.net/docs/sentence_transformer/training/distributed.html for more information."
                )

        elif self.parallel_mode == ParallelMode.DISTRIBUTED and not self.dataloader_drop_last:
            # If output_dir is "unused", then this instance is created to compare training arguments vs the defaults,
            # so we don't have to warn.
            if self.output_dir != "unused":
                logger.warning(
                    "When using DistributedDataParallel (DDP), it is recommended to set `dataloader_drop_last=True` to avoid hanging issues with an uneven last batch. "
                    "Setting `dataloader_drop_last=True`."
                )
            self.dataloader_drop_last = True
