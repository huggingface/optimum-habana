# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from accelerate.utils import DistributedDataParallelKwargs
from packaging.version import parse as parse_version
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sentence_transformers.evaluation import SentenceEvaluator, SequentialEvaluator
from sentence_transformers.losses.CoSENTLoss import CoSENTLoss
from sentence_transformers.model_card import ModelCardCallback
from sentence_transformers.models import Pooling
from sentence_transformers.models.Transformer import Transformer
from sentence_transformers.sampler import (
    DefaultBatchSampler,
    GroupByLabelBatchSampler,
    NoDuplicatesBatchSampler,
    ProportionalBatchSampler,
    RoundRobinBatchSampler,
)
from sentence_transformers.training_args import (
    BatchSamplers,
    MultiDatasetBatchSamplers,
)
from sentence_transformers.util import disable_logging, is_datasets_available
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, SubsetRandomSampler
from transformers import EvalPrediction, PreTrainedTokenizerBase, TrainerCallback
from transformers import __version__ as transformers_version
from transformers.data.data_collator import DataCollator
from transformers.integrations import WandbCallback
from transformers.trainer import TRAINING_ARGS_NAME, _is_peft_model
from transformers.trainer_utils import EvalLoopOutput
from transformers.training_args import ParallelMode

from ..transformers import GaudiConfig, GaudiTrainer
from .st_gaudi_training_args import SentenceTransformerGaudiTrainingArguments


if is_datasets_available():
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, Value

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer


class SentenceTransformerGaudiTrainer(GaudiTrainer):
    """
    SentenceTransformerGaudiTrainer is a simple but feature-complete training and eval loop for PyTorch
    based on the 🤗 Transformers :class:`~transformers.Trainer`.

    It inherits from GaudiTrainer and adapted from:
        https://github.com/UKPLab/sentence-transformers/blob/v3.3.1/sentence_transformers/trainer.py
    """

    def __init__(
        self,
        model: Optional[SentenceTransformer] = None,
        gaudi_config: Optional[GaudiConfig] = None,
        args: Optional[SentenceTransformerGaudiTrainingArguments] = None,
        train_dataset: Optional[Union[Dataset, DatasetDict, IterableDataset, Dict[str, Dataset]]] = None,
        eval_dataset: Optional[Union[Dataset, DatasetDict, IterableDataset, Dict[str, Dataset]]] = None,
        loss: Optional[
            Union[
                torch.nn.Module,
                Dict[str, torch.nn.Module],
                Callable[[SentenceTransformer], torch.nn.Module],
                Dict[str, Callable[[SentenceTransformer], torch.nn.Module]],
            ]
        ] = None,
        evaluator: Optional[Union[SentenceEvaluator, List[SentenceEvaluator]]] = None,
        data_collator: Optional[DataCollator] = None,
        tokenizer: Optional[Union[PreTrainedTokenizerBase, Callable]] = None,
        model_init: Optional[Callable[[], SentenceTransformer]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if not is_datasets_available():
            raise RuntimeError(
                "To train a SentenceTransformerGaudiTrainer model, you need to install the `datasets` module. "
                "To fix: pip install datasets"
            )

        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = SentenceTransformerGaudiTrainingArguments(
                output_dir=output_dir,
                use_habana=True,
                gaudi_config_name="Habana/distilbert-base-uncased",
                use_lazy_mode=True,
                use_hpu_graphs=True,
                use_hpu_graphs_for_inference=False,
                use_hpu_graphs_for_training=True,
            )
        elif not isinstance(args, SentenceTransformerGaudiTrainingArguments):
            raise ValueError("Please use `TrainingArguments` imported from `optimum.habana.sentence_transformers`.")

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                logger.warning(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. `model_init` will"
                    " overwrite your model when calling the `train` method."
                )
            self.model_init = model_init

        if compute_metrics is not None:
            logger.warning(
                "`compute_metrics` is currently not compatible with the SentenceTransformerGaudiTrainer. Please use the "
                "`evaluator` argument instead for detailed evaluation metrics, or the `eval_dataset` argument for "
                "the evaluation loss."
            )

        # Get a dictionary of the default training arguments, so we can determine which arguments have been changed
        # for the model card
        default_args_dict = SentenceTransformerGaudiTrainingArguments(
            output_dir="unused",
            use_habana=True,
            gaudi_config_name="Habana/distilbert-base-uncased",
            use_lazy_mode=True,
            use_hpu_graphs=True,
            use_hpu_graphs_for_inference=False,
            use_hpu_graphs_for_training=True,
        ).to_dict()

        # If the model ID is set via the SentenceTransformerGaudiTrainingArguments, but not via the
        # SentenceTransformerModelCardData, then we can set it here for the model card regardless
        if args.hub_model_id and not model.model_card_data.model_id:
            model.model_card_data.set_model_id(args.hub_model_id)

        if tokenizer is None and isinstance(model.tokenizer, PreTrainedTokenizerBase):
            tokenizer = model.tokenizer

        if data_collator is None:
            data_collator = SentenceTransformerDataCollator(tokenize_fn=model.tokenize)

        for dataset_name, dataset in zip(["train", "eval"], [train_dataset, eval_dataset]):
            if isinstance(dataset, IterableDataset) and dataset.column_names is None:
                sample = next(iter(dataset))
                naive_type_mapping = {str: "string", int: "int64", float: "float32", bool: "bool"}
                example_features = {
                    key: Value(naive_type_mapping.get(type(value), "null")) for key, value in sample.items()
                }
                raise ValueError(
                    f"The provided `{dataset_name}_dataset` must have Features. Specify them with e.g.:\n"
                    f"{dataset_name}_dataset = {dataset_name}_dataset.cast(Features({example_features}))\n"
                    "or by providing the Features to the IterableDataset initialization method. See the Datasets "
                    "documentation for more information on dataset Features: "
                    "https://huggingface.co/docs/datasets/en/about_dataset_features"
                )

        if isinstance(train_dataset, dict) and not isinstance(train_dataset, DatasetDict):
            train_dataset = DatasetDict(train_dataset)
        if isinstance(eval_dataset, dict) and not isinstance(eval_dataset, DatasetDict):
            eval_dataset = DatasetDict(eval_dataset)
        super_kwargs = {
            "model": None if self.model_init else model,
            "gaudi_config": gaudi_config,
            "args": args,
            "data_collator": data_collator,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset if eval_dataset is not None or evaluator is None else "dummy",
            "model_init": model_init,
            "compute_metrics": compute_metrics,
            "callbacks": callbacks,
            "optimizers": optimizers,
            "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
        }
        # Transformers v4.46.0 changed the `tokenizer` argument to a more general `processing_class` argument
        if parse_version(transformers_version) >= parse_version("4.46.0"):
            super_kwargs["processing_class"] = tokenizer
        else:
            super_kwargs["tokenizer"] = tokenizer
        super().__init__(**super_kwargs)

        # Transformers v4.46.0 introduced a ValueError if `eval_dataset` is None while eval_strategy is not "no",
        # but in Sentence Transformers you can also evaluate without an eval_dataset via an evaluator, so we set
        # it to "dummy" in that case to avoid the ValueError
        if self.eval_dataset == "dummy":
            self.eval_dataset = None

        # Every Sentence Transformer model can always return a loss, so we set this to True
        # to avoid having to specify it in the data collator or model's forward
        self.can_return_loss = True

        self._prompt_length_mapping = {}

        self.model: SentenceTransformer
        self.args: SentenceTransformerGaudiTrainingArguments
        self.data_collator: SentenceTransformerDataCollator
        # Set the W&B project via environment variables if it's not already set
        if any(isinstance(callback, WandbCallback) for callback in self.callback_handler.callbacks):
            os.environ.setdefault("WANDB_PROJECT", "sentence-transformers")

        if loss is None:
            logger.info("No `loss` passed, using `losses.CoSENTLoss` as a default option.")
            loss = CoSENTLoss(self.model)

        if isinstance(loss, dict):
            self.loss = {dataset_name: self.prepare_loss(loss_fn, model) for dataset_name, loss_fn in loss.items()}
            for dataset_name, dataset in zip(["train", "eval"], [train_dataset, eval_dataset]):
                if dataset is None:
                    continue
                if not isinstance(dataset, dict):
                    raise ValueError(
                        f"If the provided `loss` is a dict, then the `{dataset_name}_dataset` must be a `DatasetDict`."
                    )
                if missing := set(dataset.keys()) - set(loss.keys()):
                    raise ValueError(
                        f"If the provided `loss` is a dict, then all keys from the `{dataset_name}_dataset` dictionary must occur in `loss` also. "
                        f"Currently, {sorted(missing)} occur{'s' if len(missing) == 1 else ''} in `{dataset_name}_dataset` but not in `loss`."
                    )
        else:
            self.loss = self.prepare_loss(loss, model)

        # If evaluator is a list, we wrap it in a SequentialEvaluator
        if evaluator is not None and not isinstance(evaluator, SentenceEvaluator):
            evaluator = SequentialEvaluator(evaluator)
        self.evaluator = evaluator

        if self.train_dataset is not None:
            self.train_dataset = self.maybe_add_prompts_or_dataset_name_column(
                train_dataset, args.prompts, dataset_name="train"
            )
        if self.eval_dataset is not None:
            self.eval_dataset = self.maybe_add_prompts_or_dataset_name_column(
                eval_dataset, args.prompts, dataset_name="eval"
            )
        self.add_model_card_callback(default_args_dict)

    def add_model_card_callback(self, default_args_dict: dict[str, Any]) -> None:
        """
        Add a callback responsible for automatically tracking data required for the automatic model card generation

        This method is called in the ``__init__`` method of the
        :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` class.

        Args:
            default_args_dict (Dict[str, Any]): A dictionary of the default training arguments, so we can determine
                which arguments have been changed for the model card.

        .. note::

            This method can be overridden by subclassing the trainer to remove/customize this callback in custom uses cases
        """

        model_card_callback = ModelCardCallback(self, default_args_dict)
        self.add_callback(model_card_callback)
        model_card_callback.on_init_end(self.args, self.state, self.control, self.model)

    def _wrap_model(self, model, training=True, dataloader=None):
        """
        Differs from GaudiTrainer._wrap_model:
        - `allow_unused_input=True` was added to `ht.hpu.ModuleCacher()`
        """
        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if self.accelerator.unwrap_model(model) is not model:
            return model

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        if self.args.parallel_mode == ParallelMode.DISTRIBUTED and self.args.distribution_strategy == "ddp":
            kwargs = {}

            kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            if self.args.ddp_find_unused_parameters and self.args.gradient_checkpointing:
                logger.warning(
                    "ddp_find_unused_parameters and gradient_checkpointing are both True, which may lead to an error:"
                    " https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021"
                )
            kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb

            if self.args.use_habana:
                kwargs["gradient_as_bucket_view"] = True

            if self.args.ddp_broadcast_buffers is not None:
                kwargs["broadcast_buffers"] = self.args.ddp_broadcast_buffers

            self.accelerator.ddp_handler = DistributedDataParallelKwargs(**kwargs)

        if self.args.use_hpu_graphs_for_training:
            import habana_frameworks.torch as ht

            if _is_peft_model(model):
                base_model = model.get_base_model()
                ht.hpu.ModuleCacher()(model=base_model, allow_unused_input=True, inplace=True)
            else:
                ht.hpu.ModuleCacher()(model=model, allow_unused_input=True, inplace=True)

        return model

    def call_model_init(self, trial=None) -> SentenceTransformer:
        model = super().call_model_init(trial=trial)
        # If the Trainer already has a loss, then we'll want to override the model in the loss function
        if not hasattr(self, "loss"):
            return model

        # Multi-loss training:
        if isinstance(self.loss, dict):
            for key, loss_fn in self.loss.items():
                # If a loss function is not yet initialized, we initialize it here
                if not isinstance(loss_fn, torch.nn.Module):
                    self.loss[key] = loss_fn(model)
                # Otherwise, we override the original model with the updated model in the loss function
                elif hasattr(loss_fn, "model"):
                    self.loss = self.override_model_in_loss(self.loss, model)

        # Loss is a function accepting a model as an argument
        elif not isinstance(self.loss, torch.nn.Module):
            self.loss = self.loss(model)

        # Loss is an initialized torch.nn.Module
        elif hasattr(self.loss, "model"):
            self.loss = self.override_model_in_loss(self.loss, model)
        return model

    def override_model_in_loss(self, loss: torch.nn.Module, model: SentenceTransformer) -> torch.nn.Module:
        from sentence_transformers import SentenceTransformer

        for name, child in loss.named_children():
            if _is_peft_model(child):
                child = child.get_base_model()
            if name == "model" and isinstance(child, SentenceTransformer):
                loss.model = model
            elif isinstance(child, torch.nn.Module):
                setattr(loss, name, self.override_model_in_loss(child, model))
        return loss

    def prepare_loss(
        self,
        loss: Union[Callable[[SentenceTransformer], torch.nn.Module], torch.nn.Module],
        model: SentenceTransformer,
    ) -> torch.nn.Module:
        if isinstance(loss, torch.nn.Module):
            return loss.to(model.device)
        return loss(model).to(model.device)

    def add_dataset_name_column(self, dataset_dict: DatasetDict) -> DatasetDict:
        for key, dataset in dataset_dict.items():
            if "dataset_name" not in dataset.column_names:
                dataset_dict[key] = dataset.add_column("dataset_name", [key] * len(dataset))
        return dataset_dict

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Computes the loss for the SentenceTransformer model.

        It uses ``self.loss`` to compute the loss, which can be a single loss function or a dictionary of loss functions
        for different datasets. If the loss is a dictionary, the dataset name is expected to be passed in the inputs
        under the key "dataset_name". This is done automatically in the ``add_dataset_name_column`` method.
        Note that even if ``return_outputs = True``, the outputs will be empty, as the SentenceTransformers losses do not
        return outputs.

        Args:
            model (SentenceTransformer): The SentenceTransformer model.
            inputs (Dict[str, Union[torch.Tensor, Any]]): The input data for the model.
            return_outputs (bool, optional): Whether to return the outputs along with the loss. Defaults to False.
            num_items_in_batch (int, optional): The number of items in the batch. Defaults to None. Unused, but required by the transformers Trainer.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]: The computed loss. If `return_outputs` is True, returns a tuple of loss and outputs. Otherwise, returns only the loss.
        """
        dataset_name = inputs.pop("dataset_name", None)
        features, labels = self.collect_features(inputs)
        loss_fn = self.loss

        if isinstance(loss_fn, dict) and dataset_name:
            loss_fn = loss_fn[dataset_name]

        # Insert the wrapped (e.g. distributed or compiled) model into the loss function,
        # if the loss stores the model. Only called once per process
        if (
            model == self.model_wrapped
            and model != self.model  # Only if the model is wrapped
            and hasattr(loss_fn, "model")  # Only if the loss stores the model
            and loss_fn.model != model  # Only if the wrapped model is not already stored
        ):
            loss_fn = self.override_model_in_loss(loss_fn, model)
        loss = loss_fn(features, labels)
        if return_outputs:
            # During prediction/evaluation, `compute_loss` will be called with `return_outputs=True`.
            # However, Sentence Transformer losses do not return outputs, so we return an empty dictionary.
            # This does not result in any problems, as the SentenceTransformerGaudiTrainingArguments sets
            # `prediction_loss_only=True` which means that the output is not used.
            return loss, {}
        return loss

    def collect_features(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Tuple[List[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        """Turn the inputs from the dataloader into the separate model inputs & the labels.

        Example::

            >>> list(inputs.keys())
            ['return_loss', 'label', 'sentence_0_input_ids', 'sentence_0_token_type_ids', 'sentence_0_attention_mask', 'sentence_1_input_ids', 'sentence_1_token_type_ids', 'sentence_1_attention_mask']
            >>> features, labels = self.collect_features(inputs)
            >>> len(features)
            2
            >>> list(features[0].keys())
            ['input_ids', 'token_type_ids', 'attention_mask']
            >>> list(features[1].keys())
            ['input_ids', 'token_type_ids', 'attention_mask']
            >>> torch.equal(labels, inputs["label"])
            True
        """
        # All inputs ending with `_input_ids` (Transformers), `_sentence_embedding` (BoW), `_pixel_values` (CLIPModel)
        # are considered to correspond to a feature
        features = []
        for column in inputs:
            if column.endswith("_input_ids"):
                prefix = column[: -len("input_ids")]
            elif column.endswith("_sentence_embedding"):
                prefix = column[: -len("sentence_embedding")]
            elif column.endswith("_pixel_values"):
                prefix = column[: -len("pixel_values")]
            else:
                continue
            features.append({key[len(prefix) :]: value for key, value in inputs.items() if key.startswith(prefix)})
        labels = inputs.get("label", None)
        return features, labels

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if eval_dataset is not None:
            eval_dataset = self.maybe_add_prompts_or_dataset_name_column(
                eval_dataset, self.args.prompts, dataset_name="eval"
            )
        else:
            eval_dataset = self.eval_dataset
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # If the evaluator is not defined, we can just return the output
        if self.evaluator is None:
            return output

        # If we are training and eval_dataset is a DatasetDict, then we should
        # 1) only run the evaluator for the first dataset
        # 2) prefix that only run as "eval", rather than e.g. "eval_multi_nli"
        if self.is_in_train and isinstance(self.eval_dataset, dict) and metric_key_prefix.startswith("eval_"):
            if metric_key_prefix[5:] == list(self.eval_dataset.keys())[0]:
                metric_key_prefix = "eval"
            else:
                return output

        with nullcontext() if self.is_local_process_zero() else disable_logging(logging.INFO):
            evaluator_metrics = self.evaluator(self.model)
        if not isinstance(evaluator_metrics, dict):
            evaluator_metrics = {"evaluator": evaluator_metrics}

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(evaluator_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                evaluator_metrics[f"{metric_key_prefix}_{key}"] = evaluator_metrics.pop(key)

        output.metrics.update(evaluator_metrics)

        return output

    def _load_best_model(self) -> None:
        # We want to ensure that this does not fail, and it may change if transformers updates how checkpoints are saved
        # Loading the best model is only supported for `transformers`-based models
        if not isinstance(self.model[0], Transformer):
            logger.info("Could not load best model, as the model is not a `transformers`-based model.")
            return

        try:
            if checkpoint := self.state.best_model_checkpoint:
                step = checkpoint.rsplit("-", 1)[-1]
                self.model.model_card_data.set_best_model_step(int(step))
        except Exception:
            pass

        # Override the model with the `transformers`-based auto_model, and restore the original SentenceTransformers
        # model with the loaded `transformers` model
        full_model = self.model
        self.model = self.model[0].auto_model
        try:
            return super()._load_best_model()
        finally:
            loaded_auto_model = self.model
            self.model = full_model
            self.model[0].auto_model = loaded_auto_model

    def validate_column_names(self, dataset: Dataset, dataset_name: Optional[str] = None) -> None:
        if isinstance(dataset, dict):
            for dataset_name, dataset in dataset.items():
                self.validate_column_names(dataset, dataset_name=dataset_name)
            return

        if overlap := set(dataset.column_names) & {"return_loss", "dataset_name"}:
            raise ValueError(
                f"The following column names are invalid in your {dataset_name + ' ' if dataset_name else ''}dataset: {list(overlap)}."
                " Avoid using these column names, as they are reserved for internal use."
            )

    def get_batch_sampler(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: Optional[List[str]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Optional[BatchSampler]:
        """
        Returns the appropriate batch sampler based on the ``batch_sampler`` argument in ``self.args``.
        This batch sampler class supports ``__len__`` and ``__iter__`` methods, and is used as the ``batch_sampler``
        to create the :class:`torch.utils.data.DataLoader`.

        .. note::
            Override this method to provide a custom batch sampler.

        Args:
            dataset (Dataset): The dataset to sample from.
            batch_size (int): Number of samples per batch.
            drop_last (bool): If True, drop the last incomplete batch if the dataset size
                is not divisible by the batch size.
            valid_label_columns (List[str]): List of column names to check for labels.
                The first column name from ``valid_label_columns`` found in the dataset will
                be used as the label column.
            generator (torch.Generator, optional): Optional random number generator for shuffling
                the indices.
        """
        if isinstance(dataset, IterableDataset):
            if self.args.batch_sampler != BatchSamplers.BATCH_SAMPLER:
                logger.warning("When using an IterableDataset, you cannot specify a batch sampler.")
            return None

        if self.args.batch_sampler == BatchSamplers.NO_DUPLICATES:
            return NoDuplicatesBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                drop_last=drop_last,
                valid_label_columns=valid_label_columns,
                generator=generator,
            )

        if self.args.batch_sampler == BatchSamplers.GROUP_BY_LABEL:
            return GroupByLabelBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                drop_last=drop_last,
                valid_label_columns=valid_label_columns,
            )

        if self.args.batch_sampler == BatchSamplers.BATCH_SAMPLER:
            return DefaultBatchSampler(
                SubsetRandomSampler(range(len(dataset)), generator=generator),
                batch_size=batch_size,
                drop_last=drop_last,
            )

    def get_multi_dataset_batch_sampler(
        self,
        dataset: ConcatDataset,
        batch_samplers: list[BatchSampler],
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = 0,
    ) -> BatchSampler:
        """
        Returns the appropriate multi-dataset batch sampler based on the ``multi_dataset_batch_sampler`` argument
        in ``self.args``. This batch sampler class supports ``__len__`` and ``__iter__`` methods, and is used as the
        ``batch_sampler`` to create the :class:`torch.utils.data.DataLoader`.

        .. note::
            Override this method to provide a custom multi-dataset batch sampler.

        Args:
            dataset (ConcatDataset): The concatenation of all datasets.
            batch_samplers (List[BatchSampler]): List of batch samplers for each dataset in the concatenated dataset.
            generator (torch.Generator, optional): Optional random number generator for shuffling the indices.
            seed (int, optional): Optional seed for the random number generator
        """
        if self.args.multi_dataset_batch_sampler == MultiDatasetBatchSamplers.ROUND_ROBIN:
            return RoundRobinBatchSampler(
                dataset=dataset,
                batch_samplers=batch_samplers,
                generator=generator,
                seed=seed,
            )

        if self.args.multi_dataset_batch_sampler == MultiDatasetBatchSamplers.PROPORTIONAL:
            return ProportionalBatchSampler(
                dataset=dataset,
                batch_samplers=batch_samplers,
                generator=generator,
                seed=seed,
            )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Training requires specifying a train_dataset to the SentenceTransformerGaudiTrainer.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        generator = torch.Generator()
        if self.args.seed:
            generator.manual_seed(self.args.seed)

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        if isinstance(train_dataset, IterableDataset):
            dataloader_params.update(
                {
                    "batch_size": self.args.train_batch_size,
                    "drop_last": self.args.dataloader_drop_last,
                }
            )
            if self.args.batch_sampler != BatchSamplers.BATCH_SAMPLER:
                logger.warning("When using an IterableDataset, you cannot specify a batch sampler.")

        elif isinstance(train_dataset, IterableDatasetDict):
            raise ValueError(
                "Sentence Transformers is not compatible with IterableDatasetDict. Please use a DatasetDict instead."
            )

        elif isinstance(train_dataset, DatasetDict):
            for dataset in train_dataset.values():
                if isinstance(dataset, IterableDataset):
                    raise ValueError(
                        "Sentence Transformers is not compatible with a DatasetDict containing an IterableDataset."
                    )

            batch_samplers = [
                self.get_batch_sampler(
                    dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    valid_label_columns=data_collator.valid_label_columns,
                    generator=generator,
                )
                for dataset in train_dataset.values()
            ]

            train_dataset = ConcatDataset(train_dataset.values())
            batch_sampler = self.get_multi_dataset_batch_sampler(
                dataset=train_dataset,
                batch_samplers=batch_samplers,
                generator=generator,
                seed=self.args.seed,
            )
            dataloader_params["batch_sampler"] = batch_sampler

        elif isinstance(train_dataset, Dataset):
            batch_sampler = self.get_batch_sampler(
                train_dataset,
                batch_size=self.args.train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                valid_label_columns=data_collator.valid_label_columns,
                generator=generator,
            )
            dataloader_params["batch_sampler"] = batch_sampler
        else:
            raise ValueError(
                "Unsupported `train_dataset` type. Use a Dataset, DatasetDict, or IterableDataset for training."
            )

        # If 'even_batches' is True, it will use the initial few samples to pad out the last sample. This can
        # cause issues with multi-dataset training, so we want to set this to False.
        # For evaluation, setting 'even_batches' to False results in hanging, so we keep it as True there.
        self.accelerator.even_batches = False
        self._train_dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        return self._train_dataloader

    def get_eval_dataloader(
        self, eval_dataset: Optional[Union[Dataset, DatasetDict, IterableDataset]] = None
    ) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            # Prevent errors if the evaluator is set but no eval_dataset is provided
            if self.evaluator is not None:
                return DataLoader([])
            raise ValueError("Evaluation requires specifying an eval_dataset to the SentenceTransformerGaudiTrainer.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        generator = torch.Generator()
        if self.args.seed:
            generator.manual_seed(self.args.seed)

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }
        if isinstance(eval_dataset, IterableDataset):
            dataloader_params.update(
                {
                    "batch_size": self.args.eval_batch_size,
                    "drop_last": self.args.dataloader_drop_last,
                }
            )

        elif isinstance(eval_dataset, IterableDatasetDict):
            raise ValueError(
                "Sentence Transformers is not compatible with IterableDatasetDict. Please use a DatasetDict instead."
            )

        elif isinstance(eval_dataset, DatasetDict):
            for dataset in eval_dataset.values():
                if isinstance(dataset, IterableDataset):
                    raise ValueError(
                        "Sentence Transformers is not compatible with a DatasetDict containing an IterableDataset."
                    )

            batch_samplers = [
                self.get_batch_sampler(
                    dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    valid_label_columns=data_collator.valid_label_columns,
                    generator=generator,
                )
                for dataset in eval_dataset.values()
            ]

            eval_dataset = ConcatDataset(eval_dataset.values())
            batch_sampler = self.get_multi_dataset_batch_sampler(
                dataset=eval_dataset,
                batch_samplers=batch_samplers,
                generator=generator,
                seed=self.args.seed,
            )
            dataloader_params["batch_sampler"] = batch_sampler

        elif isinstance(eval_dataset, Dataset):
            batch_sampler = self.get_batch_sampler(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                drop_last=self.args.dataloader_drop_last,
                valid_label_columns=data_collator.valid_label_columns,
                generator=generator,
            )
            dataloader_params["batch_sampler"] = batch_sampler

        else:
            raise ValueError(
                "Unsupported `eval_dataset` type. Use a Dataset, DatasetDict, or IterableDataset for evaluation."
            )

        # If 'even_batches' is True, it will use the initial few samples to pad out the last sample. This can
        # cause issues with multi-dataset training, so we want to set this to False during training.
        # For evaluation, setting 'even_batches' to False results in hanging, so we keep it as True here.
        self.accelerator.even_batches = True
        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def get_test_dataloader(self, test_dataset: Union[Dataset, DatasetDict, IterableDataset]) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        data_collator = self.data_collator

        generator = torch.Generator()
        if self.args.seed:
            generator.manual_seed(self.args.seed)

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        if isinstance(test_dataset, IterableDataset):
            dataloader_params.update(
                {
                    "batch_size": self.args.eval_batch_size,
                    "drop_last": self.args.dataloader_drop_last,
                }
            )

        elif isinstance(test_dataset, IterableDatasetDict):
            raise ValueError(
                "Sentence Transformers is not compatible with IterableDatasetDict. Please use a DatasetDict instead."
            )

        elif isinstance(test_dataset, DatasetDict):
            for dataset in test_dataset.values():
                if isinstance(dataset, IterableDataset):
                    raise ValueError(
                        "Sentence Transformers is not compatible with a DatasetDict containing an IterableDataset."
                    )

            batch_samplers = [
                self.get_batch_sampler(
                    dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    valid_label_columns=data_collator.valid_label_columns,
                    generator=generator,
                )
                for dataset in test_dataset.values()
            ]

            test_dataset = ConcatDataset(test_dataset.values())
            batch_sampler = self.get_multi_dataset_batch_sampler(
                dataset=test_dataset,
                batch_samplers=batch_samplers,
                generator=generator,
                seed=self.args.seed,
            )
            dataloader_params["batch_sampler"] = batch_sampler

        elif isinstance(test_dataset, Dataset):
            batch_sampler = self.get_batch_sampler(
                test_dataset,
                batch_size=self.args.eval_batch_size,
                drop_last=self.args.dataloader_drop_last,
                valid_label_columns=data_collator.valid_label_columns,
                generator=generator,
            )
            dataloader_params["batch_sampler"] = batch_sampler

        else:
            raise ValueError(
                "Unsupported `test_dataset` type. Use a Dataset, DatasetDict, or IterableDataset for testing."
            )

        # If 'even_batches' is True, it will use the initial few samples to pad out the last sample. This can
        # cause issues with multi-dataset training, so we want to set this to False during training.
        # For evaluation, setting 'even_batches' to False results in hanging, so we keep it as True here.
        self.accelerator.even_batches = True
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)

        # Transformers v4.46.0 changed the `tokenizer` attribute to a more general `processing_class` attribute
        if parse_version(transformers_version) >= parse_version("4.46.0"):
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
        else:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        from sentence_transformers import SentenceTransformer

        loaded_model = SentenceTransformer(checkpoint_path, trust_remote_code=self.model.trust_remote_code)
        self.model.load_state_dict(loaded_model.state_dict())

    def _get_prompt_length(self, prompt: str) -> int:
        try:
            return self._prompt_length_mapping[prompt]
        except KeyError:
            prompt_length = self.model.tokenize([prompt])["input_ids"].shape[-1] - 1
            self._prompt_length_mapping[prompt] = prompt_length
            return prompt_length

    def _include_prompt_length(self) -> bool:
        """
        Return whether the prompt length should be passed to the model's forward method.

        True if the model does not include the prompt in the pooling layer. Can be
        overridden by the user if it's useful to include the prompt length.
        """
        for module in self.model:
            if isinstance(module, Pooling):
                return not module.include_prompt
        return False

    @staticmethod
    def add_prompts_or_dataset_name_transform(
        batch: Dict[str, List[Any]],
        prompts: Optional[Union[Dict[str, str], str]] = None,
        prompt_lengths: Optional[Union[Dict[str, int], int]] = None,
        dataset_name: Optional[str] = None,
        transform: Optional[Callable[[Dict[str, List[Any]]], Dict[str, List[Any]]]] = None,
        **kwargs,
    ) -> Dict[str, List[Any]]:
        """A transform/map function that adds prompts or dataset names to the batch.

        Args:
            batch (dict[str, list[Any]]): The batch of data, where each key is a column name and each value
                is a list of values.
            prompts (dict[str, str] | str | None, optional): An optional mapping of column names to string
                prompts, or a string prompt for all columns. Defaults to None.
            prompt_lengths (dict[str, int] | int | None, optional): An optional mapping of prompts names to
                prompt token length, or a prompt token length if the prompt is a string. Defaults to None.
            dataset_name (str | None, optional): The name of this dataset, only if there are multiple datasets
                that use a different loss. Defaults to None.
            transform (Callable[[dict[str, list[Any]]], dict[str, list[Any]]], optional): An optional transform
                function to apply on the batch before adding prompts, etc. Defaults to None.

        Returns:
            dict[str, list[Any]]: The "just-in-time" transformed batch with prompts and/or dataset names added.
        """
        # If the dataset is a Dataset(Dict), then we use set_transform and we want to also apply any
        # previous transform if it exists
        if transform:
            batch = transform(batch)

        # Return if the batch has no columns...
        if not batch:
            return batch

        # ... or if it's empty
        first_column = list(batch.keys())[0]
        if not batch[first_column]:
            return batch

        # Apply one prompt to all columns...
        if isinstance(prompts, str):
            for column_name, column in list(batch.items()):
                if isinstance(column[0], str):
                    batch[column_name] = [prompts + value for value in column]

                    if prompt_lengths is not None:
                        batch[f"{column_name}_prompt_length"] = [prompt_lengths] * len(column)

        # ... or a column-specific prompt
        if isinstance(prompts, dict):
            for column_name, prompt in prompts.items():
                if column_name in batch:
                    batch[column_name] = [prompt + value for value in batch[column_name]]

                    if prompt_lengths:
                        batch[f"{column_name}_prompt_length"] = [prompt_lengths[prompt]] * len(batch[column_name])

        # If we have multiple losses, then we need to add the dataset name to the batch
        if dataset_name:
            batch["dataset_name"] = [dataset_name] * len(batch[first_column])

        return batch

    def maybe_add_prompts_or_dataset_name_column(
        self,
        dataset_dict: Union[DatasetDict, Dataset, None],
        prompts: Optional[Union[Dict[str, Dict[str, str]], Dict[str, str], str]] = None,
        dataset_name: Optional[str] = None,
    ) -> Union[DatasetDict, Dataset, None]:
        """
        Maybe add prompts or dataset names to the dataset. We add the dataset_name column to the dataset if:

        1. The loss is a dictionary and the dataset is a DatasetDict, or
        2. The prompts contain a mapping to dataset names.

        There are 4 cases for the prompts:

        1. `str`: One prompt for all datasets and columns.
        2. `dict[str, str]`: A column to prompt mapping.
        3. `dict[str, str]`: A dataset to prompt mapping.
        4. `dict[str, dict[str, str]]`: A dataset to column to prompt mapping.

        And 2 cases for the dataset:

        A. `Dataset`: A single dataset.
        B. `DatasetDict`: A dictionary of datasets.

        3A is not allowed, and 2A doesn't make sense.

        Args:
            dataset_dict (DatasetDict | Dataset | None): The dataset to add prompts or dataset names to.

        Returns:
            DatasetDict | Dataset | None: The dataset with prompts or dataset names added.
        """
        if dataset_dict is None:
            return None

        include_dataset_name = isinstance(self.loss, dict)

        # If we've already added the transform to this (iterable) dataset, don't add it again
        if hasattr(dataset_dict, "_sentence_transformers_preprocessed"):
            return dataset_dict

        # Ensure that there's no "dataset_name"/"return_loss" columns in the unprocessed datasets
        self.validate_column_names(dataset_dict, dataset_name=dataset_name)

        # Only add if 1) we have prompts or 2) we need the dataset name for the loss dictionary
        if prompts or include_dataset_name:
            include_prompt_lengths = self._include_prompt_length()
            dataset_dict = self.add_prompts_or_dataset_name_column(
                dataset_dict,
                prompts=prompts,
                include_prompt_lengths=include_prompt_lengths,
                include_dataset_name=include_dataset_name,
            )
        return dataset_dict

    def add_prompts_or_dataset_name_column(
        self,
        dataset_dict: Union[DatasetDict, IterableDatasetDict, Dataset, IterableDataset],
        prompts: Optional[Union[Dict[str, str], str]] = None,
        dataset_name: Optional[str] = None,
        include_prompt_lengths: bool = False,
        include_dataset_name: bool = False,
    ) -> Union[DatasetDict, Dataset, None]:
        # If we have DatasetDict, recurse
        if isinstance(dataset_dict, (IterableDatasetDict, DatasetDict)):
            for dataset_name, dataset in dataset_dict.items():
                # If prompts is a dictionary that matches the dataset names, then take the nested prompts
                nested_prompts = prompts.get(dataset_name, prompts) if isinstance(prompts, dict) else prompts
                dataset_dict[dataset_name] = self.add_prompts_or_dataset_name_column(
                    dataset_dict=dataset,
                    prompts=nested_prompts,
                    dataset_name=dataset_name if include_dataset_name else None,
                    include_prompt_lengths=include_prompt_lengths,
                    include_dataset_name=include_dataset_name,
                )
            return dataset_dict

        # Get the prompt lengths if needed for the pooling layer
        prompt_lengths = None
        if prompts:
            if isinstance(prompts, str):
                if include_prompt_lengths:
                    prompt_lengths = self._get_prompt_length(prompts)
            elif isinstance(prompts, dict):
                first_key = list(prompts.keys())[0]
                if isinstance(prompts[first_key], dict):
                    raise ValueError(
                        "The prompts provided to the trainer are a nested dictionary. In this setting, the first "
                        "level of the dictionary should map to dataset names and the second level to column names. "
                        "However, as the provided dataset is a not a DatasetDict, no dataset names can be inferred. "
                        f"The keys to the provided prompts dictionary are {list(prompts.keys())!r}"
                    )
                if include_prompt_lengths:
                    # If prompt columns exist, add the prompt length column
                    prompt_lengths = {
                        prompt: self._get_prompt_length(prompt)
                        for column_name, prompt in prompts.items()
                        if column_name in dataset_dict.column_names
                    }

        # If we have a Dataset, we can set the transform directly...
        if isinstance(dataset_dict, Dataset):
            dataset_dict.set_transform(
                partial(
                    self.add_prompts_or_dataset_name_transform,
                    prompts=prompts,
                    prompt_lengths=prompt_lengths,
                    dataset_name=dataset_name,
                    **dataset_dict._format_kwargs,
                )
            )

        # ... otherwise, we have an IterableDataset and we need to map it, which performs the same operation as above
        elif isinstance(dataset_dict, IterableDataset):
            # Update the features to include the new columns
            features = dataset_dict.features
            if dataset_name:
                features["dataset_name"] = Value("string")
            if prompt_lengths:
                if isinstance(prompts, str):
                    for column_name in dataset_dict.column_names:
                        feature = features[column_name]
                        if isinstance(feature, Value) and feature.dtype in ("string", "large_string"):
                            features[f"{column_name}_prompt_length"] = Value("int16")
                elif isinstance(prompts, dict):
                    for column_name, prompt in prompts.items():
                        feature = features[column_name]
                        if (
                            prompt in prompt_lengths
                            and isinstance(feature, Value)
                            and feature.dtype in ("string", "large_string")
                        ):
                            features[f"{column_name}_prompt_length"] = Value("int16")

            dataset_dict = dataset_dict.map(
                partial(
                    self.add_prompts_or_dataset_name_transform,
                    prompts=prompts,
                    prompt_lengths=prompt_lengths,
                    dataset_name=dataset_name,
                ),
                batched=True,
                features=features,
            )

        else:
            raise ValueError("Unsupported dataset type.")

        # Add a tag to the dataset to indicate that it has been preprocessed, to ensure that we don't apply the map or
        # transform multiple times.
        dataset_dict._sentence_transformers_preprocessed = True
        return dataset_dict

    def create_model_card(
        self,
        language: Optional[str] = None,
        license: Optional[str] = None,
        tags: Optional[Union[str, List[str]]] = None,
        model_name: Optional[str] = None,
        finetuned_from: Optional[str] = None,
        tasks: Optional[Union[str, List[str]]] = None,
        dataset_tags: Optional[Union[str, List[str]]] = None,
        dataset: Optional[Union[str, List[str]]] = None,
        dataset_args: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> None:
        if not self.is_world_process_zero():
            return

        if language:
            self.model.model_card_data.set_language(language)
        if license:
            self.model.model_card_data.set_license(license)
        if tags:
            self.model.model_card_data.add_tags(tags)

        self.model._create_model_card(self.args.output_dir, model_name=model_name)

    def get_optimizer_cls_and_kwargs(
        self, args: SentenceTransformerGaudiTrainingArguments, model: Optional[SentenceTransformer] = None
    ) -> Tuple[Any, Any]:
        """
        We have to override the optimizer_grouped_parameters because the Trainer superclass bases it on the `model`
        itself, but the SentenceTransformer losses can have weights that should be updated as well, e.g.
        SoftmaxLoss (see #2872).

        This method requires `transformers` >= 4.43.0.
        """

        if isinstance(self.loss, dict):
            loss_model = torch.nn.Sequential(OrderedDict(self.loss))
        else:
            loss_model = self.loss
        optimizer_cls, optimizer_kwargs = super().get_optimizer_cls_and_kwargs(args, loss_model)

        # If the kwargs were not overridden by the super() call, then we should override them here so that the potential
        # weights in the loss(es) can also be updated.
        if not {"params", "model", "optimizer_dict"} & set(optimizer_kwargs.keys()):
            decay_parameters = self.get_decay_parameter_names(loss_model)
            optimizer_kwargs["optimizer_dict"] = [
                {
                    "params": [
                        p for n, p in loss_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in loss_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        return optimizer_cls, optimizer_kwargs
