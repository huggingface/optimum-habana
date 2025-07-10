# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import dataclasses
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate import PartialState
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    DataCollatorWithFlattening,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from trl import SFTTrainer
from transformers.utils import is_peft_available

from .sft_config import GaudiSFTConfig
from .utils import BaseDataCollatorForLanguageModeling, pad

if is_peft_available():
    import peft
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from ... import GaudiConfig, GaudiTrainer


@dataclass
class DataCollatorForLanguageModeling(BaseDataCollatorForLanguageModeling):
    """
    Copied from DataCollatorForLanguageModeling: https://github.com/huggingface/trl/blob/v0.17.0/trl/trainer/sft_trainer.py#L73
    The differences are:
        - Bucketing added. Buckets: None or emtpy list means means no bucketing
    """

    def __init__(self, pad_token_id: int, completion_only_loss: bool = True, return_tensors: str = "pt", buckets: Optional[list[int]] = None):
        super().__init__(pad_token_id=pad_token_id, return_tensors="pt", buckets=buckets)
        self.completion_only_loss = completion_only_loss

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        attention_mask = [torch.ones_like(ids) for ids in input_ids]
        labels = [torch.tensor(example["input_ids"]) for example in examples]
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = [torch.tensor(example["completion_mask"]) for example in examples]

        bucket_size = 0
        if self.buckets is not None and len(self.buckets) > 0:
            bucket_size = self._get_bucketed_len(examples)

        # Pad
        output = {}
        output["input_ids"] = pad(input_ids, padding_value=self.pad_token_id, padding_side="right", bucket_size=bucket_size)
        output["attention_mask"] = pad(attention_mask, padding_value=0, padding_side="right", bucket_size=bucket_size)
        output["labels"] = pad(labels, padding_value=-100, padding_side="right", bucket_size=bucket_size)
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = pad(completion_mask, padding_value=0, padding_side="right", bucket_size=bucket_size)
            output["labels"][completion_mask == 0] = -100  # mask everything that is not in the completion

        return output


class GaudiSFTTrainer(SFTTrainer, GaudiTrainer):
    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        args: Optional[GaudiSFTConfig] = None,
        gaudi_config: GaudiConfig = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Union[Callable[[dict], str], Callable[[dict], list[str]]]] = None,
        num_buckets: Optional[int] = -1,
    ):
        """
        Copied from SFTTrainer.__init__: https://github.com/huggingface/trl/blob/v0.17.0/trl/trainer/sft_trainer.py#L218
        The differences are:
            - add new args gaudi_config
            - use GaudiTrainer instead of Trainer
            - cast peft model to bf16
            - num_buckets: Number of buckets. > 0 means apply bucketing, <= 0  means no bucketing
        """
        if num_buckets > 0:
            assert data_collator is None, (
                "For bucketing (num_buckets > 0), we only support data_collator=None"
            )

        if args is None:
            model_name = model_id.split("/")[-1]
            args = GaudiSFTConfig(f"{model_name}-SFT")
        elif isinstance(args, TrainingArguments) and not isinstance(args, GaudiSFTConfig):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token  # to_dict hides the hub_token
            dict_args.pop("push_to_hub_token")
            args = GaudiSFTConfig(**dict_args)

        # Handle the tokenizer
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_id)

        if args.eos_token is not None:
            eos_token = args.eos_token
            eos_token_id = processing_class.convert_tokens_to_ids(eos_token)
            if eos_token_id is None:
                raise ValueError(
                    f"The specified `eos_token` ('{eos_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `eos_token` exists "
                    "in the vocabulary before using it as an EOS token."
                )
            processing_class.eos_token_id = eos_token_id

        # Model
        if args.model_init_kwargs is not None and not isinstance(model, str):
            warnings.warn(
                "You passed model_init_kwargs to the `GaudiSFTConfig`, but your model is already instantiated. "
                "The `model_init_kwargs` will be ignored."
            )
        if isinstance(model, str):
            model = self._create_model_from_path(model, args)

        # PEFT configuration and model wrapping
        if peft_config is not None:
            model = self._prepare_peft_model(model, peft_config, args)

        # Data collator
        if args.padding_free:
            if data_collator is not None:
                raise ValueError("Passing a custom data collator is not supported when using padding-free.")
            if args.packing:
                warnings.warn(
                    "You are passing `packing=True` and `padding_free=True` which is not recommended. Please refer "
                    "to the documentation to understand why this is not recommended."
                )
            if model.config._attn_implementation != "flash_attention_2":
                warnings.warn(
                    "Padding-free training is enabled, but the attention implementation is not set to "
                    "'flash_attention_2'. Padding-free training flattens batches into a single sequence, and "
                    "'flash_attention_2' is the only known attention mechanism that reliably supports this. Using "
                    "other implementations may lead to unexpected behavior. To ensure compatibility, set "
                    "`attn_implementation='flash_attention_2'` in the model configuration, or verify that your "
                    "attention mechanism can handle flattened sequences."
                )
            if args.per_device_train_batch_size == 1:
                warnings.warn(
                    "You are using a per_device_train_batch_size of 1 with padding-free training. Using a batch size "
                    "of 1 anihilate the benefits of padding-free training. Please consider increasing the batch size "
                    "to at least 2."
                )
            data_collator = DataCollatorWithFlattening()

        if args.completion_only_loss is None:
            first_example = next(iter(train_dataset))
            self.completion_only_loss = "prompt" in first_example
        else:
            self.completion_only_loss = args.completion_only_loss
        if data_collator is None:
            # Get the pad token: if not provided, use the one from the processing class or the eos token
            # if the processing class does not have a pad token.
            pad_token = args.pad_token or processing_class.pad_token or processing_class.eos_token
            pad_token_id = processing_class.convert_tokens_to_ids(pad_token)
            if pad_token_id is None:
                raise ValueError(
                    f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                    "in the vocabulary before using it as a padding token."
                )

            buckets = None
            if num_buckets > 0:
                train_dataloader = self.get_train_dataloader()
                batched_sentence_lengths = [batch["input_ids"].shape[1] for batch in train_dataloader]
                buckets = self._get_buckets(batched_sentence_lengths, num_buckets=num_buckets)
                
            data_collator = DataCollatorForLanguageModeling(
                pad_token_id=pad_token_id,
                completion_only_loss=self.completion_only_loss,
                buckets=buckets
            )

        # Dataset
        preprocess_dataset = args.dataset_kwargs is None or not args.dataset_kwargs.get("skip_prepare_dataset", False)
        if preprocess_dataset:
            train_dataset = self._prepare_dataset(
                train_dataset, processing_class, args, args.packing, formatting_func, "train"
            )
            if eval_dataset is not None:
                packing = args.packing if args.eval_packing is None else args.eval_packing
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, packing, formatting_func, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(
                        eval_dataset, processing_class, args, packing, formatting_func, "eval"
                    )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        # Initialize the Trainer. Parent class will handle:
        # - DeepSpeed configuration (through create_accelerator_and_postprocess)
        # - FSDP setup
        # - Distributed training setup
        # - Optimizer and scheduler creation
        # Some arguments are only available for transformers>=4.47.0. Can be removed when the min version is bumped.
        super_init_kwargs = {}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super_init_kwargs["optimizer_cls_and_kwargs"] = optimizer_cls_and_kwargs
        else:
            if optimizer_cls_and_kwargs is not None:
                warnings.warn(
                    "The `optimizer_cls_and_kwargs` argument is only available for `transformers>=4.47.0`. "
                    "The default optimizer will be used. "
                    "Remove the `optimizer_cls_and_kwargs` or upgrade to `transformers>=4.47.0`."
                )

        GaudiTrainer.__init__(
            self,
            model=model,
            args=args,
            gaudi_config=gaudi_config,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **super_init_kwargs,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    def _prepare_peft_model(self, model: PreTrainedModel, peft_config: Any, args: GaudiSFTConfig) -> PreTrainedModel:
        """
        Copied from SFTTrainer._prepare_peft_model: https://github.com/huggingface/trl/blob/v0.17.0/trl/trainer/sft_trainer.py#L400
        The differences are:
            - use GaudiSFTConfig instead of GaudiSFTConfig
            - cast peft model to bf16.
        """
        if not is_peft_available():
            raise ImportError("To use PeftModel, you need to install the `peft` library.")

        if not isinstance(peft_config, PeftConfig):
            raise ValueError(
                f"Expected PeftConfig object but got {type(peft_config)}. If you want to use the PeftModel, you need "
                "to pass a PeftConfig object to the SFTTrainer."
            )

        if isinstance(model, PeftModel):
            return model

        # Handle quantized models (QLoRA)
        is_qlora = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)

        is_sharded_qlora = False
        if getattr(model, "is_loaded_in_4bit", False):
            # Check if model is sharded (FSDP/DS-Zero3)
            for _, param in model.named_parameters():
                if param.__class__.__name__ == "Params4bit":
                    is_sharded_qlora = param.data.device.type in {"cpu", "meta"}
                    break

        # Prepare model for kbit training if needed
        if is_qlora and not is_sharded_qlora:
            model = self._prepare_model_for_kbit_training(model, args)
            # Disable gradient checkpointing as it's handled by prepare_model_for_kbit_training
            args = dataclasses.replace(args, gradient_checkpointing=False)
        elif args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Create PEFT model
        if (
            version.parse(peft.__version__) >= version.parse("0.12")  # autocast_adapter_dtype introduced in 0.12
            and getattr(model, "is_loaded_in_4bit", False)
            and is_sharded_qlora
        ):
            model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
        else:
            model = get_peft_model(model, peft_config)

        # Handle bf16 casting for 4-bit models
        if args.bf16 and getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora:
            model = model.to(torch.bfloat16)

        return model

    def _get_buckets(self, sentence_lengths, num_buckets):
        return np.unique(
            np.percentile(
                sentence_lengths,
                np.linspace(0, 100, num_buckets + 1),
                method="lower",
            )[1:]
        )
