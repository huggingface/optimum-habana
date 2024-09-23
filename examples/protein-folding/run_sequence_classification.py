#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import functools
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import transformers
from datasets import load_dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef
from transformers import AutoModel, AutoTokenizer, HfArgumentParser, Trainer
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments


try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


# Will error if the minimal version of Optimum Habana is not installed. Remove at your own risks.
check_optimum_habana_min_version("1.14.0.dev0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_optimizer(opt_model, lr_ratio=0.1):
    head_names = []
    for n, p in opt_model.named_parameters():
        if "classifier" in n:
            head_names.append(n)
        else:
            p.requires_grad = False
    # turn a list of tuple to 2 lists
    for n, p in opt_model.named_parameters():
        if n in head_names:
            assert p.requires_grad
    backbone_names = []
    for n, p in opt_model.named_parameters():
        if n not in head_names and p.requires_grad:
            backbone_names.append(n)

    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)  # forbidden layer norm
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    # training_args.learning_rate
    head_decay_parameters = [name for name in head_names if name in decay_parameters]
    head_not_decay_parameters = [name for name in head_names if name not in decay_parameters]
    # training_args.learning_rate * model_config.lr_ratio
    backbone_decay_parameters = [name for name in backbone_names if name in decay_parameters]
    backbone_not_decay_parameters = [name for name in backbone_names if name not in decay_parameters]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in opt_model.named_parameters() if (n in head_decay_parameters and p.requires_grad)],
            "weight_decay": training_args.weight_decay,
            "lr": training_args.learning_rate,
        },
        {
            "params": [
                p for n, p in opt_model.named_parameters() if (n in backbone_decay_parameters and p.requires_grad)
            ],
            "weight_decay": training_args.weight_decay,
            "lr": training_args.learning_rate * lr_ratio,
        },
        {
            "params": [
                p for n, p in opt_model.named_parameters() if (n in head_not_decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
            "lr": training_args.learning_rate,
        },
        {
            "params": [
                p for n, p in opt_model.named_parameters() if (n in backbone_not_decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
            "lr": training_args.learning_rate * lr_ratio,
        },
    ]
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return optimizer


def create_scheduler(training_args, optimizer):
    from transformers.optimization import get_scheduler

    return get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer if optimizer is None else optimizer,
        num_warmup_steps=training_args.get_warmup_steps(training_args.max_steps),
        num_training_steps=training_args.max_steps,
    )


def compute_metrics(eval_preds):
    probs, labels = eval_preds
    preds = np.argmax(probs, axis=-1)
    result = {"accuracy": accuracy_score(labels, preds), "mcc": matthews_corrcoef(labels, preds)}
    return result


def preprocess_logits_for_metrics(logits, labels):
    return torch.softmax(logits, dim=-1)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_length: Optional[str] = field(
        default=1024,
        metadata={"help": ("the max length that input id will be padded to")},
    )


if __name__ == "__main__":
    model_args, data_args, training_args = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, GaudiTrainingArguments)
    ).parse_args_into_dataclasses()

    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, torch_dtype=torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    def tokenize_protein(example, tokenizer=None):
        protein_seq = example["prot_seq"]
        protein_seq_str = tokenizer(protein_seq, add_special_tokens=True)
        example["input_ids"] = protein_seq_str["input_ids"]
        example["attention_mask"] = protein_seq_str["attention_mask"]
        example["labels"] = example["localization"]
        return example

    func_tokenize_protein = functools.partial(tokenize_protein, tokenizer=tokenizer)

    if data_args.dataset_name != "mila-intel/ProtST-BinaryLocalization":
        raise ValueError("preprocess is only for mila-intel/ProtST-BinaryLocalization now")
    raw_dataset = load_dataset(data_args.dataset_name)
    for split in ["train", "validation", "test"]:
        raw_dataset[split] = raw_dataset[split].map(
            func_tokenize_protein, batched=False, remove_columns=["Unnamed: 0", "prot_seq", "localization"]
        )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=data_args.max_length)

    optimizer = create_optimizer(model)
    scheduler = create_scheduler(training_args, optimizer)

    # build trainer
    gaudi_config = GaudiConfig()
    gaudi_config.use_fused_adam = True
    gaudi_config.use_fused_clip_norm = True

    trainer = GaudiTrainer(
        model=model,
        gaudi_config=gaudi_config,
        args=training_args,
        train_dataset=raw_dataset["train"],
        eval_dataset=raw_dataset["validation"],
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        # Saves the tokenizer too for easy upload
        tokenizer.save_pretrained(training_args.output_dir)

        metrics = train_result.metrics
        metrics["train_samples"] = len(raw_dataset["train"])

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate(raw_dataset["test"], metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        metrics = trainer.evaluate(raw_dataset["validation"], metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
