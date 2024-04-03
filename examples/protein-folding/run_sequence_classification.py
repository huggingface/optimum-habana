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
check_optimum_habana_min_version("1.10.0")

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


if __name__ == "__main__":
    device = torch.device("hpu")

    training_args = HfArgumentParser(GaudiTrainingArguments).parse_args_into_dataclasses()[0]

    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    model = AutoModel.from_pretrained(
        "mila-intel/protst-esm1b-for-sequential-classification", trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")

    def tokenize_protein(example, tokenizer=None):
        protein_seq = example["prot_seq"]
        protein_seq_str = tokenizer(protein_seq, add_special_tokens=True)
        example["input_ids"] = protein_seq_str["input_ids"]
        example["attention_mask"] = protein_seq_str["attention_mask"]
        example["labels"] = example["localization"]
        return example

    func_tokenize_protein = functools.partial(tokenize_protein, tokenizer=tokenizer)

    raw_dataset = load_dataset("mila-intel/ProtST-BinaryLocalization")
    for split in ["train", "validation", "test"]:
        raw_dataset[split] = raw_dataset[split].map(
            func_tokenize_protein, batched=False, remove_columns=["Unnamed: 0", "prot_seq", "localization"]
        )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=1024)

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

    train_result = trainer.train()

    trainer.save_model()
    # Saves the tokenizer too for easy upload
    tokenizer.save_pretrained(training_args.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_dataset["train"])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    metric = trainer.evaluate(raw_dataset["test"], metric_key_prefix="test")
    print("test metric: ", metric)

    metric = trainer.evaluate(raw_dataset["validation"], metric_key_prefix="valid")
    print("valid metric: ", metric)
