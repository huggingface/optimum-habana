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
"""
poly tuning script for sequence-to-sequence modeling
Adapted from the following sources:
https://github.com/huggingface/peft/blob/main/examples/poly/peft_poly_seq2seq_with_generate.ipynb
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import (
    PolyConfig,
    TaskType,
    get_peft_model,
    tuners,
)
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
)
from transformers.trainer_utils import is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from optimum.habana import GaudiConfig, GaudiSeq2SeqTrainer, GaudiSeq2SeqTrainingArguments
from optimum.habana.utils import set_seed


try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers and Optimum Habana are not installed. Remove at your own risk.
check_min_version("4.38.0")
check_optimum_habana_min_version("1.10.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set it if you want to train a model from"
                " scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    use_cache: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not the model should return the last key/values attentions (not used by all models)."
                "Only relevant if `config.is_decoder=True`."
            )
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "Setting it to True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    r: int = field(
        default=8,
        metadata={"help": ("rank of lora in poly.")},
    )
    n_skills: int = field(
        default=2,
        metadata={"help": ("number of skills in poly")},
    )
    n_splits: int = field(
        default=4,
        metadata={"help": ("number of skills in poly")},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    max_train_samples: Optional[int] = field(
        default=1000,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of train examples to this "
                "value if set."
            )
        },
    )

    max_eval_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=2,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GaudiSeq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    set_seed(training_args.seed)
    from optimum.habana.peft.layer import GaudiPolyLayerLinearForward

    tuners.poly.layer.Linear.forward = GaudiPolyLayerLinearForward
    peft_config = PolyConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        poly_type="poly",
        r=model_args.r,
        n_tasks=4,
        n_skills=model_args.n_skills,
        n_splits=model_args.n_splits,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # boolq
    boolq_dataset = (
        load_dataset("super_glue", "boolq")
        .map(
            lambda x: {
                "input": f"{x['passage']}\nQuestion: {x['question']}\nA. Yes\nB. No\nAnswer:",
                "output": ["B", "A"][int(x["label"])],
                "task_name": "boolq",
            }
        )
        .select_columns(["input", "output", "task_name"])
    )
    logger.info("boolq example: ")
    logger.info(boolq_dataset["train"][0])

    # multirc
    multirc_dataset = (
        load_dataset("super_glue", "multirc")
        .map(
            lambda x: {
                "input": (
                    f"{x['paragraph']}\nQuestion: {x['question']}\nAnswer: {x['answer']}\nIs it"
                    " true?\nA. Yes\nB. No\nAnswer:"
                ),
                "output": ["B", "A"][int(x["label"])],
                "task_name": "multirc",
            }
        )
        .select_columns(["input", "output", "task_name"])
    )
    logger.info("multirc example: ")
    logger.info(multirc_dataset["train"][0])

    # rte
    rte_dataset = (
        load_dataset("super_glue", "rte")
        .map(
            lambda x: {
                "input": (
                    f"{x['premise']}\n{x['hypothesis']}\nIs the sentence below entailed by the"
                    " sentence above?\nA. Yes\nB. No\nAnswer:"
                ),
                "output": ["A", "B"][int(x["label"])],
                "task_name": "rte",
            }
        )
        .select_columns(["input", "output", "task_name"])
    )
    logger.info("rte example: ")
    logger.info(rte_dataset["train"][0])

    # wic
    wic_dataset = (
        load_dataset("super_glue", "wic")
        .map(
            lambda x: {
                "input": (
                    f"Sentence 1: {x['sentence1']}\nSentence 2: {x['sentence2']}\nAre '{x['word']}'"
                    " in the above two sentences the same?\nA. Yes\nB. No\nAnswer:"
                ),
                # 0 - False
                # 1 - True
                "output": ["B", "A"][int(x["label"])],
                "task_name": "wic",
            }
        )
        .select_columns(["input", "output", "task_name"])
    )
    logger.info("wic example: ")
    logger.info(wic_dataset["train"][0])

    # define a task2id map
    TASK2ID = {
        "boolq": 0,
        "multirc": 1,
        "rte": 2,
        "wic": 3,
    }

    def tokenize(examples):
        inputs, targets = examples["input"], examples["output"]
        features = tokenizer(
            inputs, max_length=data_args.max_source_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = tokenizer(
            targets, max_length=data_args.max_target_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        features["labels"] = labels
        features["task_ids"] = torch.tensor([[TASK2ID[t]] for t in examples["task_name"]]).long()
        return features

    def get_superglue_dataset(
        split="train",
        n_samples=500,
    ):
        ds = concatenate_datasets(
            [
                boolq_dataset[split].shuffle().select(range(n_samples)),
                multirc_dataset[split].shuffle().select(range(n_samples)),
                rte_dataset[split].shuffle().select(range(n_samples)),
                wic_dataset[split].shuffle().select(range(n_samples)),
            ]
        )
        ds = ds.map(
            tokenize,
            batched=True,
            remove_columns=["input", "output", "task_name"],
            load_from_cache_file=False,
        )
        return ds

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = [[i for i in seq if i != -100] for seq in preds]
        labels = [[i for i in seq if i != -100] for seq in labels]
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        correct = 0
        total = 0
        for pred, true in zip(preds, labels):
            if pred.strip() == true.strip():
                correct += 1
            total += 1
        accuracy = correct / total
        return {"accuracy": accuracy}

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True if model_args.trust_remote_code else None,
        "use_cache": False if training_args.gradient_checkpointing else model_args.use_cache,
        "token": model_args.token,
    }
    # creating model
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError("Please provide value for model_name_or_path or config_name.")
    model_dtype = torch.bfloat16 if training_args.bf16 else None

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True if model_args.trust_remote_code else None,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        token=model_args.token,
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    # training and evaluation
    gaudi_config = GaudiConfig()
    gaudi_config.use_fused_adam = True
    gaudi_config.use_fused_clip_norm = True

    # Initialize our Trainer
    training_args.remove_unused_columns = False
    training_args.predict_with_generate = True
    training_args.generation_max_length = 2

    superglue_train_dataset = get_superglue_dataset(split="train", n_samples=data_args.max_train_samples)
    superglue_eval_dataset = get_superglue_dataset(split="test", n_samples=data_args.max_eval_samples)

    trainer = GaudiSeq2SeqTrainer(
        model=peft_model,
        gaudi_config=gaudi_config,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=superglue_train_dataset,
        eval_dataset=superglue_eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if is_main_process(training_args.local_rank):
        i = 5
        inputs = tokenizer(rte_dataset["validation"]["input"][i], return_tensors="pt")
        inputs["task_ids"] = torch.LongTensor([TASK2ID["rte"]])
        inputs = {k: v.to("hpu") for k, v in inputs.items()}
        logger.info(rte_dataset["validation"]["input"][i])
        logger.info(rte_dataset["validation"]["output"][i])
        logger.info(inputs)

        with torch.no_grad():
            outputs = peft_model.generate(**inputs, max_new_tokens=2)
        logger.info(outputs[0])
        logger.info(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
