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
multi-task tuning script for sequence-to-sequence modeling
Adapted from the following sources:
https://github.com/huggingface/peft/blob/main/examples/conditional_generation/multitask_prompt_tuning.ipynb
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import evaluate
import torch
import transformers
from datasets import load_dataset
from peft import (
    MultitaskPromptTuningConfig,
    MultitaskPromptTuningInit,
    TaskType,
    get_peft_model,
)
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

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
        default=16,
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

    peft_config = MultitaskPromptTuningConfig(
        tokenizer_name_or_path=model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        num_tasks=2,
        task_type=TaskType.SEQ_2_SEQ_LM,
        prompt_tuning_init=MultitaskPromptTuningInit.TEXT,
        num_virtual_tokens=50,
        num_transformer_submodules=1,
        prompt_tuning_init_text="classify the following into either positive or negative, or entailment, neutral or contradiction:",
    )

    target_dict_path = training_args.output_dir + "/adapter_model.bin"
    peft_config_target = MultitaskPromptTuningConfig(
        tokenizer_name_or_path=model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        num_tasks=1,
        task_type=TaskType.SEQ_2_SEQ_LM,
        prompt_tuning_init=MultitaskPromptTuningInit.EXACT_SOURCE_TASK,
        num_virtual_tokens=50,
        num_transformer_submodules=1,
        prompt_tuning_init_state_dict_path=target_dict_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def get_sst2(split: str):
        examples = load_dataset("sst2", split=split)
        result_examples = []
        for example in examples:
            result_examples.append({})

            result_examples[-1]["input"] = example["sentence"].strip() + "</s>"
            result_examples[-1]["output"] = (
                f"positive{tokenizer.eos_token}" if example["label"] == 1 else f"negative{tokenizer.eos_token}"
            )
            result_examples[-1]["task_id"] = 0

        return result_examples

    def get_mnli(split: str):
        examples = load_dataset("multi_nli", split=split)
        result_examples = []
        for example in examples:
            result_examples.append({})

            result_examples[-1]["input"] = example["premise"].strip() + " " + example["hypothesis"].strip() + "</s>"

            if example["label"] == 0:
                result_examples[-1]["output"] = f"entailment{tokenizer.eos_token}"
            elif example["label"] == 1:
                result_examples[-1]["output"] = f"neutral{tokenizer.eos_token}"
            else:
                result_examples[-1]["output"] = f"contradiction{tokenizer.eos_token}"

            result_examples[-1]["task_id"] = 1

        return result_examples

    class MyDataset(Dataset):
        def __init__(self, split: str, mode: str = "source") -> None:
            super().__init__()

            if split == "train":
                if mode == "source":
                    self.examples = get_sst2(split) + get_mnli(split)
                elif mode == "target":
                    self.examples = get_sst2(split)
            if split == "val":
                self.examples = get_sst2("validation")
            if split == "test":
                self.examples = get_sst2("validation")

        def __getitem__(self, index) -> dict:
            return self.examples[index]

        def __len__(self) -> int:
            return len(self.examples)

    def collate_fn(batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        input = [i["input"] for i in batch]
        input = tokenizer(
            input,
            add_special_tokens=False,
            return_tensors="pt",
            padding="max_length",
            max_length=data_args.max_source_length,
            truncation=True,
        )
        output = [i["output"] for i in batch]
        output = tokenizer(
            output,
            add_special_tokens=False,
            return_tensors="pt",
            padding="max_length",
            max_length=data_args.max_target_length,
            truncation=True,
        ).input_ids
        output[output == tokenizer.pad_token_id] = -100

        task_ids = [i["task_id"] for i in batch]
        task_ids = torch.tensor(task_ids)

        return {
            "input_ids": input.input_ids,
            "attention_mask": input.attention_mask,
            "labels": output,
            "task_ids": task_ids,
        }

    metric = evaluate.load("f1", cache_dir=model_args.cache_dir)
    POSITIVE_TOKEN_ID = tokenizer(" positive", add_special_tokens=False)["input_ids"][0]
    NEGATIVE_TOKEN_ID = tokenizer(" negative", add_special_tokens=False)["input_ids"][0]

    def compute_metrics(pred):
        scores = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        pred_ids = []
        label_ids = []
        for i in range(scores.shape[0]):
            if scores[i, 0, POSITIVE_TOKEN_ID] > scores[i, 0, NEGATIVE_TOKEN_ID]:
                pred_ids.append(POSITIVE_TOKEN_ID)
            else:
                pred_ids.append(NEGATIVE_TOKEN_ID)
            label_ids.append(pred.label_ids[i][0])

        # we do not want to group tokens when computing the metrics

        return metric.compute(predictions=pred_ids, references=label_ids, pos_label=POSITIVE_TOKEN_ID)

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

    # could remove when peft tag upgrades and contain https://github.com/huggingface/peft/pull/1662
    training_args.save_safetensors = False
    # source train
    trainer = GaudiSeq2SeqTrainer(
        model=peft_model,
        gaudi_config=gaudi_config,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=MyDataset("train"),
        eval_dataset=MyDataset("val"),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        logger.info("***source finetune***")
        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    if training_args.do_eval:
        logger.info("*** Evaluate after source finetune**")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # target train
    peft_model = get_peft_model(model, peft_config_target)
    peft_model.print_trainable_parameters()
    trainer = GaudiSeq2SeqTrainer(
        model=peft_model,
        gaudi_config=gaudi_config,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=MyDataset("train", "target"),
        eval_dataset=MyDataset("val", "target"),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        logger.info("***target finetune***")
        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    if training_args.do_eval:
        logger.info("*** Evaluate after target finetune***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
