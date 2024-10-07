#!/usr/bin/env python
# coding=utf-8

# Apache v2 license
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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

import copy
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import evaluate
import torch
import transformers
from datasets import load_dataset
from peft import (
    AdaLoraConfig,
    AdaptionPromptConfig,
    IA3Config,
    LNTuningConfig,
    LoraConfig,
    TaskType,
    VeraConfig,
    get_peft_model,
    tuners,
)
from peft.utils.other import fsdp_auto_wrap_policy
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
)
from transformers.trainer_utils import is_main_process

from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
from optimum.habana.utils import set_seed


try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


IGNORE_INDEX = -100

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

# Will error if the minimal version of Optimum Habana is not installed. Remove at your own risks.
check_optimum_habana_min_version("1.14.0.dev0")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "auth token for private models"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
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
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "When set to True, it will benefit LLM loading time and RAM consumption."
            )
        },
    )
    attn_softmax_bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to run attention softmax layer in bf16 precision for fine-tuning. The current support is limited to Llama only."
            )
        },
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use Habana flash attention for fine-tuning. The current support is limited to Llama only."
            )
        },
    )
    flash_attention_recompute: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable recompute in Habana flash attention for fine-tuning."
                " It is applicable only when use_flash_attention is True."
            )
        },
    )
    flash_attention_causal_mask: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable causal mask in Habana flash attention for fine-tuning."
                " It is applicable only when use_flash_attention is True."
            )
        },
    )
    flash_attention_fp8: bool = field(
        default=False,
        metadata={"help": ("Whether to enable flash attention in FP8.")},
    )
    use_fused_rope: bool = field(
        default=True,
        metadata={
            "help": ("Whether to use Habana fused-rope for fine-tuning. The current support is limited to Llama only.")
        },
    )
    load_meta_device: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to load the model to the device instead of the host, so it can reduce the host RAM usage."
                "https://huggingface.co/blog/accelerate-large-models"
            )
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    validation_split_percentage: Optional[int] = field(
        default=0,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={"help": "Whether to keep in memory the loaded dataset. Defaults to False."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    dataset_seed: int = field(
        default=42,
        metadata={
            "help": "Seed to use in dataset processing, different seeds might yield different datasets. This seed and the seed in training arguments are not related"
        },
    )
    dataset_cache_directory: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to directory where the processed dataset will be saved. If path exists, try to load processed dataset from this path."
        },
    )
    dataset_concatenation: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to concatenate the sentence for more efficient training."},
    )
    sql_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to have a SQL style prompt"},
    )
    chat_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to have a chat style prompt."},
    )
    save_last_ckpt: bool = field(
        default=True, metadata={"help": "Whether to save checkpoint at the end of the training."}
    )
    instruction_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the column in the dataset that describes the task that the model should perform. By "
            "default, the 'instruction' column is used for non-SQL prompts and the 'question' column is used for SQL prompts."
        },
    )
    input_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the column in the dataset that optionally provides context or input for the task. By "
            "default, the 'input' column is used for non-SQL prompts and the 'context' column is used for SQL prompts."
        },
    )
    output_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the column in the dataset with the answer to the instruction. By default, the "
            "'output' column is used for non-SQL prompts and the 'answer' column is used for SQL prompts."
        },
    )


@dataclass
class FinetuneArguments:
    """
    Arguments of finetune we are going to apply on the model.
    """

    lora_rank: int = field(
        default=8,
        metadata={"help": "Rank parameter in the LoRA method."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Alpha parameter in the LoRA method."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout parameter in the LoRA method."},
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA/AdaLoRA method."},
    )
    train_on_inputs: bool = field(
        default=True,
        metadata={"help": "if False, masks out inputs in loss"},
    )
    adalora_init_r: int = field(
        default=12,
        metadata={"help": "Initial AdaLoRA rank"},
    )
    adalora_target_r: int = field(
        default=4,
        metadata={"help": "Target AdaLoRA rank"},
    )
    adalora_tinit: int = field(
        default=50,
        metadata={"help": "Number of warmup steps for AdaLoRA wherein no pruning is performed"},
    )
    adalora_tfinal: int = field(
        default=100,
        metadata={
            "help": "Fix the resulting budget distribution and fine-tune the model for tfinal steps when using AdaLoRA"
        },
    )
    adalora_delta_t: int = field(
        default=10,
        metadata={"help": "Interval of steps for AdaLoRA to update rank"},
    )
    adalora_orth_reg_weight: float = field(
        default=0.5,
        metadata={"help": "Orthogonal regularization weight for AdaLoRA"},
    )
    peft_type: str = field(
        default="lora",
        metadata={
            "help": ("The PEFT type to use."),
            "choices": ["lora", "ia3", "adalora", "llama-adapter", "vera", "ln_tuning"],
        },
    )
    ia3_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the IA3 method."},
    )
    feedforward_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target feedforward modules for the IA3 method."},
    )
    adapter_layers: int = field(
        default=30,
        metadata={"help": "Number of adapter layers (from the top) in llama-adapter"},
    )
    adapter_len: int = field(
        default=10,
        metadata={"help": "Number of adapter tokens to insert in llama-adapter"},
    )
    vera_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the vera method."},
    )
    ln_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the ln method."},
    )


PROMPT_DICT = {
    "prompt_with_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_without_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

SQL_PROMPT = (
    "You are a text-to-SQL model. Your job is to answer questions about a database. "
    "You are given a question and a context regarding one or more tables in the database.\n\n"
    "You must output the SQL query that answers the question. The SQL query must be between [SQL] and [/SQL] tags.\n\n"
    "### Question: \n{question}\n\n### Context: \n{context}\n\n### Response:"
)


def create_prompts(examples):
    prompts = {}
    prompts["source"] = []
    prompts["target"] = []
    for example in examples:
        prompt_template = (
            PROMPT_DICT["prompt_with_input"] if example.get("input", "") != "" else PROMPT_DICT["prompt_without_input"]
        )
        source = prompt_template.format_map(example)
        prompts["source"].append(source)
        prompts["target"].append(example["output"])
    return prompts


def create_chat_prompts(examples, tokenizer):
    prompts = {}
    prompts["source"] = []
    prompts["target"] = []
    for example in examples:
        prompt = [
            {
                "role": "user",
                "content": "Answer the below Query based on the Content given below. #### Query: {instruction} #### Content: {input}".format_map(
                    example
                ),
            },
        ]
        source = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompts["source"].append(source)
        prompts["target"].append(example["output"])
    return prompts


def create_sql_prompts(examples):
    prompts = {}
    prompts["source"] = []
    prompts["target"] = []
    for example in examples:
        source = SQL_PROMPT.format_map(example)
        prompts["source"].append(source)
        prompts["target"].append(example["answer"])
    return prompts


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, GaudiTrainingArguments, FinetuneArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, finetune_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            finetune_args,
        ) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary
    b16 = training_args.fp16 or training_args.bf16
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {b16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "trust_remote_code": True if model_args.trust_remote_code else None,
        "use_cache": False if training_args.gradient_checkpointing else model_args.use_cache,
        "token": model_args.token,
        "flash_attention_fp8": model_args.flash_attention_fp8,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError("Please provide value for model_name_or_path or config_name.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "padding_side": "right",
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

        if "validation" not in raw_datasets.keys() and training_args.do_eval:
            if not data_args.validation_split_percentage:
                raise ValueError(
                    "Please set --validation_split_percentage as dataset does not contain `validation` key"
                )
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension in ("txt", "text"):
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        if extension in ("json", "jsonl"):
            extension = "json"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys() and training_args.do_eval:
            if not data_args.validation_split_percentage:
                raise ValueError(
                    "Please set --validation_split_percentage as dataset does not contain `validation` key"
                )
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
    single_column_dataset = False
    # For named dataset (timdettmers/openassistant-guanaco) or custom dataset with a single column "text"
    if (
        training_args.do_train
        and raw_datasets["train"].num_columns == 1
        or training_args.do_eval
        and raw_datasets["validation"].num_columns == 1
    ):
        single_column_dataset = True
        raw_datasets = raw_datasets.map(
            lambda x: {
                "input": "",
                "output": x["text"],
            }
        )
        if training_args.do_train:
            # Remove unused columns.
            raw_datasets = raw_datasets.remove_columns(
                [col for col in raw_datasets.column_names["train"] if col not in ["input", "output"]]
            )

        if training_args.do_eval:
            # Remove unused columns.
            raw_datasets = raw_datasets.remove_columns(
                [col for col in raw_datasets.column_names["validation"] if col not in ["input", "output"]]
            )
    else:
        # Preprocessing the datasets.
        for key in raw_datasets:
            if data_args.instruction_column_name:
                raw_datasets[key] = raw_datasets[key].rename_column(
                    data_args.instruction_column_name, "question" if data_args.sql_prompt else "instruction"
                )

            if data_args.input_column_name:
                raw_datasets[key] = raw_datasets[key].rename_column(
                    data_args.input_column_name, "context" if data_args.sql_prompt else "input"
                )

            if data_args.output_column_name:
                raw_datasets[key] = raw_datasets[key].rename_column(
                    data_args.output_column_name, "answer" if data_args.sql_prompt else "output"
                )

            if data_args.chat_prompt:
                prompts = create_chat_prompts(raw_datasets[key], tokenizer)
            elif data_args.sql_prompt:
                prompts = create_sql_prompts(raw_datasets[key])
            else:
                prompts = create_prompts(raw_datasets[key])
            columns_to_be_removed = list(raw_datasets[key].features.keys())
            raw_datasets[key] = raw_datasets[key].add_column("prompt_sources", prompts["source"])
            raw_datasets[key] = raw_datasets[key].add_column("prompt_targets", prompts["target"])
            raw_datasets[key] = raw_datasets[key].remove_columns(columns_to_be_removed)

    # Load model
    if model_args.model_name_or_path:
        model_dtype = torch.bfloat16 if training_args.bf16 else None
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            trust_remote_code=True if model_args.trust_remote_code else None,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            device_map=training_args.device.type if model_args.load_meta_device else None,
            token=model_args.token,
        )
    else:
        raise ValueError("Must provide model_name_or_path to load a pretrained CausalLM model.")

    if model.config.model_type == "llama":
        # unwind broken decapoda-research config
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        if model_args.attn_softmax_bf16:
            model.generation_config.attn_softmax_bf16 = True
        if model_args.use_flash_attention:
            model.generation_config.use_flash_attention = True
            model.generation_config.flash_attention_recompute = model_args.flash_attention_recompute
            model.generation_config.flash_attention_causal_mask = model_args.flash_attention_causal_mask

            if model_args.flash_attention_fp8:
                import habana_frameworks.torch.hpu as hthpu

                assert hthpu.get_device_name() == "GAUDI3", "Flash attention in FP8 is supported only on Gaudi3"
        if not model_args.use_fused_rope:
            model.generation_config.use_fused_rope = False

    if hasattr(model.generation_config, "pad_token_id") and model.generation_config.pad_token_id is not None:
        tokenizer.pad_token_id = model.generation_config.pad_token_id
    if hasattr(model.generation_config, "eos_token_id") and model.generation_config.eos_token_id is not None:
        tokenizer.eos_token_id = model.generation_config.eos_token_id
    if hasattr(model.generation_config, "bos_token_id") and model.generation_config.bos_token_id is not None:
        tokenizer.bos_token_id = model.generation_config.bos_token_id

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize(prompt, add_eos_token=True, add_bos_token=True):
        if hasattr(tokenizer, "add_eos_token"):
            add_eos_token_o = tokenizer.add_eos_token
        else:
            add_eos_token_o = None

        if hasattr(tokenizer, "add_bos_token"):
            add_bos_token_o = tokenizer.add_bos_token
        else:
            add_bos_token_o = None

        if not data_args.dataset_concatenation:
            tokenizer.add_eos_token = add_eos_token
            padding = "max_length"
        else:
            padding = False
        tokenizer.add_bos_token = add_bos_token
        results = tokenizer(
            prompt,
            truncation=True,
            max_length=data_args.max_seq_length,
            padding=padding,
            return_tensors=None,
        )
        # restore original value
        if add_eos_token_o is not None:
            tokenizer.add_eos_token = add_eos_token_o

        if add_bos_token_o is not None:
            tokenizer.add_bos_token = add_bos_token_o

        for i in range(len(results["input_ids"])):
            if (
                results["input_ids"][i][-1] != tokenizer.eos_token_id
                and len(results["input_ids"][i]) < data_args.max_seq_length
                and add_eos_token
            ):
                results["input_ids"][i].append(tokenizer.eos_token_id)
                results["attention_mask"][i].append(1)

        results["labels"] = copy.deepcopy(results["input_ids"])
        results["input_id_len"] = [len(result) for result in results["input_ids"]]
        return results

    def preprocess_function(examples):
        keys = list(examples.data.keys())
        if len(keys) != 2:
            raise ValueError(f"Unsupported dataset format, number of keys {keys} !=2")

        st = [s + t for s, t in zip(examples[keys[0]], examples[keys[1]])]
        add_bos_token = False if data_args.chat_prompt else True
        examples_tokenized = tokenize(st, add_bos_token=add_bos_token)
        input_ids = examples_tokenized["input_ids"]
        labels = examples_tokenized["labels"]
        if not finetune_args.train_on_inputs:
            sources_tokenized = tokenize(examples[keys[0]], add_eos_token=False, add_bos_token=add_bos_token)
            for label, source_len in zip(labels, sources_tokenized["input_id_len"]):
                label[:source_len] = [IGNORE_INDEX] * source_len
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": examples_tokenized["attention_mask"],
        }

    with training_args.main_process_first(desc="dataset map pre-processing"):
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if data_args.dataset_concatenation:

        def concatenate_data(dataset, max_seq_length):
            concatenated_dataset = {}
            for column in dataset.features:
                concatenated_data = [item for sample in dataset[column] for item in sample]
                reshaped_data = [
                    concatenated_data[i * max_seq_length : (i + 1) * max_seq_length]
                    for i in range(len(concatenated_data) // max_seq_length)
                ]
                concatenated_dataset[column] = reshaped_data
            return datasets.Dataset.from_dict(concatenated_dataset)

        if single_column_dataset:
            tokenized_datasets_ = tokenized_datasets["train"].remove_columns(["input", "output"])
            if training_args.do_eval:
                tokenized_datasets_eval_ = tokenized_datasets["validation"].remove_columns(["input", "output"])
        else:
            tokenized_datasets_ = tokenized_datasets["train"].remove_columns(["prompt_sources", "prompt_targets"])
            if training_args.do_eval:
                tokenized_datasets_eval_ = tokenized_datasets["validation"].remove_columns(
                    ["prompt_sources", "prompt_targets"]
                )
        if training_args.do_train:
            tokenized_datasets["train"] = concatenate_data(tokenized_datasets_, data_args.max_seq_length)
        if training_args.do_eval:
            tokenized_datasets["validation"] = concatenate_data(tokenized_datasets_eval_, data_args.max_seq_length)
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8, return_tensors="pt", mlm=False)
    logger.info("Using data collator of type {}".format(data_collator.__class__.__name__))

    if training_args.do_train or training_args.do_eval:
        # PEFT settings
        if finetune_args.peft_type == "lora":
            peft_config = LoraConfig(
                r=finetune_args.lora_rank,
                lora_alpha=finetune_args.lora_alpha,
                lora_dropout=finetune_args.lora_dropout,
                target_modules=finetune_args.lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
        elif finetune_args.peft_type == "adalora":
            peft_config = AdaLoraConfig(
                init_r=finetune_args.adalora_init_r,
                target_r=finetune_args.adalora_target_r,
                tinit=finetune_args.adalora_tinit,
                tfinal=finetune_args.adalora_tfinal,
                deltaT=finetune_args.adalora_delta_t,
                lora_alpha=finetune_args.lora_alpha,
                lora_dropout=finetune_args.lora_dropout,
                target_modules=finetune_args.lora_target_modules,
                orth_reg_weight=finetune_args.adalora_orth_reg_weight,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            from optimum.habana.peft.layer import GaudiAdaloraLayerSVDLinearForward

            tuners.adalora.layer.SVDLinear.forward = GaudiAdaloraLayerSVDLinearForward
        elif finetune_args.peft_type == "ia3":
            peft_config = IA3Config(
                target_modules=finetune_args.ia3_target_modules,
                feedforward_modules=finetune_args.feedforward_modules,
                task_type=TaskType.CAUSAL_LM,
            )
        elif finetune_args.peft_type == "llama-adapter":
            peft_config = AdaptionPromptConfig(
                adapter_layers=finetune_args.adapter_layers,
                adapter_len=finetune_args.adapter_len,
                task_type=TaskType.CAUSAL_LM,
            )
            from optimum.habana.peft.layer import (
                GaudiAdaptedAttention_getattr,
                GaudiAdaptedAttentionPreAttnForward,
            )

            tuners.adaption_prompt.layer.AdaptedAttention.pre_attn_forward = GaudiAdaptedAttentionPreAttnForward
            tuners.adaption_prompt.layer.AdaptedAttention.__getattr__ = GaudiAdaptedAttention_getattr
        elif finetune_args.peft_type == "vera":
            peft_config = VeraConfig(
                target_modules=finetune_args.vera_target_modules, task_type=TaskType.CAUSAL_LM, init_weights=False
            )
        elif finetune_args.peft_type == "ln_tuning":
            peft_config = LNTuningConfig(
                target_modules=finetune_args.ln_target_modules,
                task_type=TaskType.CAUSAL_LM,
            )
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
        lora_model = get_peft_model(model, peft_config)
        if training_args.bf16 and finetune_args.peft_type != "ia3":
            lora_model = lora_model.to(torch.bfloat16)
        lora_model.print_trainable_parameters()
        gaudi_config = GaudiConfig()
        gaudi_config.use_fused_adam = True
        gaudi_config.use_fused_clip_norm = True

        # Initialize our Trainer
        trainer = GaudiTrainer(
            model=lora_model,
            gaudi_config=gaudi_config,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.do_eval else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
        )

        # Solution for https://github.com/huggingface/peft/blob/v0.6.2/README.md#caveats (1)
        if training_args.fsdp and training_args.fsdp_config["auto_wrap_policy"] == "TRANSFORMER_BASED_WRAP":
            trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(lora_model)

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        if data_args.save_last_ckpt:
            trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
