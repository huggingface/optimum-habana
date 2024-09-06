# Apache v2 license
# Copyright (C) 2024 Intel Corporation
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
"""
poly tuning script for sequence-to-sequence modeling
Adapted from the following sources:
https://colab.research.google.com/drive/1rm3AGquGEYXfeeizE40bbDtcWh5S4Nlq?usp=sharing
"""

import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import Levenshtein
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForVision2Seq,
    AutoProcessor,
    HfArgumentParser,
)
from transformers.trainer_utils import is_main_process

from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments


try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

# Will error if the minimal version of Optimum Habana is not installed. Remove at your own risks.
check_optimum_habana_min_version("1.10.0")


def normalized_levenshtein(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    distance = Levenshtein.distance(s1, s2)
    return distance / max(len_s1, len_s2)


def similarity_score(a_ij, o_q_i, tau=0.5):
    nl = normalized_levenshtein(a_ij, o_q_i)
    return 1 - nl if nl < tau else 0


def average_normalized_levenshtein_similarity(ground_truth, predicted_answers):
    assert len(ground_truth) == len(predicted_answers), "Length of ground_truth and predicted_answers must match."

    N = len(ground_truth)
    total_score = 0

    for i in range(N):
        a_i = ground_truth[i]
        o_q_i = predicted_answers[i]
        if o_q_i == "":
            print("Warning: Skipped an empty prediction.")
            max_score = 0
        else:
            max_score = max(similarity_score(a_ij, o_q_i) for a_ij in a_i)

        total_score += max_score

    return total_score / N


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/processor we are going to fine-tune, or train from scratch.
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
    load_meta_device: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to load the model to the device instead of the host, so it can reduce the host RAM usage."
                "https://huggingface.co/blog/accelerate-large-models"
            )
        },
    )
    do_image_splitting: bool = field(default=False, metadata={"help": "Whether to do image split during finetune."})


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
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
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
    dataset_seed: int = field(
        default=42,
        metadata={
            "help": "Seed to use in dataset processing, different seeds might yield different datasets. This seed and the seed in training arguments are not related"
        },
    )
    save_last_ckpt: bool = field(
        default=True, metadata={"help": "Whether to save checkpoint at the end of the training."}
    )
    input_column_names: List[str] = field(
        default_factory=lambda: None,
        metadata={
            "help": "Name of the column in the dataset that optionally provides context or input for the task. By "
            "default, 'image,query' columns are used"
        },
    )
    output_column_names: List[str] = field(
        default_factory=lambda: None,
        metadata={
            "help": "Name of the column in the dataset with the answer to the instruction. By default, the "
            "'answers' column is used"
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
        default=8,
        metadata={"help": "Alpha parameter in the LoRA method."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout parameter in the LoRA method."},
    )
    lora_target_modules: str = field(
        default=None,
        metadata={"help": "Target modules for the LoRA/AdaLoRA method."},
    )


class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, examples):
        texts = []
        images = []
        keys = list(examples[0].keys())
        if not all(key in ["image", "query", "answers"] for key in keys):
            raise ValueError("Unsupported dataset format")
        for example in examples:
            image = example["image"]
            question = example["query"]["en"]
            answer = random.choice(example["answers"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        batch["labels"] = labels

        return batch


def eval(processor, model, eval_dataset, eval_batch_size):
    from tqdm import tqdm

    answers_unique = []
    generated_texts_unique = []

    for i in tqdm(range(0, len(eval_dataset), eval_batch_size)):
        examples = eval_dataset[i : i + eval_batch_size]
        answers_unique.extend(examples["answers"])
        images = [[im] for im in examples["image"]]
        texts = []
        for q in examples["query"]:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": q["en"]},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            texts.append(text.strip())
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to("hpu") for k, v in inputs.items()}
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_texts = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
        )
        generated_texts_unique.extend(generated_texts)
    generated_texts_unique = [g.strip().strip(".") for g in generated_texts_unique]
    anls = average_normalized_levenshtein_similarity(
        ground_truth=answers_unique,
        predicted_answers=generated_texts_unique,
    )
    return anls


def main():
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

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        do_image_splitting=model_args.do_image_splitting,
        padding_side="right",
    )

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "trust_remote_code": True if model_args.trust_remote_code else None,
        "use_cache": False if training_args.gradient_checkpointing else model_args.use_cache,
        "token": model_args.token,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError("Please provide value for model_name_or_path or config_name.")

        # Load model
    if model_args.model_name_or_path:
        model_dtype = torch.bfloat16 if training_args.bf16 else None
        model = AutoModelForVision2Seq.from_pretrained(
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

    lora_config = LoraConfig(
        r=finetune_args.lora_rank,
        lora_alpha=finetune_args.lora_alpha,
        lora_dropout=finetune_args.lora_dropout,
        target_modules=finetune_args.lora_target_modules,
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        split="train",
    )

    train_dataset = train_dataset.remove_columns(
        [
            col
            for col in train_dataset.column_names
            if col not in (data_args.input_column_names + data_args.output_column_names)
        ]
    )

    eval_dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        split="test",
    )

    eval_dataset = eval_dataset.remove_columns(
        [
            col
            for col in eval_dataset.column_names
            if col not in (data_args.input_column_names + data_args.output_column_names)
        ]
    )

    data_collator = MyDataCollator(processor)

    gaudi_config = GaudiConfig()
    gaudi_config.use_fused_adam = True
    gaudi_config.use_fused_clip_norm = True

    trainer = GaudiTrainer(
        model=model,
        args=training_args,
        gaudi_config=gaudi_config,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()

    if is_main_process(training_args.local_rank):
        example = eval_dataset[15]
        model.eval()
        model = model.merge_and_unload()
        if model_dtype == torch.bfloat16:
            model = model.to(torch.bfloat16)

        image = example["image"]
        query = example["query"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    {"type": "image"},
                    {"type": "text", "text": query["en"]},
                ],
            }
        ]
        processor.tokenizer.padding_side = "left"
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text.strip()], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to("hpu") for k, v in inputs.items()}
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_texts = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
        )
        logger.info(f"generated: {generated_texts}")
        if training_args.do_eval:
            anls = eval(
                processor=processor,
                model=model,
                eval_dataset=eval_dataset,
                eval_batch_size=training_args.per_device_eval_batch_size,
            )
            logger.info(f"anls = {anls}")
            if training_args.output_dir is not None:
                metrics = {"accuracy": anls}
                with open(f"{training_args.output_dir}/accuracy_metrics.json", mode="w") as file:
                    json.dump(metrics, file)


if __name__ == "__main__":
    main()
