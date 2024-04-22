# Fine-Tune Llama2-7b on SE paired dataset
# copy from https://github.com/huggingface/trl/blob/v0.7.6/examples/research_projects/stack_llama_2/scripts/sft_llama2.py, enable it for Gaudi2
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers.integrations.deepspeed import (
    is_deepspeed_available,
)
from trl.trainer import ConstantLengthDataset

from optimum.habana import GaudiConfig, GaudiTrainingArguments
from optimum.habana.trl import GaudiSFTTrainer
from optimum.habana.utils import set_seed


logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})

    # LoraConfig
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA method."},
    )


parser = HfArgumentParser((ScriptArguments, GaudiTrainingArguments))
script_args, training_args = parser.parse_args_into_dataclasses()
peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=script_args.lora_target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)

if training_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")

set_seed(training_args.seed)


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text


def create_datasets(tokenizer, args, seed=None):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=seed)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


low_cpu_mem_usage = True
if is_deepspeed_available():
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

    if is_deepspeed_zero3_enabled():
        low_cpu_mem_usage = False

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    low_cpu_mem_usage=low_cpu_mem_usage,
    torch_dtype=torch.bfloat16,
    use_auth_token=True,
)
base_model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

train_dataset, eval_dataset = create_datasets(tokenizer, script_args, seed=training_args.seed)

gaudi_config = GaudiConfig()
gaudi_config.use_fused_adam = True
gaudi_config.use_fused_clip_norm = True

trainer = GaudiSFTTrainer(
    model=base_model,
    gaudi_config=gaudi_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=script_args.packing,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
)
train_result = trainer.train()
trainer.save_model(training_args.output_dir)
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
