# copy from https://github.com/huggingface/trl/blob/v0.7.6/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py, enable it for Gaudi2
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers.integrations.deepspeed import (
    is_deepspeed_available,
)

from optimum.habana import GaudiConfig
from optimum.habana.trl import GaudiDPOConfig, GaudiDPOTrainer
from optimum.habana.utils import set_seed


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "the location of the SFT model name or path"},
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA method."},
    )
    num_buckets: Optional[int] = field(default=-1, metadata={"help": "whether to use bucketing for DPOTrainer"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    num_workers: Optional[int] = field(default=None, metadata={"help": "the number of workers to map the data"})


def get_stack_exchange_paired(
    data_dir: str = "data/rl",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        "lvwerra/stack-exchange-paired",
        split="train",
        cache_dir=cache_dir,
        data_dir=data_dir,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, GaudiDPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    low_cpu_mem_usage = True
    if is_deepspeed_available():
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        if is_deepspeed_zero3_enabled():
            low_cpu_mem_usage = False

    # 2. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=low_cpu_mem_usage,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.config.use_fused_rope = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=low_cpu_mem_usage,
        torch_dtype=torch.bfloat16,
    )
    model_ref.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Load the Stack-exchange paired dataset
    train_dataset = get_stack_exchange_paired(
        data_dir="data/rl",
        sanity_check=script_args.sanity_check,
        num_proc=script_args.num_workers,
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= training_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= training_args.max_length
    )

    # 4. Load evaluation dataset
    eval_dataset = get_stack_exchange_paired(
        data_dir="data/evaluation", sanity_check=True, num_proc=script_args.num_workers
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= training_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= training_args.max_length
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=script_args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    gaudi_config = GaudiConfig()
    gaudi_config.use_fused_adam = True
    gaudi_config.use_fused_clip_norm = True

    # 5. initialize the DPO trainer
    dpo_trainer = GaudiDPOTrainer(
        model,
        model_ref,
        gaudi_config=gaudi_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        num_buckets=script_args.num_buckets,
    )

    # 6. train
    train_result = dpo_trainer.train()

    # 7. save
    dpo_trainer.save_model(training_args.output_dir)

    # 8. save metric
    metrics = train_result.metrics
    dpo_trainer.log_metrics("train", metrics)
    dpo_trainer.save_metrics("train", metrics)
