import torch

from datasets import load_dataset
from optimum.habana.trl import GaudiGRPOTrainer, GaudiGRPOConfig
from optimum.habana import GaudiConfig, GaudiTrainer
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import ScriptArguments
from transformers.integrations.deepspeed import (
    is_deepspeed_available,
)
from dataclasses import dataclass, field
from typing import Optional


# MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
MODEL_NAME = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
REWARD_MODEL_NAME = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"

# dataset_name = "philschmid/dolly-15k-oai-style"
DATASET_NAME = "trl-lib/tldr"


@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-0.5B-Instruct", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "the dataset name"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    subset: Optional[str] = field(default=None, metadata={"help": "the subset to use"})
    streaming: Optional[bool] = field(default=False, metadata={"help": "whether to stream the dataset"})
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})
    reward_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Reward model id of a pretrained model hosted inside a model repo on huggingface.co or "
            "local path to a directory containing model weights saved using `PreTrainedModel.save_pretrained`."
        },
    )


if __name__ == "__main__":
    parser = HfArgumentParser((GaudiGRPOConfig, ScriptArguments))
    (training_args, script_args) = parser.parse_args_into_dataclasses()

    dataset = load_dataset(
        script_args.dataset_name,
        data_dir=None if script_args.subset == "None" else script_args.subset,
        num_proc=script_args.num_workers if not script_args.streaming else None,
    )

    low_cpu_mem_usage = True
    if is_deepspeed_available():
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        if is_deepspeed_zero3_enabled():
            low_cpu_mem_usage = False

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=low_cpu_mem_usage,
        torch_dtype=torch.bfloat16,
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.reward_model_name_or_path,
        trust_remote_code=True,
        num_labels=1
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    gaudi_config = GaudiConfig()

    gaudi_config.use_fused_adam = True
    gaudi_config.use_fused_clip_norm = True

    trainer = GaudiGRPOTrainer(
        model=model,
        reward_funcs=reward_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        gaudi_config=gaudi_config,
    )

    trainer.train()

    print("Done!")
