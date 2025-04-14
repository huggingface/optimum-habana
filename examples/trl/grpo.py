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
from typing import List, Optional
from peft import LoraConfig
# from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


ideal_length = 50

def reward_len(completions, **kwargs):
    return [-abs(ideal_length - len(completion)) for completion in completions]


@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-0.5B-Instruct", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "the dataset name"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "whether to use peft"})
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

    use_flash_attention: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use Habana flash attention for fine-tuning."}
    )
    flash_attention_recompute: Optional[bool] = field(
        default=False, metadata={"help": "Whether to enable recompute in Habana flash attention for fine-tuning."}
    )
    flash_attention_causal_mask: Optional[bool] = field(
        default=False, metadata={"help": "Whether to enable causal mask in Habana flash attention for fine-tuning."}
    )

    # LoraConfig
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA method."},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((GaudiGRPOConfig, ScriptArguments))
    (training_args, script_args) = parser.parse_args_into_dataclasses()

    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=script_args.lora_target_modules,
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

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

    # adapt_transformers_to_gaudi()

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=low_cpu_mem_usage,
        torch_dtype=torch.bfloat16,
    )

    model.config.use_cache = False
    if not script_args.use_flash_attention and (
        script_args.flash_attention_recompute or script_args.flash_attention_recompute
    ):
        assert "Need to enable use_flash_attention"
    model.generation_config.use_flash_attention = script_args.use_flash_attention
    model.generation_config.flash_attention_recompute = script_args.flash_attention_recompute
    model.generation_config.flash_attention_causal_mask = script_args.flash_attention_causal_mask

    reward_funcs = reward_len
    if script_args.reward_model_name_or_path:
        reward_funcs = AutoModelForSequenceClassification.from_pretrained(
            script_args.reward_model_name_or_path,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    gaudi_config = GaudiConfig()
    gaudi_config.use_fused_adam = True
    gaudi_config.use_fused_clip_norm = True

    trainer = GaudiGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        gaudi_config=gaudi_config,
        peft_config=peft_config,
    )

    trainer.train()

    print("Done!")
