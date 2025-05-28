import logging

import torch
import transformers
from datasets import load_dataset
from optimum.habana.trl import GaudiGRPOTrainer, GaudiGRPOConfig
from optimum.habana import GaudiConfig, GaudiTrainer
from optimum.habana.utils import set_seed
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import ScriptArguments
from trl.data_utils import maybe_apply_chat_template
from transformers.trainer_utils import is_main_process
from transformers.integrations.deepspeed import (
    is_deepspeed_available,
)
from dataclasses import dataclass, field
from typing import List, Optional
from peft import LoraConfig
import re
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
#from trl.data_utils import apply_chat_template

#from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
logger = logging.getLogger(__name__)
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},#question"]},#
        ],
    }

ideal_length = 50

def reward_len(completions, **kwargs):
    return [-abs(ideal_length - len(completion)) for completion in completions] #penalize response when len!=50


"""
###AI-MO/NuminaMath-TIR
def format_reward(completions, **kwargs):
    #Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags.
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]
"""
###openr1-math
def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


"""
###AI-MO/NuminaMath-TIR
def accuracy_reward(completions, **kwargs):
    #Reward function that checks if the completion is the same as the ground truth.
    solutions = kwargs["solution"]#["answer"]#
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        try:
            gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                try:
                    rewards.append(float(verify(answer_parsed, gold_parsed)))
                except ValueError as ve: # Catch the specific SymPy error
                    print(f"  [VERIFY ERROR - ValueError] For content='{content}', solution='{solution}': {ve}")
                    rewards.append(0.0) # Keep current behavior of scoring 0
                except Exception as e_verify: # Catch other potential errors from verify
                    print(f"  [VERIFY ERROR - Other] For content='{content}', solution='{solution}': {e_verify}")
                    rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception as e_outer: # Catch errors from parsing or other steps
            print(f"  [OUTER ERROR] For content='{content}', solution='{solution}': {e_outer}")
            rewards.append(0.0)
    return rewards
"""
###openr1-math
def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards

def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


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
        default=True, metadata={"help": "Whether to use Habana flash attention for fine-tuning."}
    )
    flash_attention_recompute: Optional[bool] = field(
        default=False, metadata={"help": "Whether to enable recompute in Habana flash attention for fine-tuning."}
    )
    flash_attention_causal_mask: Optional[bool] = field(
        default=False, metadata={"help": "Whether to enable causal mask in Habana flash attention for fine-tuning."}
    )

    # LoraConfig
    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA method."},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((GaudiGRPOConfig, ScriptArguments))
    (training_args, script_args) = parser.parse_args_into_dataclasses()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")
    # Set seed before initializing model.
    set_seed(training_args.seed)

    use_deepspeed = training_args.world_size > 1

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

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    #train_dataset, test_dataset = load_dataset(
    dataset = load_dataset(
        script_args.dataset_name, #name=script_args.dataset_config,#'default',#'main',#
        data_dir=None if script_args.subset == "None" else script_args.subset,
        #num_proc=script_args.num_workers if not script_args.streaming else None,
        #split=["train[:5%]", "test[:5%]"] ###disabled for openr1-math
    )
    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    #train_dataset = train_dataset.map(make_conversation)
    #test_dataset = test_dataset.map(make_conversation)
    #train_dataset = train_dataset.remove_columns(["messages", "problem"])
    
    """
    ###apply template for gsm8k and deepseek-r1-base
    ###only question was reformatted 'answer' has to be processed later
    dataset = dataset.map(
        lambda x: { 
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": x["question"]},
                ],
            }
    )
    dataset = dataset.map(lambda x: apply_chat_template(x, tokenizer))
    """

    low_cpu_mem_usage = True
    if is_deepspeed_available() and use_deepspeed:
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
        import deepspeed
        logger.info("DeepSpeed is enabled.")
        deepspeed.init_distributed(dist_backend="hccl")
        
        if is_deepspeed_zero3_enabled():
            low_cpu_mem_usage = False

    #adapt_transformers_to_gaudi()

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

    reward_funcs = [format_reward, accuracy_reward, tag_count_reward]#reward_len
    if script_args.reward_model_name_or_path:
        reward_funcs = AutoModelForSequenceClassification.from_pretrained(
            script_args.reward_model_name_or_path,
            trust_remote_code=True,
        )

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
