import logging

import torch
#from unsloth import FastModel #pip install unsloth --no-deps this only supports nvidia gpu and intel xpu
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
            {"role": "user", "content": example["problem"]},# problem for others, question for gsm
        ],
    }

ideal_length = 50

def reward_len(completions, **kwargs):
    return [-abs(ideal_length - len(completion)) for completion in completions] #penalize response when len!=50

"""
###mini r-1
def format_reward(completions, target, **kwargs):
    
    #Format: <think>...</think><answer>...</answer>
    #Args:
    #    completions (list[str]): Generated outputs
    #    target (list[str]): Expected answers
      
    #  Returns:
    #      list[float]: Reward scores

    rewards = []
 
    for completion, gt in zip(completions, target):
 
      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion        
        # Check if the format is correct
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
 
        match = re.search(regex, completion, re.DOTALL) 
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
      except Exception:
        rewards.append(0.0)
    return rewards
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
    #Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags.
    #pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
"""

"""
####Mini r-1
def accuracy_reward(completions, target, nums, **kwargs):
    #Evaluates completions based on:
    #2. Mathematical correctness of the answer
 
    #Args:
    #    completions (list[str]): Generated outputs
    #    target (list[str]): Expected answers
    #    nums (list[str]): Available numbers
    
    #Returns:
    #    list[float]: Reward scores

    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            rewards.append(0.0)
            continue
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        
        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(numbers):
            rewards.append(0.0)
            continue
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
           rewards.append(0.0)
           continue
        
        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(gt)) < 1e-5:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
      except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0) 
    return rewards
"""
###AI-MO/NuminaMath-TIR
def accuracy_reward(completions, **kwargs):
    #Reward function that checks if the completion is the same as the ground truth.
    solutions = kwargs["solution"] #for others, answer for gsm8k ["answer"]#
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards
"""
###openr1-math
def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    #Reward function that checks if the completion is the same as the ground truth.
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
    #Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90


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
"""

@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-0.5B-Instruct", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "the dataset name"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "whether to use peft"})
    #num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
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
            target_modules=script_args.lora_target_modules,#"all-linear",#
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    #dataset = load_dataset( ####open-r1
    train_dataset, test_dataset = load_dataset( ####ai-o1
        script_args.dataset_name, #split='train',#name=script_args.dataset_config,#'default',#'main', #
        data_dir=None if script_args.subset == "None" else script_args.subset,
        #num_proc=script_args.num_workers if not script_args.streaming else None,
        split=["train[:10%]", "test[:10%]"] ###disabled for openr1-math
        #split=["train", "test"]
    )
    #dataset = dataset.shuffle(seed=42).select(range(50000)) #for minir1
    
    #dataset = dataset.map(make_conversation) #for openr1
    """
    def generate_r1_prompt(numbers, target):
        r1_prefix = [{
                "role": "system",
                "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
            },
            { 
                "role": "user",
                "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
            }]
        return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target}


    dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    """

    #for split in dataset:
    #    if "messages" in dataset[split].column_names:
    #        dataset[split] = dataset[split].remove_columns("messages")
    
    train_dataset = train_dataset.map(make_conversation)
    test_dataset = test_dataset.map(make_conversation)
    train_dataset = train_dataset.remove_columns(["messages", "problem"])
    
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
        
        if is_deepspeed_zero3_enabled():
            low_cpu_mem_usage = False

    #adapt_transformers_to_gaudi()

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=low_cpu_mem_usage,
        torch_dtype=torch.bfloat16,
    )
    """
    model = FastModel.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=low_cpu_mem_usage,
        torch_dtype=torch.bfloat16,
    )
    import pdb;pdb.set_trace()
    """

    model.config.use_cache = False
    if not script_args.use_flash_attention and (
        script_args.flash_attention_recompute or script_args.flash_attention_recompute
    ):
        assert "Need to enable use_flash_attention"
    model.generation_config.use_flash_attention = script_args.use_flash_attention
    model.generation_config.flash_attention_recompute = script_args.flash_attention_recompute
    model.generation_config.flash_attention_causal_mask = script_args.flash_attention_causal_mask

    #reward_funcs = [format_reward, accuracy_reward, tag_count_reward]#for openr1
    reward_funcs = [format_reward, accuracy_reward]
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
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        #train_dataset=dataset[script_args.dataset_train_split],
        #eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        gaudi_config=gaudi_config,
        peft_config=peft_config,
    )

    trainer.train()

    print("Done!")
