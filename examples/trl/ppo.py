# copy from https://github.com/huggingface/trl/blob/v0.7.6/examples/research_projects/stack_llama/scripts/rl_training.py, enable it for Gaudi2
import json
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, pipeline
from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from optimum.habana.accelerate import GaudiAccelerator
from optimum.habana.trl import GaudiPPOConfig, GaudiPPOTrainer, adapt_PreTrainedModelWrapper_to_gaudi
from optimum.habana.utils import HabanaGenerationTime, set_seed
from optimum.habana.utils.environment import disable_kernel


tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    tokenizer_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": "the tokenizer name"}
    )
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum output length for generation"})
    input_max_length: Optional[int] = field(default=512, metadata={"help": "maximum input length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    use_habana: Optional[bool] = field(default=True, metadata={"help": "use habana for RL training"})
    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA method."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


disable_kernel("FusedRMSNorm")
adapt_PreTrainedModelWrapper_to_gaudi()
parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
reward_model_name = script_args.reward_model_name
dataset_name = "lvwerra/stack-exchange-paired"
config = GaudiPPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name_or_path,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
    use_habana=script_args.use_habana,
    pad_max_len=script_args.input_max_length + script_args.output_max_length,
    pad_max_input_len=script_args.input_max_length,
)

train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
if script_args.max_train_samples is not None:
    max_train_samples = min(len(train_dataset), script_args.max_train_samples)
    train_dataset = train_dataset.select(range(max_train_samples))
original_columns = train_dataset.column_names

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}
if config.pad_for_acceleration:
    sent_kwargs["padding"] = "max_length"
    sent_kwargs["max_length"] = script_args.input_max_length + script_args.output_max_length

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name_or_path)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    tokenizer,
    dataset_name="lvwerra/stack-exchange-paired",
    input_max_length=512,
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < input_max_length, batched=False)

    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer, input_max_length=script_args.input_max_length)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = GaudiAccelerator().local_process_index
lora_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=script_args.lora_target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    peft_config=lora_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model.config.use_fused_rope = False
model.config.use_fused_rms_norm = False
optimizer = None
model = model.to(torch.bfloat16)

if script_args.use_habana:
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
else:
    ref_model = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = GaudiPPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)
# We then build the sentiment analysis pipeline using our reward model, passing the
# model name and the sentiment analysis pipeline arguments. Let's also make sure to
# set the device to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device

reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_name,
    num_labels=1,
    low_cpu_mem_usage=True,
)

if config.use_habana:
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph

    reward_model = wrap_in_hpu_graph(reward_model)

if device.type == "hpu":
    device = "hpu"

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model,
    tokenizer=tokenizer,
    return_token_type_ids=False,
    device=device,
    model_kwargs={
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    },
)

if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id
# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
if not config.pad_for_acceleration:
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
else:
    output_length_sampler = LengthSampler(output_max_length, output_max_length + 1)
timer = HabanaGenerationTime()
timer.start()
sample = 0
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= config.total_ppo_epochs:
        break
    question_tensors = batch["input_ids"]
    sample = sample + len(question_tensors)
    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute reward score (using the sentiment analysis pipeline)
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
timer.step()

ppo_trainer.save_pretrained(script_args.output_dir)
metrics = {"train_runtime": timer.last_duration, "train_samples_per_second": sample / timer.last_duration}
with open(f"{script_args.output_dir}/all_results.json", mode="w") as file:
    json.dump(metrics, file)
