# copy from https://github.com/huggingface/trl/blob/v0.7.6/examples/research_projects/stack_llama/scripts/rl_training.py, enable it for Gaudi2
import json
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    pipeline
)
from trl.core import LengthSampler
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from optimum.habana import GaudiConfig
# from optimum.habana.accelerate import GaudiAccelerator
from optimum.habana.trl import GaudiPPOConfig, GaudiPPOTrainer, adapt_PreTrainedModelWrapper_to_gaudi
from optimum.habana.utils import set_seed


tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/rl", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    num_buckets: Optional[int] = field(default=-1, metadata={"help": "whether to use bucketing for SFTTrainer"})
    tokenizer_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": "the tokenizer name"}
    )
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum output length for generation"})
    input_max_length: Optional[int] = field(default=512, metadata={"help": "maximum input length for generation"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
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


adapt_PreTrainedModelWrapper_to_gaudi()
parser = HfArgumentParser((ScriptArguments, GaudiPPOConfig))
script_args, training_args = parser.parse_args_into_dataclasses()

dataset = load_dataset(
    script_args.dataset_name,
    data_dir=None if script_args.subset == "None" else script_args.subset,
    split=script_args.split,
    num_proc=script_args.num_workers,
)
if script_args.max_train_samples is not None:
    max_train_samples = min(len(dataset), script_args.max_train_samples)
    dataset = dataset.select(range(max_train_samples))

eval_samples = 100
train_dataset = dataset.select(range(len(dataset) - eval_samples))
eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
dataset_text_field = "question"

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}
# if training_args.pad_for_acceleration:
sent_kwargs["padding"] = "max_length"
sent_kwargs["max_length"] = script_args.input_max_length + script_args.output_max_length

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, padding_side="left")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
# def build_dataset(
#     tokenizer,
#     dataset_name="lvwerra/stack-exchange-paired",
#     input_max_length=512,
# ):
#     """
#     Build dataset for training. This builds the dataset from `load_dataset`, one should
#     customize this function to train the model on its own dataset.

#     Args:
#         dataset_name (`str`):
#             The name of the dataset to be loaded.

#     Returns:
#         dataloader (`torch.utils.data.DataLoader`):
#             The dataloader for the dataset.
#     """

#     num_proc = 24

#     def preprocess_function(examples):
#         new_examples = {
#             "query": [],
#             "input_ids": [],
#         }
#         for question in examples["question"]:
#             query = "Question: " + question + "\n\nAnswer: "
#             tokenized_question = tokenizer(query, truncation=True)
#             new_examples["query"].append(query)
#             new_examples["input_ids"].append(tokenized_question["input_ids"])

#         return new_examples

#     ds = train_dataset.map(
#         preprocess_function,
#         batched=True,
#         num_proc=num_proc,
#         remove_columns=original_columns,
#     )
#     ds = ds.filter(lambda x: len(x["input_ids"]) < input_max_length, batched=False)

#     ds.set_format(type="torch")
#     return ds

def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=training_args.dataset_num_proc,
    )


# Compute that only on the main process for faster data processing.
# see: https://github.com/huggingface/trl/pull/1255
with PartialState().local_main_process_first():
    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)

# set seed before initializing value head for deterministic eval
set_seed(training_args.seed)

# Now let's build the model, the reference model, and the tokenizer.
# current_device = GaudiAccelerator().local_process_index
peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=script_args.lora_target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model.config.use_fused_rope = False
model.config.use_fused_rms_norm = False
optimizer = None
model = model.to(torch.bfloat16)

value_model = AutoModelForSequenceClassification.from_pretrained(
    training_args.reward_model_path,
    num_labels=1,
    low_cpu_mem_usage=True,
)

reward_model = AutoModelForSequenceClassification.from_pretrained(
    script_args.reward_model_name,
    num_labels=1,
    low_cpu_mem_usage=True,
)

ref_model = AutoModelForCausalLM.from_pretrained(
    training_args.sft_model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

optimizer = None
if training_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=training_args.learning_rate,
    )

gaudi_config = GaudiConfig()
# gaudi_config.use_fused_adam = True
# gaudi_config.use_fused_clip_norm = True

trainer = GaudiPPOTrainer(
    args=training_args,
    processing_class=tokenizer,
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    value_model=value_model,
    gaudi_config=gaudi_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(optimizer, None),
    peft_config=peft_config,
    num_buckets=script_args.num_buckets,
)
# We then build the sentiment analysis pipeline using our reward model, passing the
# model name and the sentiment analysis pipeline arguments. Let's also make sure to
# set the device to the same device as the PPOTrainer.
# device = ppo_trainer.accelerator.device

# # if training_args.use_habana:
# #     from habana_frameworks.torch.hpu import wrap_in_hpu_graph

# #     reward_model = wrap_in_hpu_graph(reward_model)

# if device.type == "hpu":
#     device = "hpu"

# sentiment_pipe = pipeline(
#     "sentiment-analysis",
#     model=reward_model,
#     tokenizer=tokenizer,
#     return_token_type_ids=False,
#     device=device,
#     model_kwargs={
#         "low_cpu_mem_usage": True,
#         "torch_dtype": torch.bfloat16,
#     },
# )

# if sentiment_pipe.model.training_args.pad_token_id is None:
#     sentiment_pipe.model.training_args.pad_token_id = tokenizer.pad_token_id
# # We then define the arguments to pass to the `generate` function. These arguments
# # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# # the `generate` function of the trained model.
# generation_kwargs = {
#     # "min_length": -1,
#     "top_k": 0.0,
#     "top_p": 1.0,
#     "do_sample": True,
#     "pad_token_id": tokenizer.pad_token_id,
#     "eos_token_id": 100_000,
# }
# output_min_length = 32
# output_max_length = script_args.output_max_length
# # if not training_args.pad_for_acceleration:
# #     output_length_sampler = LengthSampler(output_min_length, output_max_length)
# # else:
# output_length_sampler = LengthSampler(output_max_length, output_max_length + 1)
# timer = HabanaGenerationTime()
# timer.start()
# sample = 0
# for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
#     if epoch >= training_args.total_ppo_epochs:
#         break
#     question_tensors = batch["input_ids"]
#     sample = sample + len(question_tensors)
#     response_tensors = ppo_trainer.generate(
#         question_tensors,
#         return_prompt=False,
#         length_sampler=output_length_sampler,
#         **generation_kwargs,
#     )
#     batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

#     # Compute reward score (using the sentiment analysis pipeline)
#     texts = [q + r for q, r in zip(batch["query"], batch["response"])]
#     pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
#     rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]

#     # Run PPO step
#     stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
#     ppo_trainer.log_stats(stats, batch, rewards)

#     if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
#         ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
# timer.step()

trainer.train()

trainer.save_model(training_args.output_dir)
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# metrics = {"train_runtime": timer.last_duration, "train_samples_per_second": sample / timer.last_duration}
# with open(f"{script_args.output_dir}/all_results.json", mode="w") as file:
#     json.dump(metrics, file)
