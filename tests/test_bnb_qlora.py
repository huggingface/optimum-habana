# coding=utf-8
# Copyright 2022 the HuggingFace Inc. team.
#
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

import subprocess

import pytest
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling

from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
from optimum.habana.transformers import modeling_utils

from .utils import OH_DEVICE_CONTEXT


modeling_utils.adapt_transformers_to_gaudi()


MODEL_ID = "meta-llama/Llama-3.2-1B"


def print_model_size(model):
    """
    Prints the model size in GB.
    """
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_GB = model_size / (1024**3)
    print(f" Model size : {model_size_GB} GB")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def get_data(tokenizer, dataset_name):
    dataset = load_dataset(dataset_name)
    dataset = dataset.shuffle(seed=42)
    data = dataset.map(lambda example: tokenizer(example["text"]), batched=True)
    split_data = data["train"].train_test_split(test_size=0.1, seed=42)

    return split_data


def get_model(token: str):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=nf4_config, device_map={"": "hpu"}, torch_dtype=torch.bfloat16, token=token.value
    )

    return model


@pytest.mark.skipif("gaudi1" == OH_DEVICE_CONTEXT, reason="execution not supported on gaudi1")
def test_nf4_quantization_inference(token: str, baseline):
    try:
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "peft==0.12.0"])
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except subprocess.CalledProcessError:
        pytest.fail("Failed to install peft==0.12.0")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token.value)
    # needed for llama tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    model = get_model(token)
    model.gradient_checkpointing_enable()
    print_model_size(model)

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=4,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    data = get_data(tokenizer, dataset_name="tatsu-lab/alpaca")

    gaudi_config = GaudiConfig(
        use_fused_adam=True,
        use_fused_clip_norm=True,
        use_torch_autocast=True,
    )

    training_args = GaudiTrainingArguments(
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        max_steps=5,
        eval_steps=3,
        warmup_steps=3,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="results",
        lr_scheduler_type="linear",
        use_habana=True,
        use_lazy_mode=True,
        pipelining_fwd_bwd=True,
    )

    trainer = GaudiTrainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        gaudi_config=gaudi_config,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    trainer.train()

    baseline.assertRef(
        compare=lambda actual, ref: abs(actual - ref) < 5e2,
        context=[OH_DEVICE_CONTEXT],
        eval_loss=trainer.evaluate()["eval_loss"],
    )
