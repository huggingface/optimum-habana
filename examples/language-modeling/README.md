<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Language Model Training

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset.
GPT-2 is trained or fine-tuned using a causal language modeling (CLM) loss while ALBERT, BERT, DistilBERT and RoBERTa are trained or fine-tuned using a masked language modeling (MLM) loss. You can find more information about the differences between those objectives in our [model summary](https://huggingface.co/transformers/model_summary.html).

The following examples will run on datasets hosted on our [hub](https://huggingface.co/datasets) or with your own
text files for training and validation. We give examples of both below.

## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

## GPT2/GPT-J/GPT-NeoX and causal language modeling

The following examples fine-tune GPT-2, GPT-J-6B and GPT-NeoX-20B on WikiText-2. We're using the raw WikiText-2 (no tokens were replaced before the tokenization). The loss here is the one of causal language modeling.


### Single-card Training (GPT2)

```bash
PT_HPU_LAZY_MODE=1 python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --throughput_warmup_steps 3
```

This takes about 13 minutes to train on a single HPU. It reaches
a perplexity of about 20.9963 once fine-tuned on the dataset.

To run on your own training and validation files, use the following command:

```bash
PT_HPU_LAZY_MODE=1 python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --throughput_warmup_steps 3
```


### Multi-card Training (GPT2)

```bash
PT_HPU_LAZY_MODE=1 python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gradient_checkpointing \
    --use_cache False \
    --throughput_warmup_steps 3
```

This takes about 4 minutes to train on 8 HPUs. It reaches
a perplexity of 21.7968 once fine-tuned on the dataset.


### Multi-card Training with Deepspeed (GPT-J)

The following command triggers the fine-tuning of [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b) on WikiText-2 with DeepSpeed ZeRO-2.
Fine tuning on 8 HPU cards takes around 6 minutes with a batch size of 32 (4 per device).
It reaches a perplexity of 14.011.

```bash
PT_HPU_LAZY_MODE=1 python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_clm.py \
    --model_name_or_path EleutherAI/gpt-j-6b \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm-xl-1 \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode \
    --gradient_checkpointing \
    --use_hpu_graphs_for_inference \
    --throughput_warmup_steps 3 \
    --deepspeed path_for_deepspeed_config
```

This example has been validated with the following DeepSpeed ZeRO-2 config: https://github.com/huggingface/optimum-habana/blob/main/tests/configs/deepspeed_zero_2.json


## Multi-Node Training with Deepspeed (GPT-NeoX)

The following command triggers the fine-tuning of [GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b) on WikiText-2 with Deepspeed ZeRO-2.
Fine-tuning on 16 HPU cards (2 Gaudi2 nodes) takes around 9 minutes with a batch size of 32 (2 per device).
It reaches a perplexity of 10.469.

> [!NOTE]
>  For GPT-NeoX-20B model, please switch to jemalloc in case of host OOM issues using ``` export LD_PRELOAD=<path>/libjemalloc.so.2 ```

> Please refer to [this page](https://github.com/huggingface/optimum-habana/tree/main/examples/multi-node-training) for performing multi-node training properly.

```bash
PT_HPU_LAZY_MODE=1 python ../gaudi_spawn.py \
    --hostfile path_to_my_hostfile --use_deepspeed run_clm.py \
    --model_name_or_path EleutherAI/gpt-neox-20b \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 2\
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm-xl-bs2 \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode \
    --gradient_checkpointing \
    --use_hpu_graphs_for_inference \
    --throughput_warmup_steps 3 \
    --deepspeed path_for_deepspeed_config
```

This example has been validated with the following DeepSpeed ZeRO-2 config: https://github.com/huggingface/optimum-habana/blob/main/tests/configs/deepspeed_zero_2.json


## RoBERTa/BERT/DistilBERT and masked language modeling

The following examples fine-tune RoBERTa on WikiText-2. Here too, we're using the raw WikiText-2. The loss is different as BERT/RoBERTa have a bidirectional mechanism; we're therefore using the same loss that was used during their pre-training: masked language modeling.
Following the RoBERTa paper, we use dynamic masking rather than static masking. The model may, therefore,
converge slightly slower (over-fitting takes more epochs).


### Multi-card Training

```bash
PT_HPU_LAZY_MODE=1 python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gaudi_config_name Habana/roberta-base \
    --throughput_warmup_steps 3 \
    --bf16
```

If your dataset is organized with one sample per line, you can use the `--line_by_line` flag (otherwise the script
concatenates all texts and then splits them into blocks of the same length).

**Note:** On HPU, you should use the flag `--pad_to_max_length` in conjunction with the `--line_by_line` flag to make sure all your batches have the same length.


### Training in torch.compile mode
RoBERTa-Large model training in [torch.compile](pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) mode is enabled by applying the following changes to your command,
a) Set the following environment variable `PT_ENABLE_INT64_SUPPORT=1`.
b) Run the above commands with `--model_name_or_path roberta-large`, `--use_lazy_mode False` and add `--torch_compile`, `--torch_compile_backend hpu_backend` and remove `--use_hpu_graphs_for_inference` flags.


## Pretraining

You can easily train a model from scratch by replacing `--model_name_or_path my_model_name` by `--config_name my_model_name --tokenizer_name my_model_name`.

For example with GPT2:
```bash
PT_HPU_LAZY_MODE=1 python run_clm.py \
    --config_name gpt2 \
    --tokenizer_name gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --throughput_warmup_steps 3 \
    --bf16
```

## Inference

To run only inference, you can start from the commands above and you just have to remove the training-only arguments such as `--do_train`, `--per_device_train_batch_size`, `--num_train_epochs`, etc...

For instance, you can run inference with GPT2 on the Wikitext dataset on 1 Gaudi card with the following command:
```bash
PT_HPU_LAZY_MODE=1 python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_eval_batch_size 4 \
    --do_eval \
    --output_dir /tmp/test-clm \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --bf16
```


## PEFT

### LORA/ADALORA/IA3/LLAMA_ADAPTER/VERA/LN_TUNING

To run LoRA finetuning, you can use `run_lora_clm.py`.
Here are single-/multi-device command examples for Llama1-7B, Falcon-40B, Llama2-70B, Llama3-8B and Llama3-70B.
You can also use multicard version for Falcon-180B:

- Single-card finetuning of Llama1-7B:
```bash
PT_HPU_LAZY_MODE=1 python3 run_lora_clm.py \
    --model_name_or_path huggyllama/llama-7b \
    --dataset_name tatsu-lab/alpaca \
    --bf16 True \
    --output_dir ./model_lora_llama \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-4 \
    --warmup_ratio  0.03 \
    --lr_scheduler_type "constant" \
    --max_grad_norm  0.3 \
    --logging_steps 1 \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --throughput_warmup_steps 3 \
    --lora_rank=8 \
    --lora_alpha=16 \
    --lora_dropout=0.05 \
    --lora_target_modules "q_proj" "v_proj" \
    --dataset_concatenation \
    --max_seq_length 512 \
    --low_cpu_mem_usage True \
    --validation_split_percentage 4 \
    --adam_epsilon 1e-08
```

- Multi-card finetuning of gemma2 using chat template:
```bash
PT_HPU_LAZY_MODE=1 python ../gaudi_spawn.py \
    --world_size 2 --use_mpi run_lora_clm.py \
    --model_name_or_path google/gemma-2b-it \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_train \
    --do_eval \
    --num_train_epochs 15 \
    --output_dir ./output/2b_2hpu_16bs_15ep \
    --save_total_limit 1 \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --gradient_checkpointing \
    --throughput_warmup_steps 3 \
    --use_lazy_mode \
    --pipelining_fwd_bwd \
    --bf16 \
    --logging_strategy epoch \
    --eval_strategy epoch \
    --lora_target_modules "q_proj" "o_proj" "k_proj" "v_proj" "gate_proj" "up_proj" "down_proj" \
    --lora_rank=8 \
    --lora_alpha=16 \
    --lora_dropout=0.05 \
    --dataset_name mamamiya405/finred \
    --chat_prompt True
```

- Multi-card finetuning of Falcon-40B:
```bash
PT_HPU_LAZY_MODE=1 PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST=ops_bf16.txt python3 ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_lora_clm.py \
    --model_name_or_path tiiuae/falcon-40b \
    --dataset_name timdettmers/openassistant-guanaco \
    --bf16 True \
    --output_dir ./model_lora_falcon_ddp \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 4e-4 \
    --max_grad_norm  0.3 \
    --warmup_ratio  0.03 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --do_train \
    --use_habana \
    --use_lazy_mode \
    --pipelining_fwd_bwd \
    --throughput_warmup_steps 3 \
    --lora_rank=64 \
    --lora_alpha=16 \
    --lora_dropout=0.1 \
    --lora_target_modules "query_key_value" "dense" "dense_h_to_4h" "dense_4h_to_h" \
    --dataset_concatenation \
    --max_seq_length 256 \
    --ddp_bucket_cap_mb 50 \
    --adam_epsilon 1e-08 \
    --do_eval \
    --low_cpu_mem_usage True \
    --validation_split_percentage 6
```

- Multi-card finetuning of Llama3.1-8B with Deepspeed ZeRO-1 optimization, LoRA and FP8 precision:
```bash
PT_TE_CUSTOM_OP=1 PT_HPU_LAZY_MODE=0 python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_lora_clm.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_name tatsu-lab/alpaca \
    --bf16 False \
    --output_dir ./model_lora_llama_8B \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 3e-4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant" \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode False \
    --throughput_warmup_steps 3 \
    --lora_rank=8 \
    --lora_alpha=16 \
    --lora_dropout=0.05 \
    --lora_target_modules "q_proj" "v_proj" \
    --dataset_concatenation \
    --max_seq_length 4096 \
    --adam_epsilon 1e-08 \
    --validation_split_percentage 4 \
    --deepspeed llama3_ds_zero1_config.json \
    --torch_compile_backend hpu_backend \
    --torch_compile \
    --fp8 \
    --use_flash_attention True \
    --flash_attention_causal_mask True  \
    --per_device_eval_batch_size 4  \
    --cache_size_limit 64 \
    --use_regional_compilation \
    --compile_from_sec_iteration \
    --allow_unspec_int_on_nn_module True
```

- Multi-card finetuning of Llama2-70B with DeepSpeed ZeRO-3 optimization, LoRA and FP8 precision:

  > The following command requires Habana DeepSpeed 1.13.0 or later.

```bash
PT_HPU_LAZY_MODE=1 PT_HPU_MAX_COMPOUND_OP_SIZE=10 \
python3 ../gaudi_spawn.py --use_deepspeed --world_size 8 run_lora_clm.py \
  --model_name_or_path meta-llama/Llama-2-70b-hf \
  --deepspeed llama2_ds_zero3_config.json \
  --dataset_name tatsu-lab/alpaca \
  --bf16 True \
  --output_dir ./lora_out \
  --num_train_epochs 2 \
  --max_seq_len 2048 \
  --per_device_train_batch_size 10 \
  --per_device_eval_batch_size 1 \
  --gradient_checkpointing \
  --eval_strategy epoch \
  --eval_delay 2 \
  --save_strategy no \
  --learning_rate 0.0018 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --dataset_concatenation \
  --attn_softmax_bf16 True \
  --do_train \
  --do_eval \
  --use_habana \
  --use_lazy_mode \
  --pipelining_fwd_bwd \
  --throughput_warmup_steps 3 \
  --lora_rank 4 \
  --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj" \
  --validation_split_percentage 4 \
  --use_flash_attention True \
  --flash_attention_causal_mask True \
  --fp8 True
```

- Multi-card finetuning of Llama2-70B with FSDP and LoRA:

```bash
PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST=ops_bf16.txt \
python3 ../gaudi_spawn.py --world_size 8 --use_mpi run_lora_clm.py \
  --model_name_or_path meta-llama/Llama-2-70b-hf \
  --dataset_name tatsu-lab/alpaca \
  --bf16 True \
  --output_dir ./lora_out \
  --max_seq_len 2048 \
  --gradient_checkpointing \
  --per_device_train_batch_size 5 \
  --save_strategy no \
  --learning_rate 0.0004 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "constant" \
  --logging_steps 1 \
  --dataset_concatenation \
  --do_train \
  --use_habana \
  --throughput_warmup_steps 3 \
  --lora_rank 4 \
  --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj" \
  --attn_softmax_bf16 True \
  --validation_split_percentage 4 \
  --use_lazy_mode False \
  --fsdp_config fsdp_config.json \
  --fsdp auto_wrap \
  --num_train_epochs 2 \
  --eval_strategy epoch \
  --per_device_eval_batch_size 1 \
  --eval_delay 2 \
  --do_eval \
  --pipelining_fwd_bwd False \
  --use_fused_rope False \
  --torch_compile_backend hpu_backend \
  --torch_compile \
  --gradient_accumulation_steps 2 \
  --use_flash_attention True \
  --flash_attention_causal_mask True
```

Default `peft_type` is `lora`, you could enable adalora or ia3 using `--peft_type adalora` or `--peft_type ia3`, or enable llama-adapter for llama model using `--peft_type llama-adapter`, or enable ln-tuning using `--peft_type ln_tuning`, or enable vera using `--peft_type vera`.

#### Custom Files

To run on your own training and validation files, use the following command:

```bash
PT_HPU_LAZY_MODE=1 python run_lora_clm.py \
    --model_name_or_path bigcode/starcoder \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-lora-clm \
    --bf16 \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --dataset_concatenation \
    --throughput_warmup_steps 3
```

The format of the jsonlines files (with extensions .json or .jsonl) is expected to be

```json
{"text": "<text>"}
{"text": "<text>"}
{"text": "<text>"}
{"text": "<text>"}
```

The format of the text files (with extensions .text or .txt) is expected to be

```json
"<text>"
"<text>"
"<text>"
"<text>"
```

> Note: When using both custom files i.e `--train_file` and `--validation_file`, all files are expected to be of the same type i.e json or text.

### Prompt/Prefix/P-tuning

To run prompt tuning finetuning, you can use `run_prompt_tuning_clm.py`.
Here are single-card command examples for Llama2-7B:
- single-card finetuning of meta-llama/Llama-2-7b-hf with dataset "ought/raft" and config "twitter_complaints":
```bash
PT_HPU_LAZY_MODE=1 python3 run_prompt_tuning_clm.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --output_dir prompt_tuning_out \
    --bf16 True \
    --report_to=none \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --low_cpu_mem_usage True \
    --logging_steps 1 \
    --do_train \
    --num_train_epochs 50 \
    --do_eval  \
    --use_habana  \
    --use_lazy_mode
```

Default `peft_type` is `prompt_tuning`, you could enable prefix-tuning or p-tuning using `--peft_type prefix_tuning` or `--peft_type p_tuning`.

Use the prompt finetuned model for text-generation:
```bash
PT_HPU_LAZY_MODE=1 python3 ../text-generation/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --max_new_tokens 128 \
    --bf16 \
    --use_kv_cache \
    --batch_size 1 \
    --use_hpu_graphs \
    --no-ignore_eos \
    --peft_model prompt_tuning_out \
    --prompt "@SEPTA_SOCIAL Ok. Thanks. Label :"
```

### Multitask Prompt/Poly seq2seq tuning

To run multitask prompt seq2seq finetuning, you can use `run_multitask_prompt_tuning.py`.
Here is a multi-device command example for [google/flan-t5-base](https://huggingface.co/google/flan-t5-base):
```bash
PT_HPU_LAZY_MODE=1 python3 ../gaudi_spawn.py --world_size 8 --use_mpi run_multitask_prompt_tuning.py \
    --model_name_or_path google/flan-t5-base \
    --do_train \
    --report_to=none \
    --num_train_epochs 3 \
    --output_dir out_multi_peft \
    --use_habana \
    --use_lazy_mode \
    --eval_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "no" \
    --learning_rate 1e-4  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --use_hpu_graphs_for_inference \
    --use_hpu_graphs_for_training \
    --bf16
```

To run poly seq2seq finetuning, you can use `peft_poly_seq2seq_with_generate.py`.
Here is a multi-device command example for [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl):
```bash
PT_HPU_LAZY_MODE=1 python3 ../gaudi_spawn.py --world_size 8 --use_mpi peft_poly_seq2seq_with_generate.py \
    --model_name_or_path google/flan-t5-xl \
    --do_train \
    --report_to=none \
    --num_train_epochs 1 \
    --output_dir out_poly \
    --use_habana \
    --use_lazy_mode \
    --eval_strategy "epoch" \
    --logging_strategy "epoch" \
    --save_strategy "no" \
    --learning_rate 5e-5  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --bf16 \
    --use_hpu_graphs_for_inference \
    --use_hpu_graphs_for_training \
    --trust_remote_code
```

### Training models with Long Sequence lengths
We have added support for [Deepspeed Ulysses](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md). This allows us to train large transformer models using very long sequence length inputs with limited HW resources. This feature has been tested using LLama3.1-8B & LLama3.1-70B fine-tuning with input sequence lengths of 32k on 8xGaudi3 cards. Reference command for LLama3.1-8B fine-tuning is shared below.

`--context_parallel_size` sets the number of cards single input sequences will get mapped to, e.g., setting `context_parallel_size=4` with `max_seq_len=32k` will result in each card processing input chunks of length 8k each (thereby reducing memory requirement for activations). This feature can be combined with Zero-3 to enable scaling not only to large sequence lengths but also to large size models.

> [!NOTE]
> This feature is still in beta version and may not work out of the box for all transformer model architectures and configurations.

```bash
PT_HPU_LAZY_MODE=1 python3 ../gaudi_spawn.py  \
        --world_size 8  --use_deepspeed run_lora_clm.py \
        --model_name_or_path meta-llama/Llama-3.1-8B \
        --dataset_name tatsu-lab/alpaca \
        --bf16 True \
        --output_dir /tmp/lora_out \
        --max_seq_len 32768 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --save_strategy no \
        --learning_rate 0.0004 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "constant" \
        --logging_steps 1 \
        --dataset_concatenation \
        --do_train \
        --use_habana \
        --throughput_warmup_steps 3 \
        --lora_rank 8 \
        --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj" \
        --attn_softmax_bf16 True \
        --validation_split_percentage 4 \
        --flash_attention_causal_mask True \
        --eval_strategy epoch \
        --pipelining_fwd_bwd \
        --use_lazy_mode \
        --use_flash_attention True \
        --deepspeed llama3_ds_zero1_config.json \
        --num_train_epochs 3 \
        --eval_delay 3 \
        --do_eval \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --gradient_accumulation_steps 4 \
        --flash_attention_recompute True \
        --context_parallel_size 4
```

## Streaming

To use the streaming dataset mode which can be very useful for large datasets, add `--streaming` with `--max_steps` specified in the command line. This is supported by `run_mlm.py` and `run_clm.py`.

For example:
```bash
PT_HPU_LAZY_MODE=1 python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --output_dir /tmp/test-clm \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --throughput_warmup_steps 3 \
    --streaming \
    --max_steps 1000 \
    --do_eval
```


## Creating a model on the fly

When training a model from scratch, configuration values may be overridden with the help of `--config_overrides`:

```bash
PT_HPU_LAZY_MODE=1 python run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --config_overrides="n_embd=1024,n_head=16,n_layer=48,n_positions=1024" \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --gradient_checkpointing \
    --use_cache False \
    --output_dir /tmp/test-clm \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gaudi_config_name Habana/gpt2 \
    --throughput_warmup_steps 3
```

<!-- This feature is only available in `run_clm.py` and `run_mlm.py`. -->


## Low Cpu Memory Usage

To use low cpu memory mode which can be very useful for LLM, add `--low_cpu_mem_usage` to the command line.
