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
python run_clm.py \
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
python run_clm.py \
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
python ../gaudi_spawn.py \
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
python ../gaudi_spawn.py \
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

> Please refer to [this page](https://github.com/huggingface/optimum-habana/tree/main/examples/multi-node-training) for performing multi-node training properly.

```bash
python ../gaudi_spawn.py \
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


### Single-card Training

```bash
python run_mlm.py \
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

To run on your own training and validation files, use the following command:

```bash
python run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
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


### Multi-card Training

```bash
python ../gaudi_spawn.py \
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


### Training in torch.compile mode
RoBERTa-Large model training in [torch.compile](pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) mode is enabled by applying the following changes to your command,
a) Set the following environment variables `PT_HPU_LAZY_MODE=0` and `PT_ENABLE_INT64_SUPPORT=1`.
b) Run the above commands with `--model_name_or_path roberta-large`, `--use_lazy_mode False` and add `--torch_compile`, `--torch_compile_backend hpu_backend` and remove `--use_hpu_graphs_for_inference` flags.


## Pretraining

You can easily train a model from scratch by replacing `--model_name_or_path my_model_name` by `--config_name my_model_name --tokenizer_name my_model_name`.

For example with GPT2:
```bash
python run_clm.py \
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


## Using DeepSpeed

Multi-card examples can be simply adapted to be run with DeepSpeed. Here is the CLM example with GPT2-XL:

```bash
python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_clm.py \
    --model_name_or_path gpt2-xl \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --learning_rate 4e-4 \
    --output_dir /tmp/test-clm \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gradient_checkpointing \
    --use_cache False \
    --throughput_warmup_steps 3 \
    --deepspeed path_to_my_deepspeed_config
```

You can look at the [documentation](https://huggingface.co/docs/optimum/habana/usage_guides/deepspeed) for more information about how to use DeepSpeed in Optimum Habana.
Here is a DeepSpeed configuration you can use to train your models on Gaudi:
```json
{
    "steps_per_print": 64,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "bf16": {
        "enabled": true
    },
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": false,
        "reduce_scatter": false,
        "contiguous_gradients": false
    }
}
```

Here is another example with Bloom-7B1:

```bash
DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1 PT_HPU_MAX_COMPOUND_OP_SYNC=1 PT_HPU_MAX_COMPOUND_OP_SIZE=1 python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_clm.py \
    --model_name_or_path bigscience/bloom-7b1 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --do_train \
    --output_dir /tmp/test-clm \
    --gaudi_config_name Habana/roberta-base \
    --use_habana \
    --use_lazy_mode \
    --gradient_checkpointing \
    --use_cache False \
    --throughput_warmup_steps 3 \
    --save_strategy "no" \
    --learning_rate 1e-04 \
    --deepspeed path_to_my_deepspeed_config
```
[This](https://github.com/huggingface/optimum-habana/blob/main/tests/configs/deepspeed_zero_3_gaudi1.json) is a DeepSpeed configuration you can use to train this model on Gaudi1.


## Inference

To run only inference, you can start from the commands above and you just have to remove the training-only arguments such as `--do_train`, `--per_device_train_batch_size`, `--num_train_epochs`, etc...

For instance, you can run inference with GPT2 on the Wikitext dataset on 1 Gaudi card with the following command:
```bash
python run_clm.py \
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

### LORA/ADALORA/IA3/LLAMA_ADAPTER

To run LoRA finetuning, you can use `run_lora_clm.py`.
Here are single-/multi-device command examples for Llama1-7B, Falcon-40B, Llama2-70B, Llama3-8B and Llama3-70B.
You can also use multicard version for Falcon-180B:

- Single-card finetuning of Llama1-7B:
```bash
python3 run_lora_clm.py \
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
- Single-card finetuning of Falcon-40B:
```bash
LOWER_LIST=ops_bf16.txt python3 run_lora_clm.py \
    --model_name_or_path tiiuae/falcon-40b \
    --dataset_name timdettmers/openassistant-guanaco \
    --bf16 True \
    --output_dir ./model_lora_falcon \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 3e-4 \
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
    --low_cpu_mem_usage True \
    --adam_epsilon 1e-08 \
    --do_eval \
    --validation_split_percentage 5
```

- Multi-card finetuning of Llama1-7B:
```bash
python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_lora_clm.py \
    --model_name_or_path huggyllama/llama-7b \
    --dataset_name tatsu-lab/alpaca \
    --bf16 True \
    --output_dir ./model_lora_llama_ddp \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 3e-4 \
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
    --ddp_bucket_cap_mb 50 \
    --adam_epsilon 1e-08 \
    --validation_split_percentage 4 \
    --low_cpu_mem_usage True
```

- Multi-card finetuning of Llama2-7B with FP8:
```bash
LOWER_LIST=ops_bf16.txt python ../gaudi_spawn.py \
	--world_size 8 --use_mpi run_lora_clm.py \
	--model_name_or_path meta-llama/Llama-2-7b-hf \
	--dataset_name tatsu-lab/alpaca \
	--bf16 True \
	--output_dir ./model_lora_llama \
	--num_train_epochs 3 \
	--per_device_train_batch_size 16 \
	--gradient_accumulation_steps 1 \
	--eval_strategy "no" \
	--save_strategy "no" \
	--learning_rate 3e-4 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "constant" \
	--max_grad_norm 0.3 \
	--logging_steps 20 \
	--do_train \
	--do_eval \
	--use_habana \
	--use_lazy_mode \
	--throughput_warmup_steps 18 \
	--lora_rank=8 \
	--lora_alpha=16 \
	--lora_dropout=0.05 \
	--lora_target_modules "q_proj" "v_proj" \
	--dataset_concatenation \
	--max_seq_length 512 \
	--ddp_bucket_cap_mb 50 \
	--adam_epsilon 1e-08 \
	--validation_split_percentage 10 \
	--low_cpu_mem_usage True \
	--pipelining_fwd_bwd \
	--fp8 True
```

- Multi-card finetuning of codegen-16B-mono:
```bash
python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_lora_clm.py \
    --model_name_or_path Salesforce/codegen-16B-mono \
    --dataset_name b-mc2/sql-create-context \
    --sql_prompt \
    --bf16 True \
    --output_dir ./finetuned-models/codegen-finetune-on-sql-create-context-hpu8-lora8-bs4 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-4 \
    --logging_steps 1 \
    --dataset_concatenation \
    --do_train \
    --use_habana \
    --use_lazy_mode \
    --throughput_warmup_steps 3 \
    --use_hpu_graphs_for_inference \
    --lora_target_modules "qkv_proj" \
    --lora_rank 8 \
    --do_eval \
    --validation_split_percentage 10 \
    --use_cache False
```

- Multi-card finetuning of Falcon-40B:
```bash
LOWER_LIST=ops_bf16.txt python3 ../gaudi_spawn.py \
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

- Multi-card finetuning of Llama2-70B with DeepSpeed ZeRO-3 optimization, LoRA and FP8 precision:

  > The following command requires Habana DeepSpeed 1.13.0 or later.

```bash
PT_HPU_MAX_COMPOUND_OP_SIZE=10 \
python3 ../gaudi_spawn.py --use_deepspeed  --world_size 8  run_lora_clm.py \
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
LOWER_LIST=ops_bf16.txt PT_HPU_LAZY_MODE=0 \
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

- Multi-card finetuning of Falcon-180B:
  - Falcon-180B example command saves only the LoRA parameters at end
  - For inference we need to merge the pretrained model and LoRA weights
```bash
DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1 LOWER_LIST=ops_bf16.txt python3 ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_lora_clm.py \
    --model_name_or_path tiiuae/falcon-180B \
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
    --adam_epsilon 1e-08 \
    --do_eval \
    --validation_split_percentage 5 \
    --deepspeed ds_falcon_180b_z3.json
```
Default `peft_type` is `lora`, you could enable adalora or ia3 using `--peft_type adalora` or `--peft_type ia3`, or enable llama-adapter for llama model using `--peft_type llama-adapter`.

#### Custom Files

To run on your own training and validation files, use the following command:

```bash
python run_lora_clm.py \
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
Here are single-/multi-device command examples for Llama2-7B:
- single-card finetuning of meta-llama/Llama-2-7b-hf with dataset "ought/raft" and config "twitter_complaints":
```bash
python3 run_prompt_tuning_clm.py \
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

- multi-card finetuning of meta-llama/Llama-2-7b-hf with dataset "ought/raft" and config "twitter_complaints":
```bash
python3 ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_prompt_tuning_clm.py \
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
python3 ../text-generation/run_generation.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --max_new_tokens 128 \
    --bf16 \
    --use_kv_cache \
    --batch_size 1 \
    --use_hpu_graphs \
    --ignore_eos \
    --peft_model prompt_tuning_out \
    --prompt "@SEPTA_SOCIAL Ok. Thanks. Label :"

```
### Multitask Prompt/Poly seq2seq tuning

To run multitask prompt seq2seq finetuning, you can use `run_multitask_prompt_tuning.py`.
Here is a multi-device command example for [google/flan-t5-base](https://huggingface.co/google/flan-t5-base):
```bash
python3 ../gaudi_spawn.py --world_size 8 --use_mpi run_multitask_prompt_tuning.py \
    --model_name_or_path google/flan-t5-base \
    --do_train \
    --report_to=none \
    --num_train_epochs 3 \
    --output_dir out_multi_peft \
    --use_habana \
    --use_lazy_mode \
    --evaluation_strategy "steps" \
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
python3 ../gaudi_spawn.py --world_size 8 --use_mpi peft_poly_seq2seq_with_generate.py \
    --model_name_or_path google/flan-t5-xl \
    --do_train \
    --report_to=none \
    --num_train_epochs 1 \
    --output_dir out_poly \
    --use_habana \
    --use_lazy_mode \
    --evaluation_strategy "epoch" \
    --logging_strategy "epoch" \
    --save_strategy "no" \
    --learning_rate 5e-5  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --bf16 \
    --use_hpu_graphs_for_inference \
    --use_hpu_graphs_for_training
```


## Streaming

To use the streaming dataset mode which can be very useful for large datasets, add `--streaming` with `--max_steps` specified in the command line. This is supported by `run_mlm.py` and `run_clm.py`.

For example:
```bash
python run_clm.py \
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
python run_clm.py \
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
