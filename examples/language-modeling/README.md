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
    --per_device_train_batch_size 4 \
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
    --hostfile path_to_my_hostfile --world_size 16 --use_deepspeed run_clm.py \
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
    --throughput_warmup_steps 3
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
    --throughput_warmup_steps 3
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
    --throughput_warmup_steps 3
```


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
    --throughput_warmup_steps 3
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
    --use_hpu_graphs_for_inference
```


## PEFT

To run LoRA finetuning and inference. you could use `run_lora_clm.py` as an example. Multi-card examples can be simply adapted to run LoRA finetuning. Here is the CLM example with Llama:

Single-card finetuning:
```bash
python3 run_lora_clm.py \
    --model_name_or_path huggyllama/llama-7b \
    --dataset_name tatsu-lab/alpaca \
    --bf16 True \
    --output_dir ./model_lora_llama \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --logging_steps 1 \
    --dataset_concatenation \
    --do_train \
    --use_habana \
    --use_lazy_mode \
    --throughput_warmup_steps 3
```

Multi-card finetuning:
```bash
python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_lora_clm.py \
    --model_name_or_path huggyllama/llama-7b \
    --dataset_name tatsu-lab/alpaca \
    --bf16 True \
    --output_dir ./model_lora_llama \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --logging_steps 1 \
    --dataset_concatenation \
    --do_train \
    --use_habana \
    --use_lazy_mode \
    --throughput_warmup_steps 3
```


## Streaming

To use the streaming dataset mode which can be very useful for large datasets, add `--streaming` with `--max_steps` specified in the command line. This is currently supported by `run_mlm.py` and `run_clm.py`.

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
