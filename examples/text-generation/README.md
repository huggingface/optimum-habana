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

# Language generation

Conditional text generation with [BLOOM](https://huggingface.co/bigscience/bloom) and [BLOOMZ](https://huggingface.co/bigscience/bloomz) on Habana Gaudi. You can find more information about it in [this blog post](https://huggingface.co/blog/habana-gaudi-2-bloom).


## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

Then, since this example relies on [DeepSpeed-inference](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Inference_Using_DeepSpeed.html), you should install DeepSpeed as follows:
```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.9.0
```


## Usage

In this section, we present how to benchmark BLOOM and BLOOMZ on Habana Gaudi with this script. We also show how to use it to run generation on any dataset from the [Hugging Face Hub](https://huggingface.co/datasets).

> The present script is currently limited to greedy generation.


### Benchmark

To run a benchmark and get the throughput of your model, you can run:
```
python ../gaudi_spawn.py --use_deepspeed --world_size number_of_devices run_generation.py \
--model_name_or_path path_to_model \
--max_new_tokens number_of_tokens_to_generate \
--batch_size batch_size \
--n_iterations number_of_iterations \
--use_hpu_graphs \
--use_kv_cache
```
with
- `number_of_devices` the number of HPUs you want to use
- `path_to_model` a model name on the Hugging Face Hub or a path to a model saved locally
- `number_of_tokens_to_generate` the number of tokens to generate for each prompt
- `batch_size` the size of the batches provided to the model
- `number_of_iterations` the number of iterations to perform in the benchmark
- `use_hpu_graphs` enables HPU graphs which are recommended for faster latencies
- `use_kv_cache` enables a key-value cache to speed up the generation process.

For example, you can reproduce the results presented in [this blog post](https://huggingface.co/blog/habana-gaudi-2-bloom) with the following command:
```
python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_generation.py \
--model_name_or_path bigscience/bloom \
--batch_size 1 \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 100
```


### Use any dataset from the Hugging Face Hub

You can also provide the name of a dataset from the Hugging Face Hub to perform generation on it with the argument `--dataset_name`.

By default, the first column in the dataset of type `string` will be used as prompts. You can also select the column you want with the argument `--column_name`.

Here is an example with [JulesBelveze/tldr_news](https://huggingface.co/datasets/JulesBelveze/tldr_news):
```
python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_generation.py \
--model_name_or_path bigscience/bloom \
--batch_size 2 \
--max_new_tokens 100 \
--use_hpu_graphs \
--use_kv_cache \
--dataset_name JulesBelveze/tldr_news \
--column_name content
```

> The prompt length is limited to 16 tokens. Prompts longer than this will be truncated.
