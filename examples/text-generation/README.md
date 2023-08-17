<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

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

Conditional text generation on Habana Gaudi/Gaudi2. You can find more information about it in [this blog post](https://huggingface.co/blog/habana-gaudi-2-bloom).


## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

Then, if you plan to use [DeepSpeed-inference](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Inference_Using_DeepSpeed.html) (e.g. to use BLOOM/BLOOMZ), you should install DeepSpeed as follows:
```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.10.0
```


## Usage

In this section, we present how to benchmark a model on Habana Gaudi/Gaudi2 with this script. We also show how to use it to run generation on any dataset from the [Hugging Face Hub](https://huggingface.co/datasets).

To run generation with DeepSpeed-inference, you must launch the script as follows:

```bash
python ../gaudi_spawn.py --use_deepspeed --world_size number_of_devices run_generation.py ARGS
```

Without DeepSpeed-inference, you can run the script with:

```bash
python run_generation.py ARGS
```

The list of all possible arguments can be obtained running:
```bash
python run_generation.py --help
```


### Single prompt

If you want to generate a sequence of text from a prompt of your choice, you should use the `--prompt` argument.
For example:
```
python run_generation.py \
--model_name_or_path gpt2 \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 100 \
--do_sample \
--prompt "Here is my prompt"
```


### Benchmark

The default behaviour of this script (i.e. if no dataset is specified with `--dataset_name`) is to benchmark the given model with a few pre-defined prompts or with the prompt you gave with `--prompt`.
Here are a few settings you may be interested in:
- `--max_new_tokens` to specify the number of tokens to generate
- `--batch_size` to specify the batch size
- `--bf16` to run generation in bfloat16 precision (or to be specified in your DeepSpeed configuration if using DeepSpeed)
- `--use_hpu_graphs` to use [HPU graphs](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html) to speed up generation
- `--use_kv_cache` to use the [key/value cache](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.use_cache) to speed up generation
- `--do_sample` or `--num_beams` to generate new tokens doing sampling or beam search (greedy search is the default)
- `--prompt` to benchmark the model on a prompt of your choice

For example, you can reproduce the results presented in [this blog post](https://huggingface.co/blog/habana-gaudi-2-bloom) with the following command:
```bash
python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_generation.py \
--model_name_or_path bigscience/bloom \
--batch_size 1 \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 100
```

> To be able to run gated models like [StarCoder](https://huggingface.co/bigcode/starcoder), you should:
> - have a HF account
> - agree to the terms of use of the model in its model card on the HF Hub
> - set a read token as explained [here](https://huggingface.co/docs/hub/security-tokens)
> - login to your account using the HF CLI: run `huggingface-cli login` before launching your script
>
> And then you can run it as any other model:
> ```
> python run_generation.py \
> --model_name_or_path bigcode/starcoder \
> --batch_size 1 \
> --use_hpu_graphs \
> --use_kv_cache \
> --max_new_tokens 100 \
> --bf16
> ```


### Use any dataset from the Hugging Face Hub

You can also provide the name of a dataset from the Hugging Face Hub to perform generation on it with the argument `--dataset_name`.

By default, the first column in the dataset of type `string` will be used as prompts. You can also select the column you want with the argument `--column_name`.

Here is an example with [JulesBelveze/tldr_news](https://huggingface.co/datasets/JulesBelveze/tldr_news):
```bash
python run_generation.py \
--model_name_or_path gpt2 \
--batch_size 2 \
--max_new_tokens 100 \
--use_hpu_graphs \
--use_kv_cache \
--dataset_name JulesBelveze/tldr_news \
--column_name content \
--bf16
```

> The prompt length is limited to 16 tokens. Prompts longer than this will be truncated.


### Use PEFT models for generation

You can also provide the path to a PEFT model to perform generation with the argument `--peft_model`.

For example:
```bash
python run_generation.py \
--model_name_or_path huggyllama/llama-7b \
--use_hpu_graphs \
--use_kv_cache \
--batch_size 1 \
--bf16 \
--max_new_tokens 100 \
--prompt "Here is my prompt" \
--peft_model trl-lib/llama-7b-se-rm-peft
```
