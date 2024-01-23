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
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.13.0
```


## Usage

In this section, we present how to benchmark a model on Habana Gaudi/Gaudi2 with this script. We also show how to use it to run generation on any dataset from the [Hugging Face Hub](https://huggingface.co/datasets).

To run generation with DeepSpeed-inference, you must launch the script as follows:

```bash
python ../gaudi_spawn.py --use_deepspeed --world_size number_of_devices run_generation.py ARGS
```

To run multiple DeepSpeed tasks simultaneously, you can launch them with different `master_port` and [`HABANA_VISIBLE_MODULES`](https://docs.habana.ai/en/latest/PyTorch/PT_Multiple_Tenants_on_HPU/Multiple_Dockers_each_with_Single_Workload.html#running-distributed-workload-inside-the-docker-container), for example:

```bash
# the following tasks could run simultaneously in a container with 8 HPUs
HABANA_VISIBLE_MODULES="0,1" python ../gaudi_spawn.py --use_deepspeed --world_size 2 run_generation.py ARGS     # using the default master_port=29500
HABANA_VISIBLE_MODULES="2,3,4,5" python ../gaudi_spawn.py --use_deepspeed --world_size 4 --master_port 29501 run_generation.py ARGS
HABANA_VISIBLE_MODULES="6,7" python ../gaudi_spawn.py --use_deepspeed --world_size 2 --master_port 29502 run_generation.py ARGS
```

Without DeepSpeed-inference, you can run the script with:

```bash
python run_generation.py ARGS
```

The list of all possible arguments can be obtained running:
```bash
python run_generation.py --help
```


### Single and multiple prompts

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

If you want to provide several prompts as inputs, here is how to do it:
```
python run_generation.py \
--model_name_or_path gpt2 \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 100 \
--do_sample \
--batch_size 2 \
--prompt "Hello world" "How are you?"
```

> The batch size should be larger than or equal to the number of prompts. Otherwise, only the first N prompts are kept with N being equal to the batch size.


### Benchmark

The default behaviour of this script (i.e. if no dataset is specified with `--dataset_name`) is to benchmark the given model with a few pre-defined prompts or with the prompt you gave with `--prompt`.
Here are a few settings you may be interested in:
- `--max_new_tokens` to specify the number of tokens to generate
- `--max_input_tokens` to specify the max input tokens to pad and truncate input sequences
- `--batch_size` to specify the batch size
- `--bf16` to run generation in bfloat16 precision (or to be specified in your DeepSpeed configuration if using DeepSpeed)
- `--use_hpu_graphs` to use [HPU graphs](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html) to speed up generation
- `--limit_hpu_graphs` to skip HPU Graph usage for first token to save memory
- `--use_kv_cache` to use the [key/value cache](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.use_cache) to speed up generation
- `--do_sample` or `--num_beams` to generate new tokens doing sampling or beam search (greedy search is the default)
- `--prompt` to benchmark the model on one or several prompts of your choice
- `--attn_softmax_bf16` to run attention softmax layer in bfloat16 precision provided that the model (such as Llama) supports it
- `--trim_logits` to calculate logits only for the last token in the first time step provided that the model (such as Llama) supports it
- `--kv_cache_fp8` Store kv-cache in float8 when kv-cache is used
- `--fp8` Enable Quantization to fp8

For example, you can reproduce the results presented in [this blog post](https://huggingface.co/blog/habana-gaudi-2-bloom) with the following command:
```bash
python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_generation.py \
--model_name_or_path bigscience/bloom \
--batch_size 1 \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 100
```

You can also run Llama2-70B on Gaudi2 with all optimizations enabled using the following command:
```bash
python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_generation.py \
--model_name_or_path meta-llama/Llama-2-70b-hf \
--max_new_tokens 4096 \
--bf16 \
--use_hpu_graphs \
--use_kv_cache \
--batch_size 52 \
--attn_softmax_bf16 \
--limit_hpu_graphs \
--reuse_cache \
--trim_logits
```

To run Falcon inference, use the following command. Please note that the option `--skip_hash_with_views` is added to the command to disable the `hash_with_views` feature in HPU graphs, which requires SynapseAI 1.13.0 or later:
```bash
python run_generation.py \
 --model_name_or_path tiiuae/falcon-7b \
 --bf16 \
 --use_hpu_graphs \
 --use_kv_cache \
 --batch_size 1 \
 --max_new_tokens 128 \
 --do_sample \
 --skip_hash_with_views
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
--model_name_or_path meta-llama/Llama-2-7b-hf \
--use_hpu_graphs \
--use_kv_cache \
--batch_size 1 \
--bf16 \
--max_new_tokens 100 \
--prompt "Here is my prompt" \
--peft_model goliaro/llama-2-7b-lora-full
```

### Using growing bucket optimization
With `--bucket_size`, instead of padding up the kv-cache up to full size before starting, we grow the cache/input in multiples of `bucket_size`. This helps increase throughput and also reduce number of compilations if the dataset has varying prompt lengths.

> For now, it is available only for greedy and beam search generation, and cannot be used with `--reuse_cache`.

Here is an example:
```bash
python run_generation.py \
--model_name_or_path path_to_model    \
--use_hpu_graphs \
--use_kv_cache \
--bf16 \
--max_new_tokens 200 \
--batch_size=2 \
--bucket_size 50
```

`--bucket_size` option is especially useful when processing an input stream with varying lengths, that is when you have something like `--dataset_name squad --column_name context --max_input_tokens -1`. `--max_input_tokens -1` specifies no truncation of input prompt in the dataset.

Another way to simulate dynamic input is to use `--simulate_dyn_prompt`. For example `--simulate_dyn_prompt 25,35,45` will extend or crop the default prompt (or the prompt passed in using `--prompt`) to sizes 25, 35, and 45, and throughput will be measured for these 3 lengths. If `--simulate_dyn_prompt` is used, the min and max input lengths from it are computed to perform warmup as well. One final optimization that can be used in case of dynamic inputs is `--reduce_recompile`. Thus the suggested configuration to simulate dynamicity after warmup is to use all three arguments: `--simulate_dyn_prompt 25 35 45 --reduce_recompile --bucket_size 30`
### Running with FP8

Llama2-7b in FP8 is enabled. Use `--fp8` to enable quantization in fp8.
Add the `--kv_cache_fp8` argument to run the model with a KV cache allocated in fp8.
More information on enabling fp8 in SynapseAI is available here:
https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html

Here is an example:
```bash
USE_DEFAULT_QUANT_PARAM=true UPDATE_GRAPH_OUTPUT_MME=false ENABLE_CALC_DYNAMIC_RANGE=false ENABLE_EXPERIMENTAL_FLAGS=true \
python run_generation.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--use_hpu_graphs \
--use_kv_cache \
--reuse_cache \
--trim_logits \
--attn_softmax_bf16 \
--max_new_tokens 200 \
--batch_size=2 \
--kv_cache_fp8 \
--fp8
```



## Language Model Evaluation Harness

The evaluation of LLMs can be done using the `lm_eval.py` script. It utilizes the [LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness)
 framework and provides the possibility to run one of four tasks: HellaSwag, Lambada_openai, PiQA, WinoGrande.



For a more detailed description of parameters, please see the help message:
```
./run_lm_eval.py -h
```


### LM Eval Requirements

First, you should install the requirements:
```bash
pip install -r requirements_lm_eval.txt
```


### Examples

Evaluate Llama 7B on Gaudi on task PiQA, using the BF16 data type:
```
python run_lm_eval.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--use_hpu_graphs \
--use_kv_cache \
--bf16 \
--batch_size=1 \
--tasks piqa \
-o eval.json
```

Evaluate Llama 70B on 8 Gaudi2 cards on task WinoGrande, using the BF16 data type:
```
deepspeed --num_gpus 8 run_lm_eval.py \
--model_name_or_path meta-llama/Llama-2-70b-hf \
--use_hpu_graphs \
--use_kv_cache \
--bf16 \
--batch_size=1 \
--tasks winogrande \
-o eval.json
```

## Text-Generation Pipeline

A Transformers-like pipeline is defined and provided [here](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation/text-generation-pipeline). It is optimized for Gaudi and can be called to generate text in your scripts.
