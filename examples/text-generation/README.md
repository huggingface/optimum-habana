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

Conditional text generation on Intel® Gaudi® AI Accelerators. You can find more information about it in [this blog post](https://huggingface.co/blog/habana-gaudi-2-bloom).


## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

Then, if you plan to use [DeepSpeed-inference](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Inference_Using_DeepSpeed.html) (e.g. to use BLOOM/BLOOMZ), you should install DeepSpeed as follows:
```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.17.0
```


## Usage

In this section, we present how to benchmark a model on Intel Gaudi AI Accelerators with this script. We also show how to use it to run generation on any dataset from the [Hugging Face Hub](https://huggingface.co/datasets).

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

### Run Speculative Sampling on Gaudi

If you want to generate a sequence of text from a prompt of your choice using assisted decoding, you can use the following command as an example:

```
python run_generation.py \
--model_name_or_path gpt2 \
--assistant_model distilgpt2 \
--batch_size 1 \
--max_new_tokens 100 \
--use_hpu_graphs \
--use_kv_cache \
--num_return_sequences 1 \
--temperature 0 \
--prompt "Alice and Bob"
```

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
- `--top_k` and `--penalty_alpha` to generate new tokens doing contrastive search (greedy search is the default)
- `--prompt` to benchmark the model on one or several prompts of your choice
- `--attn_softmax_bf16` to run attention softmax layer in bfloat16 precision provided that the model (such as Llama) supports it
- `--trim_logits` to calculate logits only for the last token in the first time step provided that the model (such as Llama) supports it

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
--batch_size 180 \
--attn_softmax_bf16 \
--limit_hpu_graphs \
--reuse_cache \
--trim_logits
```

To run Falcon-7B inference, use the following command:
```bash
python run_generation.py \
 --model_name_or_path tiiuae/falcon-7b \
 --bf16 \
 --use_hpu_graphs \
 --use_kv_cache \
 --batch_size 1 \
 --max_new_tokens 128 \
 --do_sample
```

To run Falcon-40B inference on 8 Gaudi2 cards, use the following command:
```bash
python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_generation.py \
--model_name_or_path tiiuae/falcon-40b \
--max_new_tokens 2048 \
--bf16 \
--use_hpu_graphs \
--use_kv_cache \
--batch_size 1 \
--do_sample \
--use_flash_attention \
--flash_attention_causal_mask
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

Another way to simulate dynamic input is to use `--simulate_dyn_prompt`. For example `--simulate_dyn_prompt 25 35 45` will extend or crop the default prompt (or the prompt passed in using `--prompt`) to sizes 25, 35, and 45, and throughput will be measured for these 3 lengths. If `--simulate_dyn_prompt` is used, the min and max input lengths from it are computed to perform warmup as well. One final optimization that can be used in case of dynamic inputs is `--reduce_recompile`. Thus the suggested configuration to simulate dynamicity after warmup is to use all three arguments: `--simulate_dyn_prompt 25 35 45 --reduce_recompile --bucket_size 30`

While `--bucket_size` works for any model without model file changes, an even more optimized version of bucketing is supported for certain models like Llama. This can be enabled by setting `--bucket_internal` flag (along with `--bucket_size` to specify the bucket size)


### Running with torch.compile

> [!NOTE]
> For `GPTBigCodeForCausalLM` architecture models, such as [ibm-granite/granite-20b-code-instruct](https://huggingface.co/ibm-granite/granite-20b-code-instruct), performance may have degradation with `--use_flash_attention`. Please remove it from the command line.

torch.compile is an experimental feature. It has not been validated for all models. To enable torch.compile, please
set the following environment variables before running the command: `PT_ENABLE_INT64_SUPPORT=1` and `PT_HPU_LAZY_MODE=0`.

You will also need to add `--torch_compile` in your command.

### Running with tensor-parallel strategy

> [!NOTE]
> This strategy includes code from the [foundation-model-stack](https://github.com/foundation-model-stack/foundation-model-stack) repository, which is licensed under the Apache License 2.0. See the `LICENSE` file for more details.

> [!WARNING]
> torch.compile with tensor parallel strategy is an experimental feature. It has not been validated for all models.

To enable torch.compile with tensor parallel strategy, please set the following environment variables before running the
command: `PT_ENABLE_INT64_SUPPORT=1` and `PT_HPU_LAZY_MODE=0`. This will enable tensor parallel strategy without deepspeed.

You will also need to add `--torch_compile` and `--parallel_strategy="tp"` in your command.

Here is an example:
```bash
PT_ENABLE_INT64_SUPPORT=1 PT_HPU_LAZY_MODE=0 python ../gaudi_spawn.py  --world_size 8 run_generation.py \
--model_name_or_path meta-llama/Llama-2-70b-hf  \
--trim_logits \
--use_kv_cache \
--attn_softmax_bf16 \
--bf16 \
--bucket_internal  \
--bucket_size=128  \
--use_flash_attention \
--flash_attention_recompute \
--batch_size 246 \
--max_input_tokens 2048 \
--max_new_tokens 2048 \
--torch_compile \
--parallel_strategy="tp"
```

### Running with FP8

Llama2-70b, Llama2-7b, Llama3-70b, Llama3-8b, Mixtral-8x7B, Falcon-7B, Falcon-40B, Falcon-180B and phi-2 in FP8 are enabled using the [Intel Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html), which provides model measurement and quantization capabilities in PyTorch. From synapse 1.17 / optimum-habana 1.13 release, INC is used by default for measuring and quantization. Habana Quantization Toolkit (HQT), which was used earlier, will be removed in future releases. To use HQT, disable INC by setting the following environment variable: `USE_INC=0`.

More information on enabling fp8 in SynapseAI is available here:
https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html

Here is an example to measure the tensor quantization statistics on LLama2-70b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python ../gaudi_spawn.py \
--use_deepspeed --world_size 8 run_lm_eval.py \
-o acc_70b_bs1_measure.txt \
--model_name_or_path meta-llama/Llama-2-70b-hf \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--bucket_size=128 \
--bucket_internal \
--use_flash_attention \
--flash_attention_recompute \
--bf16 \
--batch_size 1
```

Here is an example to quantize the model based on previous measurements for LLama2-70b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant.json python ../gaudi_spawn.py \
--use_deepspeed --world_size 8 run_lm_eval.py \
-o acc_70b_bs1_quant.txt \
--model_name_or_path meta-llama/Llama-2-70b-hf \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--bucket_size=128 \
--bucket_internal \
--use_flash_attention \
--flash_attention_recompute \
--bf16 \
--batch_size 1
```

Alternatively, here is another example to quantize the model based on previous measurements for LLama2-70b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant.json python ../gaudi_spawn.py \
--use_deepspeed --world_size 8 run_generation.py \
--model_name_or_path meta-llama/Llama-2-70b-hf \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--reuse_cache \
--use_flash_attention \
--flash_attention_recompute \
--bf16 \
--batch_size 350 \
--max_new_tokens 2048 \
--max_input_tokens 2048 \
--limit_hpu_graphs
```

Here is an example to measure the tensor quantization statistics on Mixtral-8x7B with 1 card:
```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python run_generation.py \
--model_name_or_path mistralai/Mixtral-8x7B-v0.1 \
--use_hpu_graphs \
--use_kv_cache \
--limit_hpu_graphs \
--bucket_size 128 \
--max_new_tokens 128 \
--batch_size 1 \
--bf16
```

Here is an example to quantize the model based on previous measurements for Mixtral-8x7B with 1 card:
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant_mixtral.json python run_generation.py \
--model_name_or_path mistralai/Mixtral-8x7B-v0.1 \
--use_hpu_graphs \
--use_kv_cache \
--limit_hpu_graphs \
--bucket_size 128 \
--max_new_tokens 2048 \
--batch_size 16 \
--bf16
```

Here is an example to measure the tensor quantization statistics on Falcon-180B with 8 cards:
> Please note that Falcon-180B is a gated model, and users are required to request access to it. Please refer to the instructions provided in the StarCoder example above.
```bash
QUANT_CONFIG=./quantization_config/maxabs_measure_include_outputs.json python ../gaudi_spawn.py \
--use_deepspeed --world_size 8 run_lm_eval.py \
-o acc_falcon180b_bs1_quant.txt \
--model_name_or_path tiiuae/falcon-180B \
--use_hpu_graphs \
--use_kv_cache \
--trim_logits \
--batch_size 1 \
--bf16 \
--reuse_cache \
--use_flash_attention \
--flash_attention_recompute \
--flash_attention_causal_mask
```

Here is an example to quantize the model based on previous measurements for Falcon-180B with 8 cards:
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant.json python ../gaudi_spawn.py \
--use_deepspeed --world_size 8 run_generation.py \
--model_name_or_path tiiuae/falcon-180B \
--use_hpu_graphs \
--use_kv_cache \
--limit_hpu_graphs \
--max_input_tokens 128 \
--max_new_tokens 2048 \
--batch_size 110 \
--bf16 \
--reuse_cache \
--trim_logits \
--use_flash_attention \
--flash_attention_recompute \
--flash_attention_causal_mask
```

Here is an example to measure the tensor quantization statistics on phi-2 with 1 card:

```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python run_lm_eval.py \
-o acc_phi-2_bs1_measure.txt  \
--model_name_or_path microsoft/phi-2 \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 100 \
--batch_size 1 \
--trim_logits \
--reuse_cache \
--bf16
```

Here is an example to quantize the model based on previous measurements for phi-2 with 1 card:
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant_phi.json python run_generation.py \
--model_name_or_path microsoft/phi-2 \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 100 \
--batch_size 1 \
--bf16 \
--trim_logits \
--reuse_cache
```


### Running FP8 models on single device

Some bf16 models don't fit on one card due to hpu memory limitation, but in fp8 precision they do fit.
As measurement is being calculated in bf16 precision, to be able to run fp8 model on single card you should use `unify_measurements` script.
Here are the steps:
1. Measure the model on a number of cards that are enough for the model to fit in BF16.
2. Quantize the model on the same amount of cards for scales to be saved.
3. Run unify_measurements.py script using the measurement files created after running steps 1 and 2. A unified measurement is then calculated.
```bash
python quantization_tools/unify_measurements.py -g 01234567 -m *path_to_8x_measurements* -o *path_to_output_1x_measurement*
```
In the above example, the measurements of cards 0-7 will be unified to a single measurement. For example, if you specify `-g 0123 4567`,
cards 0-3 and cards 4-7 will be unified in two different measurement files. All different group combinations are supported.
4. Run quantization using the unified measurement file/s.

More information on usage of the unifier script can be found in fp8 Habana docs: https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html



### CPU memory reduction on single card

Some models can fit on HPU DRAM but can't fit on the CPU RAM.
When we run a model on single card and don't use deepspeed, the `--disk_offload` flag allows to offload weights to disk during model quantization in INC. When this flag is mentioned, during the quantization process, each weight first is loaded from disk to CPU RAM, when brought to HPU DRAM and quantized there. This way not all the model is on the CPU RAM but only one weight each time.
To enable this weights offload mechanism, add `--disk_offload` flag to the topology command line.
Here is an example of using disk_offload in quantize command.
Please follow the "Running FP8 models on single device" section first before running the cmd below.

```bash
QUANT_CONFIG=./quantization_config/maxabs_quant.json TQDM_DISABLE=1 \
python run_generation.py \
--model_name_or_path meta-llama/Llama-2-70b-hf \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--limit_hpu_graphs \
--bucket_size=128 \
--bucket_internal \
--max_new_tokens 2048 \
--max_input_tokens 2048 \
--bf16 \
--batch_size 1 \
--disk_offload \
--use_flash_attention \
--flash_attention_recompute
```


### Loading 4 Bit Checkpoints from Hugging Face

You can load pre-quantized 4bit models with the argument `--load_quantized_model_with_inc`.
Currently, uint4 checkpoints and single device are supported.
More information on enabling 4 bit inference in SynapseAI is available here:
https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_UINT4.html.

Below is an example to load a model with 4bit checkpoints from Hugging Face.
Please note that model name is denoted as `<model_path_in_hugging_face>`.
Additionally, the below env vars are used for performance optimizations, and are planned to be removed in future version:
`SRAM_SLICER_SHARED_MME_INPUT_EXPANSION_ENABLED=false ENABLE_EXPERIMENTAL_FLAGS=1`
```bash
SRAM_SLICER_SHARED_MME_INPUT_EXPANSION_ENABLED=false ENABLE_EXPERIMENTAL_FLAGS=1 \
python run_lm_eval.py \
-o acc_load_uint4_model.txt \
--model_name_or_path <model_path_in_hugging_face> \
--use_hpu_graphs \
--use_kv_cache \
--trim_logits \
--batch_size 1 \
--bf16 \
--attn_softmax_bf16 \
--bucket_size=128 \
--bucket_internal \
--load_quantized_model_with_inc
```

### Loading 4 Bit Checkpoints from Neural Compressor (INC)

You can load a pre-quantized 4-bit checkpoint with the argument `--quantized_inc_model_path`, supplied with the original model with the argument `--model_name_or_path`.
Currently, only uint4 checkpoints and single-device configurations are supported.
**Note:** In this process, you can load a checkpoint that has been quantized using INC.
More information on enabling 4-bit inference in SynapseAI is available here:
https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_INT4.html.

Below is an example of loading a llama7b model with a 4bit checkpoint quantized in INC.
Please note that the model checkpoint name is denoted as `<local_model_path_from_inc>`.
Additionally, the following environment variables are used for performance optimizations and are planned to be removed in future versions:
`SRAM_SLICER_SHARED_MME_INPUT_EXPANSION_ENABLED=false ENABLE_EXPERIMENTAL_FLAGS=1`
```bash
SRAM_SLICER_SHARED_MME_INPUT_EXPANSION_ENABLED=false ENABLE_EXPERIMENTAL_FLAGS=1 \
python run_lm_eval.py \
-o acc_load_uint4_model.txt \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--use_hpu_graphs \
--use_kv_cache \
--trim_logits \
--batch_size 1 \
--bf16 \
--attn_softmax_bf16 \
--bucket_size=128 \
--bucket_internal \
--quantized_inc_model_path <local_model_path_from_inc> \
```

### Using Habana Flash Attention

Habana Flash Attention addresses large sequence lengths on prompt stage of inference. Using causal attention mask on prompt stage requires input sequences in batch to be of the same length, but can provide a memory saving, thus enabling higher batch sizes.

Below example uses `flash_attention_recompute` mode in order to reduce memory consumption on prompt stage. Additionally since all sequences in a batch are of the same length it uses `flash_attention_causal_mask` which will further improve performance by taking advantage of specific lower-diagonal shape of inputs to softmax operation.

```bash
python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_generation.py \
--model_name_or_path meta-llama/Llama-2-70b-hf \
--use_hpu_graphs \
--limit_hpu_graphs \
--use_kv_cache \
--bf16 \
--trim_logits \
--attn_softmax_bf16 \
--bucket_size=128 \
--bucket_internal \
--batch_size 10 \
--max_input_tokens 40960 \
--max_new_tokens 5120 \
--use_flash_attention \
--flash_attention_recompute \
--flash_attention_causal_mask \
--book_source
```

For more details see [documentation](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_PyTorch_Models.html#using-fused-sdpa).

### Running with UINT4 weight quantization using AutoGPTQ


Llama2-7b in UINT4 weight only quantization is enabled using [AutoGPTQ Fork](https://github.com/HabanaAI/AutoGPTQ), which provides quantization capabilities in PyTorch.
Currently, the support is for UINT4 inference of pre-quantized models only.

You can run a *UINT4 weight quantized* model using AutoGPTQ by setting the following environment variables:
`SRAM_SLICER_SHARED_MME_INPUT_EXPANSION_ENABLED=false ENABLE_EXPERIMENTAL_FLAGS=true` before running the command,
and by adding the argument `--load_quantized_model_with_autogptq`.

***Note:***
Setting the above environment variables improves performance. These variables will be removed in future releases.
 

Here is an example to run a quantized model <quantized_gptq_model>:
```bash
SRAM_SLICER_SHARED_MME_INPUT_EXPANSION_ENABLED=false \
ENABLE_EXPERIMENTAL_FLAGS=true python run_generation.py \
--attn_softmax_bf16 \
--model_name_or_path <quantized_gptq_model> \
--use_hpu_graphs \
--limit_hpu_graphs \
--use_kv_cache \
--bucket_size 128 \
--bucket_internal \
--trim_logits \
--max_new_tokens 128 \
--batch_size 1 \
--bf16 \
--load_quantized_model_with_autogptq
```

## Language Model Evaluation Harness

The evaluation of LLMs can be done using the `lm_eval.py` script. It utilizes the [LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness)
 framework and provides the possibility to run one of four tasks: HellaSwag, Lambada_openai, PiQA, WinoGrande.

For a more detailed description of parameters, please see the help message:
```
python run_lm_eval.py --help
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
