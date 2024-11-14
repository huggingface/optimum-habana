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

# Text-Generation Pipeline

The text-generation pipeline can be used to perform text-generation by providing single or muliple prompts as input.

## Requirements

If you plan to use [DeepSpeed-inference](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Inference_Using_DeepSpeed.html), you should install DeepSpeed as follows:
```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.18.0
```

If you would like to use the pipeline with LangChain classes, you can install LangChain as follows:
```bash
pip install langchain==0.2.5
pip install langchain-huggingface
```

## Usage

To run generation with DeepSpeed-inference, you must launch the script as follows:

```bash
python ../../gaudi_spawn.py --use_deepspeed --world_size number_of_devices run_pipeline.py ARGS
```

Without DeepSpeed-inference, you can run the script with:

```bash
python run_pipeline.py ARGS
```

The list of all possible arguments can be obtained running:
```bash
python run_pipeline.py --help
```


### Single and multiple prompts

If you want to generate a sequence of text from a prompt of your choice, you should use the `--prompt` argument.
For example:
```
python run_pipeline.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 100 \
--do_sample \
--prompt "Here is my prompt"
```

If you want to provide several prompts as inputs, here is how to do it:
```
python run_pipeline.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 100 \
--do_sample \
--batch_size 2 \
--prompt "Hello world" "How are you?"
```

If you want to perform generation on default prompts, do not pass the `--prompt` argument.
```
python run_pipeline.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 100 \
--do_sample
```

If you want to change the temperature and top_p values, make sure to include the `--do_sample` argument. Here is a sample command.
```
python run_pipeline.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 100 \
--do_sample \
--temperature 0.5 \
--top_p 0.95 \
--batch_size 2 \
--prompt "Hello world" "How are you?"
```

### Multi-card runs

To run a large model such as Llama-2-70b via DeepSpeed, run the following command.
```
python ../../gaudi_spawn.py --use_deepspeed --world_size 8 run_pipeline.py \
--model_name_or_path meta-llama/Llama-2-70b-hf \
--max_new_tokens 100 \
--bf16 \
--use_hpu_graphs \
--use_kv_cache \
--batch_size 4 \
--prompt "Hello world" "How are you?" "Here is my prompt" "Once upon a time"
```

To change the temperature and top_p values, run the following command.
```
python ../../gaudi_spawn.py --use_deepspeed --world_size 8 run_pipeline.py \
--model_name_or_path meta-llama/Llama-2-70b-hf \
--max_new_tokens 100 \
--bf16 \
--use_hpu_graphs \
--use_kv_cache \
--do_sample \
--temperature 0.5 \
--top_p 0.95 \
--batch_size 4 \
--prompt "Hello world" "How are you?" "Here is my prompt" "Once upon a time"
```

### Usage with LangChain

To run a Q&A example with LangChain, use the script `run_pipeline_langchain.py`. It supports a similar syntax to `run_pipeline.py`. For example, you can use following command:
```
python run_pipeline_langchain.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --bf16 \
    --use_hpu_graphs \
    --use_kv_cache \
    --batch_size 32 \
    --max_input_tokens 200 \
    --max_new_tokens 1024 \
    --do_sample \
    --device=hpu
```

> The pipeline class has been validated for LangChain version 0.2.5 and may not work with other versions of the package.
