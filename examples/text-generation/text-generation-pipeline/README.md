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

Update `PYTHONPATH` as follows.
```bash
export OPTIMUM_HABANA_PATH=/path/to/optimum-habana
export PYTHONPATH=${PYTHONPATH}:${OPTIMUM_HABANA_PATH}/examples/text-generation
```

If you plan to use [DeepSpeed-inference](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Inference_Using_DeepSpeed.html), you should install DeepSpeed as follows:
```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.15.0
```

If you would like to use the pipeline with LangChain classes, you can install LangChain as follows:
```bash
pip install langchain==0.0.191
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
--prompt "Hello world" "How are you?" "Here is my prompt" "Once upon a time"
```

### Usage with LangChain

The text-generation pipeline can be fed as input to LangChain classes via the `use_with_langchain` constructor argument. Here is a sample snippet that shows how the pipeline class can be used with LangChain.
```python
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the pipeline
pipe = GaudiTextGenerationPipeline(args, logger, use_with_langchain=True)

# Create LangChain object
llm = HuggingFacePipeline(pipeline=pipe)

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
just say that you don't know, don't try to make up an answer.

Context: Large Language Models (LLMs) are the latest models used in NLP.
Their superior performance over smaller models has made them incredibly
useful for developers building NLP enabled applications. These models
can be accessed via Hugging Face's `transformers` library, via OpenAI
using the `openai` library, and via Cohere using the `cohere` library.

Question: {question}
Answer: """

prompt = PromptTemplate(input_variables=["question"], template=template)
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Use LangChain object
question = "Which libraries and model providers offer LLMs?"
response = llm_chain(prompt.format(question=question))
print(f"Question: {question}")
print(f"Response: {response['text']}")
```
> The pipeline class has been validated for LangChain version 0.0.191 and may not work with other versions of the package.
