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

Based on the script [`run_generation.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py).

Conditional text generation using the auto-regressive models of the library: GPT2 and BLOOM.
A similar script is used for our official demo [Write With Transfomer](https://transformer.huggingface.co), where you can try out the different models available in the library.

First, you should install the requirements:
```bash
pip install git+https://github.com/huggingface/transformers.git
pip install -r requirements.txt
```

Example usage on 1 HPU:

```bash
python run_generation.py \
    --model_name_or_path=bigscience/bloom-1b3 \
    --batch_size 4 \
    --length 64 \
    --input_size 64 \
    --n_iterations 20 \
    --use_hpu_graphs
```

Example usage on 8 HPUs:

```bash
python ../gaudi_spawn.py --world_size 8 --use_mpi \
    run_generation.py \
    --model_name_or_path=bigscience/bloom-1b3 \
    --batch_size 4 \
    --length 64 \
    --input_size 64 \
    --n_iterations 80 \
    --use_hpu_graphs
```

Example usage on 8 HPUs with DeepSpeed:

```bash
python ../gaudi_spawn.py --world_size 8 --use_deepspeed \
    run_generation.py \
    --model_name_or_path=bigscience/bloom-7b1 \
    --batch_size 4 \
    --length 64 \
    --input_size 64 \
    --n_iterations 80 \
    --use_hpu_graphs
```
