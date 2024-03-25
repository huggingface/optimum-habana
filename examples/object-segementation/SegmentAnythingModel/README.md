<!---
Copyright 2024 The HuggingFace Team. All rights reserved.
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

# Segment Anything Model Examples

This directory contains an example script that demonstrates using SAM with graph mode.

## Single-HPU inference

```bash
python3 run_example.py \
    --model_name_or_path "facebook/sam-vit-huge" \
    --image_path "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png" \
    --point_prompt "450,600" \
    --warmup 3 \
    --n_iterations 20 \
    --use_hpu_graphs \
    --bf16 \
    --print_result
```
Models that have been validated:
  - [facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base)
  - [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge)