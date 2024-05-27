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

# Object Segmentation Examples

This directory contains an example script that demonstrates how to perform object segmentation on Gaudi with graph mode.

## Single-HPU inference

```bash
python3 run_example.py \
    --model_name_or_path "CIDAS/clipseg-rd64-refined" \
    --image_path "http://images.cocodataset.org/val2017/000000039769.jpg" \
    --prompt "cat, remote, blanket" \
    --warmup 3 \
    --n_iterations 20 \
    --use_hpu_graphs \
    --bf16 \
    --print_result
```
Models that have been validated:
  - [clipseg-rd64-refined ](https://huggingface.co/CIDAS/clipseg-rd64-refined)