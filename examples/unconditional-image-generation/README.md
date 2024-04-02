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

# Unconditional Image Generation Example

This directory contains a python script which demonstrates how to perform unconditional-image-generation on Gaudi/HPU.

Original unconditional image generation pipeline is shared in here: [Unconditional Image Generation](https://huggingface.co/docs/diffusers/using-diffusers/unconditional_image_generation)


### Single Card Inference

You can launch an inference task on one HPU card using the following script 

```bash
python3 run_generation.py \
    --model_name_or_path "google/ddpm-ema-celebahq-256" \
    --batch_size 16 \
    --use_habana \
    --use_gaudi_optimized_scheduler \
    --use_hpu_graphs \
    --bf16 \
    --save_outputs \
    --output_dir "/tmp/" 
```
