<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

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

# Image to Text Examples

This directory contains a script that showcases how to use the Transformers pipeline API to run image to text task on HPUs.

## Single-HPU inference

```bash
python3 run_pipeline.py \
    --model_name_or_path Salesforce/blip-image-captioning-large \
    --image_path "https://ankur3107.github.io/assets/images/image-captioning-example.png" \
    --use_hpu_graphs \
    --bf16
```
Models that have been validated:
  - [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)
  - [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large)
  - [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)