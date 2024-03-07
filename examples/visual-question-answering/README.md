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

# Visual Question Answering Examples

This directory contains a script that showcases how to use the Transformers pipeline API to run visual question answering task on HPUs.

## Single-HPU inference

```bash
python3 run_pipeline.py \
    --model_name_or_path Salesforce/blip-vqa-capfilt-large \
    --image_path "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg" \
    --question "how many dogs are in the picture?" \
    --use_hpu_graphs \
    --bf16
```

Models that have been validated:
  - [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base)
  - [dandelin/vilt-b32-finetuned-vqa](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
  - [Salesforce/blip-vqa-capfilt-large](https://huggingface.co/Salesforce/blip-vqa-capfilt-large)