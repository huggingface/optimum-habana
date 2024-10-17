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

# Text to Speech Examples

This directory contains a script that showcases how to use the Transformers pipeline API to run text to speech task on HPUs.

## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

## Single-HPU inference

```bash
python3 run_pipeline.py \
    --model_name_or_path microsoft/speecht5_tts \
    --text "Hello, my dog is cooler than you!" \
    --use_hpu_graphs \
    --bf16
```
Models that have been validated:
  - [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts)
  - [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng)
