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

# Text to Video Examples

This directory contains a script that showcases how to use the `GaudiTextToVideoSDPipeline` to run text-to-video generation tasks on HPUs.

## Requirements

First, you should install the requirements:

```bash
pip install -r requirements.txt
```

## Single-HPU inference

```bash
python3 text_to_video_generation.py \
    --model_name_or_path ali-vilab/text-to-video-ms-1.7b \
    --prompts "An astronaut riding a horse" \
    --use_habana \
    --use_hpu_graphs \
    --dtype bf16
```

Models that have been validated:
  - [ali-vilab/text-to-video-ms-1.7b](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b)
