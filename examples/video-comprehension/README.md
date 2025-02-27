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

#  Examples

This directory contains example scripts that demonstrate how to perform video comprehension on Gaudi with graph mode.

## Single-HPU inference

<<<<<<< HEAD:examples/video-comprehension/README.md
### Video-LLaVA Model
=======
```bash
PT_HPU_LAZY_MODE=1 python3 text_to_video_generation.py \
    --model_name_or_path ali-vilab/text-to-video-ms-1.7b \
    --prompts "An astronaut riding a horse" \
    --use_habana \
    --use_hpu_graphs \
    --dtype bf16
```
>>>>>>> b56bafaf ([SW-218526] Updated Readme files for explicite lazy mode part2 (#177)):examples/text-to-video/README.md

```bash
python3 run_example.py \
    --model_name_or_path "LanguageBind/Video-LLaVA-7B-hf" \
    --warmup 3 \
    --n_iterations 5 \
    --batch_size 1 \
    --use_hpu_graphs \
    --bf16 \
    --output_dir ./
```
Models that have been validated:
  - [LanguageBind/Video-LLaVA-7B-hf ](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf)
