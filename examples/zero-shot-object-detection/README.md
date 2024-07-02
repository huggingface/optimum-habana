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

# Zero Shot Object Detection Example

This folder contains an example script which demonstrates the usage of OWL-ViT to run zero shot object detection task on Gaudi platform.

## Single-HPU inference

```bash
python3 run_example.py \
    --model_name_or_path google/owlvit-base-patch32 \
    --image_path "http://images.cocodataset.org/val2017/000000039769.jpg" \
    --prompt "a photo of a cat, a photo of a dog" \
    --use_hpu_graphs \
    --bf16
```

Model that have been validated:
  - [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32)