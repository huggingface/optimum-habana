<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

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

# ControlNet Examples

This directory contains a script that showcases how to perform text-to-image generation using ControlNet on Habana Gaudi.

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models ](https://huggingface.co/papers/2302.05543) by Lvmin Zhang and Maneesh Agrawala.


## Text-to-image Generation with Canny Control

```python
python text_to_image_generation_canny.py \
    --model_name_or_path runwayml/stable-diffusion-v1-5 \
    --controlnet_model_name_or_path lllyasviel/sd-controlnet-canny \
    --prompts "futuristic-looking woman" \
    --num_images_per_prompt 20 \
    --batch_size 4 \
    --image_save_dir /tmp/controlnet_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --bf16
```

