<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

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

# Stable Diffusion Examples

This directory contains a script that showcases how to perform text-to-image generation using Stable Diffusion on Habana Gaudi.

Stable Diffusion was proposed in [Stable Diffusion Announcement](https://stability.ai/blog/stable-diffusion-announcement) by Patrick Esser and Robin Rombach and the Stability AI team.


## Text-to-image Generation

### Single Prompt

Here is how to generate images with one prompt:
```python
python text_to_image_generation.py \
    --model_name_or_path runwayml/stable-diffusion-v1-5 \
    --prompts "An image of a squirrel in Picasso style" \
    --num_images_per_prompt 20 \
    --batch_size 4 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --bf16
```

> HPU graphs are recommended when generating images by batches to get the fastest possible generations.
> The first batch of images entails a performance penalty. All subsequent batches will be generated much faster.
> You can enable this mode with `--use_hpu_graphs`.


### Multiple Prompts

Here is how to generate images with several prompts:
```python
python text_to_image_generation.py \
    --model_name_or_path runwayml/stable-diffusion-v1-5 \
    --prompts "An image of a squirrel in Picasso style" "A shiny flying horse taking off" \
    --num_images_per_prompt 20 \
    --batch_size 8 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --bf16
```

> HPU graphs are recommended when generating images by batches to get the fastest possible generations.
> The first batch of images entails a performance penalty. All subsequent batches will be generated much faster.
> You can enable this mode with `--use_hpu_graphs`.


### Stable Diffusion 2

[Stable Diffusion 2](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion_2) can also be used to generate images with this script. Here is an example for a single prompt:

```python
python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-2-1 \
    --prompts "An image of a squirrel in Picasso style" \
    --num_images_per_prompt 10 \
    --batch_size 2 \
    --height 768 \
    --width 768 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion-2
```

> There are two different checkpoints for Stable Diffusion 2:
> - use [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) for generating 768x768 images
> - use [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) for generating 512x512 images


### Latent Diffusion Model for 3D (LDM3D)

[LDM3D](https://arxiv.org/abs/2305.10853) generates both image and depth map data from a given text prompt, allowing users to generate RGBD images from text prompts. 

[Original checkpoint](https://huggingface.co/Intel/ldm3d) and [latest checkpoint](https://huggingface.co/Intel/ldm3d-4c) are open source.


```python
python text_to_image_generation.py \
    --ldm3d_model_name_or_path "Intel/ldm3d-4c" \
    --prompts "An image of a squirrel in Picasso style" \
    --num_images_per_prompt 10 \
    --batch_size 2 \
    --height 768 \
    --width 768 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion-2
    --ldm3d
```

> There are three different checkpoints for LDM3D:
> - use [original checkpoint](https://huggingface.co/Intel/ldm3d) to generate outputs from the paper
> - use [the latest checkpoint](https://huggingface.co/Intel/ldm3d-4c) for generating improved results
> - use [the pano checkpoint](https://huggingface.co/Intel/ldm3d-pano) to generate panoramic view

