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

This directory contains a script that showcases how to perform text-to-image generation using Stable Diffusion on Intel® Gaudi® AI Accelerators.

Stable Diffusion was proposed in [Stable Diffusion Announcement](https://stability.ai/blog/stable-diffusion-announcement) by Patrick Esser and Robin Rombach and the Stability AI team.


## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

## Text-to-image Generation

### Single Prompt

Here is how to generate images with one prompt:

```bash
python text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4 \
    --prompts "An image of a squirrel in Picasso style" \
    --num_images_per_prompt 28 \
    --batch_size 7 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

> HPU graphs are recommended when generating images by batches to get the fastest possible generations.
> The first batch of images entails a performance penalty. All subsequent batches will be generated much faster.
> You can enable this mode with `--use_hpu_graphs`.

### Multiple Prompts

Here is how to generate images with several prompts:

```bash
python text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4 \
    --prompts "An image of a squirrel in Picasso style" "A shiny flying horse taking off" \
    --num_images_per_prompt 32 \
    --batch_size 8 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

### Distributed inference with multiple HPUs

Here is how to generate images with two prompts on two HPUs:

```bash
python ../gaudi_spawn.py \
    --world_size 2 text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4 \
    --prompts "An image of a squirrel in Picasso style" "A shiny flying horse taking off" \
    --num_images_per_prompt 20 \
    --batch_size 4 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16 \
    --distributed
```

> HPU graphs are recommended when generating images by batches to get the fastest possible generations.
> The first batch of images entails a performance penalty. All subsequent batches will be generated much faster.
> You can enable this mode with `--use_hpu_graphs`.

### Stable Diffusion 2

[Stable Diffusion 2](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion_2) can also be used to generate images with this script. Here is an example for a single prompt:

```bash
python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-2-1 \
    --prompts "An image of a squirrel in Picasso style" \
    --num_images_per_prompt 28 \
    --batch_size 7 \
    --height 768 \
    --width 768 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion-2 \
    --sdp_on_bf16 \
    --bf16
```

> There are two different checkpoints for Stable Diffusion 2:
>
> - use [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) for generating 768x768 images
> - use [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) for generating 512x512 images

### Latent Diffusion Model for 3D (LDM3D)

[LDM3D](https://arxiv.org/abs/2305.10853) generates both image and depth map data from a given text prompt, allowing users to generate RGBD images from text prompts.

[Original checkpoint](https://huggingface.co/Intel/ldm3d) and [latest checkpoint](https://huggingface.co/Intel/ldm3d-4c) are open source.
A [demo](https://huggingface.co/spaces/Intel/ldm3d) is also available. Here is how to run this model:

```bash
python text_to_image_generation.py \
    --model_name_or_path "Intel/ldm3d-4c" \
    --prompts "An image of a squirrel in Picasso style" \
    --num_images_per_prompt 28 \
    --batch_size 7 \
    --height 768 \
    --width 768 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion-2 \
    --ldm3d
```

Here is how to generate images and depth maps with two prompts on two HPUs:

```bash
python ../gaudi_spawn.py \
    --world_size 2 text_to_image_generation.py \
    --model_name_or_path "Intel/ldm3d-4c" \
    --prompts "An image of a squirrel in Picasso style" "A shiny flying horse taking off" \
    --num_images_per_prompt 10 \
    --batch_size 2 \
    --height 768 \
    --width 768 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion-2 \
    --ldm3d \
    --distributed
```

> There are three different checkpoints for LDM3D:
>
> - use [original checkpoint](https://huggingface.co/Intel/ldm3d) to generate outputs from the paper
> - use [the latest checkpoint](https://huggingface.co/Intel/ldm3d-4c) for generating improved results
> - use [the pano checkpoint](https://huggingface.co/Intel/ldm3d-pano) to generate panoramic view

### Stable Diffusion XL (SDXL)

Stable Diffusion XL was proposed in [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/pdf/2307.01952.pdf) by the Stability AI team.

Here is how to generate SDXL images with a single prompt:

```bash
python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --prompts "Sailing ship painting by Van Gogh" \
    --num_images_per_prompt 28 \
    --batch_size 7 \
    --image_save_dir /tmp/stable_diffusion_xl_images \
    --scheduler euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

> HPU graphs are recommended when generating images by batches to get the fastest possible generations.
> The first batch of images entails a performance penalty. All subsequent batches will be generated much faster.
> You can enable this mode with `--use_hpu_graphs`.

Here is how to generate SDXL images with several prompts:

```bash
python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --prompts "Sailing ship painting by Van Gogh" "A shiny flying horse taking off" \
    --num_images_per_prompt 32 \
    --batch_size 8 \
    --image_save_dir /tmp/stable_diffusion_xl_images \
    --scheduler euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

SDXL combines a second text encoder (OpenCLIP ViT-bigG/14) with the original text encoder to significantly
increase the number of parameters. Here is how to generate images with several prompts for both `prompt`
and `prompt_2` (2nd text encoder), as well as their negative prompts:

```bash
python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --prompts "Sailing ship painting by Van Gogh" "A shiny flying horse taking off" \
    --prompts_2 "Red tone" "Blue tone" \
    --negative_prompts "Low quality" "Sketch" \
    --negative_prompts_2 "Clouds" "Clouds" \
    --num_images_per_prompt 32 \
    --batch_size 8 \
    --image_save_dir /tmp/stable_diffusion_xl_images \
    --scheduler euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

Here is how to generate SDXL images with two prompts on two HPUs:

```bash
python ../gaudi_spawn.py \
    --world_size 2 text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --prompts "Sailing ship painting by Van Gogh" "A shiny flying horse taking off" \
    --prompts_2 "Red tone" "Blue tone" \
    --negative_prompts "Low quality" "Sketch" \
    --negative_prompts_2 "Clouds" "Clouds" \
    --num_images_per_prompt 32 \
    --batch_size 8 \
    --image_save_dir /tmp/stable_diffusion_xl_images \
    --scheduler euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16 \
    --distributed
```

Here is how to generate SDXL images with optimized pipeline:
```bash
python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --prompts "Sailing ship painting by Van Gogh" \
    --num_images_per_prompt 28 \
    --batch_size 7 \
    --image_save_dir /tmp/stable_diffusion_xl_images \
    --scheduler euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16 \
    --optimize
```

Here is how to generate SDXL images with optimized pipeline in fp8:
```bash
QUANT_CONFIG=./quantization/quant_config.json python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --prompts "Sailing ship painting by Van Gogh" \
    --num_images_per_prompt 28 \
    --batch_size 7 \
    --image_save_dir /tmp/stable_diffusion_xl_images \
    --scheduler euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16 \
    --optimize
```

> HPU graphs are recommended when generating images by batches to get the fastest possible generations.
> The first batch of images entails a performance penalty. All subsequent batches will be generated much faster.
> You can enable this mode with `--use_hpu_graphs`.

### SDXL-Turbo

SDXL-Turbo is a distilled version of SDXL 1.0, trained for real-time synthesis.

Here is how to generate images with multiple prompts:

```bash
python text_to_image_generation.py \
    --model_name_or_path stabilityai/sdxl-turbo \
    --prompts "Sailing ship painting by Van Gogh" "A shiny flying horse taking off" \
    --num_images_per_prompt 32 \
    --batch_size 8 \
    --image_save_dir /tmp/stable_diffusion_xl_turbo_images \
    --scheduler euler_ancestral_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16 \
    --num_inference_steps 1 \
    --guidance_scale 1.000001 \
    --timestep_spacing trailing
```

> HPU graphs are recommended when generating images by batches to get the fastest possible generations.
> The first batch of images entails a performance penalty. All subsequent batches will be generated much faster.
> You can enable this mode with `--use_hpu_graphs`.

> Note: there is a regression with "--guidance_scale 0.0" in current release which will be addressed in later releases. Setting `--guidance_scale` to a value larger than 1 resolves the regression.

### Stable Diffusion 3 (SD3)

Stable Diffusion 3 was introduced by Stability AI [here](https://stability.ai/news/stable-diffusion-3).
It uses Diffusion Transformer instead of UNet for denoising, which yields improved image quality.

Before running SD3 pipeline, you need to:

1. Agree to the Terms and Conditions for using SD3 model at [HuggingFace model page](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
2. Authenticate with HuggingFace using your HF Token. For authentication, run:

```bash
huggingface-cli login
```

Here is how to generate SD3 images with a single prompt:

```bash
PT_HPU_MAX_COMPOUND_OP_SIZE=1 \
python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
    --prompts "Sailing ship painting by Van Gogh" \
    --num_images_per_prompt 10 \
    --batch_size 1 \
    --num_inference_steps 28 \
    --image_save_dir /tmp/stable_diffusion_3_images \
    --scheduler default \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

> For improved performance of the SD3 pipeline on Gaudi, it is recommended to configure the environment
> by setting PT_HPU_MAX_COMPOUND_OP_SIZE to 1.

### FLUX.1

FLUX.1 was introduced by Black Forest Labs [here](https://blackforestlabs.ai/announcing-black-forest-labs/).

Here is how to run FLUX.1-schnell model (fast version of FLUX.1):

```bash
python text_to_image_generation.py \
    --model_name_or_path black-forest-labs/FLUX.1-schnell \
    --prompts "A cat holding a sign that says hello world" \
    --num_images_per_prompt 10 \
    --batch_size 1 \
    --num_inference_steps 4 \
    --image_save_dir /tmp/flux_1_images \
    --scheduler flow_match_euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

Before running FLUX.1-dev model, you need to:

1. Agree to the Terms and Conditions for using FLUX.1-dev model at [HuggingFace model page](https://huggingface.co/black-forest-labs/FLUX.1-dev)
2. Authenticate with HuggingFace using your HF Token. For authentication, run:

```bash
huggingface-cli login
```

Here is how to run FLUX.1-dev model:

```bash
python text_to_image_generation.py \
    --model_name_or_path black-forest-labs/FLUX.1-dev \
    --prompts "A cat holding a sign that says hello world" \
    --num_images_per_prompt 10 \
    --batch_size 1 \
    --num_inference_steps 30 \
    --image_save_dir /tmp/flux_1_images \
    --scheduler flow_match_euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

This model can also be quantized with some ops running in FP8 precision.

Before quantization, run stats collection using measure mode:

```bash
QUANT_CONFIG=quantization/flux/measure_config.json \
python text_to_image_generation.py \
    --model_name_or_path black-forest-labs/FLUX.1-dev \
    --prompts "A cat holding a sign that says hello world" \
    --num_images_per_prompt 10 \
    --batch_size 1 \
    --num_inference_steps 30 \
    --image_save_dir /tmp/flux_1_images \
    --scheduler flow_match_euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16 \
    --quant_mode measure
```

After stats collection, here is how to run FLUX.1-dev in quantization mode:

```bash
QUANT_CONFIG=quantization/flux/quantize_config.json \
python text_to_image_generation.py \
    --model_name_or_path black-forest-labs/FLUX.1-dev \
    --prompts "A cat holding a sign that says hello world" \
    --num_images_per_prompt 10 \
    --batch_size 1 \
    --num_inference_steps 30 \
    --image_save_dir /tmp/flux_1_images \
    --scheduler flow_match_euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16 \
    --quant_mode quantize
```

## ControlNet

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang and Maneesh Agrawala.
It is a type of model for controlling StableDiffusion by conditioning the model with an additional input image.

Here is how to generate images conditioned by canny edge model:

```bash
python text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4 \
    --controlnet_model_name_or_path lllyasviel/sd-controlnet-canny \
    --prompts "futuristic-looking woman" \
    --control_image https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png \
    --num_images_per_prompt 28 \
    --batch_size 7 \
    --image_save_dir /tmp/controlnet_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

Here is how to generate images conditioned by canny edge model and with multiple prompts:

```bash
python text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4 \
    --controlnet_model_name_or_path lllyasviel/sd-controlnet-canny \
    --prompts "futuristic-looking woman" "a rusty robot" \
    --control_image https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png \
    --num_images_per_prompt 28 \
    --batch_size 7 \
    --image_save_dir /tmp/controlnet_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

Here is how to generate images conditioned by canny edge model and with two prompts on two HPUs:

```bash
python ../gaudi_spawn.py \
    --world_size 2 text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4 \
    --controlnet_model_name_or_path lllyasviel/sd-controlnet-canny \
    --prompts "futuristic-looking woman" "a rusty robot" \
    --control_image https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png \
    --num_images_per_prompt 16 \
    --batch_size 4 \
    --image_save_dir /tmp/controlnet_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16 \
    --distributed
```

Here is how to generate images conditioned by open pose model:

```bash
python text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4 \
    --controlnet_model_name_or_path lllyasviel/sd-controlnet-openpose \
    --prompts "Chef in the kitchen" \
    --control_image https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/pose.png \
    --control_preprocessing_type "none" \
    --num_images_per_prompt 28 \
    --batch_size 7 \
    --image_save_dir /tmp/controlnet_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

Here is how to generate images with conditioned by canny edge model using Stable Diffusion 2

```bash
python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-2-1 \
    --controlnet_model_name_or_path thibaud/controlnet-sd21-canny-diffusers \
    --control_image https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png \
    --control_preprocessing_type "none" \
    --prompts "bird" \
    --seed 0 \
    --num_images_per_prompt 28 \
    --batch_size 7 \
    --image_save_dir /tmp/controlnet-2-1_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion-2 \
    --sdp_on_bf16 \
    --bf16
```

## Inpainting

Inpainting replaces or edits specific areas of an image. For more details,
please refer to [Hugging Face Diffusers doc](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint).

### Stable Diffusion Inpainting

```bash
python text_to_image_generation.py \
    --model_name_or_path  stabilityai/stable-diffusion-2-inpainting \
    --base_image https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png \
    --mask_image https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png \
    --prompts "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k" \
    --seed 0 \
    --num_images_per_prompt 12 \
    --batch_size 4 \
    --image_save_dir /tmp/inpaiting_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

### Stable Diffusion XL Inpainting

```bash
python text_to_image_generation.py \
    --model_name_or_path  diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
    --base_image https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png \
    --mask_image https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png \
    --prompts "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k" \
    --seed 0 \
    --scheduler euler_discrete \
    --num_images_per_prompt 12 \
    --batch_size 4 \
    --image_save_dir /tmp/xl_inpaiting_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

## Image-to-image Generation

### Single Prompt

Here is how to generate images with one prompt and one image.
Take instruct-pix2pix as an example.

```bash
python image_to_image_generation.py \
    --model_name_or_path "timbrooks/instruct-pix2pix" \
    --src_image_path "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg" \
    --prompts "turn him into cyborg" \
    --num_images_per_prompt 20 \
    --batch_size 4 \
    --guidance_scale 7.5 \
    --image_guidance_scale 1 \
    --num_inference_steps 10 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

> HPU graphs are recommended when generating images by batches to get the fastest possible generations.
> The first batch of images entails a performance penalty. All subsequent batches will be generated much faster.
> You can enable this mode with `--use_hpu_graphs`.

### Multiple Prompts

Here is how to generate images with several prompts and one image.

```bash
python image_to_image_generation.py \
    --model_name_or_path "timbrooks/instruct-pix2pix" \
    --src_image_path "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg" \
    --prompts "turn him into cyborg" "a strong soldier"\
    --num_images_per_prompt 20 \
    --batch_size 4 \
    --guidance_scale 7.5 \
    --image_guidance_scale 1 \
    --num_inference_steps 10 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

> HPU graphs are recommended when generating images by batches to get the fastest possible generations.
> The first batch of images entails a performance penalty. All subsequent batches will be generated much faster.
> You can enable this mode with `--use_hpu_graphs`.

### Stable Diffusion XL Refiner

Here is how to generate SDXL images with a single prompt and one image:

```bash
python image_to_image_generation.py \
    --model_name_or_path "stabilityai/stable-diffusion-xl-refiner-1.0" \
    --src_image_path "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg" \
    --prompts "turn him into cyborg" \
    --num_images_per_prompt 20 \
    --batch_size 4 \
    --guidance_scale 7.5 \
    --num_inference_steps 10 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

### FLUX.1 Image to Image

Here is how to generate FLUX.1 images with a single prompt and one input image:

```bash
python image_to_image_generation.py \
    --model_name_or_path "black-forest-labs/FLUX.1-dev" \
    --src_image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png" \
    --prompts "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k" \
    --num_images_per_prompt 40 \
    --batch_size 10 \
    --strength 0.9 \
    --guidance_scale 3.5 \
    --num_inference_steps 30 \
    --image_save_dir /tmp/flux_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

### Stable Diffusion Image Variations

Here is how to generate images with one image, it does not accept prompt input

```bash
python image_to_image_generation.py \
    --model_name_or_path "lambdalabs/sd-image-variations-diffusers" \
    --src_image_path "https://github.com/SHI-Labs/Versatile-Diffusion/blob/master/assets/demo/reg_example/ghibli.jpg?raw=true" \
    --num_images_per_prompt 20 \
    --batch_size 4 \
    --image_save_dir /tmp/stable_diffusion_images \
    --guidance_scale 3 \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

### Depth to Image Generation

Here is how to generate a depth2img-guided image generation using HPU graphs with BF16:

```bash
python depth_to_image_generation.py \
    --model_name_or_path "stabilityai/stable-diffusion-2-depth" \
    --prompts "two tigers" \
    --base_image "http://images.cocodataset.org/val2017/000000039769.jpg" \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --sdp_on_bf16 \
    --bf16
```

## Unconditional Image Generation Example

Here is how to perform unconditional-image-generation on Gaudi/HPU.

Original unconditional image generation pipeline is shared in here: [Unconditional Image Generation](https://huggingface.co/docs/diffusers/using-diffusers/unconditional_image_generation)

```bash
python unconditional_image_generation.py \
    --model_name_or_path "google/ddpm-ema-celebahq-256" \
    --batch_size 16 \
    --use_habana \
    --use_gaudi_ddim_scheduler \
    --use_hpu_graphs \
    --sdp_on_bf16 \
    --bf16 \
    --save_outputs \
    --output_dir "/tmp/"
```

## Additional inference techniques

Here is how to run the diffusers examples of inference techniques. For more details,
please refer to [Hugging Face Diffusers doc](https://huggingface.co/docs/diffusers/main/en/using-diffusers/overview_techniques).

### Controlling brightness

Here is how to run the example of controlling brightness. For more details,
please refer to [Hugging Face Diffusers doc](https://huggingface.co/docs/diffusers/main/en/using-diffusers/control_brightness).

```bash
PT_HPU_MAX_COMPOUND_OP_SIZE=1 python text_to_image_generation.py \
    --model_name_or_path ptx0/pseudo-journey-v2 \
    --prompts "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k" \
    --num_images_per_prompt 1 \
    --batch_size 1 \
    --use_habana \
    --use_hpu_graphs \
    --image_save_dir /tmp/stable_diffusion_images_brightness \
    --seed 33 \
    --use_zero_snr \
    --guidance_scale 0.7 \
    --timestep_spacing trailing
```

### Prompt weighting

Here is how to run the example of prompt weighting. For more details,
please refer to [Hugging Face Diffusers doc](https://huggingface.co/docs/diffusers/main/en/using-diffusers/weighted_prompts).

```bash
python text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4 \
    --prompts "a red cat playing with a ball+++" "a red cat playing with a ball---" \
    --num_images_per_prompt 4 \
    --batch_size 4 \
    --use_habana --use_hpu_graphs \
    --image_save_dir /tmp/stable_diffusion_images_compel \
    --seed 33 \
    --sdp_on_bf16 \
    --bf16 \
    --num_inference_steps 20 \
    --use_compel
```

### Controlling image quality

Here is how to run the example of improving image quality. For more details,
please refer to [Hugging Face Diffusers doc](https://huggingface.co/docs/diffusers/main/en/using-diffusers/image_quality).

```bash
python text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4 \
    --prompts "A squirrel eating a burger" \
    --num_images_per_prompt 4 \
    --batch_size 4 \
    --use_habana \
    --image_save_dir /tmp/stable_diffusion_images_freeu \
    --seed 33 \
    --use_freeu \
    --sdp_on_bf16 \
    --bf16
```
# Stable Video Diffusion Examples

Stable Video Diffusion (SVD) was unveiled in [Stable Video Diffusion Announcement](https://stability.ai/news/stable-video-diffusion-open-ai-video-model)
by the Stability AI team. Stable Video Diffusion XT version (SVD-XT) is tuned to generate 25 frames of video from a single image.

## Image-to-video Generation

Script `image_to_video_generation.py` showcases how to perform image-to-video generation using Stable Video Diffusion on Intel Gaudi.

### Single Image Prompt

Here is how to generate video with one image prompt:

```bash
PT_HPU_MAX_COMPOUND_OP_SIZE=1 \
python image_to_video_generation.py \
    --model_name_or_path "stabilityai/stable-video-diffusion-img2vid-xt" \
    --image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png" \
    --num_videos_per_prompt 1 \
    --video_save_dir /tmp/stable_video_diffusion_xt \
    --save_frames_as_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

> For improved performance of the image-to-video pipeline on Gaudi, it is recommended to configure the environment
> by setting PT_HPU_MAX_COMPOUND_OP_SIZE to 1.

### Multiple Image Prompts

Here is how to generate videos with several image prompts:

```bash
PT_HPU_MAX_COMPOUND_OP_SIZE=1 \
python image_to_video_generation.py \
    --model_name_or_path "stabilityai/stable-video-diffusion-img2vid-xt" \
    --image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png" \
                 "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png" \
                 "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png" \
                 "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png" \
    --num_videos_per_prompt 1 \
    --video_save_dir /tmp/stable_video_diffusion_xt \
    --save_frames_as_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

> For improved performance of the image-to-video pipeline on Gaudi, it is recommended to configure the environment
> by setting PT_HPU_MAX_COMPOUND_OP_SIZE to 1.

### Image-to-video ControlNet

Here is how to generate video conditioned by depth:

```
python image_to_video_generation.py \
    --model_name_or_path "stabilityai/stable-video-diffusion-img2vid" \
    --controlnet_model_name_or_path "CiaraRowles/temporal-controlnet-depth-svd-v1" \
    --control_image_path "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/depth/frame_0.png?raw=true" \
             "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/depth/frame_1.png?raw=true" \
             "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/depth/frame_2.png?raw=true" \
             "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/depth/frame_3.png?raw=true" \
             "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/depth/frame_4.png?raw=true" \
             "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/depth/frame_5.png?raw=true" \
             "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/depth/frame_6.png?raw=true" \
             "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/depth/frame_7.png?raw=true" \
             "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/depth/frame_8.png?raw=true" \
             "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/depth/frame_9.png?raw=true" \
             "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/depth/frame_10.png?raw=true" \
             "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/depth/frame_11.png?raw=true" \
             "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/depth/frame_12.png?raw=true" \
             "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/depth/frame_13.png?raw=true" \
    --image_path "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/chair.png?raw=true" \
    --video_save_dir SVD_controlnet \
    --save_frames_as_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --bf16 \
    --sdp_on_bf16 \
    --num_frames 14 \
    --motion_bucket_id=14 \
    --width=512 \
    --height=512
```

> [!NOTE]
> For Gaudi3 only:
> 1. Due to a known issue, batch sizes for models needs to be reduced. It will be fixed in the future release.
> 2. The Image-to-video ControlNet command is not enabled on Gaudi3.
