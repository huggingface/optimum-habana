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

# Stable Diffusion Examples (Legacy Models)

This directory contains sample scripts demonstrating how to perform diffusion-based generative tasks on Intel® Gaudi® AI Accelerators.

Stable Diffusion was introduced in [Stable Diffusion Announcement](https://stability.ai/blog/stable-diffusion-announcement) by Patrick Esser,
Robin Rombach and the Stability AI team.

## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

## Text-to-Image Generation

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

> [!NOTE]
> HPU graphs are recommended when generating images by batches to get the fastest possible generations.
> The first batch of images entails a performance penalty. All subsequent batches will be generated much faster.
> You can enable this mode with `--use_hpu_graphs`.

To input multiple prompts, pass prompt strings separated by spaces. Additionally, you can run inference on multiple
HPUs by replacing `python text_to_image_generation.py`
with `python ../gaudi_spawn.py --world_size <number-of-HPUs> text_to_image_generation.py`.

You can run other other older Stable Diffusion models in a similar manner. For example, to generate images with
Stable Diffusion 1.5, use the option: `--model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5`.
Examples showcasing Stable Diffusion 2 are provided next.

### Stable Diffusion 2

[Stable Diffusion 2](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion_2) can also be used
to generate images with this script. Here is an example demonstrating image generation with a single prompt:

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

> [!NOTE]
> There are two different checkpoints for Stable Diffusion 2:
> - use [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) for generating 768x768 images
> - use [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) for generating 512x512 images

To input multiple prompts, pass prompt strings separated by spaces. Additionally, you can run inference on multiple HPUs by replacing
`python text_to_image_generation.py` with `python ../gaudi_spawn.py --world_size <number-of-HPUs> text_to_image_generation.py` and
adding option `--distributed`.

### Latent Diffusion Model for 3D (LDM3D)

[LDM3D](https://arxiv.org/abs/2305.10853) generates both image and depth map data from a given text prompt, allowing users
to generate RGBD images from text prompts.

[Original checkpoint](https://huggingface.co/Intel/ldm3d) and [latest checkpoint](https://huggingface.co/Intel/ldm3d-4c)
are open source. A [demo](https://huggingface.co/spaces/Intel/ldm3d) is also available. Here is how to run this model:

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

> [!NOTE]
> There are three different checkpoints for LDM3D:
> - use [original checkpoint](https://huggingface.co/Intel/ldm3d) to generate outputs from the paper
> - use [the latest checkpoint](https://huggingface.co/Intel/ldm3d-4c) for generating improved results
> - use [the pano checkpoint](https://huggingface.co/Intel/ldm3d-pano) to generate panoramic view

To input multiple prompts, pass prompt strings separated by spaces. Additionally, you can run inference on multiple HPUs by replacing
`python text_to_image_generation.py` with `python ../gaudi_spawn.py --world_size <number-of-HPUs> text_to_image_generation.py` and
adding option `--distributed`

## ControlNet

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543)
by Lvmin Zhang and Maneesh Agrawala, enables conditioning the Stable Diffusion model with an additional input image.
This allows for precise control over the composition of generated images using various features such as edges,
pose, depth, and more.

Here is how to generate images conditioned by Canny edge model:

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

You can run inference on multiple HPUs by replacing `python text_to_image_generation.py` with
`python ../gaudi_spawn.py --world_size <number-of-HPUs> text_to_image_generation.py` and adding option `--distributed`.

This ControlNet example will preprocess the input image to derive Canny edges. Alternatively, you can use `--control_preprocessing_type none`
to supply a preprocessed control image directly, enabling many additional use cases.

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

## Additional Stable Diffusion-based Inference Techniques

This section provides examples of additional inference techniques based on Stable Diffusion. For more details, please refer to
[Hugging Face Diffusers documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/overview_techniques).

### Unconditional Image Generation

Here is how to perform unconditional image generation on Intel Gaudi. For more details,  please refer to the 
[Unconditional Image Generation](https://huggingface.co/docs/diffusers/using-diffusers/unconditional_image_generation)
section in the Hugging Face documentation.

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

### Controlling Brightness

Here is an example of how to control brightness. For more information, please refer to the
[Control Brightness](https://huggingface.co/docs/diffusers/main/en/using-diffusers/control_brightness)
section in the Hugging Face documentation.

```bash
PT_HPU_MAX_COMPOUND_OP_SIZE=1 \
python text_to_image_generation.py \
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

### Prompt Weighting

Here is an example of how to run prompt weighting. For more information, please refer to the
[Weighted Prompts](https://huggingface.co/docs/diffusers/main/en/using-diffusers/weighted_prompts)
section in the Hugging Face documentation.

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

### Controlling Image Quality

Here is an example of how to improve image quality. For more details, please refer to the
[Image Quality](https://huggingface.co/docs/diffusers/main/en/using-diffusers/image_quality)
section in the Hugging Face documentation.

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

## Image-to-Image Generation

Images can also be generated using initial input images to guide the diffusion-based image generation process.

### Stable Diffusion-based Image-to-Image

Here is how to generate images using a single prompt and an input image with the `timbrooks/instruct-pix2pix` model, which is based on Stable Diffusion:

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
> You can pass multiple prompts strings separated via space.

### Stable Diffusion XL Refiner

Here is how to refine SDXL images using a single image and prompt:

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

### Stable Diffusion Image Variations

Here is how to generate image variations of a single image (without any input prompts):

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

Here is an example of performing depth-guided image generation:

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