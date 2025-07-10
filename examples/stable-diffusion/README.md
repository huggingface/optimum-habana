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

This directory contains sample scripts demonstrating how to perform diffusion-based generative tasks on Intel® Gaudi® AI Accelerators.

Stable Diffusion was introduced in [Stable Diffusion Announcement](https://stability.ai/blog/stable-diffusion-announcement) by Patrick Esser,
Robin Rombach and the Stability AI team.

## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

## Text-to-Image Generation

Optimum for Intel Gaudi supports state-of-the-art diffusion-based text-to-image generation models, including SDXL, SD3/3.5, and FLUX. We provide
brief inference examples for these models. For running legacy Stable Diffusion (SD) models, please refer to [this](README_legacy.md) document.

### Stable Diffusion XL (SDXL)

Stable Diffusion XL was proposed in [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/pdf/2307.01952.pdf)
by the Stability AI team.

Here is how to generate SDXL images with a single prompt:

```bash
PT_HPU_LAZY_MODE=1 python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --prompts "Sailing ship painting by Van Gogh" \
    --num_images_per_prompt 28 \
    --batch_size 7 \
    --num_inference_steps 30 \
    --image_save_dir /tmp/stable_diffusion_xl_images \
    --scheduler euler_discrete \
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

> [!WARNING]
> There is a regression with `--guidance_scale 0.0` in current release which will be addressed in later releases.
> Setting `--guidance_scale` to a value larger than 1 resolves the regression.

To input multiple prompts, pass prompt strings separated by spaces. SDXL improves text-to-image generation by combining
OpenCLIP ViT-bigG/14 with the original Stable Diffusion text encoder, thus allowing for more descriptive prompts.
You can pass single or multiple prompts for both `prompt` and `prompt_2` (2nd text encoder), as well as their negative prompts.

Additionally, you can run inference on multiple HPUs by replacing `python text_to_image_generation.py`
with `python ../gaudi_spawn.py --world_size <num-HPUs> text_to_image_generation.py` and adding option `--distributed`.

A version of the SDXL pipeline optimized for FP8 on Intel Gaudi is also available. Set
`QUANT_CONFIG=quantization/stable-diffusion-xl/quantize_config.json` enviromement variable and use option `--optimize`
to run FP8-optimized SDXL pipeline.

To run SDXL-Turbo, the distilled version of SDXL, use `--model_name_or_path stabilityai/sdxl-turbo` in the input.

### Stable Diffusion 3 and 3.5 (SD3)

Stable Diffusion 3 was introduced by Stability AI [here](https://stability.ai/news/stable-diffusion-3).
It uses Diffusion Transformer instead of UNet for denoising, which yields improved image quality.

```bash
PT_HPU_LAZY_MODE=1 \
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

> [!NOTE]
> The access to SD3 requires agreeing to its terms and conditions at [HuggingFace model page](https://huggingface.co/stabilityai/stable-diffusion-3-medium),
> and then authenticating using your HF token via `huggingface-cli login`.

This model can also be quantized with some ops running in FP8 precision. Before quantization, run stats collection once using measure mode by setting
runtime variable `QUANT_CONFIG=quantization/stable-diffusion-3/measure_config.json` and `--quant_mode measure`. After stats collection, you can run
SD3 in quantization mode by setting runtime variable `QUANT_CONFIG=quantization/stable-diffusion-3/quantize_config.json` and `--quant_mode quantize`.

> [!NOTE]
> If you are running SD3 Gaudi pipeline as a service, run quantization mode only once and pipeline in memory will be quantized to use FP8 precision.
> Running quantization mode multiple times on the same pipeline object may cause errors.

To run Stable Diffusion 3.5 Large, use `--model_name_or_path stabilityai/stable-diffusion-3.5-large` in the input.

#### SD3 Distributed CFG Inference

SD3 family of models use classifier-free guidance (CFG), which processes both conditional and unconditional latents during denoising, typically in
a single forward pass with batch size 2. With the `--use_distributed_cfg` option, we parallelize this step across two HPU devices by splitting the
batch and running each branch independently, then synchronizing to apply guidance. While this mode uses 2 HPUs per unique generated image, it
achieves roughly 2x faster inference. Here is an example of running SD3.5-Large model with 2 HPU devices in disributed CFG mode:

```bash
PT_HPU_LAZY_MODE=1 \
python ../gaudi_spawn.py --world_size 2 text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-3.5-large \
    --prompts "Sailing ship painting by Van Gogh" \
    --num_images_per_prompt 4 \
    --batch_size 1 \
    --num_inference_steps 28 \
    --image_save_dir /tmp/stable_diffusion_3_images \
    --scheduler default \
    --use_habana \
    --use_hpu_graphs \
    --use_distributed_cfg \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

> [!NOTE]
> Distributed CFG mode requires even number of devices in the `world_size`.

### FLUX

FLUX.1 was introduced by Black Forest Labs [here](https://blackforestlabs.ai/announcing-black-forest-labs/).

Here is how to run FLUX.1-dev model:

```bash
PT_HPU_LAZY_MODE=1 python text_to_image_generation.py \
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

> [!NOTE]
> The access to FLUX.1-dev model requires agreeing to its terms and conditions at [HuggingFace model page](https://huggingface.co/black-forest-labs/FLUX.1-dev),
> and then authenticating using your HF token via `huggingface-cli login`.

This model can also be quantized with some ops running in FP8 precision. Before quantization, run stats collection once using measure mode by setting
runtime variable `QUANT_CONFIG=quantization/flux/measure_config.json` and `--quant_mode measure`. After stats collection, you can run
FLUX in quantization mode by setting runtime variable `QUANT_CONFIG=quantization/flux/quantize_config.json` and `--quant_mode quantize`.

> [!NOTE]
> If you are running Flux Gaudi pipeline as a service, run quantization mode only once and pipeline in memory will be quantized to use FP8 precision.
> Running quantization mode multiple times on the same pipeline object may cause errors.

To run with FLUX.1-schnell model, a distilled version of FLUX.1 (which is not gated), use `--model_name_or_path black-forest-labs/FLUX.1-schnell`.

## ControlNet

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543)
by Lvmin Zhang and Maneesh Agrawala, enables conditioning the Stable Diffusion model with an additional input image.
This allows for precise control over the composition of generated images using various features such as edges,
pose, depth, and more.

Here is how to generate images conditioned by Canny edge model:

```bash
PT_HPU_LAZY_MODE=1 python text_to_image_generation.py \
    --model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
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

### Stable Diffusion XL Inpainting

```bash
PT_HPU_LAZY_MODE=1 python text_to_image_generation.py \
    --model_name_or_path diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
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

### Controlling Brightness

Here is an example of how to control brightness. For more information, please refer to the
[Control Brightness](https://huggingface.co/docs/diffusers/main/en/using-diffusers/control_brightness)
section in the Hugging Face documentation.

```bash
PT_HPU_LAZY_MODE=1 PT_HPU_MAX_COMPOUND_OP_SIZE=1 \
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
PT_HPU_LAZY_MODE=1 python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --prompts "a red cat--- playing with a ball+++" "a red cat+++ playing with a ball---" \
    --num_images_per_prompt 4 \
    --batch_size 4 \
    --use_habana --use_hpu_graphs \
    --image_save_dir /tmp/stable_diffusion_xl_images_compel \
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
PT_HPU_LAZY_MODE=1 python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --prompts "A squirrel eating a burger" \
    --num_images_per_prompt 4 \
    --batch_size 4 \
    --use_habana \
    --image_save_dir /tmp/stable_diffusion_xl_images_freeu \
    --seed 33 \
    --use_freeu \
    --sdp_on_bf16 \
    --bf16
```

## Image-to-Image Generation

Images can also be generated using initial input images to guide the diffusion-based image generation process.

### Stable Diffusion XL Image-to-Image

Here is how to refine SDXL images using a single image and prompt:

```bash
PT_HPU_LAZY_MODE=1 python image_to_image_generation.py \
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

### FLUX Image-to-Image

Here is how to generate a FLUX.1 image using a single input image and prompt:

```bash
PT_HPU_LAZY_MODE=1 python image_to_image_generation.py \
    --model_name_or_path "black-forest-labs/FLUX.1-dev" \
    --src_image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png" \
    --prompts "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k" \
    --num_images_per_prompt 10 \
    --batch_size 1 \
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

## Text-to-Video Generation

This section demonstrates how to use the `GaudiTextToVideoSDPipeline` for text-to-video generation tasks on HPUs.
The pipeline employs a UNet3D structure and generates videos through an iterative denoising process.

```bash
PT_HPU_LAZY_MODE=1 python text_to_video_generation.py \
    --model_name_or_path ali-vilab/text-to-video-ms-1.7b \
    --prompts "An astronaut riding a horse" \
    --use_habana \
    --use_hpu_graphs \
    --dtype bf16
```

# Stable Video Diffusion Examples

Stable Video Diffusion (SVD) was unveiled in [Stable Video Diffusion Announcement](https://stability.ai/news/stable-video-diffusion-open-ai-video-model)
by the Stability AI team. Stable Video Diffusion XT version (SVD-XT) is tuned to generate 25 frames of video from a single image.


## Image-to-Video Generation

Script `image_to_video_generation.py` showcases how to perform image-to-video generation using Stable Video Diffusion on Intel Gaudi.

Here is how to generate video with one image prompt:

```bash
PT_HPU_LAZY_MODE=1 PT_HPU_MAX_COMPOUND_OP_SIZE=1 \
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

> [!NOTE]
> For improved performance of the image-to-video pipeline on Gaudi, it is recommended to set the following env variable: `PT_HPU_MAX_COMPOUND_OP_SIZE=1`.

You can pass multiple image prompts strings separated via space, i.e.
`--image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png" "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"`.

### Image-to-Video ControlNet

Here is how to generate video conditioned by depth:

```bash
PT_HPU_LAZY_MODE=1 python image_to_video_generation.py \
    --model_name_or_path "stabilityai/stable-video-diffusion-img2vid" \
    --controlnet_model_name_or_path "CiaraRowles/temporal-controlnet-depth-svd-v1" \
    --control_image_path \
        "https://github.com/CiaraStrawberry/svd-temporal-controlnet/blob/main/validation_demo/depth/frame_0.png?raw=true" \
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
    --sdp_on_bf16 \
    --bf16 \
    --num_frames 14 \
    --motion_bucket_id=14 \
    --width=512 \
    --height=512
```

### Image-to-Video with I2vgen-xl
I2vgen-xl is high quality Image-to-Video synthesis via cascaded diffusion models. Please refer to  [Huggingface i2vgen-xl doc](https://huggingface.co/ali-vilab/i2vgen-xl).

Here is how to generate video with one image and text prompt:

```bash
PT_HPU_LAZY_MODE=1 PT_HPU_MAX_COMPOUND_OP_SIZE=1 \
python image_to_video_generation.py \
    --model_name_or_path "ali-vilab/i2vgen-xl" \
    --image_path "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png" \
    --num_videos_per_prompt 1 \
    --video_save_dir ./i2vgen_xl \
    --num_inference_steps 50 \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --gif \
    --num_frames 16 \
    --prompts "Papers were floating in the air on a table in the library" \
    --negative_prompts "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms" \
    --seed 8888  \
    --sdp_on_bf16 \
    --bf16
```

### Text-to-Video with CogvideoX

CogVideoX is an open-source version of the video generation model originating from QingYing, unveiled in https://huggingface.co/THUDM/CogVideoX-5b.

```bash
PT_HPU_LAZY_MODE=1 python text_to_video_generation.py \
    --model_name_or_path "THUDM/CogVideoX-2b" \
    --pipeline_type "cogvideox" \
    --prompts "An astronaut riding a horse" \
    --use_habana \
    --use_hpu_graphs \
    --num_videos_per_prompt 1 \
    --num_inference_steps 50 \
    --num_frames 49 \
    --guidance_scale 6 \
    --dtype bf16
```

# Important Notes for Gaudi3 Users

 - **Batch Size Limitation**: Due to a known issue, batch sizes for some Stable Diffusion models need to be reduced.
   This issue is expected to be resolved in a future release.

- **Image-to-Video ControlNet**: The Image-to-Video ControlNet command is currently not supported on Gaudi3.
