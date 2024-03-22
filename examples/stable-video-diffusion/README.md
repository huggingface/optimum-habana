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

# Stable Video Diffusion XT Examples

This directory contains a script that showcases how to perform image-to-video generation using Stable Video Diffusion on Habana Gaudi.

Stable Video Diffusion (SVD) was unveiled in [Stable Video Diffusion Announcement](https://stability.ai/news/stable-video-diffusion-open-ai-video-model) by the Stability AI team.

Stable Video Diffusion XT version (SVD-XT) is tuned to generate 25 frames of video from a single image.


## Image-to-video Generation

### Single Image Prompt

Here is how to generate video with one image prompt:
```bash
python image_to_video_generation.py \
    --model_name_or_path "stabilityai/stable-video-diffusion-img2vid-xt" \
    --image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png" \
    --num_videos_per_prompt 1 \
    --video_save_dir /tmp/stable_video_diffusion_xt \
    --save_frames_as_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --bf16
```

### Multiple Image Prompts

Here is how to generate videos with several image prompts:
```bash
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
    --bf16
```
