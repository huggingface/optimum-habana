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
A [demo](https://huggingface.co/spaces/Intel/ldm3d) is also available. Here is how to run this model:

```python
python text_to_image_generation.py \
    --model_name_or_path "Intel/ldm3d-4c" \
    --prompts "An image of a squirrel in Picasso style" \
    --num_images_per_prompt 10 \
    --batch_size 2 \
    --height 768 \
    --width 768 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion-2 \
    --ldm3d
```

> There are three different checkpoints for LDM3D:
> - use [original checkpoint](https://huggingface.co/Intel/ldm3d) to generate outputs from the paper
> - use [the latest checkpoint](https://huggingface.co/Intel/ldm3d-4c) for generating improved results
> - use [the pano checkpoint](https://huggingface.co/Intel/ldm3d-pano) to generate panoramic view

### Stable Diffusion XL (SDXL)

Stable Diffusion XL was proposed in [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/pdf/2307.01952.pdf) by the Stability AI team.

Here is how to generate SDXL images with a single prompt:
```python
python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --prompts "Sailing ship painting by Van Gogh" \
    --num_images_per_prompt 20 \
    --batch_size 4 \
    --image_save_dir /tmp/stable_diffusion_xl_images \
    --scheduler euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --bf16
```

> HPU graphs are recommended when generating images by batches to get the fastest possible generations.
> The first batch of images entails a performance penalty. All subsequent batches will be generated much faster.
> You can enable this mode with `--use_hpu_graphs`.

Here is how to generate SDXL images with several prompts:
```python
python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --prompts "Sailing ship painting by Van Gogh" "A shiny flying horse taking off" \
    --num_images_per_prompt 20 \
    --batch_size 8 \
    --image_save_dir /tmp/stable_diffusion_xl_images \
    --scheduler euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --bf16
```

SDXL combines a second text encoder (OpenCLIP ViT-bigG/14) with the original text encoder to significantly
increase the number of parameters. Here is how to generate images with several prompts for both `prompt`
and `prompt_2` (2nd text encoder), as well as their negative prompts:
```python
python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --prompts "Sailing ship painting by Van Gogh" "A shiny flying horse taking off" \
    --prompts_2 "Red tone" "Blue tone" \
    --negative_prompts "Low quality" "Sketch" \
    --negative_prompts_2 "Clouds" "Clouds" \
    --num_images_per_prompt 20 \
    --batch_size 8 \
    --image_save_dir /tmp/stable_diffusion_xl_images \
    --scheduler euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --bf16
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
    --num_images_per_prompt 20 \
    --batch_size 8 \
    --image_save_dir /tmp/stable_diffusion_xl_turbo_images \
    --scheduler euler_ancestral_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --bf16 \
    --num_inference_steps 1 \
    --guidance_scale 0.0 \
    --timestep_spacing trailing
```

> HPU graphs are recommended when generating images by batches to get the fastest possible generations.
> The first batch of images entails a performance penalty. All subsequent batches will be generated much faster.
> You can enable this mode with `--use_hpu_graphs`.

### ControlNet

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models ](https://huggingface.co/papers/2302.05543) by Lvmin Zhang and Maneesh Agrawala.
It is a type of model for controlling StableDiffusion by conditioning the model with an additional input image.

Here is how to generate images conditioned by canny edge model:
```bash
pip install -r requirements.txt
python text_to_image_generation.py \
    --model_name_or_path runwayml/stable-diffusion-v1-5 \
    --controlnet_model_name_or_path lllyasviel/sd-controlnet-canny \
    --prompts "futuristic-looking woman" \
    --control_image https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png \
    --num_images_per_prompt 20 \
    --batch_size 4 \
    --image_save_dir /tmp/controlnet_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --bf16
```

Here is how to generate images conditioned by canny edge model and with multiple prompts:
```bash
pip install -r requirements.txt
python text_to_image_generation.py \
    --model_name_or_path runwayml/stable-diffusion-v1-5 \
    --controlnet_model_name_or_path lllyasviel/sd-controlnet-canny \
    --prompts "futuristic-looking woman" "a rusty robot" \
    --control_image https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png \
    --num_images_per_prompt 10 \
    --batch_size 4 \
    --image_save_dir /tmp/controlnet_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --bf16
```

Here is how to generate images conditioned by open pose model:
```bash
pip install -r requirements.txt
python text_to_image_generation.py \
    --model_name_or_path runwayml/stable-diffusion-v1-5 \
    --controlnet_model_name_or_path lllyasviel/sd-controlnet-openpose \
    --prompts "Chef in the kitchen" \
    --control_image https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/pose.png \
    --control_preprocessing_type "none" \
    --num_images_per_prompt 20 \
    --batch_size 4 \
    --image_save_dir /tmp/controlnet_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --bf16
```

Here is how to generate images with conditioned by canny edge model using Stable Diffusion 2
```bash
pip install -r requirements.txt
python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-2-1 \
    --controlnet_model_name_or_path thibaud/controlnet-sd21-canny-diffusers \
    --control_image https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png \
    --control_preprocessing_type "none" \
    --prompts "bird" \
    --seed 0 \
    --num_images_per_prompt 10 \
    --batch_size 2 \
    --image_save_dir /tmp/controlnet-2-1_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion-2
```

## Textual Inversion

[Textual Inversion](https://arxiv.org/abs/2208.01618) is a method to personalize text2image models like Stable Diffusion on your own images using just 3-5 examples.
The `textual_inversion.py` script shows how to implement the training procedure on Habana Gaudi.


### Cat toy example

Let's get our dataset. For this example, we will use some cat images: https://huggingface.co/datasets/diffusers/cat_toy_example .

Let's first download it locally:

```py
from huggingface_hub import snapshot_download

local_dir = "./cat"
snapshot_download("diffusers/cat_toy_example", local_dir=local_dir, repo_type="dataset", ignore_patterns=".gitattributes")
```

This will be our training data.
Now we can launch the training using:

```bash
python textual_inversion.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --train_data_dir ./cat \
  --learnable_property object \
  --placeholder_token "<cat-toy>" \
  --initializer_token toy \
  --resolution 512 \
  --train_batch_size 4 \
  --max_train_steps 3000 \
  --learning_rate 5.0e-04 \
  --scale_lr \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --output_dir /tmp/textual_inversion_cat \
  --save_as_full_pipeline \
  --gaudi_config_name Habana/stable-diffusion \
  --throughput_warmup_steps 3
```

> Change `--resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.

> As described in [the official paper](https://arxiv.org/abs/2208.01618), only one embedding vector is used for the placeholder token, *e.g.* `"<cat-toy>"`. However, one can also add multiple embedding vectors for the placeholder token to increase the number of fine-tuneable parameters. This can help the model to learn more complex details. To use multiple embedding vectors, you can define `--num_vectors` to a number larger than one, *e.g.*: `--num_vectors 5`. The saved textual inversion vectors will then be larger in size compared to the default case.


### Multi-card Run

You can run this fine-tuning script in a distributed fashion as follows:
```bash
python ../gaudi_spawn.py --use_mpi --world_size 8 textual_inversion.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --train_data_dir ./cat \
  --learnable_property object \
  --placeholder_token '"<cat-toy>"' \
  --initializer_token toy \
  --resolution 512 \
  --train_batch_size 4 \
  --max_train_steps 375 \
  --learning_rate 5.0e-04 \
  --scale_lr \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --output_dir /tmp/textual_inversion_cat \
  --save_as_full_pipeline \
  --gaudi_config_name Habana/stable-diffusion \
  --throughput_warmup_steps 3
```


### Inference

Once you have trained a model as described right above, inference can be done simply using the `GaudiStableDiffusionPipeline`. Make sure to include the `placeholder_token` in your prompt.

```python
import torch
from optimum.habana.diffusers import GaudiStableDiffusionPipeline

model_id = "path-to-your-trained-model"
pipe = GaudiStableDiffusionPipeline.from_pretrained(
  model_id,
  torch_dtype=torch.bfloat16,
  use_habana=True,
  use_hpu_graphs=True,
  gaudi_config="Habana/stable-diffusion",
)

prompt = "A <cat-toy> backpack"

image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("cat-backpack.png")
```
