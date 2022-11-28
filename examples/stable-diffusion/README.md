# Stable Diffusion

This directory contains a script that showcases how to perform text-to-image generation using Stable Diffusion on Habana Gaudi.

Stable Diffusion was proposed in [Stable Diffusion Announcement](https://stability.ai/blog/stable-diffusion-announcement) by Patrick Esser and Robin Rombach and the Stability AI team.


## Text-to-image Generation

### Single Prompt

Here is how to generate images with one prompt:
```python
python text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4 \
    --prompts "An image of a squirrel in Picasso style" \
    --num_images_per_prompt 20 \
    --batch_size 4 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion
```

> Lazy mode or HPU graphs are recommended when generating images by batches to get the fastest possible generations.
> The first batch of images entails a performance penalty. All subsequent batches will be generated much faster.
> You can enable these mode with `--use_lazy_mode` or `--use_hpu_graphs`.


### Multiple Prompts

Here is how to generate images with several prompts:
```python
python text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4 \
    --prompts "An image of a squirrel in Picasso style" "A shiny flying horse taking off" \
    --num_images_per_prompt 20 \
    --batch_size 8 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion
```

> Lazy mode or HPU graphs are recommended when generating images by batches to get the fastest possible generations.
> The first batch of images entails a performance penalty. All subsequent batches will be generated much faster.
> You can enable these mode with `--use_lazy_mode` or `--use_hpu_graphs`.


## Stable Diffusion 2

[Stable Diffusion 2](https://huggingface.co/docs/diffusers/v0.9.0/en/api/pipelines/stable_diffusion_2) can also be used to generate images with this script. Here is an example for a single prompt:

```python
python text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-2-base \
    --prompts "An image of a squirrel in Picasso style" \
    --num_images_per_prompt 20 \
    --batch_size 4 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion
```
