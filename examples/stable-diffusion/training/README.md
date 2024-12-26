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

# Stable Diffusion Training Examples

This directory contains scripts that showcase how to perform training/fine-tuning of Stable Diffusion models on Habana Gaudi.


## Textual Inversion

[Textual Inversion](https://arxiv.org/abs/2208.01618) is a method to personalize text2image models like Stable Diffusion on your own images using just 3-5 examples.
The `textual_inversion.py` script shows how to implement the training procedure on Habana Gaudi.


### Cat Toy Example

In the examples below, we will use a set of cat images from the following dataset:
[https://huggingface.co/datasets/diffusers/cat_toy_example](https://huggingface.co/datasets/diffusers/cat_toy_example)

Let's first download this dataset locally:

```python
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil

local_dir = './cat'
snapshot_download(
    'diffusers/cat_toy_example',
    local_dir=local_dir,
    repo_type='dataset',
    ignore_patterns='.gitattributes',
)
cache_dir = Path(local_dir, '.cache')
if cache_dir.is_dir():
    shutil.rmtree(cache_dir)
```

This will be our training data.
Now we can launch the training using:

```bash
python textual_inversion.py \
  --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
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

The following example shows how to run inference using the fine-tuned model:

```python
from optimum.habana.diffusers import GaudiStableDiffusionPipeline
import torch

model_id = "/tmp/textual_inversion_cat"
pipe = GaudiStableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    use_habana=True,
    use_hpu_graphs=True,
    gaudi_config="Habana/stable-diffusion",
)

prompt = "A <cat-toy> backpack"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save(f"cat-backpack.png")
```

> Change `--resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.

> As described in [the official paper](https://arxiv.org/abs/2208.01618), only one embedding vector is used for the placeholder token, *e.g.* `"<cat-toy>"`.
> However, one can also add multiple embedding vectors for the placeholder token to increase the number of fine-tuneable parameters.
> This can help the model to learn more complex details. To use multiple embedding vectors, you can define `--num_vectors` to a number larger than one,
> *e.g.*: `--num_vectors 5`. The saved textual inversion vectors will then be larger in size compared to the default case.


## Textual Inversion XL

The `textual_inversion_sdxl.py` script shows how to implement textual inversion fine-tuning on Gaudi for XL diffusion models
such as `stabilityai/stable-diffusion-xl-base-1.0` or `cagliostrolab/animagine-xl-3.1` for example.

Assuming the afforemenioned cat toy dataset has been obtained, we can launch textual inversion XL training using:

```bash
python textual_inversion_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --train_data_dir ./cat \
  --learnable_property object \
  --placeholder_token "<cat-toy>" \
  --initializer_token toy \
  --resolution 768 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 500 \
  --learning_rate 5.0e-04 \
  --scale_lr \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --output_dir /tmp/textual_inversion_cat_sdxl \
  --save_as_full_pipeline \
  --gaudi_config_name Habana/stable-diffusion \
  --throughput_warmup_steps 3
```

> As described in [the official paper](https://arxiv.org/abs/2208.01618), only one embedding vector is used for the placeholder token, *e.g.* `"<cat-toy>"`.
> However, one can also add multiple embedding vectors for the placeholder token to increase the number of fine-tuneable parameters.
> This can help the model to learn more complex details. To use multiple embedding vectors, you can define `--num_vectors` to a number larger than one,
> *e.g.*: `--num_vectors 5`. The saved textual inversion vectors will then be larger in size compared to the default case.

The script also supports training of both text encoders of SDXL, so inference can be executed by inserting a placeholder token into one or both prompts.
The following example shows how to run inference using the fine tuned-model with both text encoders, separately and in combination:

```python
from optimum.habana.diffusers import GaudiStableDiffusionXLPipeline
import torch

model_id = "/tmp/textual_inversion_cat_sdxl"
pipe = GaudiStableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    use_habana=True,
    use_hpu_graphs=True,
    gaudi_config="Habana/stable-diffusion",
)

prompt = "A <cat-toy> backpack"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save(f"cat-backpack.png")

image = pipe(prompt="", prompt_2=prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save(f"cat-backpack_p2.png")

prompt_2 = "A <cat-toy> colored backpack"
image = pipe(prompt=prompt, prompt_2=prompt_2, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save(f"cat-backpack_p1and2.png")
```

> [!NOTE]
> Change `--resolution` to 768 if you are using [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.

> [!NOTE]
> As described in [the official paper](https://arxiv.org/abs/2208.01618), only one embedding vector is used for the placeholder token,
> e.g. `"<cat-toy>"`. However, one can also add multiple embedding vectors for the placeholder token to increase the number of fine-tuneable
> parameters. This can help the model to learn more complex details. To use multiple embedding vectors, you can define `--num_vectors` to
> a number larger than one, e.g.: `--num_vectors 5`. The saved textual inversion vectors will then be larger in size compared to the default case.


## ControlNet Training

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models ](https://huggingface.co/papers/2302.05543)
by Lvmin Zhang and Maneesh Agrawala. It is a type of model for controlling StableDiffusion by conditioning the model with an additional input image.
This example is adapted from [controlnet example in the diffusers repository](https://github.com/huggingface/diffusers/tree/main/examples/controlnet#training).

First, download the conditioning images as shown below:

```bash
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

Then proceed to training with command:

```bash
python train_controlnet.py \
 --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4\
 --output_dir=/tmp/stable_diffusion1_5 \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=4 \
 --throughput_warmup_steps=3 \
 --use_hpu_graphs \
 --sdp_on_bf16 \
 --bf16 \
 --trust_remote_code
```

### Multi-card Run

You can run these fine-tuning scripts in a distributed fashion as follows:
```bash
python ../../gaudi_spawn.py --use_mpi --world_size 8 train_controlnet.py \
  --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
  --output_dir=/tmp/stable_diffusion1_5 \
  --dataset_name=fusing/fill50k \
  --resolution=512 \
  --learning_rate=1e-5 \
  --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
  --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
  --train_batch_size=4 \
  --throughput_warmup_steps 3 \
  --use_hpu_graphs \
  --sdp_on_bf16 \
  --bf16 \
  --trust_remote_code
```


### Inference

Once you have trained a model as described right above, inference can be done simply using the `GaudiStableDiffusionPipeline`.
Make sure to include the `placeholder_token` in your prompt.

```python
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
from optimum.habana.diffusers import GaudiStableDiffusionControlNetPipeline

base_model_path = "CompVis/stable-diffusion-v1-4"
controlnet_path = "/tmp/stable_diffusion1_5"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.bfloat16)
pipe = GaudiStableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.bfloat16,
    use_habana=True,
    use_hpu_graphs=True,
    gaudi_config="Habana/stable-diffusion",
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

control_image = load_image("./conditioning_image_1.png")
prompt = "pale golden rod circle with old lace background"

# generate image
generator = torch.manual_seed(0)
image = pipe(
    prompt, num_inference_steps=20, generator=generator, image=control_image
).images[0]
image.save("./output.png")
```


## Fine-Tuning for Stable Diffusion XL

The `train_text_to_image_sdxl.py` script shows how to implement the fine-tuning of Stable Diffusion XL models on Gaudi.

### Requirements

Install the requirements:
```bash
pip install -r requirements.txt
```

### Single-card Training

To train Stable Diffusion XL on a single Gaudi card, use:
```bash
python train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
  --dataset_name lambdalabs/naruto-blip-captions \
  --resolution 512 \
  --crop_resolution 512 \
  --center_crop \
  --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size 16 \
  --max_train_steps 2500 \
  --learning_rate 1e-05 \
  --max_grad_norm 1 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --output_dir sdxl_model_output \
  --gaudi_config_name Habana/stable-diffusion \
  --throughput_warmup_steps 3 \
  --dataloader_num_workers 8 \
  --sdp_on_bf16 \
  --bf16 \
  --use_hpu_graphs_for_training \
  --use_hpu_graphs_for_inference \
  --validation_prompt="a cute naruto creature" \
  --validation_epochs 48 \
  --checkpointing_steps 2500 \
  --logging_step 10 \
  --adjust_throughput
```


### Multi-Card Training

To train Stable Diffusion XL on a multi-card Gaudi system, use:
```bash
PT_HPU_RECIPE_CACHE_CONFIG=/tmp/stdxl_recipe_cache,True,1024  \
python ../../gaudi_spawn.py --world_size 8 --use_mpi train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
  --dataset_name lambdalabs/naruto-blip-captions \
  --resolution 512 \
  --crop_resolution 512 \
  --center_crop \
  --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size 16 \
  --max_train_steps 336 \
  --learning_rate 1e-05 \
  --max_grad_norm 1 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --output_dir sdxl_model_output \
  --gaudi_config_name Habana/stable-diffusion \
  --throughput_warmup_steps 3 \
  --dataloader_num_workers 8 \
  --sdp_on_bf16 \
  --bf16 \
  --use_hpu_graphs_for_training \
  --use_hpu_graphs_for_inference \
  --validation_prompt="a cute naruto creature" \
  --validation_epochs 48 \
  --checkpointing_steps 336 \
  --mediapipe dataset_sdxl_mediapipe \
  --adjust_throughput
```

### Single-Card Training on Gaudi1

To train Stable Diffusion XL on a single Gaudi1 card, use:
```bash
python train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
  --dataset_name lambdalabs/naruto-blip-captions \
  --resolution 256 \
  --center_crop \
  --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 3000 \
  --learning_rate 1e-05 \
  --max_grad_norm 1 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --output_dir sdxl_model_output \
  --gaudi_config_name Habana/stable-diffusion \
  --throughput_warmup_steps 3 \
  --use_hpu_graphs_for_training \
  --use_hpu_graphs_for_inference \
  --checkpointing_steps 3000 \
  --sdp_on_bf16 \
  --bf16
```

> [!NOTE]
> There is a known issue that in the first 2 steps, graph compilation takes longer than 10 seconds.
> This will be fixed in a future release.

> [!NOTE]
> `--mediapipe` only works on Gaudi2.


## DreamBooth

DreamBooth is a technique for personalizing text-to-image models like Stable Diffusion using only a few images (typically 3-5)
of a specific subject. The `train_dreambooth.py` script demonstrates how to implement this training process and adapt it for
Stable Diffusion.

### Dog Toy Example

For DreamBooth examples we will use a set of dog images from the following dataset:
[https://huggingface.co/datasets/diffusers/dog-example](https://huggingface.co/datasets/diffusers/dog-example).

Let's first download this dataset locally:

```python
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil

local_dir = './dog'
snapshot_download(
    'diffusers/dog-example',
    local_dir=local_dir,
    repo_type='dataset',
    ignore_patterns='.gitattributes',
)
cache_dir = Path(local_dir, '.cache')
if cache_dir.is_dir():
    shutil.rmtree(cache_dir)
```

### Full Model Fine-Tuning

To launch the multi-card Stable Diffusion training, use:
```bash
python ../../gaudi_spawn.py --world_size 8 --use_mpi train_dreambooth.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="dog" \
  --output_dir="dog_sd" \
  --class_data_dir="path-to-class-images" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --num_class_images=200 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --mixed_precision=bf16 \
  --use_hpu_graphs_for_training \
  --use_hpu_graphs_for_inference \
  --gaudi_config_name Habana/stable-diffusion \
  full
```

Prior preservation is used to prevent overfitting and language drift. For more details, refer to the original paper.
In this process, we first generate images using the model with a class prompt and then use those images during training
alongside our data. According to the paper, it's recommended to generate `num_epochs * num_samples` images for prior
preservation, with 200-300 images being effective in most cases. The `num_class_images` flag controls how many images
are generated with the class prompt. You can place existing images in the `class_data_dir`, and the training script will
generate any additional images needed to meet the `num_class_images` requirement during training.

### PEFT Model Fine-Tuning

We provide DreamBooth examples demonstrating how to use LoRA, LoKR, LoHA, and OFT adapters to fine-tune the
UNet or text encoder.

To run the multi-card training, use:
```bash
python ../../gaudi_spawn.py --world_size 8 --use_mpi train_dreambooth.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir="dog" \
  --output_dir="dog_sd" \
  --class_data_dir="path-to-class-images" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --num_class_images=200 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --mixed_precision=bf16 \
  --use_hpu_graphs_for_training \
  --use_hpu_graphs_for_inference \
  --gaudi_config_name Habana/stable-diffusion \
  lora --unet_r 8 --unet_alpha 8
```
> [!NOTE]
> When using PEFT method we can use a much higher learning rate compared to vanilla dreambooth.
> Here we use `1e-4` instead of the usual `5e-6`

Similar command could be applied with `loha`, `lokr`, or `oft` adapters.

You could check each adapter's specific arguments with `--help`, for example:

```bash
python3 train_dreambooth.py oft --help
```
> [!NOTE]
> Currently, the `oft` adapter is not supported in HPU graph mode, as it triggers `torch.inverse`,
> causing a CPU fallback that is incompatible with HPU graph capturing.

After training completes, you can use `text_to_image_generation.py` sample for inference as follows:

```bash
python ../text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4  \
    --unet_adapter_name_or_path dog_sd/unet \
    --prompts "a sks dog" \
    --num_images_per_prompt 5 \
    --batch_size 1 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

### DreamBooth LoRA Fine-Tuning with Stable Diffusion XL

We can use the same `dog` dataset for the following examples.

To launch Stable Diffusion XL LoRA training on a multi-card Gaudi system, use:"
```bash
python train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  \
  --instance_data_dir="dog" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --output_dir="lora-trained-xl" \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed=0 \
  --use_hpu_graphs_for_inference \
  --use_hpu_graphs_for_training \
  --gaudi_config_name Habana/stable-diffusion
```

To launch Stable Diffusion XL LoRA training on a multi-card Gaudi system, use:"
```bash
python ../../gaudi_spawn.py --world_size 8 --use_mpi train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  \
  --instance_data_dir="dog" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --output_dir="lora-trained-xl" \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed=0 \
  --use_hpu_graphs_for_inference \
  --use_hpu_graphs_for_training \
  --gaudi_config_name Habana/stable-diffusion
```
> [!NOTE]
> To use DeepSpeed instead of MPI, replace `--use_mpi` with `--deepspeed` in the previous example

After training completes, you can run inference with a simple python script like this:
```python
import torch
from optimum.habana import GaudiConfig
from optimum.habana.diffusers import GaudiStableDiffusionXLPipeline

pipe = GaudiStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,
    use_hpu_graphs=True,
    use_habana=True,
    gaudi_config="Habana/stable-diffusion",
)
pipe.load_lora_weights("lora-trained-xl")

prompt = "A photo of sks dog in a bucket"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=30,
    max_sequence_length=512,
).images[0]
image.save("sdxl-lora.png")
```

Alternatively, you could directly use `text_to_image_generation.py` sample for inference as follows:
```bash
python ../text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0  \
    --lora_id lora-trained-xl \
    --prompts "A picture of a sks dog in a bucket" \
    --num_images_per_prompt 5 \
    --batch_size 1 \
    --image_save_dir /tmp/stable_diffusion_xl_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --bf16
```

### DreamBooth LoRA Fine-Tuning with FLUX.1-dev

We can use the same `dog` dataset for the following examples.

To launch FLUX.1-dev LoRA training on a single Gaudi card, use:"
```bash
python train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --dataset="dog" \
  --prompt="a photo of sks dog" \
  --output_dir="dog_lora_flux" \
  --mixed_precision="bf16" \
  --weighting_scheme="none" \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --guidance_scale=1 \
  --report_to="tensorboard" \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --cache_latents \
  --rank=4 \
  --max_train_steps=500 \
  --seed="0" \
  --use_hpu_graphs_for_inference \
  --use_hpu_graphs_for_training \
  --gaudi_config_name="Habana/stable-diffusion"
```

To launch FLUX.1-dev LoRA training on a multi-card Gaudi system, use:"
```bash
python ../../gaudi_spawn.py --world_size 8 --use_mpi train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --dataset="dog" \
  --prompt="a photo of sks dog" \
  --output_dir="dog_lora_flux" \
  --mixed_precision="bf16" \
  --weighting_scheme="none" \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --guidance_scale=1 \
  --report_to="tensorboard" \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --cache_latents \
  --rank=4 \
  --max_train_steps=500 \
  --seed="0" \
  --use_hpu_graphs_for_inference \
  --use_hpu_graphs_for_training \
  --gaudi_config_name="Habana/stable-diffusion"
```
> [!NOTE]
> To use DeepSpeed instead of MPI, replace `--use_mpi` with `--use_deepspeed` in the previous example

After training completes, you can run inference on Gaudi system with a simple python script like this:
```python
import torch
from optimum.habana import GaudiConfig
from optimum.habana.diffusers import GaudiFluxPipeline

pipe = GaudiFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    use_hpu_graphs=True,
    use_habana=True,
    gaudi_config="Habana/stable-diffusion",
)
pipe.load_lora_weights("dog_lora_flux")

prompt = "A photo of sks dog in a bucket"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=30,
).images[0]
image.save("flux-dev.png")
```

Alternatively, you could directly use `text_to_image_generation.py` sample for inference as follows:
```bash
python ../text_to_image_generation.py \
    --model_name_or_path "black-forest-labs/FLUX.1-dev" \
    --lora_id dog_lora_flux \
    --prompts "A picture of a sks dog in a bucket" \
    --num_images_per_prompt 5 \
    --batch_size 1 \
    --image_save_dir /tmp/flux_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```
