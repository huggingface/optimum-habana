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


## ControlNet Training

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models ](https://huggingface.co/papers/2302.05543) by Lvmin Zhang and Maneesh Agrawala. It is a type of model for controlling StableDiffusion by conditioning the model with an additional input image.
This example is adapted from [controlnet example in the diffusers repository](https://github.com/huggingface/diffusers/tree/main/examples/controlnet#training).

First, download the conditioning images as shown below:

```bash
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

Then proceed to training with command:

```bash
python train_controlnet.py \
 --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5\
 --output_dir=/tmp/stable_diffusion1_5 \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=4 \
 --throughput_warmup_steps=3 \
 --use_hpu_graphs \
 --bf16
```

### Multi-card Run

You can run these fine-tuning scripts in a distributed fashion as follows:
```bash
python ../../gaudi_spawn.py --use_mpi --world_size 8 train_controlnet.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --output_dir=/tmp/stable_diffusion1_5 \
  --dataset_name=fusing/fill50k \
  --resolution=512 \
  --learning_rate=1e-5 \
  --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
  --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
  --train_batch_size=4 \
  --throughput_warmup_steps 3 \
  --use_hpu_graphs \
  --bf16
```


### Inference

Once you have trained a model as described right above, inference can be done simply using the `GaudiStableDiffusionPipeline`. Make sure to include the `placeholder_token` in your prompt.

```python
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
from optimum.habana.diffusers import GaudiStableDiffusionControlNetPipeline

base_model_path = "runwayml/stable-diffusion-v1-5"
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

The `train_text_to_image_sdxl.py` script shows how to implement the fine-tuning of Stable Diffusion models on Habana Gaudi.

### Requirements

Install the requirements:
```bash
pip install -r requirements.txt
```

### Single-card Training

```bash
python train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path stabilityai/sdxl-vae \
  --dataset_name lambdalabs/pokemon-blip-captions \
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
  --output_dir sdxl-pokemon-model \
  --gaudi_config_name Habana/stable-diffusion \
  --throughput_warmup_steps 3 \
  --dataloader_num_workers 8 \
  --bf16 \
  --use_hpu_graphs_for_training \
  --use_hpu_graphs_for_inference \
  --validation_prompt="a robotic cat with wings" \
  --validation_epochs 48 \
  --checkpointing_steps 2500 \
  --logging_step 10 \
  --adjust_throughput
```


### Multi-card Training
```bash
PT_HPU_RECIPE_CACHE_CONFIG=/tmp/stdxl_recipe_cache,True,1024  \
python ../../gaudi_spawn.py --world_size 8 --use_mpi train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path stabilityai/sdxl-vae \
  --dataset_name lambdalabs/pokemon-blip-captions \
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
  --output_dir sdxl-pokemon-model \
  --gaudi_config_name Habana/stable-diffusion \
  --throughput_warmup_steps 3 \
  --dataloader_num_workers 8 \
  --bf16 \
  --use_hpu_graphs_for_training \
  --use_hpu_graphs_for_inference \
  --validation_prompt="a robotic cat with wings" \
  --validation_epochs 48 \
  --checkpointing_steps 336 \
  --mediapipe dataset_sdxl_pokemon \
  --adjust_throughput
```

### Single-card Training on Gaudi1
```bash
PT_HPU_MAX_COMPOUND_OP_SIZE=5 python train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path stabilityai/sdxl-vae \
  --dataset_name lambdalabs/pokemon-blip-captions \
  --resolution 512 \
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
  --output_dir sdxl-pokemon-model \
  --gaudi_config_name Habana/stable-diffusion \
  --throughput_warmup_steps 3 \
  --use_hpu_graphs_for_training \
  --use_hpu_graphs_for_inference \
  --bf16
```

> [!NOTE]
> There is a known issue that in the first 2 steps, graph compilation takes longer than 10 seconds. This will be fixed in a future release.

> [!NOTE]
> `--mediapipe` only works on Gaudi2.
