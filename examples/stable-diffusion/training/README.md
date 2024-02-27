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
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5\
 --output_dir=/tmp/stable_diffusion1_5 \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=4 \
 --throughput_warmup_steps=3
```

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


## Fine-Tuning

The `train_text_to_image_sdxl.py` script shows how to implement the fine-tuning of Stable Diffusion models on Habana Gaudi.

### Requirements

Install the requirements:
```bash
pip install -r requirements.txt
```

### Example for SDXL
We can launch the fine-tuning of SDXL model using:

```bash
python train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path stabilityai/sdxl-vae \
  --dataset_name lambdalabs/pokemon-blip-captions \
  --resolution 1024 \
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
  --use_hpu_graphs \
  --bf16
```

### Example for LoRA SDXL

Low-Rank Adaption (LoRA) allows adapting a pretrained model by adding pairs of rank-decomposition matrices to
existing weights and only training those newly added weights.

We can launch the LoRA based fine-tuning of SDXL model using:

```bash
python train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --dataset_name="lambdalabs/pokemon-blip-captions" \
  --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd-pokemon-model-lora-sdxl" \
  --finetuning_method="lora" \
  --gaudi_config_name="Habana/stable-diffusion" \
  --throughput_warmup_steps=3 \
  --use_hpu_graphs \
  --bf16
```

> [!NOTE]
> SDXL's VAE is known to suffer from numerical instability issues. This is why we also expose a CLI argument namely `--pretrained_vae_model_name_or_path` that lets you specify the location of a better VAE (such as this one).

#### LoRA SDXL Inference

Once you have trained a LoRA weights as in the example above, inference can be done
by using the `GaudiStableDiffusionXLPipeline`.

```python
import torch
from optimum.habana.diffusers import (
    GaudiStableDiffusionXLPipeline,
    GaudiEulerDiscreteScheduler,
)

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lora_model_id = "sd-pokemon-model-lora-sdxl"
pipe = GaudiStableDiffusionXLPipeline.from_pretrained(
  model_id,
  scheduler=GaudiEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler"),
  torch_dtype=torch.bfloat16,
  use_habana=True,
  use_hpu_graphs=True,
  gaudi_config="Habana/stable-diffusion",
)
pipe.load_lora_weights(lora_model_id)

prompt = "cute dragon creature"
image = pipe(prompt).images[0]
image.save("green-pokemon.png")
```
