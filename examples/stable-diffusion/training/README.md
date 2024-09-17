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
  --bf16 \
  --trust_remote_code
```


### Inference

Once you have trained a model as described right above, inference can be done simply using the `GaudiStableDiffusionPipeline`. Make sure to include the `placeholder_token` in your prompt.

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
  --bf16 \
  --use_hpu_graphs_for_training \
  --use_hpu_graphs_for_inference \
  --validation_prompt="a cute naruto creature" \
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
  --bf16 \
  --use_hpu_graphs_for_training \
  --use_hpu_graphs_for_inference \
  --validation_prompt="a cute naruto creature" \
  --validation_epochs 48 \
  --checkpointing_steps 336 \
  --mediapipe dataset_sdxl_mediapipe \
  --adjust_throughput
```

### Single-card Training on Gaudi1
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
  --bf16
```

> [!NOTE]
> There is a known issue that in the first 2 steps, graph compilation takes longer than 10 seconds. This will be fixed in a future release.

> [!NOTE]
> `--mediapipe` only works on Gaudi2.


## DreamBooth
DreamBooth is a method to personalize text-to-image models like Stable Diffusion given just a few (3~5) images of a subject. The `train_dreambooth.py` script shows how to implement the training procedure and adapt it for Stable Diffusion.

### Dog toy example

Now let's get our dataset. For this example we will use some dog images: https://huggingface.co/datasets/diffusers/dog-example.

Let's first download it locally:

```python
from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

### Full model finetune
And launch the multi-card training using:
```bash

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="out"

python ../../gaudi_spawn.py --world_size 8 --use_mpi train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
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
Prior-preservation is used to avoid overfitting and language-drift. Refer to the paper to learn more about it. For prior-preservation we first generate images using the model with a class prompt and then use those during training along with our data.
According to the paper, it's recommended to generate `num_epochs * num_samples` images for prior-preservation. 200-300 works well for most cases. The `num_class_images` flag sets the number of images to generate with the class prompt. You can place existing images in `class_data_dir`, and the training script will generate any additional images so that `num_class_images` are present in `class_data_dir` during training time.

### PEFT model finetune
We provide example for dreambooth to use lora/lokr/loha/oft to finetune unet or text encoder.

**___Note: When using peft method we can use a much higher learning rate compared to vanilla dreambooth. Here we
use *1e-4* instead of the usual *5e-6*.___**

Launch the multi-card training using:
```bash

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="out"

python ../../gaudi_spawn.py --world_size 8 --use_mpi train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
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
Similar command could be applied to loha, lokr, oft.
You could check each adapter specific args by "--help", like you could use following command to check oft specific args.

```bash
python3 train_dreambooth.py oft --help

```

**___Note: oft could not work with hpu graphs mode. since "torch.inverse" need to fallback to cpu.
there's error like "cpu fallback is not supported during hpu graph capturing"___**


You could use text_to_image_generation.py to generate picture using the peft adapter like

```bash
python ../text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4  \
    --prompts "a sks dog" \
    --num_images_per_prompt 5 \
    --batch_size 1 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --unet_adapter_name_or_path out/unet \
    --gaudi_config Habana/stable-diffusion \
    --bf16
```

### DreamBooth training example for Stable Diffusion XL
You could use the dog images as example as well.
You can launch training using:
```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="lora-trained-xl"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

python ../../gaudi_spawn.py --world_size 8 --use_mpi train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
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

You could use text_to_image_generation.py to generate picture using the peft adapter like

```bash
python ../text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0  \
    --prompts "A picture of a sks dog in a bucket" \
    --num_images_per_prompt 5 \
    --batch_size 1 \
    --image_save_dir /tmp/stable_diffusion_xl_images \
    --use_habana \
    --use_hpu_graphs \
    --lora_id  lora-trained-xl \
    --gaudi_config Habana/stable-diffusion \
    --bf16
```
