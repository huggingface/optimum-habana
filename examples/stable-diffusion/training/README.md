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

## Textual Inversion XL

The `textual_inversion_sdxl.py` script shows how to implement textual inversion fine-tuning on Gaudi for XL diffusion models
such as `stabilityai/stable-diffusion-xl-base-1.0` or `cagliostrolab/animagine-xl-3.1` for example.

For this example we will use a set of cat toy images from the following dataset:
[https://huggingface.co/datasets/diffusers/cat_toy_example](https://huggingface.co/datasets/diffusers/cat_toy_example).

To download this and other example training datasets locally, run:

```bash
python download_train_datasets.py
```

Assuming the afforemenioned cat toy dataset has been obtained, we can launch textual inversion XL training using:

```bash
PT_HPU_LAZY_MODE=1 python textual_inversion_sdxl.py \
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
    --sdp_on_bf16 \
    --throughput_warmup_steps 3
```

> [!NOTE]
> As described in [the official paper](https://arxiv.org/abs/2208.01618), only one embedding vector is used for the placeholder token,
> e.g. `"<cat-toy>"`. However, one can also add multiple embedding vectors for the placeholder token to increase the number of fine-tuneable
> parameters. This can help the model to learn more complex details. To use multiple embedding vectors, you can define `--num_vectors` to
> a number larger than one, e.g.: `--num_vectors 5`. The saved textual inversion vectors will then be larger in size compared to the default case.

The script also supports training of both text encoders of SDXL, so inference can be executed by inserting a placeholder token into one or both prompts.

For example, after training you can use `text_to_image_generation.py` sample to run inference with the fine-tuned model as follows:

```bash
PT_HPU_LAZY_MODE=1 python ../text_to_image_generation.py \
    --model_name_or_path /tmp/textual_inversion_cat_sdxl \
    --prompts "A <cat-toy> backpack" \
    --num_images_per_prompt 5 \
    --batch_size 1 \
    --image_save_dir /tmp/textual_inversion_cat_sdxl_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

## ControlNet Training

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models ](https://huggingface.co/papers/2302.05543)
by Lvmin Zhang and Maneesh Agrawala. It is a type of model for controlling StableDiffusion by conditioning the model with an additional input image.
This example is adapted from [controlnet example in the diffusers repository](https://github.com/huggingface/diffusers/tree/main/examples/controlnet#training).

To download the example conditioning images locally, run:
```bash
python download_train_datasets.py
```

Then proceed to training with command:

```bash
PT_HPU_LAZY_MODE=1 python train_controlnet.py \
   --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1 \
   --output_dir=/tmp/stable_diffusion2_1 \
   --dataset_name=fusing/fill50k \
   --resolution=512 \
   --learning_rate=1e-5 \
   --validation_image "./cnet/conditioning_image_1.png" "./cnet/conditioning_image_2.png" \
   --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
   --train_batch_size=4 \
   --throughput_warmup_steps=3 \
   --use_hpu_graphs \
   --sdp_on_bf16 \
   --bf16 \
   --max_train_steps 2500 \
   --trust_remote_code
```

You can run inference on multiple HPUs by replacing `python train_controlnet.py`
with `python ../../gaudi_spawn.py --world_size <num-HPUs> train_controlnet.py`.

### Inference

After training completes, you can use `text_to_image_generation.py` sample to run inference with the fine-tuned ControlNet model:

```bash
PT_HPU_LAZY_MODE=1 python ../text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-2-1 \
    --controlnet_model_name_or_path /tmp/stable_diffusion2_1 \
    --prompts "pale golden rod circle with old lace background" \
    --control_image "./cnet/conditioning_image_1.png" \
    --num_images_per_prompt 5 \
    --batch_size 1 \
    --image_save_dir /tmp/controlnet_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

## Fine-Tuning for Stable Diffusion XL

The `train_text_to_image_sdxl.py` script shows how to implement the fine-tuning of Stable Diffusion XL models on Gaudi.

### Requirements

Install the requirements:
```bash
pip install -r requirements.txt
```

### Single Card Training

To train Stable Diffusion XL on a single Gaudi card, use:

```bash
PT_HPU_LAZY_MODE=1 python train_text_to_image_sdxl.py \
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

> [!WARNING]
> There is a known issue that in the first 2 steps, graph compilation takes longer than 10 seconds. This will be fixed in a future release.

You can run inference on multiple HPUs by replacing `python train_text_to_image_sdxl.py`
with `PT_HPU_RECIPE_CACHE_CONFIG=/tmp/stdxl_recipe_cache,True,1024 python ../../gaudi_spawn.py --world_size <num-HPUs> train_text_to_image_sdxl.py`.

### Inference

After training is finished, you can run inference using `text_to_image_generation.py` script as follows:

```bash
PT_HPU_LAZY_MODE=1 python ../text_to_image_generation.py \
    --model_name_or_path sdxl_model_output \
    --prompts "a cute naruto creature" \
    --num_images_per_prompt 5 \
    --batch_size 1 \
    --image_save_dir /tmp/stable_diffusion_xl_images \
    --scheduler euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

## DreamBooth

DreamBooth is a technique for personalizing text-to-image models like Stable Diffusion using only a few images (typically 3-5)
of a specific subject. The `train_dreambooth.py` script demonstrates how to implement this training process and adapt it for
Stable Diffusion.

For DreamBooth examples we will use a set of dog images from the following dataset:
[https://huggingface.co/datasets/diffusers/dog-example](https://huggingface.co/datasets/diffusers/dog-example).

To download this and other example training datasets locally, run:

```bash
python download_train_datasets.py
```

### Full Model Fine-Tuning

To launch the multi-card Stable Diffusion training, use:

```bash
PT_HPU_LAZY_MODE=1 python ../../gaudi_spawn.py --world_size 8 --use_mpi train_dreambooth.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1"  \
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
    --sdp_on_bf16 \
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

We provide DreamBooth examples demonstrating how to use LoRA, LoKR, LoHA, OFT and BOFT adapters to fine-tune the
UNet or text encoder.

To run the multi-card training, use:

```bash
PT_HPU_LAZY_MODE=1 python ../../gaudi_spawn.py --world_size 8 --use_mpi train_dreambooth.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1"  \
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
    --sdp_on_bf16 \
    --gaudi_config_name Habana/stable-diffusion \
    lora --unet_r 8 --unet_alpha 8
```

> [!NOTE]
> When using PEFT method we can use a much higher learning rate compared to vanilla dreambooth.
> Here we use `1e-4` instead of the usual `5e-6`

Similar command could be applied with `loha`, `lokr`, `oft` or `boft` adapters.

You could check each adapter's specific arguments with `--help`, for example:

```bash
python train_dreambooth.py oft --help
```

> [!WARNING]
> Currently, the `oft` and `boft` adapter are not supported in HPU graph mode, as it triggers `torch.inverse`  `torch.linalg.solve`,

> causing a CPU fallback that is incompatible with HPU graph capturing.

After training completes, you can use `text_to_image_generation.py` sample for inference as follows:

```bash
PT_HPU_LAZY_MODE=1 python ../text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-2-1  \
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

To launch Stable Diffusion XL LoRA training on a single card Gaudi system, use:

```bash
PT_HPU_LAZY_MODE=1 python train_dreambooth_lora_sdxl.py \
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
    --sdp_on_bf16 \
    --gaudi_config_name Habana/stable-diffusion
```

> [!NOTE]
> You can run training on multiple HPUs by replacing `python train_dreambooth_lora_sdxl.py` with 
> `python ../../gaudi_spawn.py --world_size <num-HPUs> train_dreambooth_lora_sdxl.py`. To use MPI for multi-card training,
> add `--use_mpi` after `--world_size <num-HPUs>`. To use DeepSpeed instead of MPI, replace `--use_mpi` with `--use_deepspeed`.

After training is completed, you can directly use `text_to_image_generation.py` sample for inference, as shown below:

```bash
PT_HPU_LAZY_MODE=1 python ../text_to_image_generation.py \
    --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0  \
    --lora_id lora-trained-xl \
    --prompts "A picture of a sks dog in a bucket" \
    --num_images_per_prompt 5 \
    --batch_size 1 \
    --image_save_dir /tmp/stable_diffusion_xl_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

### DreamBooth LoRA Fine-Tuning with FLUX.1-dev

We can use the same `dog` dataset for the following examples.

To launch FLUX.1-dev LoRA training on a single Gaudi card, use:
```bash
PT_HPU_LAZY_MODE=1 python train_dreambooth_lora_flux.py \
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
> You can run training on multiple HPUs by replacing `python train_dreambooth_lora_flux.py` with 
> `python ../../gaudi_spawn.py --world_size <num-HPUs> train_dreambooth_lora_flux.py`. To use MPI for multi-card training,
> add `--use_mpi` after `--world_size <num-HPUs>`. To use DeepSpeed instead of MPI, replace `--use_mpi` with `--use_deepspeed`.

After training completes, you could directly use `text_to_image_generation.py` sample for inference as follows:
```bash
PT_HPU_LAZY_MODE=1 python ../text_to_image_generation.py \
    --model_name_or_path "black-forest-labs/FLUX.1-dev" \
    --lora_id dog_lora_flux \
    --prompts "A picture of a sks dog in a bucket" \
    --num_images_per_prompt 5 \
    --batch_size 1 \
    --num_inference_steps 30 \
    --image_save_dir /tmp/flux_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

### DreamBooth LoRA Fine-Tuning with Stable Diffusion 3 and 3.5 (SD3)

We can use the same `dog` dataset for the following example.

To launch SD3 LoRA training on a single Gaudi card, use:
```bash
python train_dreambooth_lora_sd3.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \
    --dataset_name="dog" \
    --instance_prompt="a photo of sks dog" \
    --validation_prompt="a photo of sks dog in a bucket" \
    --output_dir="dog_lora_sd3" \
    --mixed_precision="bf16" \
    --rank=8 \
    --resolution=1024 \
    --train_batch_size=1 \
    --guidance_scale=7 \
    --learning_rate=5e-4 \
    --max_grad_norm=0.5 \
    --report_to="tensorboard" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1500 \
    --validation_epochs=50 \
    --save_validation_images \
    --use_hpu_graphs_for_inference \
    --use_hpu_graphs_for_training \
    --gaudi_config_name="Habana/stable-diffusion" \
    --sdp_on_bf16 \
    --bf16
```

You can run training on multiple HPUs by replacing `python train_text_to_image_sd3.py`
with `python ../../gaudi_spawn.py --world_size <num-HPUs> train_text_to_image_sd3.py`.

> [!NOTE]
> To use MPI for multi-card training, add `--use_mpi` after `--world_size <num-HPUs>`.
> To use DeepSpeed instead of MPI, replace `--use_mpi` with `--use_deepspeed`.

After training completes, you could directly use `text_to_image_generation.py` sample for inference as follows:
```bash
python ../text_to_image_generation.py \
    --model_name_or_path "stabilityai/stable-diffusion-3-medium-diffusers" \
    --lora_id dog_lora_sd3 \
    --prompts "A picture of a sks dog in a bucket" \
    --scheduler flow_match_euler_discrete \
    --num_images_per_prompt 5 \
    --batch_size 1 \
    --num_inference_steps 28 \
    --image_save_dir /tmp/sd3_lora_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

## Full Model Fine-Tuning for Stable Diffusion 3 and 3.5 (SD3)

We can use the `dog` dataset for the following example.

To launch SD3 full model training on single Gaudi card, use:
```bash
python train_text_to_image_sd3.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \
    --dataset_name="dog" \
    --instance_prompt="a photo of sks dog" \
    --validation_prompt="a photo of sks dog in a bucket" \
    --output_dir="dog_ft_sd3" \
    --mixed_precision="bf16" \
    --resolution=1024 \
    --train_batch_size=1 \
    --guidance_scale=7 \
    --learning_rate=5e-4 \
    --max_grad_norm=1 \
    --report_to="tensorboard" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1500 \
    --validation_epochs=50 \
    --save_validation_images \
    --use_hpu_graphs_for_inference \
    --use_hpu_graphs_for_training \
    --gaudi_config_name="Habana/stable-diffusion" \
    --sdp_on_bf16 \
    --bf16
```
You can run training on multiple HPUs by replacing `python train_text_to_image_sd3.py`
with `python ../../gaudi_spawn.py --world_size <num-HPUs> train_text_to_image_sd3.py`.

> [!NOTE]
> To use MPI for multi-card training, add `--use_mpi` after `--world_size <num-HPUs>`.
> To use DeepSpeed instead of MPI, replace `--use_mpi` with `--use_deepspeed`.
> Fine-tuning the full SD3.5-Large model requires multiple HPU cards.

After training completes, you could directly use `text_to_image_generation.py` sample for inference as follows:
```bash
python ../text_to_image_generation.py \
    --model_name_or_path "dog_ft_sd3" \
    --prompts "A picture of a sks dog in a bucket" \
    --scheduler flow_match_euler_discrete \
    --num_images_per_prompt 5 \
    --batch_size 1 \
    --num_inference_steps 28 \
    --image_save_dir /tmp/sd3_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```
