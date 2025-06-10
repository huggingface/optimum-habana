<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

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

# Image to Text Examples
This directory contains a script that showcases how to perform image to text generation on Intel® Gaudi® AI Accelerators.

Habana FusedSDPA is a fused and optimized implementation of torch.nn.functional.scaled_dot_product_attention() for Gaudi. For more details, refer to [Gaudi online documentation](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_PyTorch_Models.html?highlight=fusedsdpa#using-fused-scaled-dot-product-attention-fusedsdpa). We optimized many models with FusedSDPA implementation as in [optimum/habana/transformers/models](https://github.com/huggingface/optimum-habana/tree/main/optimum/habana/transformers/models). If a model is not optimized with FusedSDPA, it uses [SDPA implementation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

## Inference with mixed-precision (BF16)

### Single card inference with BF16
To run Llama inference with SDPA, use the following command:

```bash
PT_HPU_LAZY_MODE=1 python3 run_pipeline.py \
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```
> SDPA may introduce [reduced precison](https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-in-scaled-dot-product-attention-sdpa)

To run inference with THUDM/glm-4v-9b, use the following command (Note that you need to set the environment variable `GLM=4v` to distinguish between glm4v and chatglm, as these models are customized and share the same model type named "chatglm"):
```bash
PT_HPU_LAZY_MODE=1 GLM=4v python3 run_pipeline.py \
    --model_name_or_path THUDM/glm-4v-9b \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16 \
    --use_flash_attention \
    --use_kv_cache
```

### Multi-cards inference with BF16

Use the following commands to run Llama-3.2-90B-Vision-Instruct BF16 inference with FusedSDPA on 8 HPUs:
```bash
PT_HPU_LAZY_MODE=1 PT_HPU_ENABLE_LAZY_COLLECTIVES=true python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_pipeline.py \
    --model_name_or_path meta-llama/Llama-3.2-90B-Vision-Instruct \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --use_flash_attention \
    --flash_attention_recompute
```

## Inference with FP8

Inference with FP8 precision is enabled using [Intel Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/index.html?highlight=inc), which provides model measurement and quantization capabilities in PyTorch.
More information on enabling FP8 in SynapseAI is available here:
[Run Inference Using FP8](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html?highlight=fp8)

### Single card inference with FP8
Here is an example to measure the tensor quantization statistics on Llava-v1.6-vicuna-13b with SDPA:
```bash
PT_HPU_LAZY_MODE=1 QUANT_CONFIG=./quantization_config/maxabs_measure.json python run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-vicuna-13b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

Here is an example to quantize the model based on previous measurements for Llava-v1.6-vicuna-13b with SDPA:
```bash
PT_HPU_LAZY_MODE=1 QUANT_CONFIG=./quantization_config/maxabs_quant_scale_format_const.json python run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-vicuna-13b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

### Multi-cards inference with FP8
Here is an example of measuring the tensor quantization statistics on Llava-v1.6-mistral-7b with FusedSDPA on 8 HPUs:
```bash
PT_HPU_LAZY_MODE=1 QUANT_CONFIG=./quantization_config/maxabs_measure.json python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --use_flash_attention \
    --flash_attention_recompute
```

Here is an example of quantizing the model based on previous measurements for Llava-v1.6-mistral-7b with FusedSDPA on 8 HPUs:
```bash
PT_HPU_LAZY_MODE=1 QUANT_CONFIG=./quantization_config/maxabs_quant_scale_format_const.json python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --use_flash_attention \
    --flash_attention_recompute
```

## LORA Finetune

Here are single-/multi-device command examples for meta-llama/Llama-3.2-11B-Vision-Instruct.

```bash
PT_HPU_LAZY_MODE=1 python3 run_image2text_lora_finetune.py \
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --dataset_name nielsr/docvqa_1200_examples \
    --bf16 True \
    --output_dir ./model_lora_llama \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --weight_decay 0.01 \
    --logging_steps 25 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 5e-5 \
    --warmup_steps  50 \
    --lr_scheduler_type "constant" \
    --input_column_names 'image' 'query' \
    --output_column_names 'answers' \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --lora_rank=8 \
    --lora_alpha=8 \
    --lora_dropout=0.1 \
    --low_cpu_mem_usage True \
    --max_seq_length=512 \
    --use_hpu_graphs_for_inference True \
    --lora_target_modules ".*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"
```

```bash
PT_HPU_LAZY_MODE=1 python3 ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_image2text_lora_finetune.py \
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --dataset_name nielsr/docvqa_1200_examples \
    --bf16 True \
    --output_dir ./model_lora_llama \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --weight_decay 0.01 \
    --logging_steps 25 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 5e-5 \
    --warmup_steps  50 \
    --lr_scheduler_type "constant" \
    --input_column_names 'image' 'query' \
    --output_column_names 'answers' \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --lora_rank=8 \
    --lora_alpha=8 \
    --lora_dropout=0.1 \
    --low_cpu_mem_usage True \
    --max_seq_length=512 \
    --use_hpu_graphs_for_inference True \
    --lora_target_modules '".*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"'
```

The single card training command for llava-hf/llava-1.5-7b-hf is similar.

>  For different models, please adjust training parameters and `lora_target_modules`. Such as replace `lora_target_modules`
>  with below for HuggingFaceM4/idefics2-8b.
>  '".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"'
