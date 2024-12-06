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

## Single-HPU inference

Models that have been validated:
  - [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)
  - [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large)
  - [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
  - [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
  - [llava-hf/llava-1.5-13b-hf](https://huggingface.co/llava-hf/llava-1.5-13b-hf)
  - [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
  - [llava-hf/llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf)
  - [llava-hf/llava-v1.6-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf)
  - [llava-hf/llava-v1.6-34b-hf](https://huggingface.co/llava-hf/llava-v1.6-34b-hf)
  - [llava-hf/llama3-llava-next-8b-hf](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf)
  - [HuggingFaceM4/idefics2-8b](https://huggingface.co/HuggingFaceM4/idefics2-8b)
  - [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
  - [meta-llama/Llama-3.2-90B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct)
  - [tiiuae/falcon-11B-vlm](https://huggingface.co/tiiuae/falcon-11B-vlm)
  - [google/paligemma-3b-mix-224](https://huggingface.co/google/paligemma-3b-mix-224)

### Inference with BF16

To run Salesforce/blip-image-captioning-large inference, use the following command:
```bash
python3 run_pipeline.py \
    --model_name_or_path Salesforce/blip-image-captioning-large \
    --image_path "https://ankur3107.github.io/assets/images/image-captioning-example.png" \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

To run Llava-1.5-7b inference, use the following command:
```bash
python3 run_pipeline.py \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

To run Llava-1.5-13b inference, use the following command:
```bash
python3 run_pipeline.py \
    --model_name_or_path llava-hf/llava-1.5-13b-hf \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

To run Llava-v1.6-mistral-7b inference, use the following command:
```bash
python3 run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

To run Llava-v1.6-vicuna-13b inference, use the following command:
```bash
python3 run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-vicuna-13b-hf \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

To run Llava-hf/llava-v1.6-34b-hf inference, use the following command:
```bash
python3 run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-34b-hf \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

To run google/paligemma-3b-mix-224 inference, use the following command:
```bash
python3 run_pipeline.py \
    --model_name_or_path google/paligemma-3b-mix-224 \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

To run Llava-hf/llama3-llava-next-8b-hf inference, use the following command:
```bash
python3 run_pipeline.py \
    --model_name_or_path llava-hf/llama3-llava-next-8b-hf \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

To run idefics2 inference, use the following command:

```bash
python3 run_pipeline.py \
    --model_name_or_path HuggingFaceM4/idefics2-8b \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

To run mllama inference using reduced precision in the SDPA, use the following command:

```bash
python3 run_pipeline.py \
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

### Inference with FP8
Inference for Llava-1.5-7b, Llava-1.5-13b, Llava-v1.6-mistral-7b and Llava-v1.6-vicuna-13b in FP8 precision are enabled using  [Intel Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html), which provides model measurement and quantization capabilities in PyTorch.

More information on enabling FP8 in SynapseAI is available here:
https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html

Here is an example to measure the tensor quantization statistics on Llava-1.5-7b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python run_pipeline.py \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

Here is an example to quantize the model based on previous measurements for Llava-1.5-7b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant_scale_format_const.json python run_pipeline.py \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```


Here is an example to measure the tensor quantization statistics on Llava-v1.6-mistral-7b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

Here is an example to quantize the model based on previous measurements for Llava-v1.6-mistral-7b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant_scale_format_const.json python run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

Here is an example to measure the tensor quantization statistics on Llava-v1.6-vicuna-13b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-vicuna-13b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

Here is an example to quantize the model based on previous measurements for Llava-v1.6-vicuna-13b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant_scale_format_const.json python run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-vicuna-13b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --sdp_on_bf16
```

### Inference with FusedSDPA

Habana FusedSDPA is a fused and optimized implementation of torch.nn.functional.scaled_dot_product_attention() for Gaudi. For more details, refer to [Gaudi online documentation](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_PyTorch_Models.html?highlight=fusedsdpa#using-fused-scaled-dot-product-attention-fusedsdpa).

Use the following command to run Llava-1.5-7b BF16 inference with FusedSDPA
```bash
python3 run_pipeline.py \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --use_flash_attention \
    --flash_attention_recompute
```


Use the following command to run Llava-v1.6-mistral-7b BF16 inference with FusedSDPA
```bash
python3 run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --use_flash_attention \
    --flash_attention_recompute
```


Use the following commands to run Llava-v1.6-mistral-7b FP8 inference with FusedSDPA

Here is an example of measuring the tensor quantization statistics on Llava-v1.6-mistral-7b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --use_flash_attention \
    --flash_attention_recompute
```

Here is an example of quantizing the model based on previous measurements for Llava-v1.6-mistral-7b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant_scale_format_const.json python run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --use_flash_attention \
    --flash_attention_recompute
```
## LORA Finetune

To run LoRA finetuning, you can use `run_image2text_lora_finetune.py`.
Here are single-/multi-device command examples for HuggingFaceM4/idefics2-8b.

```bash
python3 run_image2text_lora_finetune.py \
    --model_name_or_path HuggingFaceM4/idefics2-8b \
    --dataset_name nielsr/docvqa_1200_examples \
    --bf16 True \
    --output_dir ./model_lora_llama \
    --num_train_epochs 1 \
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
    --max_seq_length=512 \
    --use_hpu_graphs_for_inference \
    --low_cpu_mem_usage True \
    --lora_target_modules '.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$'
```

```bash
python3 ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_image2text_lora_finetune.py \
    --model_name_or_path HuggingFaceM4/idefics2-8b \
    --dataset_name nielsr/docvqa_1200_examples \
    --bf16 True \
    --output_dir ./model_lora_llama \
    --num_train_epochs 1 \
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
    --max_seq_length=512 \
    --use_hpu_graphs_for_inference \
    --low_cpu_mem_usage True \
    --lora_target_modules '".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"'
```

Here are single-/multi-device command examples for meta-llama/Llama-3.2-11B-Vision-Instruct.

```bash
python3 run_image2text_lora_finetune.py \
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
python3 ../gaudi_spawn.py \
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

## Multi-HPU inference

### BF16 Inference with FusedSDPA on 8 HPUs

Use the following commands to run Llava-v1.6-mistral-7b BF16 inference with FusedSDPA on 8 HPUs:
```bash
python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --use_flash_attention \
    --flash_attention_recompute
```

Use the following commands to run Llama-3.2-90B-Vision-Instruct BF16 inference with FusedSDPA on 8 HPUs:
```bash
PT_HPU_ENABLE_LAZY_COLLECTIVES=true python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_pipeline.py \
    --model_name_or_path meta-llama/Llama-3.2-90B-Vision-Instruct \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --use_flash_attention \
    --flash_attention_recompute
```


### FP8 Inference with FusedSDPA on 8 HPUs

Use the following commands to run Llava-v1.6-mistral-7b FP8 inference with FusedSDPA on 8 HPUs.
Here is an example of measuring the tensor quantization statistics on Llava-v1.6-mistral-7b on 8 HPUs:
```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --use_flash_attention \
    --flash_attention_recompute
```

Here is an example of quantizing the model based on previous measurements for Llava-v1.6-mistral-7b on 8 HPUs:
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant_scale_format_const.json python ../gaudi_spawn.py --use_deepspeed --world_size 8 run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --use_flash_attention \
    --flash_attention_recompute
```
