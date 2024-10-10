#!/bin/bash
QUANT_CONFIG=quantize/quant_config_500.json \
PT_HPU_WEIGHT_SHARING=0 \
python text_to_image_generation.py \
     --model_name_or_path black-forest-labs/FLUX.1-dev \
     --prompts "A cat holding a sign that says hello world" \
     --num_images_per_prompt 1 \
     --batch_size 1 \
     --num_inference_steps 30 \
     --image_save_dir /tmp/flux_1_images_fp8_500 \
     --scheduler flow_match_euler_discrete \
     --use_habana \
     --use_hpu_graphs \
     --gaudi_config Habana/stable-diffusion \
     --bf16 \
     --quant_mode quantize
