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

### Inference with BF16

To run Salesforce/blip-image-captioning-large inference, use the following command:
```bash
python3 run_pipeline.py \
    --model_name_or_path Salesforce/blip-image-captioning-large \
    --image_path "https://ankur3107.github.io/assets/images/image-captioning-example.png" \
    --use_hpu_graphs \
    --bf16
```

To run Llava-1.5-7b inference, use the following command:
```bash
python3 run_pipeline.py \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --use_hpu_graphs \
    --bf16
```

To run Llava-1.5-13b inference, use the following command:
```bash
python3 run_pipeline.py \
    --model_name_or_path llava-hf/llava-1.5-13b-hf \
    --use_hpu_graphs \
    --bf16
```

To run Llava-v1.6-mistral-7b inference, use the following command:
```bash
python3 run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --use_hpu_graphs \
    --bf16
```

To run Llava-v1.6-vicuna-13b inference, use the following command:
```bash
python3 run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-vicuna-13b-hf \
    --use_hpu_graphs \
    --bf16
```

### Inference with FP8

Inference for Llava-1.5-7b, Llava-1.5-13b, Llava-v1.6-mistral-7b and Llava-v1.6-vicuna-13b in FP8 precision are enabled using the Quantization Toolkit (HQT), which provides model measurement and quantization capabilities in PyTorch.

More information on enabling FP8 in SynapseAI is available here:
https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html

Here is an example to measure the tensor quantization statistics on Llava-1.5-7b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python run_pipeline.py \
--model_name_or_path llava-hf/llava-1.5-7b-hf \
--image_path "https://llava-vl.github.io/static/images/view.jpg" \
--use_hpu_graphs \
--bf16
```

Here is an example to quantize the model based on previous measurements for Llava-1.5-7b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant.json python run_pipeline.py \
--model_name_or_path llava-hf/llava-1.5-7b-hf \
--image_path "https://llava-vl.github.io/static/images/view.jpg" \
--use_hpu_graphs \
--bf16
```


Here is an example to measure the tensor quantization statistics on Llava-v1.6-mistral-7b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python run_pipeline.py \
--model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
--image_path "https://llava-vl.github.io/static/images/view.jpg" \
--use_hpu_graphs \
--bf16
```

Here is an example to quantize the model based on previous measurements for Llava-v1.6-mistral-7b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant.json python run_pipeline.py \
--model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
--image_path "https://llava-vl.github.io/static/images/view.jpg" \
--use_hpu_graphs \
--bf16
```

Here is an example to measure the tensor quantization statistics on Llava-v1.6-vicuna-13b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python run_pipeline.py \
--model_name_or_path llava-hf/llava-v1.6-vicuna-13b-hf \
--image_path "https://llava-vl.github.io/static/images/view.jpg" \
--use_hpu_graphs \
--bf16
```

Here is an example to quantize the model based on previous measurements for Llava-v1.6-vicuna-13b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant.json python run_pipeline.py \
--model_name_or_path llava-hf/llava-v1.6-vicuna-13b-hf \
--image_path "https://llava-vl.github.io/static/images/view.jpg" \
--use_hpu_graphs \
--bf16
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
    --use_flash_attention
```


Use the following command to run Llava-v1.6-mistral-7b BF16 inference with FusedSDPA
```bash
python3 run_pipeline.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --use_flash_attention
```


Use the following commands to run Llava-v1.6-mistral-7b FP8 inference with FusedSDPA

Here is an example of measuring the tensor quantization statistics on Llava-v1.6-mistral-7b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python run_pipeline.py \
--model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
--image_path "https://llava-vl.github.io/static/images/view.jpg" \
--use_hpu_graphs \
--bf16 --use_flash_attention
```

Here is an example of quantizing the model based on previous measurements for Llava-v1.6-mistral-7b:
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant.json python run_pipeline.py \
--model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
--image_path "https://llava-vl.github.io/static/images/view.jpg" \
--use_hpu_graphs \
--bf16 --use_flash_attention
```
