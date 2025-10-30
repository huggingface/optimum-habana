<!---
Copyright 2025 The HuggingFace Team. All rights reserved.

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

# Vision-Language Model Fine-tuning with LoRA

Fine-tuning multimodal vision-language models using Low-Rank Adaptation (LoRA) on Intel Gaudi HPUs. This approach provides parameter-efficient training of vision-language models while maintaining high performance across different hardware configurations.

The script supports vision-language datasets from the [Hugging Face Hub](https://huggingface.co/datasets) and handles automatic dataset schema normalization, train/validation splitting, and multimodal data processing.

## Supported Models

The following vision-language models are tested:

- **LLaVA-1.6**: `llava-hf/llava-v1.6-mistral-7b-hf` (7B parameters)
- **Gemma-3**: `google/gemma-3-12b-it` (12B parameters)  
- **Qwen2.5-VL**: `Qwen/Qwen2.5-VL-7B-Instruct` (7B parameters)

The script automatically detects and configures model-specific settings for optimal training.

## Supported Datasets

The script automatically handles various vision-language dataset schemas and supports datasets from the Hugging Face Hub.

### Example Dataset

- **ChartQA**: `HuggingFaceM4/ChartQA` - Chart-based visual question answering

The script can be extended to work with other visual question-answering (VQA) datasets that follow standard image-question-answer formats. See the [HuggingFace VQA Datasets](https://huggingface.co/datasets?task_categories=visual-question-answering) for more options.

## Requirements

First, install the requirements:
```bash
pip install -r requirements.txt
```

## Quick Start

### LLaVA-1.6 (7B) - Single Card, Eager Mode

```bash
python run_lora_vlm.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --dataset_name HuggingFaceM4/ChartQA \
    --output_dir ./output_llava \
    --do_train --max_train_samples 1000 \
    --do_eval --max_eval_samples 200 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --use_habana --bf16 \
    --gaudi_config_name Habana/gpt2
```

### Gemma-3 (12B) - Single Card, Eager Mode

```bash
python run_lora_vlm.py \
    --model_name_or_path google/gemma-3-12b-it \
    --dataset_name HuggingFaceM4/ChartQA \
    --output_dir ./output_gemma3 \
    --do_train --max_train_samples 1000 \
    --do_eval --max_eval_samples 200 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --use_habana --bf16 \
    --gaudi_config_name Habana/gpt2
```

## Execution Modes

The script supports Eager and Lazy execution modes on Gaudi HPUs. **Eager mode is recommended** for production use as it provides universal compatibility and reliable performance.
Eager mode is currently default and doesnt need any specific settings. 

> **Important**: The environment variable `PT_HPU_LAZY_MODE` and the `--use_lazy_mode` flag must be consistent:
> - Eager mode: `PT_HPU_LAZY_MODE=0` (without `--use_lazy_mode` flag)
> - Lazy mode: `PT_HPU_LAZY_MODE=1` (with `--use_lazy_mode` flag)

### 1. Eager Mode (RECOMMENDED)

Eager mode provides immediate execution without graph compilation, making it the most reliable and debuggable option. It works consistently across all vision-language models and configurations.

**Benefits:**
- Universal compatibility with all supported models
- Stable across different batch sizes and gradient accumulation settings
- Works with both single-card and multi-card (MPI/DeepSpeed) setups
- No compilation overhead or graph-related errors
- Predictable memory usage and behavior

**Usage:**
```bash
python run_lora_vlm.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --dataset_name HuggingFaceM4/ChartQA \
    --output_dir ./output_eager \
    --do_train --max_train_samples 1000 \
    --do_eval --max_eval_samples 200 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --use_habana --bf16 \
    --gaudi_config_name Habana/gpt2
```

### 2. Lazy Mode (Experimental - Limited Support)

Lazy mode defers graph execution for potential memory optimization, but has known stability issues with certain configurations.

** Current Known Limitations:**
- Only works with `--gradient_accumulation_steps 1`
- `gradient_accumulation_steps > 1` causes `synStatus 1 [Invalid argument]` during concat operations
- Must use `--gradient_checkpointing` for memory efficiency

**Working Example (gradient_accumulation_steps=1 only):**
```bash
PT_HPU_LAZY_MODE=1 python run_lora_vlm.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --dataset_name HuggingFaceM4/ChartQA \
    --output_dir ./output_lazy \
    --do_train --max_train_samples 1000 \
    --do_eval --max_eval_samples 200 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --gradient_checkpointing \
    --use_habana --bf16 \
    --gaudi_config_name Habana/gpt2 \
    --use_lazy_mode
```

*Note: Must include `--use_lazy_mode` flag when using Lazy mode.*

## Multi-Card Training

Multi-card training enables distributed training across multiple HPUs for faster processing of larger datasets.

### Data Parallel with MPI (8 HPUs)

```bash
PT_HPU_LAZY_MODE=0 python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_lora_vlm.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --dataset_name HuggingFaceM4/ChartQA \
    --output_dir ./output_mpi \
    --do_train --max_train_samples 1000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --use_habana --bf16 \
    --gaudi_config_name Habana/gpt2
```

### DeepSpeed (Data Scaling)

For distributed data parallel training using DeepSpeed:

```bash
PT_HPU_LAZY_MODE=0 deepspeed --num_nodes 1 --num_gpus 8 \
    run_lora_vlm.py \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --dataset_name HuggingFaceM4/ChartQA \
    --output_dir ./output_deepspeed \
    --do_train --max_train_samples 1000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --use_habana --bf16 \
    --gaudi_config_name Habana/gpt2
```

## Key Features

### **Automatic Configuration**
- **Model Detection**: Automatically detects and configures model-specific settings
- **Dataset Normalization**: Handles different dataset schemas (query/label, question/answer)
- **Gaudi Config**: Auto-generates Gaudi configuration if not provided
- **Train/Eval Split**: Automatically creates validation split if not available

### **LoRA Fine-tuning**
- **Parameter Efficient**: Only trains a small subset of parameters
- **Flexible Target Modules**: Supports custom LoRA target modules per model
- **Rank Configuration**: Configurable LoRA rank (4, 8, 16, 32)
- **Memory Efficient**: Significantly reduces memory requirements

### **Performance Optimization**
- **Multiple Execution Modes**: Eager, Torch Compile, and Lazy modes
- **Gradient Checkpointing**: Reduces memory usage during training
- **Mixed Precision**: BF16 support for faster training
- **Multi-card Scaling**: Data parallel training across 8 HPUs

### **Comprehensive Evaluation**
- **Training Metrics**: Loss, learning rate, throughput tracking
- **Evaluation Metrics**: Perplexity and custom metrics
- **Automatic Validation**: Configurable evaluation strategy
- **Progress Logging**: Detailed logging with timestamps


### Common Issues:

1. **Out of Memory Errors**:
   - Reduce `--per_device_train_batch_size` to 1
   - Increase `--gradient_accumulation_steps`
   - Use `--lora_rank 4` or `8` instead of 16
   - Use multi-card training (MPI or DeepSpeed) for larger datasets

2. **Dataset Loading Issues**:
   - Remove `trust_remote_code=True` for standard datasets
   - Check dataset schema matches expected format
   - Verify dataset availability on Hugging Face Hub


### Execution Mode Compatibility:

| Mode | Status | Single-Card | Multi-Card | Notes |
|------|--------|-------------|------------|-------|
| **Eager** | Recommended | Yes | Yes | Tested |
| **Lazy** | Limited | Yes (GA=1) | Yes (GA=1) | Fails with GA>1, use Eager instead |


---