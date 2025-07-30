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

# pyTorch-IMage-Models (TIMM) Examples with HPUs

This directory contains the scripts that showcase how to inference/fine-tune the TIMM models on Intel's HPUs with the lazy/graph modes. Training is supported for single/multiple HPU cards. Currently we can support first 10 most downloadable models from [Hugging Face timm link](https://huggingface.co/timm). In our example below for inference/training we will use [timm/resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k) as our testing model and same usage for other models.

## Requirements

First, you should install the requirements:

```bash
pip install -r requirements.txt
```

## Training

### Datasets

The following datasets [imagenette2-320 dataset](https://huggingface.co/datasets/johnowhitaker/imagenette2-320) and model [timm/resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k) from Hugging Face will be used.

### Using graph mode

```bash
PT_HPU_LAZY_MODE=1 python train_hpu_graph.py \
    --data-dir ./ \
    --dataset hfds/johnowhitaker/imagenette2-320 \
    --device 'hpu' \
    --model resnet50.a1_in1k \
    --train-split train \
    --val-split train \
    --dataset-download \
    --epochs 100
```

To run fine-tuning on multiple HPU replace `python train_hpu_graph.py` with
`torchrun --nnodes 1 --nproc_per_node <number-of-HPUs> train_hpu_graph.py`.

## inference

### Using graph mode

```bash
PT_HPU_LAZY_MODE=1 python inference.py \
    --data-dir='./' \
    --dataset hfds/johnowhitaker/imagenette2-320 \
    --device='hpu' \
    --model resnet50.a1_in1k \
    --split train \
    --graph_mode
```
