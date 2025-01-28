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

First, you should install the pytorch-image-models (Timm):
```bash
git clone https://github.com/huggingface/pytorch-image-models.git
cd pytorch-image-models
pip install .
```

## Single-HPU training

### Using datasets from Hub

Here we show how to fine-tune the [imagenette2-320 dataset](https://huggingface.co/datasets/johnowhitaker/imagenette2-320) and model with [timm/resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k) from Hugging Face.

### Training with HPU graph mode

```bash
python train_hpu_graph.py \
    --data-dir ./ \
    --dataset hfds/johnowhitaker/imagenette2-320 \
    --device 'hpu' \
    --model resnet50.a1_in1k \
    --train-split train \
    --val-split train \
    --dataset-download 
```

## Multi-HPU training

Here we show how to fine-tune the [imagenette2-320 dataset](https://huggingface.co/datasets/johnowhitaker/imagenette2-320) and model with [timm/resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k) from Hugging Face.

### Training with HPU graph mode

```bash
torchrun --nnodes 1 --nproc_per_node 2 \
    train_hpu_graph.py \
    --data-dir ./ \
    --dataset hfds/johnowhitaker/imagenette2-320 \
    --device 'hpu' \
    --model resnet50.a1_in1k \
    --train-split train \
    --val-split train \
    --dataset-download
```


## Single-HPU inference

Here we show how to fine-tune the [imagenette2-320 dataset](https://huggingface.co/datasets/johnowhitaker/imagenette2-320) and model with [timm/resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k) from Hugging Face.

### HPU with graph mode
```bash
python inference.py \
    --data-dir='./' \
    --dataset hfds/johnowhitaker/imagenette2-320 \
    --device='hpu' \
    --model resnet50.a1_in1k \
    --split train \
    --graph_mode
```




