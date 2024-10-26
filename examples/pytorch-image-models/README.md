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

This directory contains the scripts that showcases how to inference/fine-tune the TIMM models on intel's HPUs with the lazy/graph modes.  We support the trainging for single/multiple HPU cards both two. Currently we support several most downloadable models from Hugging Face as below list.

- [timm/resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k)
- [timm/resnet18.a1_in1k](https://huggingface.co/timm/resnet18.a1_in1k)
- [timm/resnet18.fb_swsl_ig1b_ft_in1k](https://huggingface.co/timm/resnet18.fb_swsl_ig1b_ft_in1k)
- [timm/wide_resnet50_2.racm_in1k](https://huggingface.co/timm/wide_resnet50_2.racm_in1k)
- [timm/efficientformerv2_s2.snap_dist_in1k](https://huggingface.co/timm/efficientformerv2_s2.snap_dist_in1k)
- [timm/efficientnet_b3.ra2_in1k](https://huggingface.co/timm/efficientnet_b3.ra2_in1k)
- [timm/efficientnet_lite0.ra_in1k](https://huggingface.co/timm/efficientnet_lite0.ra_in1k)
- [timm/efficientnet_b0.ra_in1k](https://huggingface.co/timm/efficientnet_b0.ra_in1k)
- [timm/nf_regnet_b1.ra2_in1k](https://huggingface.co/timm/nf_regnet_b1.ra2_in1k)
- [timm/mobilenetv3_large_100.ra_in1k](https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k)
- [timm/tf_mobilenetv3_large_minimal_100.in1k](https://huggingface.co/timm/tf_mobilenetv3_large_minimal_100.in1k)
- [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
- [timm/vgg19.tv_in1k]()

## Requirements

First, you should install the pytorch-image-models (Timm):
```bash
git clone https://github.com/huggingface/pytorch-image-models.git
cd pytorch-image-models
pip install .
```

## Single-HPU training

### Using datasets from Hub

Here we show how to fine-tune the [imagenette2-320 dataset](https://www.kaggle.com/datasets/xbinchen/imagenette2-320) and model with [timm/resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k) from Hugging Face.

1) training with hpu lazy mode
   
```bash
python train_hpu_lazy.py \
    --data-dir ./imagenette2-320/ \
    --device 'hpu' \
    --model resnet50.a1_in1k
```
2) training with hpu graph mode

```bash
python train_hpu_graph.py \
    --data-dir ./imagenette2-320/ \
    --device 'hpu' \
    --model resnet50.a1_in1k
```

Here the results for lazy mode is shown below for example:


Train: 0 [   0/73 (  1%)]  Loss: 6.86 (6.86)  Time: 9.575s,   13.37/s  (9.575s,   13.37/s)  LR: 1.000e-05  Data: 0.844 (0.844)
Train: 0 [  50/73 ( 70%)]  Loss: 6.77 (6.83)  Time: 0.320s,  400.32/s  (0.470s,  272.39/s)  LR: 1.000e-05  Data: 0.217 (0.047)
Test: [   0/30]  Time: 6.593 (6.593)  Loss:   6.723 ( 6.723)  Acc@1:   0.000 (  0.000)  Acc@5:   0.000 (  0.000)
Test: [  30/30]  Time: 3.856 (0.732)  Loss:   6.615 ( 6.691)  Acc@1:   0.000 (  0.076)  Acc@5:   1.176 (  3.287)
Current checkpoints:
 ('./output/train/20241016-034927-resnet50_a1_in1k-224/checkpoint-0.pth.tar', 0.07643312101910828)

Train: 1 [   0/73 (  1%)]  Loss: 6.69 (6.69)  Time: 0.796s,  160.74/s  (0.796s,  160.74/s)  LR: 1.001e-02  Data: 0.685 (0.685)
Train: 1 [  50/73 ( 70%)]  Loss: 3.23 (3.76)  Time: 0.160s,  798.85/s  (0.148s,  863.22/s)  LR: 1.001e-02  Data: 0.053 (0.051)
Test: [   0/30]  Time: 0.663 (0.663)  Loss:   1.926 ( 1.926)  Acc@1:  46.094 ( 46.094)  Acc@5:  85.938 ( 85.938)
Test: [  30/30]  Time: 0.022 (0.126)  Loss:   1.462 ( 1.867)  Acc@1:  63.529 ( 39.261)  Acc@5:  83.529 ( 85.096)
Current checkpoints:
 ('./output/train/20241016-034927-resnet50_a1_in1k-224/checkpoint-1.pth.tar', 39.26114640448503)
 ('./output/train/20241016-034927-resnet50_a1_in1k-224/checkpoint-0.pth.tar', 0.07643312101910828)



## Multi-HPU training

Here we show how to fine-tune the [imagenette2-320 dataset](https://www.kaggle.com/datasets/xbinchen/imagenette2-320) and model with [timm/resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k) from Hugging Face.

```bash
torchrun --nnodes 1 --nproc_per_node 2 \
    train_hpu_lazy.py \
    --data-dir ./imagenette2-320/ \
    --device 'hpu' \
    --model resnet50.a1_in1k
```
2) training with hpu graph mode

```bash
torchrun --nnodes 1 --nproc_per_node 2 \
    train_hpu_graph.py \
    --data-dir ./imagenette2-320/ \
    --device 'hpu' \
    --model resnet50.a1_in1k
```

Here the results for lazy mode is shown below for example:

Train: 0 [   0/36 (  3%)]  Loss: 6.88 (6.88)  Time: 10.036s,   25.51/s  (10.036s,   25.51/s)  LR: 1.000e-05  Data: 0.762 (0.762)
Distributing BatchNorm running means and vars
Test: [   0/15]  Time: 7.796 (7.796)  Loss:   6.915 ( 6.915)  Acc@1:   0.000 (  0.000)  Acc@5:   0.000 (  0.000)
Test: [  15/15]  Time: 1.915 (1.263)  Loss:   6.847 ( 6.818)  Acc@1:   0.000 (  0.000)  Acc@5:   0.000 (  0.688)
Current checkpoints:
 ('./output/train/20241016-034443-resnet50_a1_in1k-224/checkpoint-0.pth.tar', 0.0)

Train: 1 [   0/36 (  3%)]  Loss: 6.84 (6.84)  Time: 6.687s,   38.28/s  (6.687s,   38.28/s)  LR: 2.001e-02  Data: 0.701 (0.701)
Distributing BatchNorm running means and vars
Test: [   0/15]  Time: 1.315 (1.315)  Loss:   2.463 ( 2.463)  Acc@1:  14.062 ( 14.062)  Acc@5:  48.828 ( 48.828)
Test: [  15/15]  Time: 0.020 (0.180)  Loss:   1.812 ( 1.982)  Acc@1:  52.326 ( 32.934)  Acc@5:  66.279 ( 75.064)
Current checkpoints:
 ('./output/train/20241016-034443-resnet50_a1_in1k-224/checkpoint-1.pth.tar', 32.93428432485976)
 ('./output/train/20241016-034443-resnet50_a1_in1k-224/checkpoint-0.pth.tar', 0.0)



## Inference

# graph_mode
python inference.py --data-dir='./download_ds/imagenette2-320' --device='hpu' --graph_mode

# lazy mode
python inference.py --data-dir='./download_ds/imagenette2-320' --device='hpu'

To run only inference, you can start from the commands above and you just have to remove the training-only arguments such as `--do_train`, `--per_device_train_batch_size`, `--num_train_epochs`, etc...

For instance, you can run inference with ViT on Cifar10 on 1 Gaudi card with the following command:
```bash
python run_image_classification.py \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --dataset_name cifar10 \
    --output_dir /tmp/outputs/ \
    --remove_unused_columns False \
    --image_column_name img \
    --do_eval \
    --per_device_eval_batch_size 64 \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gaudi_config_name Habana/vit \
    --dataloader_num_workers 1 \
    --bf16
```

## TIMM/FastViT Examples

This directory contains an example script that demonstrates using FastViT with graph mode.

### Single-HPU inference

```bash
python3 run_timm_example.py \
    --model_name_or_path "timm/fastvit_t8.apple_in1k" \
    --image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png" \
    --warmup 3 \
    --n_iterations 20 \
    --use_hpu_graphs \
    --bf16 \
    --print_result
```
Models that have been validated:
  - [timm/fastvit_t8.apple_dist_in1k](https://huggingface.co/timm/fastvit_t8.apple_dist_in1k)
  - [timm/fastvit_t8.apple_in1k](https://huggingface.co/timm/fastvit_t8.apple_in1k)
  - [timm/fastvit_sa12.apple_in1k](https://huggingface.co/timm/fastvit_sa12.apple_in1k)
