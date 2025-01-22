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

# Image Classification Examples

This directory contains a script that showcases how to fine-tune any model supported by the [`AutoModelForImageClassification` API](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForImageClassification) (such as [ViT](https://huggingface.co/docs/transformers/main/en/model_doc/vit) or [Swin Transformer](https://huggingface.co/docs/transformers/main/en/model_doc/swin)) on HPUs. They can be used to fine-tune models on both [datasets from the hub](#using-datasets-from-hub) as well as on [your own custom data](#using-your-own-data). This directory also contains a script to demonstrate a single HPU inference for [PyTorch-Image-Models/TIMM](https://huggingface.co/docs/timm/index).


## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

## Single-HPU training

### Using datasets from Hub

Here we show how to fine-tune a Vision Transformer (`ViT`) on Cifar10:

```bash
PT_HPU_LAZY_MODE=0 python run_image_classification.py \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --dataset_name cifar10 \
    --output_dir /tmp/outputs/ \
    --remove_unused_columns False \
    --image_column_name img \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 64 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --use_habana \
    --use_lazy_mode False \
    --torch_compile_backend hpu_backend \
    --torch_compile \
    --gaudi_config_name Habana/vit \
    --throughput_warmup_steps 6 \
    --dataloader_num_workers 1 \
    --sdp_on_bf16 \
    --bf16
```

For Swin, you need to change/add the following arguments:
- `--model_name_or_path microsoft/swin-base-patch4-window7-224-in22k`
- `--gaudi_config_name Habana/swin`
- `--ignore_mismatched_sizes`

> If your model classification head dimensions do not fit the number of labels in the dataset, you can specify `--ignore_mismatched_sizes` to adapt it.


### Using your own data

To use your own dataset, there are 2 ways:
- you can either provide your own folders as `--train_dir` and/or `--validation_dir` arguments
- you can upload your dataset to the hub (possibly as a private repo, if you prefer so), and simply pass the `--dataset_name` argument.

Below, we explain both in more detail.

#### Provide them as folders

If you provide your own folders with images, the script expects the following directory structure:

```bash
root/dog/xxx.png
root/dog/xxy.png
root/dog/[...]/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/[...]/asd932_.png
```

In other words, you need to organize your images in subfolders, based on their class. You can then run the script like this:

```bash
PT_HPU_LAZY_MODE=0 python run_image_classification.py \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --train_dir <path-to-train-root> \
    --output_dir /tmp/outputs/ \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode False \
    --torch_compile_backend hpu_backend \
    --torch_compile \
    --gaudi_config_name Habana/vit \
    --throughput_warmup_steps 3 \
    --dataloader_num_workers 1 \
    --sdp_on_bf16 \
    --bf16
```

Internally, the script will use the [`ImageFolder`](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder) feature which will automatically turn the folders into 🤗 Dataset objects.

##### 💡 The above will split the train dir into training and evaluation sets
  - To control the split amount, use the `--train_val_split` flag.
  - To provide your own validation split in its own directory, you can pass the `--validation_dir <path-to-val-root>` flag.

#### Upload your data to the hub, as a (possibly private) repo

It's very easy (and convenient) to upload your image dataset to the hub using the [`ImageFolder`](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder) feature available in 🤗 Datasets. Simply do the following:

```python
from datasets import load_dataset

# example 1: local folder
dataset = load_dataset("imagefolder", data_dir="path_to_your_folder")

# example 2: local files (supported formats are tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset("imagefolder", data_files="path_to_zip_file")

# example 3: remote files (supported formats are tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset("imagefolder", data_files="https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip")

# example 4: providing several splits
dataset = load_dataset("imagefolder", data_files={"train": ["path/to/file1", "path/to/file2"], "test": ["path/to/file3", "path/to/file4"]})
```

`ImageFolder` will create a `label` column, and the label name is based on the directory name.

Next, push it to the hub!

```python
# assuming you have ran the huggingface-cli login command in a terminal
dataset.push_to_hub("name_of_your_dataset")

# if you want to push to a private repo, simply pass private=True:
dataset.push_to_hub("name_of_your_dataset", private=True)
```

and that's it! You can now train your model by simply setting the `--dataset_name` argument to the name of your dataset on the hub (as explained in [Using datasets from the 🤗 hub](#using-datasets-from-hub)).

More on this can also be found in [this blog post](https://huggingface.co/blog/image-search-datasets).

### Sharing your model on 🤗 Hub

0. If you haven't already, [sign up](https://huggingface.co/join) for a 🤗 account.

1. Make sure you have `git-lfs` installed and git set up.

```bash
$ apt install git-lfs
$ git config --global user.email "you@example.com"
$ git config --global user.name "Your Name"
```

2. Log in with your HuggingFace account credentials using `huggingface-cli`:

```bash
$ huggingface-cli login
# ...follow the prompts
```

3. When running the script, pass the following arguments:

```bash
python run_image_classification.py \
    --push_to_hub \
    --push_to_hub_model_id <name-your-model> \
    ...
```


## Multi-HPU training

Here is how you would fine-tune ViT on Cifar10 using 8 HPUs:

```bash
PT_HPU_LAZY_MODE=0 python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_image_classification.py \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --dataset_name cifar10 \
    --output_dir /tmp/outputs/ \
    --remove_unused_columns False \
    --image_column_name img \
    --do_train \
    --do_eval \
    --learning_rate 2e-4 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 64 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --use_habana \
    --use_lazy_mode False \
    --torch_compile_backend hpu_backend \
    --torch_compile \
    --gaudi_config_name Habana/vit \
    --throughput_warmup_steps 8 \
    --dataloader_num_workers 1 \
    --sdp_on_bf16 \
    --bf16
```

For Swin, you need to change/add the following arguments:
- `--model_name_or_path microsoft/swin-base-patch4-window7-224-in22k`
- `--gaudi_config_name Habana/swin`
- `--ignore_mismatched_sizes`

> If your model classification head dimensions do not fit the number of labels in the dataset, you can specify `--ignore_mismatched_sizes` to adapt it.


## Using DeepSpeed

Similarly to multi-HPU training, here is how you would fine-tune ViT on Cifar10 using 8 HPUs with DeepSpeed:

```bash
PT_HPU_LAZY_MODE=0 python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_image_classification.py \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --dataset_name cifar10 \
    --output_dir /tmp/outputs/ \
    --remove_unused_columns False \
    --image_column_name img \
    --do_train \
    --do_eval \
    --learning_rate 2e-4 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 64 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --use_habana \
    --use_lazy_mode False \
    --torch_compile_backend hpu_backend \
    --torch_compile \
    --gaudi_config_name Habana/vit \
    --throughput_warmup_steps 3 \
    --dataloader_num_workers 1 \
    --deepspeed path_to_my_deepspeed_config
```

You can look at the [documentation](https://huggingface.co/docs/optimum/habana/usage_guides/deepspeed) for more information about how to use DeepSpeed in Optimum Habana.
Here is a DeepSpeed configuration you can use to train your models on Gaudi:
```json
{
    "steps_per_print": 64,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "bf16": {
        "enabled": true
    },
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": false,
        "reduce_scatter": false,
        "contiguous_gradients": false
    }
}
```

> If your model classification head dimensions do not fit the number of labels in the dataset, you can specify `--ignore_mismatched_sizes` to adapt it.


## Inference

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
    --sdp_on_bf16 \
    --bf16
```

## TIMM/FastViT Examples

This directory contains an example script that demonstrates using FastViT with graph mode.

```bash
pip install --no-deps -r requirements_no_deps.txt
```

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
