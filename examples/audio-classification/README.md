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

# Audio Classification Examples

The following examples showcase how to fine-tune `Wav2Vec2` for audio classification on Habana Gaudi.

Speech recognition models that have been pretrained in an unsupervised fashion on audio data alone, *e.g.* [Wav2Vec2](https://huggingface.co/transformers/main/model_doc/wav2vec2.html), have shown to require only very little annotated data to yield good performance on speech classification datasets.

## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

## Single-HPU

The following command shows how to fine-tune [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) on the 🗣️ [Keyword Spotting subset](https://huggingface.co/datasets/superb#ks) of the SUPERB dataset on a single HPU.

```bash
python run_audio_classification.py \
    --model_name_or_path facebook/wav2vec2-base \
    --dataset_name superb \
    --dataset_config_name ks \
    --output_dir /tmp/wav2vec2-base-ft-keyword-spotting \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --max_length_seconds 1 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --dataloader_num_workers 4 \
    --seed 27 \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_training \
    --use_hpu_graphs_for_inference \
    --gaudi_config_name Habana/wav2vec2 \
    --throughput_warmup_steps 3 \
    --sdp_on_bf16 \
    --bf16 \
    --trust_remote_code True
```

On a single HPU, this script should run in ~13 minutes and yield an accuracy of **97.96%**.

> If your model classification head dimensions do not fit the number of labels in the dataset, you can specify `--ignore_mismatched_sizes` to adapt it.


## Multi-HPU

The following command shows how to fine-tune [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) for 🌎 **Language Identification** on the [CommonLanguage dataset](https://huggingface.co/datasets/anton-l/common_language) on 8 HPUs.

```bash
PT_HPU_LAZY_MODE=0 python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_audio_classification.py \
    --model_name_or_path facebook/wav2vec2-base \
    --dataset_name common_language \
    --audio_column_name audio \
    --label_column_name language \
    --output_dir /tmp/wav2vec2-base-lang-id \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 3e-4 \
    --max_length_seconds 8 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --seed 0 \
    --use_habana \
    --use_lazy_mode False\
    --gaudi_config_name Habana/wav2vec2 \
    --throughput_warmup_steps 3 \
    --sdp_on_bf16 \
    --bf16 \
    --trust_remote_code True \
    --torch_compile \
    --torch_compile_backend hpu_backend
```

On 8 HPUs, this script should run in ~12 minutes and yield an accuracy of **80.49%**.

> If your model classification head dimensions do not fit the number of labels in the dataset, you can specify `--ignore_mismatched_sizes` to adapt it.

> If you get an error reporting unused parameters in the model, you can specify `--ddp_find_unused_parameters True`. Using this parameter might affect the training speed.

## Inference

To run only inference, you can start from the commands above and you just have to remove the training-only arguments such as `--do_train`, `--per_device_train_batch_size`, `--num_train_epochs`, etc...

For instance, you can run inference with Wav2Vec2 on the Keyword Spotting subset on 1 Gaudi card with the following command:
```bash
python run_audio_classification.py \
    --model_name_or_path facebook/wav2vec2-base \
    --dataset_name superb \
    --dataset_config_name ks \
    --output_dir /tmp/wav2vec2-base-ft-keyword-spotting \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --bf16 \
    --do_eval \
    --max_length_seconds 1 \
    --attention_mask False \
    --per_device_eval_batch_size 256 \
    --dataloader_num_workers 4 \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --throughput_warmup_steps 3 \
    --gaudi_config_name Habana/wav2vec2 \
    --trust_remote_code
```


## Sharing your model on 🤗 Hub

0. If you haven't already, [sign up](https://huggingface.co/join) for a 🤗 account

1. Make sure you have `git-lfs` installed and git set up.

```bash
$ apt install git-lfs
```

2. Log in with your HuggingFace account credentials using `huggingface-cli`

```bash
$ huggingface-cli login
# ...follow the prompts
```

3. When running the script, pass the following arguments:

```bash
python run_audio_classification.py \
    --push_to_hub \
    --hub_model_id <username/model_id> \
    ...
```
