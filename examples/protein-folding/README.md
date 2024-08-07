<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

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

# ESMFold Example

ESMFold ([paper link](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2)) is a recently released protein folding model from FAIR. Unlike other protein folding models, it does not require external databases or search tools to predict structures, and is up to 60X faster as a result.

The port to the Hugging Face Transformers library is even easier to use, as we've removed the dependency on tools like openfold - once you run `pip install transformers`, you're ready to use this model!

Note that all the code that follows will be running the model locally, rather than calling an external API. This means that no rate limiting applies here - you can predict as many structures as your computer can handle.

## Single-HPU inference

Here we show how to predict the folding of a single chain on HPU:

```bash
python run_esmfold.py
```
The predicted protein structure will be stored in save-hpu.pdb file. We can use some tools like py3Dmol to visualize it.


# Mila-Intel protST example

## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

## Single-HPU inference for zero shot evaluation
Here we show how to run zero shot evaluation of protein ST model on HPU:

```bash
python run_zero_shot_eval.py --bf16 --max_seq_length 1024
```
## Multi-HPU finetune for sequence classification task

```bash
python ../gaudi_spawn.py --world_size 8 --use_mpi run_sequence_classification.py \
    --output_dir ./out \
    --model_name_or_path mila-intel/protst-esm1b-for-sequential-classification \
    --tokenizer_name facebook/esm1b_t33_650M_UR50S \
    --trust_remote_code \
    --dataset_name mila-intel/ProtST-BinaryLocalization \
    --torch_dtype bfloat16 \
    --overwrite_output_dir \
    --do_train \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-05 \
    --weight_decay 0 \
    --num_train_epochs 100 \
    --lr_scheduler_type constant \
    --do_eval \
    --eval_strategy epoch \
    --per_device_eval_batch_size 32 \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_steps 820 \
    --dataloader_num_workers 0 \
    --report_to none \
    --optim adamw_torch \
    --label_names labels \
    --load_best_model_at_end \
    --metric_for_best_model accuracy \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --use_hpu_graphs_for_training
```

