<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

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

# Token Classification Example

`run_token_classification.py` is a lightweight example of how to download and preprocess a dataset from the [🤗 Datasets](https://github.com/huggingface/datasets) library or use your own files (jsonlines or csv), then fine-tune `lilt-roberta-en-base`.


# Single-card Training

Here is an example of a translation fine-tuning with a FUNSD dataset.

1. Install the required dependencies
```bash
apt install -y tesseract-ocr
pip install -r requirements.txt
```

2. Run Fine-tuning on FUND dataset for SCUT-DLVCLab/lilt-roberta-en-base

```bash
python run_token_classification.py \
  --model_name_or_path SCUT-DLVCLab/lilt-roberta-en-base \
  --gaudi_config_name Habana/roberta-base \
  --dataset_name nielsr/funsd-layoutlmv3 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --output_dir ./results \
  --use_habana \
  --bf16 \
  --logging_strategy epoch \
  --evaluation_strategy epoch \
  --num_train_epochs 10 \
  --save_strategy epoch \
  --save_total_limit 2 \
  --load_best_model_at_end \
  --metric_for_best_model overall_f1 \
  --dataloader_num_workers 4 \
  --non_blocking_data_copy \
  --gradient_checkpointing \
  --use_hpu_graphs \
  --use_lazy_mode
```

## Multi-card using DeepSpeed

Here is an example with DeepSpeed on 8 HPUs:

```bash
python ../gaudi_spawn.py \
    --world_size 8 \
    --use_deepspeed run_token_classification.py \
    --model_name_or_path SCUT-DLVCLab/lilt-roberta-en-base \
    --gaudi_config Habana/roberta-base  \
    --do_train  \
    --do_eval \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8  \
    --learning_rate 5e-5  \
    --num_train_epochs 3 \
    --output_dir /tmp/tkn_output/ \
    --overwrite_output_dir \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --throughput_warmup_steps 3  \
    --deepspeed path-to-deepspeed-config
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

## Inference

And here is how you would use the Fined-tuned model to perform inferencing, in this case on FUNSD dataset's validation dataset.

```bash
python run_token_classification_inference.py \
  --device_type hpu \
  --dataset_id "nielsr/funsd-layoutlmv3" \
  --model_path "./results/" \
  --num_images 10 \
  --batch_size 4 \
  --precision bf16 \
  --use_hpu_graphs
```