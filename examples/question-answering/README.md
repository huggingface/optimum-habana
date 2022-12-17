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

# Question Answering Examples on SQuAD

Based on the script [`run_qa.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py).

**Note:** This script only works with models that have a fast tokenizer (backed by the 🤗 Tokenizers library) as it
uses special features of those tokenizers. You can check if your favorite model has a fast tokenizer in
[this table](https://huggingface.co/transformers/index.html#supported-frameworks).

`run_qa.py` allows you to fine-tune any supported model on the SQUAD dataset or another question-answering dataset of the `datasets` library or your own csv/jsonlines files as long as they are structured the same way as SQUAD. You might need to tweak the data processing inside the script if your data is structured differently.

Note that if your dataset contains samples with no possible answers (like SQUAD version 2), you need to pass along the flag `--version_2_with_negative`.

## Fine-tuning BERT on SQuAD1.1

For the following cases, an example of Gaudi configuration file is given
[here](https://github.com/huggingface/optimum-habana#how-to-use-it).


### Single-card Training

This example code fine-tunes BERT on the SQuAD1.1 dataset.
It runs in 63 minutes with BERT-large.


```bash
python run_qa.py \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --gaudi_config_name Habana/bert-large-uncased-whole-word-masking \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 24 \
  --per_device_eval_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/squad/ \
  --use_habana \
  --use_lazy_mode \
  --throughput_warmup_steps 2
```

Training with the previously defined hyper-parameters yields the following results:
```bash
f1 = 92.9397
exact_match = 86.6887
```


### Multi-card Training

Here is how you would fine-tune the BERT large model (with whole word masking) on the SQuAD dataset using the `run_qa` script, with 8 HPUs:

```bash
python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_qa.py \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --gaudi_config_name Habana/bert-large-uncased-whole-word-masking \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 8 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir /tmp/squad_output/ \
    --use_habana \
    --use_lazy_mode \
    --throughput_warmup_steps 2
```

It runs in 11 minutes with BERT-large and yields the following results:
```bash
f1 = 93.1666
exact_match = 86.8874
```


## Using DeepSpeed

Similarly to multi-card training, here is how you would fine-tune the BERT large model (with whole word masking) on the SQuAD dataset using DeepSpeed with 8 HPUs:

```bash
python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_qa.py \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --gaudi_config_name Habana/bert-large-uncased-whole-word-masking \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 8 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir /tmp/squad_output/ \
    --use_habana \
    --use_lazy_mode \
    --throughput_warmup_steps 2 \
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


### Recommended Hyperparameters for Mixed Precision

| | learning_rate | num_train_epochs | per_device_train_batch_size | per_device_eval_batch_size |
|----------------------------|:----:|:--:|:-:|:-:|
| BERT base                  | 3e-5 | 2 | 24 | 8 |
| BERT large                 | 3e-5 | 2 | 24 | 8 |
| RoBERTa base               | 3e-5 | 2 | 12 | 8 |
| RoBERTa large              | 3e-5 | 2 | 12 | 8 |
| ALBERT large (single-card) | 5e-5 | 2 | 32 | 4 |
| ALBERT large (multi-card)  | 6e-5 | 2 | 32 | 4 |
| ALBERT XXL (single-card)   | 5e-6 | 2 | 12 | 2 |
| ALBERT XXL (multi-card)    | 5e-5 | 2 | 12 | 2 |
| DistilBERT                 | 5e-5 | 3 | 8  | 8 |
