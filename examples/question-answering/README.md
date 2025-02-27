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

## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

## Fine-tuning Llama on SQuAD1.1

> [!NOTE]
>   Llama/Llama2 for question answering requires Transformers v4.38.0 or newer, which supports the `LlamaForQuestionAnswering` class.

Here is a command you can run to train a Llama model for question answering:
```bash
python ../gaudi_spawn.py \
  --world_size 8 --use_deepspeed run_qa.py \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --gaudi_config_name Habana/bert-large-uncased-whole-word-masking \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/squad_output/ \
  --use_habana \
  --use_lazy_mode \
  --use_hpu_graphs_for_inference \
  --throughput_warmup_steps 3 \
  --max_train_samples 45080 \
  --deepspeed ../../tests/configs/deepspeed_zero_2.json \
  --sdp_on_bf16
```


## Inference

To run only inference, you can start from the commands above and you just have to remove the training-only arguments such as `--do_train`, `--per_device_train_batch_size`, `--num_train_epochs`, etc...
<<<<<<< HEAD
=======

For instance, you can run inference with BERT on SQuAD on 1 Gaudi card with the following command:
```bash
PT_HPU_LAZY_MODE=1 python run_qa.py \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --gaudi_config_name Habana/bert-large-uncased-whole-word-masking \
  --dataset_name squad \
  --do_eval \
  --per_device_eval_batch_size 8 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/squad/ \
  --use_habana \
  --use_lazy_mode \
  --use_hpu_graphs_for_inference \
  --bf16 \
  --sdp_on_bf16
```


## Recommended Hyperparameters for Mixed Precision

| | learning_rate | num_train_epochs | per_device_train_batch_size | per_device_eval_batch_size |
|----------------------------|:----:|:--:|:-:|:-:|
| BERT base                  | 3e-5 | 2 | 24 | 8 |
| BERT large                 | 3e-5 | 2 | 24 | 8 |
| RoBERTa base               | 3e-5 | 2 | 12 | 8 |
| RoBERTa large              | 3e-5 | 2 | 12 | 8 |
| ALBERT large (single-card) | 5e-5 | 2 | 32 | 4 |
| ALBERT large (multi-card)  | 6e-5 | 2 | 32 | 4 |
| ALBERT XXL (single-card)   | 5e-6 | 2 | 16 | 2 |
| ALBERT XXL (multi-card)    | 5e-5 | 2 | 16 | 2 |
| DistilBERT                 | 5e-5 | 3 | 8  | 8 |
| meta-llama/Llama-2-13b-chat-hf (multi-card) | 3e-5 | 2 | 8 | 8 |
| FlagAlpha/Llama2-Chinese-13b-Chat (multi-card) | 3e-5 | 2 | 8 | 8 |


## Fine-tuning T5 on SQuAD2.0

The [`run_seq2seq_qa.py`](https://github.com/huggingface/optimum-habana/blob/main/examples/question-answering/run_seq2seq_qa.py) script is meant for encoder-decoder (also called seq2seq) Transformer models, such as T5 or BART. These models are generative, rather than discriminative. This means that they learn to generate the correct answer, rather than predicting the start and end position of the tokens of the answer.

The following command fine-tunes T5 on the SQuAD2.0 dataset:

```bash
PT_HPU_LAZY_MODE=1 python run_seq2seq_qa.py \
  --model_name_or_path t5-small \
  --gaudi_config_name Habana/t5 \
  --dataset_name squad_v2 \
  --version_2_with_negative \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 33 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/seq2seq_squad/ \
  --predict_with_generate \
  --use_habana \
  --use_lazy_mode \
  --use_hpu_graphs_for_inference \
  --ignore_pad_token_for_loss False \
  --pad_to_max_length \
  --save_strategy epoch \
  --throughput_warmup_steps 3 \
  --sdp_on_bf16 \
  --bf16
```

For multi-card and DeepSpeed runs, you can use `python ../gaudi_spawn.py --world_size 8 --use_mpi` and `python ../gaudi_spawn.py --world_size 8 --use_deepspeed` as shown in the previous sections.
>>>>>>> b56bafaf ([SW-218526] Updated Readme files for explicite lazy mode part2 (#177))
