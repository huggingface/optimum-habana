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
  --model_name_or_path FlagAlpha/Llama2-Chinese-13b-Chat \
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
