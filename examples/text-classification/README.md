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

# Text classification examples

## GLUE tasks

Based on the script [`run_glue.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py).

Fine-tuning the library models for sequence classification on the GLUE benchmark: [General Language Understanding
Evaluation](https://gluebenchmark.com/). This script can fine-tune any of the models on the [hub](https://huggingface.co/models)
and can also be used for a dataset hosted on our [hub](https://huggingface.co/datasets) or your own data in a csv or a JSON file
(the script might need some tweaks in that case, refer to the comments inside for help).

GLUE is made up of a total of 9 different tasks where task name can be one of cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli.


## Fine-tuning BERT on MRPC

For the following cases, an example of Gaudi configuration file is given
[here](https://github.com/huggingface/optimum-habana#how-to-use-it).


### Single-card Training

The following example fine-tunes BERT Large (lazy mode) on the `mrpc` dataset hosted on our [hub](https://huggingface.co/datasets):

```bash
python run_glue.py \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --gaudi_config_name gaudi_config_name_or_path \
  --task_name mrpc \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 128 \
  --output_dir ./output/mrpc/ \
  --use_habana \
  --use_lazy_mode
```

### Multi-card Training

Here is how you would fine-tune the BERT large model (with whole word masking) on the text classification MRPC task using the `run_glue`
script, with 8 HPUs:

```bash
python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_glue.py \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --gaudi_config_name gaudi_config_name_or_path \
    --task_name mrpc \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length 128 \
    --output_dir /tmp/mrpc_output/ \
    --use_habana \
    --use_lazy_mode
```
