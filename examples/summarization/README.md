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

# Summarization Examples

This directory contains examples for finetuning and evaluating transformers on summarization tasks.

`run_summarization.py` is a lightweight example of how to download and preprocess a dataset from the [🤗 Datasets](https://github.com/huggingface/datasets) library or use your own files (jsonlines or csv), then fine-tune and evaluate T5 (or predict using BART) on it.

For custom datasets in `jsonlines` format please see: https://huggingface.co/docs/datasets/loading_datasets#json-files.
You will also find examples of these below.

## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

## Single-card Training

Here is an example of a summarization task with T5:

```bash
python run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gaudi_config_name Habana/t5 \
    --ignore_pad_token_for_loss False \
    --pad_to_max_length \
    --save_strategy epoch \
    --throughput_warmup_steps 3 \
    --bf16
```

Only T5 models `t5-small`, `t5-base`, `t5-large`, `t5-3b` and `t5-11b` must use an additional argument: `--source_prefix "summarize: "`.

We used CNN/DailyMail dataset in this example as `t5-small` was trained on it and one can get good scores even when pre-training with a very small sample.

Extreme Summarization (XSum) Dataset is another commonly used dataset for the task of summarization. To use it replace `--dataset_name cnn_dailymail --dataset_config "3.0.0"` with  `--dataset_name xsum`.

And here is how you would use it on your own files, after adjusting the values for the arguments
`--train_file`, `--validation_file`, `--text_column` and `--summary_column` to match your setup:

```bash
python run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gaudi_config_name Habana/t5 \
    --ignore_pad_token_for_loss False \
    --pad_to_max_length \
    --throughput_warmup_steps 3 \
    --bf16
```

The task of summarization also supports custom CSV and JSONLINES formats.

### Custom CSV Files

If it's a csv file the training and validation files should have a column for the inputs texts and a column for the summaries.

If the csv file has just two columns as in the following example:

```csv
text,summary
"I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder","I'm sitting in a room where I'm waiting for something to happen"
"I see trees so green, red roses too. I see them bloom for me and you. And I think to myself what a wonderful world. I see skies so blue and clouds so white. The bright blessed day, the dark sacred night. And I think to myself what a wonderful world.","I'm a gardener and I'm a big fan of flowers."
"Christmas time is here. Happiness and cheer. Fun for all that children call. Their favorite time of the year. Snowflakes in the air. Carols everywhere. Olden times and ancient rhymes. Of love and dreams to share","It's that time of year again."
```

The first column is assumed to be for `text` and the second is for the summary.

If the csv file has multiple columns, you can then specify the names of the columns to use:

```bash
    --text_column text_column_name \
    --summary_column summary_column_name \
```

For example, if the columns were:

```csv
id,date,text,summary
```

and you wanted to select only `text` and `summary`, then you'd pass these additional arguments:

```bash
    --text_column text \
    --summary_column summary \
```

### Custom JSONLINES Files

The second supported format is jsonlines. Here is an example of a jsonlines custom data file.


```json
{"text": "I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder", "summary": "I'm sitting in a room where I'm waiting for something to happen"}
{"text": "I see trees so green, red roses too. I see them bloom for me and you. And I think to myself what a wonderful world. I see skies so blue and clouds so white. The bright blessed day, the dark sacred night. And I think to myself what a wonderful world.", "summary": "I'm a gardener and I'm a big fan of flowers."}
{"text": "Christmas time is here. Happiness and cheer. Fun for all that children call. Their favorite time of the year. Snowflakes in the air. Carols everywhere. Olden times and ancient rhymes. Of love and dreams to share", "summary": "It's that time of year again."}
```

Same as with the CSV files, by default the first value will be used as the text record and the second as the summary record. Therefore you can use any key names for the entries, in this example `text` and `summary` were used.

And as with the CSV files, you can specify which values to select from the file, by explicitly specifying the corresponding key names. In our example this again would be:

```bash
    --text_column text \
    --summary_column summary \
```


## Multi-card Training

Here is an example on 8 HPUs:
```bash
python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config '"3.0.0"' \
    --source_prefix '"summarize: "' \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gaudi_config_name Habana/t5 \
    --ignore_pad_token_for_loss False \
    --pad_to_max_length \
    --save_strategy epoch \
    --throughput_warmup_steps 3 \
    --bf16
```


## Using DeepSpeed

Here is an example on 8 HPUs on Gaudi2 with DeepSpeed-ZeRO3 to fine-tune [FLAN-T5 XXL](https://huggingface.co/google/flan-t5-xxl):
```bash
PT_HPU_MAX_COMPOUND_OP_SIZE=512 python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_summarization.py \
    --model_name_or_path google/flan-t5-xxl \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config '"3.0.0"' \
    --source_prefix '"summarize: "' \
    --output_dir ./tst-summarization \
    --per_device_train_batch_size 22 \
    --per_device_eval_batch_size 22 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --overwrite_output_dir \
    --predict_with_generate \
    --use_habana \
    --use_lazy_mode \
    --gaudi_config_name Habana/t5 \
    --ignore_pad_token_for_loss False \
    --pad_to_max_length \
    --generation_max_length 129 \
    --save_strategy epoch \
    --throughput_warmup_steps 3 \
    --gradient_checkpointing \
    --adam_epsilon 1e-08 --logging_steps 1 \
    --deepspeed ds_flan_t5_z3_config_bf16.json
```

You can look at the [documentation](https://huggingface.co/docs/optimum/habana/usage_guides/deepspeed) for more information about how to use DeepSpeed in Optimum Habana.


## Inference

To run only inference, you can start from the commands above and you just have to remove the training-only arguments such as `--do_train`, `--per_device_train_batch_size`, `--num_train_epochs`, etc...

For instance, you can run inference with T5 on the CNN-DailyMail dataset on 1 Gaudi card with the following command:
```bash
python run_summarization.py \
    --model_name_or_path t5-small \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gaudi_config_name Habana/t5 \
    --ignore_pad_token_for_loss False \
    --pad_to_max_length \
    --bf16 \
    --bf16_full_eval
```

You can run inference with BART on the CNN-DailyMail dataset on 1 Gaudi card with the following command:
```bash
python run_summarization.py \
    --model_name_or_path facebook/bart-large-cnn \
    --do_predict \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir /tmp/tst-summarization \
    --per_device_eval_batch_size 2 \
    --overwrite_output_dir \
    --predict_with_generate \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gaudi_config_name Habana/bart \
    --ignore_pad_token_for_loss False \
    --pad_to_max_length \
    --num_beams 1
```
