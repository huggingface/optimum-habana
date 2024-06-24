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

# Translation Examples

`run_translation.py` is a lightweight example of how to download and preprocess a dataset from the [🤗 Datasets](https://github.com/huggingface/datasets) library or use your own files (jsonlines or csv), then fine-tune one of the architectures above on it.

For custom datasets in `jsonlines` format please see: https://huggingface.co/docs/datasets/loading_datasets#json-files.
You will also find examples of these below.

## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

## Single-card Training

Here is an example of a translation fine-tuning with a T5 model.
T5 models `t5-small`, `t5-base`, `t5-large`, `t5-3b` and `t5-11b` must use an additional argument: `--source_prefix "translate {source_lang} to {target_lang}"`. For instance:

```bash
python run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
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

If you get a terrible BLEU score, make sure that you didn't forget to use the `--source_prefix` argument.

For the aforementioned group of T5 models, it's important to remember that if you switch to a different language pair, make sure to adjust the source and target values in all 3 language-specific command line arguments: `--source_lang`, `--target_lang` and `--source_prefix`.

In lazy mode, make sure to use the arguments `--pad_to_max_length` and `--ignore_pad_token_for_loss False` to pad batches to max length and to avoid negative pad tokens.

And here is how you would use the translation finetuning on your own files, after adjusting the
values for the arguments `--train_file`, `--validation_file` to match your setup:

```bash
python run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --train_file path_to_jsonlines_file \
    --validation_file path_to_jsonlines_file \
    --output_dir /tmp/tst-translation \
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
    --throughput_warmup_steps 3 \
    --bf16
```

The task of translation supports only custom JSONLINES files, with each line being a dictionary with the key `"translation"` and its value another dictionary whose keys is the language pair. For example:

```json
{ "translation": { "en": "Others have dismissed him as a joke.", "ro": "Alții l-au numit o glumă." } }
{ "translation": { "en": "And some are holding out for an implosion.", "ro": "Iar alții așteaptă implozia." } }
```
Here the languages are Romanian (`ro`) and English (`en`).

If you want to use a pre-processed dataset that leads to high BLEU scores, but for the `en-de` language pair, you can use `--dataset_name stas/wmt14-en-de-pre-processed`, as follows:

```bash
python run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang de \
    --source_prefix "translate English to German: " \
    --dataset_name stas/wmt14-en-de-pre-processed \
    --output_dir /tmp/tst-translation \
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
    --throughput_warmup_steps 3 \
    --bf16
 ```


 ## Multi-card Training

 Here is an example of distributing training on 8 HPUs:

 ```bash
python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix '"translate English to Romanian: "' \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
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

 Here is an example with DeepSpeed on 8 HPUs:

 ```bash
python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix '"translate English to Romanian: "' \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
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


## Inference

To run only inference, you can start from the commands above and you just have to remove the training-only arguments such as `--do_train`, `--per_device_train_batch_size`, `--num_train_epochs`, etc...

For instance, you can run inference with BERT on GLUE on 1 Gaudi card with the following command:
```bash
python run_translation.py \
    --model_name_or_path t5-small \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gaudi_config_name Habana/t5 \
    --ignore_pad_token_for_loss False \
    --pad_to_max_length \
    --bf16
```
