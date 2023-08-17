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

# Automatic Speech Recognition Examples

## Connectionist Temporal Classification

The script [`run_speech_recognition_ctc.py`](https://github.com/huggingface/optimum-habana/tree/main/examples/speech-recognition/run_speech_recognition_ctc.py) can be used to fine-tune any pretrained [Connectionist Temporal Classification Model](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCTC) for automatic speech recognition on one of the [official speech recognition datasets](https://huggingface.co/datasets?task_ids=task_ids:automatic-speech-recognition) or a custom dataset.

Speech recognition models that have been pretrained in an unsupervised fashion on audio data alone, *e.g.* [Wav2Vec2](https://huggingface.co/transformers/main/model_doc/wav2vec2.html), have shown to require only very little annotated data to yield good performance on automatic speech recognition datasets.

In the script [`run_speech_recognition_ctc`](https://github.com/huggingface/optimum-habana/tree/main/examples/speech-recognition/run_speech_recognition_ctc.py), we first create a vocabulary from all unique characters of both the training data and evaluation data. Then, we preprocess the speech recognition dataset, which includes correct resampling, normalization and padding. Finally, the pretrained speech recognition model is fine-tuned on the annotated speech recognition datasets using CTC loss.

<!-- ---
**NOTE**

If you encounter problems with data preprocessing by setting `--preprocessing_num_workers` > 1,
you might want to set the environment variable `OMP_NUM_THREADS` to 1 as follows:

```bash
OMP_NUM_THREADS=1 python run_speech_recognition_ctc ...
```

If the environment variable is not set, the training script might freeze, *i.e.* see: https://github.com/pytorch/audio/issues/1021#issuecomment-726915239

--- -->

### Single-HPU CTC

The following command shows how to fine-tune [wav2vec2-large-lv60](https://huggingface.co/facebook/wav2vec2-large-lv60) on [Librispeech](https://huggingface.co/datasets/librispeech_asr) using a single HPU.

```bash
python run_speech_recognition_ctc.py \
    --dataset_name="librispeech_asr" \
    --model_name_or_path="facebook/wav2vec2-large-lv60" \
    --dataset_config_name="clean" \
    --train_split_name="train.100" \
    --eval_split_name="validation" \
    --output_dir="/tmp/wav2vec2-librispeech-clean-100h-demo-dist" \
    --preprocessing_num_workers="64" \
    --overwrite_output_dir \
    --num_train_epochs="3" \
    --per_device_train_batch_size="4" \
    --learning_rate="3e-4" \
    --warmup_steps="500" \
    --text_column_name="text" \
    --layerdrop="0.0" \
    --freeze_feature_encoder \
    --chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gaudi_config_name="Habana/wav2vec2" \
    --throughput_warmup_steps="3"
```

On a single HPU, this script should run in *ca.* 6 hours and yield a CTC loss of **0.059** and a word error rate of **0.0423**.

> If your data has a sampling rate which is different from the one of the data the model was trained on, this script will raise an error.
> Resampling with the `datasets` library is not supported on HPUs yet.

### Multi-HPU CTC

The following command shows how to fine-tune [wav2vec2-large-lv60](https://huggingface.co/facebook/wav2vec2-large-lv60) on [Librispeech](https://huggingface.co/datasets/librispeech_asr) using 8 HPUs.

```bash
python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_speech_recognition_ctc.py \
    --dataset_name librispeech_asr \
    --model_name_or_path facebook/wav2vec2-large-lv60 \
    --dataset_config_name clean \
    --train_split_name train.100 \
    --eval_split_name validation \
    --output_dir /tmp/wav2vec2-librispeech-clean-100h-demo-dist \
    --preprocessing_num_workers 64 \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 3e-4 \
    --warmup_steps 500 \
    --text_column_name text \
    --layerdrop 0.0 \
    --freeze_feature_encoder \
    --chars_to_ignore '",?.!-;:\"“%‘”"' \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gaudi_config_name Habana/wav2vec2 \
    --throughput_warmup_steps 3
```

On 8 HPUs, this script should run in *ca.* 49 minutes and yield a CTC loss of **0.0613** and a word error rate of **0.0458**.

> If your data has a sampling rate which is different from the one of the data the model was trained on, this script will raise an error.
> Resampling with the `datasets` library is not supported on HPUs yet.


## DeepSpeed

> You need to install DeepSpeed with:
> ```bash
> pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.11.0
> ```

DeepSpeed can be used with almost the same command as for a multi-card run:
- `use_mpi` should be replaced by `use_deepspeed`,
- an additional `--deepspeed path_to_my_deepspeed config` argument should be provided, for instance `--deepspeed ../../tests/configs/deepspeed_zero_2.json`.

For example:
```bash
python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_speech_recognition_ctc.py \
    --dataset_name librispeech_asr \
    --model_name_or_path facebook/wav2vec2-large-lv60 \
    --dataset_config_name clean \
    --train_split_name train.100 \
    --eval_split_name validation \
    --output_dir /tmp/wav2vec2-librispeech-clean-100h-demo-dist \
    --preprocessing_num_workers 64 \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 3e-4 \
    --warmup_steps 500 \
    --text_column_name text \
    --layerdrop 0.0 \
    --freeze_feature_encoder \
    --chars_to_ignore '",?.!-;:\"“%‘”"' \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --gaudi_config_name Habana/wav2vec2 \
    --throughput_warmup_steps 3\
    --deepspeed ../../tests/configs/deepspeed_zero_2.json
```

[The documentation](https://huggingface.co/docs/optimum/habana/usage_guides/deepspeed) provides more information about how to use DeepSpeed within Optimum Habana.

> If your data has a sampling rate which is different from the one of the data the model was trained on, this script will raise an error.
> Resampling with the `datasets` library is not supported on HPUs yet.


## Inference

To run only inference, you can start from the commands above and you just have to remove the training-only arguments such as `--do_train`, `--per_device_train_batch_size`, `--num_train_epochs`, etc...

For instance, you can run inference with Wav2Vec2 on the Librispeech dataset on 1 Gaudi card with the following command:
```bash
python run_speech_recognition_ctc.py \
    --dataset_name="librispeech_asr" \
    --model_name_or_path="facebook/wav2vec2-large-lv60" \
    --dataset_config_name="clean" \
    --train_split_name="train.100" \
    --eval_split_name="validation" \
    --output_dir="/tmp/wav2vec2-librispeech-clean-100h-demo-dist" \
    --preprocessing_num_workers="64" \
    --overwrite_output_dir \
    --text_column_name="text" \
    --chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --gaudi_config_name="Habana/wav2vec2"
```
