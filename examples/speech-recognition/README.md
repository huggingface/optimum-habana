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

## Table of Contents

- [Automatic Speech Recognition with CTC](#connectionist-temporal-classification)
	- [Single HPU example](#single-hpu-ctc)
	- [Multi HPU example](#multi-hpu-ctc)
- [Automatic Speech Recognition with Sequence-to-Sequence](#sequence-to-sequence)
	- [Whisper Model](#whisper-model)
	- [Fine tuning](#single-hpu-whisper-fine-tuning-with-seq2seq)
	- [Inference](#single-hpu-seq2seq-inference)


## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

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
PT_HPU_LAZY_MODE=1 python run_speech_recognition_ctc.py \
    --dataset_name="librispeech_asr" \
    --model_name_or_path="facebook/wav2vec2-large-lv60" \
    --dataset_config_name="clean" \
    --train_split_name="train.100" \
    --eval_split_name="validation" \
    --output_dir="/tmp/wav2vec2-librispeech-clean-100h-demo-dist" \
    --preprocessing_num_workers="64" \
    --dataloader_num_workers 8 \
    --overwrite_output_dir \
    --num_train_epochs="3" \
    --per_device_train_batch_size="4" \
    --learning_rate="3e-4" \
    --warmup_steps="500" \
    --text_column_name="text" \
    --layerdrop="0.0" \
    --freeze_feature_encoder \
    --chars_to_ignore '",?.!-;:“%‘”' \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --gaudi_config_name="Habana/wav2vec2" \
    --throughput_warmup_steps="3" \
    --sdp_on_bf16 \
    --bf16 \
    --use_hpu_graphs_for_training \
    --use_hpu_graphs_for_inference \
    --attn_implementation sdpa \
    --trust_remote_code True
```

On a single HPU, this script should run in *ca.* 6 hours and yield a CTC loss of **0.059** and a word error rate of **0.0423**.

> If your data has a sampling rate which is different from the one of the data the model was trained on, this script will raise an error.
> Resampling with the `datasets` library is not supported on HPUs yet. HPU graphs are supported only on Gaudi2 and from SynapseAI v1.15.

### Multi-HPU CTC

The following command shows how to fine-tune [wav2vec2-large-lv60](https://huggingface.co/facebook/wav2vec2-large-lv60) on [Librispeech](https://huggingface.co/datasets/librispeech_asr) using 8 HPUs.

```bash
PT_HPU_LAZY_MODE=1 python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_speech_recognition_ctc.py \
    --dataset_name librispeech_asr \
    --model_name_or_path facebook/wav2vec2-large-lv60 \
    --dataset_config_name clean \
    --train_split_name train.100 \
    --eval_split_name validation \
    --output_dir /tmp/wav2vec2-librispeech-clean-100h-demo-dist \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 8 \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 3e-4 \
    --warmup_steps 500 \
    --text_column_name text \
    --layerdrop 0.0 \
    --freeze_feature_encoder \
    --chars_to_ignore '",?.!-;:“%‘”' \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --gaudi_config_name Habana/wav2vec2 \
    --throughput_warmup_steps 3 \
    --bf16 \
    --sdp_on_bf16 \
    --use_hpu_graphs_for_training \
    --use_hpu_graphs_for_inference \
    --attn_implementation sdpa \
    --trust_remote_code True
```

On 8 HPUs, this script should run in *ca.* 49 minutes and yield a CTC loss of **0.0613** and a word error rate of **0.0458**.

> If your data has a sampling rate which is different from the one of the data the model was trained on, this script will raise an error.
> Resampling with the `datasets` library is not supported on HPUs yet. HPU graphs are supported only on Gaudi2 and from SynapseAI v1.15.


## DeepSpeed

> You need to install DeepSpeed with:
> ```bash
> pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.21.0
> ```

DeepSpeed can be used with almost the same command as for a multi-card run:
- `use_mpi` should be replaced by `use_deepspeed`,
- an additional `--deepspeed path_to_my_deepspeed config` argument should be provided, for instance `--deepspeed ../../tests/configs/deepspeed_zero_2.json`.

For example:
```bash
PT_HPU_LAZY_MODE=1 python ../gaudi_spawn.py \
    --world_size 8 --use_deepspeed run_speech_recognition_ctc.py \
    --dataset_name librispeech_asr \
    --model_name_or_path facebook/wav2vec2-large-lv60 \
    --dataset_config_name clean \
    --train_split_name train.100 \
    --eval_split_name validation \
    --output_dir /tmp/wav2vec2-librispeech-clean-100h-demo-dist \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 8 \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 3e-4 \
    --warmup_steps 500 \
    --text_column_name text \
    --layerdrop 0.0 \
    --freeze_feature_encoder \
    --chars_to_ignore '",?.!-;:“%‘”' \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --gaudi_config_name Habana/wav2vec2 \
    --throughput_warmup_steps 3 \
    --deepspeed ../../tests/configs/deepspeed_zero_2.json \
    --sdp_on_bf16 \
    --attn_implementation sdpa \
    --trust_remote_code True
```

[The documentation](https://huggingface.co/docs/optimum/habana/usage_guides/deepspeed) provides more information about how to use DeepSpeed within Optimum Habana.

> If your data has a sampling rate which is different from the one of the data the model was trained on, this script will raise an error.
> Resampling with the `datasets` library is not supported on HPUs yet.


## Inference

To run only inference, you can start from the commands above and you just have to remove the training-only arguments such as `--do_train`, `--per_device_train_batch_size`, `--num_train_epochs`, etc...

For instance, you can run inference with Wav2Vec2 on the Librispeech dataset on 1 Gaudi card with the following command:
```bash
PT_HPU_LAZY_MODE=1 python run_speech_recognition_ctc.py \
    --dataset_name="librispeech_asr" \
    --model_name_or_path="facebook/wav2vec2-large-lv60" \
    --dataset_config_name="clean" \
    --train_split_name="train.100" \
    --eval_split_name="validation" \
    --output_dir="/tmp/wav2vec2-librispeech-clean-100h-demo-dist" \
    --preprocessing_num_workers="64" \
    --dataloader_num_workers 8 \
    --overwrite_output_dir \
    --text_column_name="text" \
    --chars_to_ignore '",?.!-;:“%‘”' \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --gaudi_config_name="Habana/wav2vec2" \
    --sdp_on_bf16 \
    --bf16 \
    --use_hpu_graphs_for_inference \
    --trust_remote_code True
```

## Sequence to Sequence

The script [`run_speech_recognition_seq2seq.py`](https://github.com/huggingface/optimum-habana/examples/speech-recognition/run_speech_recognition_seq2seq.py) can be used to fine-tune any [Whisper Sequence-to-Sequence Model](https://huggingface.co/docs/transformers/main/en/model_doc/whisper#whisper) for automatic speech
recognition on one of the well known speech recognition datasets similar to shown below or a custom dataset. Examples of two datasets using the Whisper model from OpenAI are included below.

### Whisper Model
We can load all components of the Whisper model directly from the pretrained checkpoint, including the pretrained model weights, feature extractor and tokenizer. We simply have to specify our fine-tuning dataset and training hyperparameters.

### Single HPU Whisper Fine tuning with Seq2Seq
The following example shows how to fine-tune the [Whisper small](https://huggingface.co/openai/whisper-small) checkpoint on the Hindi subset of [Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) using a single HPU device in bf16 precision:
```bash
PT_HPU_LAZY_MODE=1 python run_speech_recognition_seq2seq.py \
    --model_name_or_path="openai/whisper-small" \
    --dataset_name="mozilla-foundation/common_voice_11_0" \
    --trust_remote_code \
    --dataset_config_name="hi" \
    --language="hindi" \
    --task="transcribe" \
    --train_split_name="train+validation" \
    --eval_split_name="test" \
    --gaudi_config_name="Habana/whisper" \
    --max_steps="5000" \
    --output_dir="/tmp/whisper-small-hi" \
    --per_device_train_batch_size="48" \
    --per_device_eval_batch_size="2" \
    --logging_steps="25" \
    --learning_rate="1e-5" \
    --warmup_steps="500" \
    --eval_strategy="steps" \
    --eval_steps="1000" \
    --save_strategy="steps" \
    --save_steps="1000" \
    --generation_max_length="225" \
    --preprocessing_num_workers="1" \
    --max_duration_in_seconds="30" \
    --text_column_name="sentence" \
    --freeze_feature_encoder="False" \
    --sdp_on_bf16 \
    --bf16 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --use_habana \
    --use_hpu_graphs_for_inference \
    --label_features_max_length 128 \
    --dataloader_num_workers 8 \
    --throughput_warmup_steps 3 \
    --sdp_on_bf16
```

If training on a different language, you should be sure to change the `language` argument. The `language` and `task` arguments should be omitted for English speech recognition.


### Multi HPU Whisper Training with Seq2Seq
The following example shows how to fine-tune the [Whisper large](https://huggingface.co/openai/whisper-large) checkpoint on the Hindi subset of [Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) using 8 HPU devices in half-precision:
```bash
PT_HPU_LAZY_MODE=1 python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_speech_recognition_seq2seq.py \
    --model_name_or_path="openai/whisper-large" \
    --dataset_name="mozilla-foundation/common_voice_11_0" \
    --trust_remote_code \
    --dataset_config_name="hi" \
    --language="hindi" \
    --task="transcribe" \
    --train_split_name="train+validation" \
    --eval_split_name="test" \
    --gaudi_config_name="Habana/whisper" \
    --max_steps="625" \
    --output_dir="/tmp/whisper-large-hi" \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="2" \
    --logging_steps="25" \
    --learning_rate="1e-5" \
    --generation_max_length="225" \
    --preprocessing_num_workers="1" \
    --max_duration_in_seconds="30" \
    --text_column_name="sentence" \
    --freeze_feature_encoder="False" \
    --sdp_on_bf16 \
    --bf16 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --use_habana \
    --use_hpu_graphs_for_inference \
    --label_features_max_length 128 \
    --dataloader_num_workers 8 \
    --gradient_checkpointing \
    --throughput_warmup_steps 3
```

#### Single HPU Seq2Seq Inference

The following example shows how to do inference with the [Whisper small](https://huggingface.co/openai/whisper-small) checkpoint on the Hindi subset of [Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) using 1 HPU devices in half-precision:

```bash
PT_HPU_LAZY_MODE=1 python run_speech_recognition_seq2seq.py \
    --model_name_or_path="openai/whisper-small" \
    --dataset_name="mozilla-foundation/common_voice_11_0" \
    --trust_remote_code \
    --dataset_config_name="hi" \
    --language="hindi" \
    --task="transcribe" \
    --eval_split_name="test" \
    --gaudi_config_name="Habana/whisper" \
    --output_dir="./results/whisper-small-clean" \
    --per_device_eval_batch_size="32" \
    --generation_max_length="225" \
    --preprocessing_num_workers="1" \
    --max_duration_in_seconds="30" \
    --text_column_name="sentence" \
    --freeze_feature_encoder="False" \
    --sdp_on_bf16 \
    --bf16 \
    --overwrite_output_dir \
    --do_eval \
    --predict_with_generate \
    --use_habana \
    --use_hpu_graphs_for_inference \
    --label_features_max_length 128 \
    --dataloader_num_workers 8 \
    --sdp_on_bf16
```
