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

<!-- ## Table of Contents

- [Automatic Speech Recognition with CTC](#connectionist-temporal-classification)
	- [Single-HPU example](#single-gpu-ctc)
	- [Multi-HPU example](#multi-gpu-ctc)
- [Automatic Speech Recognition with Sequence-to-Sequence](#sequence-to-sequence)
	- [Single GPU example](#single-gpu-seq2seq)
	- [Multi GPU example](#multi-gpu-seq2seq)
	- [Examples](#examples-seq2seq)
		- [Librispeech](#librispeech-seq2seq) -->

## Connectionist Temporal Classification

The script [`run_speech_recognition_ctc.py`](https://github.com/huggingface/optimum-habana/tree/main/examples/speech-recognition/run_speech_recognition_ctc.py) can be used to fine-tune any pretrained [Connectionist Temporal Classification Model](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCTC) for automatic speech recognition on one of the [official speech recognition datasets](https://huggingface.co/datasets?task_ids=task_ids:automatic-speech-recognition) or a custom dataset.

Speech recognition models that have been pretrained in unsupervised fashion on audio data alone, *e.g.* [Wav2Vec2](https://huggingface.co/transformers/main/model_doc/wav2vec2.html), have shown to require only
very little annotated data to yield good performance on automatic speech recognition datasets.

In the script [`run_speech_recognition_ctc`], we first create a vocabulary from all unique characters of both the training data and evaluation data. Then, we preprocesses the speech recognition dataset, which includes correct resampling, normalization and padding. Finally, the pretrained speech recognition model is fine-tuned on the annotated speech recognition datasets using CTC loss.

---
**NOTE**

If you encounter problems with data preprocessing by setting `--preprocessing_num_workers` > 1,
you might want to set the environment variable `OMP_NUM_THREADS` to 1 as follows:

```bash
OMP_NUM_THREADS=1 python run_speech_recognition_ctc ...
```

If the environment variable is not set, the training script might freeze, *i.e.* see: https://github.com/pytorch/audio/issues/1021#issuecomment-726915239

---

### Single-HPU CTC

The following command shows how to fine-tune [Wav2Vec2](https://huggingface.co/transformers/main/model_doc/wav2vec2.html) on [Librispeech](https://huggingface.co/datasets/librispeech_asr) using a single HPU.

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
    --per_device_train_batch_size="8" \
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
    --gaudi_config_name="Habana/wav2vec2" \
    --throughput_warmup_steps="2"
```

On a single HPU, this script should run in *ca.* 1 hour 20 minutes and yield a CTC loss of **0.39** and word error rate of **0.35**.

### Multi-GPU CTC

The following command shows how to fine-tune [Wav2Vec2](https://huggingface.co/transformers/main/model_doc/wav2vec2.html) on [Librispeech](https://huggingface.co/datasets/librispeech_asr) using 8 HPUs.

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
    --gaudi_config_name Habana/wav2vec2 \
    --throughput_warmup_steps 2
```

On 8 HPUs, this script should run in *ca.* 49 minutes and yield a CTC loss of **0.0615** and word error rate of **0.0448**.


<!-- ### Multi GPU CTC with Dataset Streaming

The following command shows how to use [Dataset Streaming mode](https://huggingface.co/docs/datasets/dataset_streaming.html)
to fine-tune [XLS-R](https://huggingface.co/transformers/main/model_doc/xls_r.html)
on [Common Voice](https://huggingface.co/datasets/common_voice) using 4 GPUs in half-precision.

Streaming mode imposes several constraints on training:
1. We need to construct a tokenizer beforehand and define it via `--tokenizer_name_or_path`.
2. `--num_train_epochs` has to be replaced by `--max_steps`. Similarly, all other epoch-based arguments have to be replaced by step-based ones.
3. Full dataset shuffling on each epoch is not possible, since we don't have the whole dataset available at once.
However, the `--shuffle_buffer_size` argument controls how many examples we can pre-download before shuffling them.


```bash
**python -m torch.distributed.launch \
	--nproc_per_node 4 run_speech_recognition_ctc_streaming.py \
	--dataset_name="common_voice" \
	--model_name_or_path="facebook/wav2vec2-xls-r-300m" \
	--tokenizer_name_or_path="anton-l/wav2vec2-tokenizer-turkish" \
	--dataset_config_name="tr" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--output_dir="wav2vec2-xls-r-common_voice-tr-ft" \
	--overwrite_output_dir \
	--max_steps="5000" \
	--per_device_train_batch_size="8" \
	--gradient_accumulation_steps="2" \
	--learning_rate="5e-4" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--text_column_name="sentence" \
	--save_steps="500" \
	--eval_steps="500" \
	--logging_steps="1" \
	--layerdrop="0.0" \
	--eval_metrics wer cer \
	--save_total_limit="1" \
	--mask_time_prob="0.3" \
	--mask_time_length="10" \
	--mask_feature_prob="0.1" \
	--mask_feature_length="64" \
	--freeze_feature_encoder \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
	--max_duration_in_seconds="20" \
	--shuffle_buffer_size="500" \
	--fp16 \
	--push_to_hub \
	--do_train --do_eval \
	--gradient_checkpointing**
```

On 4 V100 GPUs, this script should run in *ca.* 3h 31min and yield a CTC loss of **0.35** and word error rate
of **0.29**. -->


<!-- ## Sequence to Sequence

The script [`run_speech_recognition_seq2seq.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py) can be used to fine-tune any [Speech Sequence-to-Sequence Model](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForSpeechSeq2Seq) for automatic speech recognition on one of the [official speech recognition datasets](https://huggingface.co/datasets?task_ids=task_ids:automatic-speech-recognition) or a custom dataset.

A very common use case is to leverage a pretrained speech [encoding model](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModel),
*e.g.* [Wav2Vec2](https://huggingface.co/transformers/main/model_doc/wav2vec2.html), with a pretrained [text decoding model](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModel), *e.g.* [Bart](https://huggingface.co/docs/transformers/main/en/model_doc/bart#transformers.BartForCausalLM) to create a [SpeechEncoderDecoderModel](https://huggingface.co/docs/transformers/main/en/model_doc/speechencoderdecoder#speech-encoder-decoder-models).
Consequently, the warm-started Speech-Encoder-Decoder model can be fine-tuned in this script.

As an example, let's instantiate a *Wav2Vec2-2-Bart* model with the `SpeechEncoderDecoderModel` framework:

First create an empty repo on `hf.co`:

```bash
huggingface-cli repo create wav2vec2-2-bart-base
git clone https://huggingface.co/<your-user-name>/wav2vec2-2-bart-base
cd wav2vec2-2-bart-base
```

Next, run the following script **inside** the just cloned repo:

```py
from transformers import SpeechEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer, Wav2Vec2Processor

# checkpoints to leverage
encoder_id = "facebook/wav2vec2-base"
decoder_id = "facebook/bart-base"

# load and save speech-encoder-decoder model
# set some hyper-parameters for training and evaluation
model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_id, decoder_id, encoder_add_adapter=True, encoder_feat_proj_dropout=0.0, encoder_layerdrop=0.0, max_length=200, num_beams=5)
model.config.decoder_start_token_id = model.decoder.config.bos_token_id
model.config.pad_token_id = model.decoder.config.pad_token_id
model.config.eos_token_id = model.decoder.config.eos_token_id
model.save_pretrained("./")

# load and save processor
feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
tokenizer = AutoTokenizer.from_pretrained(decoder_id)
processor = Wav2Vec2Processor(feature_extractor, tokenizer)
processor.save_pretrained("./")
```

Finally, we can upload all files:
```bash
git lfs install
git add . && git commit -m "upload model files" && git push
```

and link the official `run_speech_recognition_seq2seq.py` script to the folder:

```bash
ln -s $(realpath <path/to/transformers>/examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py) ./
```

Note that we have added a randomly initialized adapter to `wav2vec2-base` with
`encoder_add_adapter=True` which further samples the output sequence of
`wav2vec2-base` along the time dimension. The reason is that by default a single
output vector of `wav2vec2-base` has a receptive field of *ca.* 25ms (*cf.* with
section *4.2* of the [official Wav2Vec2 paper](https://arxiv.org/pdf/2006.11477.pdf)), which represents a little less a single character. BART on the other hand
makes use of a sentence-piece tokenizer as an input processor so that a single
hidden vector of `bart-base` represents *ca.* 4 characters. To better align
the output of *Wav2Vec2* and *BART*'s hidden vectors for the cross-attention
mechanism, we further subsample *Wav2Vec2*'s output by a factor of 8 by
adding a convolution-based adapter.

Having warm-started the speech-encoder-decoder model `<your-user-name>/wav2vec2-2-bart`, we can now fine-tune it on speech recognition.

In the script [`run_speech_recognition_seq2seq`], we load the warm-started model,
the feature extractor, and the tokenizer, process a speech recognition dataset,
and then make use of the [`Seq2SeqTrainer`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Seq2SeqTrainer).
Note that it is important to also align the decoder's vocabulary with
the speech transcriptions of the dataset. *E.g.* the [`Librispeech`](https://huggingface.co/datasets/librispeech_asr) has only captilized letters in the transcriptions,
whereas BART was pretrained mostly on normalized text. Thus it is recommended to add
`--do_lower_case` to the fine-tuning script when using a warm-started `SpeechEncoderDecoderModel`. The model is fine-tuned on the standard cross-entropy language modeling
loss for sequence-to-sequence (just like *T5* or *BART* in natural language processing).

---
**NOTE**

If you encounter problems with data preprocessing by setting `--preprocessing_num_workers` > 1,
you might want to set the environment variable `OMP_NUM_THREADS` to 1 as follows:

```bash
OMP_NUM_THREADS=1 python run_speech_recognition_ctc ...
```

If the environment variable is not set, the training script might freeze, *i.e.* see: https://github.com/pytorch/audio/issues/1021#issuecomment-726915239

---

### Single GPU Seq2Seq

The following command shows how to fine-tune [XLSR-Wav2Vec2](https://huggingface.co/transformers/main/model_doc/xlsr_wav2vec2.html) on [Common Voice](https://huggingface.co/datasets/common_voice) using a single GPU in half-precision.

```bash
python run_speech_recognition_seq2seq.py \
 	--nproc_per_node 8 run_speech_recognition_seq2seq.py \
	--dataset_name="librispeech_asr" \
	--model_name_or_path="./" \
	--dataset_config_name="clean" \
	--train_split_name="train.100" \
	--eval_split_name="validation" \
	--output_dir="./" \
	--preprocessing_num_workers="16" \
	--length_column_name="input_length" \
	--overwrite_output_dir \
	--num_train_epochs="5" \
	--per_device_train_batch_size="8" \
	--per_device_eval_batch_size="8" \
	--gradient_accumulation_steps="8" \
	--learning_rate="3e-4" \
	--warmup_steps="400" \
	--evaluation_strategy="steps" \
	--text_column_name="text" \
	--save_steps="400" \
	--eval_steps="400" \
	--logging_steps="10" \
	--save_total_limit="1" \
	--freeze_feature_encoder \
	--gradient_checkpointing \
	--fp16 \
	--group_by_length \
	--predict_with_generate \
	--generation_max_length="40" \
	--generation_num_beams="1" \
	--do_train --do_eval \
	--do_lower_case
```

On a single V100 GPU, this script should run in *ca.* 5 hours and yield a
cross-entropy loss of **0.405** and word error rate of **0.0728**.

### Multi GPU Seq2Seq

The following command shows how to fine-tune [XLSR-Wav2Vec2](https://huggingface.co/transformers/main/model_doc/xlsr_wav2vec2.html) on [Common Voice](https://huggingface.co/datasets/common_voice) using 8 GPUs in half-precision.

```bash
python -m torch.distributed.launch \
 	--nproc_per_node 8 run_speech_recognition_seq2seq.py \
	--dataset_name="librispeech_asr" \
	--model_name_or_path="./" \
	--dataset_config_name="clean" \
	--train_split_name="train.100" \
	--eval_split_name="validation" \
	--output_dir="./" \
	--preprocessing_num_workers="16" \
	--length_column_name="input_length" \
	--overwrite_output_dir \
	--num_train_epochs="5" \
	--per_device_train_batch_size="8" \
	--per_device_eval_batch_size="8" \
	--gradient_accumulation_steps="1" \
	--learning_rate="3e-4" \
	--warmup_steps="400" \
	--evaluation_strategy="steps" \
	--text_column_name="text" \
	--save_steps="400" \
	--eval_steps="400" \
	--logging_steps="10" \
	--save_total_limit="1" \
	--freeze_feature_encoder \
	--gradient_checkpointing \
	--fp16 \
	--group_by_length \
	--predict_with_generate \
	--do_train --do_eval \
	--do_lower_case
```

On 8 V100 GPUs, this script should run in *ca.* 45 minutes and yield a cross-entropy loss of **0.405** and word error rate of **0.0728**

### Examples Seq2Seq

#### Librispeech Seq2Seq

- [Librispeech](https://huggingface.co/datasets/librispeech_asr)

| Dataset | Dataset Config | Pretrained Model | Word error rate on eval | Phoneme error rate on eval | GPU setup | Training time | Fine-tuned Model & Logs | Command to reproduce |
|-------|------------------------------|-------------|---------------|---------------|----------------------|-------------| -------------| ------- |
| [Librispeech](https://huggingface.co/datasets/librispeech_asr)| `"clean"` - `"train.100"` |  [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) and [facebook/bart-base](https://huggingface.co/facebook/bart-base) | 0.0728 | - | 8 GPU V100 | 45min  | [here](https://huggingface.co/patrickvonplaten/wav2vec2-2-bart-base) | [create_model.py](https://huggingface.co/patrickvonplaten/wav2vec2-2-bart-base/blob/main/create_model.py) & [run.sh](https://huggingface.co/patrickvonplaten/wav2vec2-2-bart-base/blob/main/run_librispeech.sh) |
| [Librispeech](https://huggingface.co/datasets/librispeech_asr)| `"clean"` - `"train.100"` |  [facebook/wav2vec2-large-lv60](https://huggingface.co/facebook/wav2vec2-large-lv60) and [facebook/bart-large](https://huggingface.co/facebook/bart-large) | 0.0486 | - | 8 GPU V100 | 1h20min  | [here](https://huggingface.co/patrickvonplaten/wav2vec2-2-bart-large) | [create_model.py](https://huggingface.co/patrickvonplaten/wav2vec2-2-bart-large/blob/main/create_model.py) & [run.sh](https://huggingface.co/patrickvonplaten/wav2vec2-2-bart-large/blob/main/run_librispeech.sh) | -->
