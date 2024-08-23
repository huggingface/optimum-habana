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

<a href="https://github.com/huggingface/optimum-habana#gh-light-mode-only">
  <img src="https://github.com/huggingface/optimum-habana/blob/main/readme_logo_light.png"/>
</a>

<a href="https://github.com/huggingface/optimum-habana#gh-dark-mode-only">
  <img src="https://github.com/huggingface/optimum-habana/blob/main/readme_logo_dark.png"/>
</a>


# Optimum for Intel® Gaudi® Accelerators

Optimum for Intel Gaudi - a.k.a. `optimum-habana` - is the interface between the Transformers and Diffusers libraries and [Intel Gaudi AI Accelerators (HPU)](https://docs.habana.ai/en/latest/index.html).
It provides a set of tools enabling easy model loading, training and inference on single- and multi-HPU settings for different downstream tasks.
The list of officially validated models and tasks is available [here](https://github.com/huggingface/optimum-habana#validated-models). Users can try other of the thousands of Hugging Face models on Intel Gaudi accelerators and tasks with only few changes.


## What are Intel Gaudi AI Accelerators (HPUs)?

HPUs offer fast model training and inference as well as a great price-performance ratio.
Check out [this blog post about BLOOM inference](https://huggingface.co/blog/habana-gaudi-2-bloom) and [this post benchmarking Intel Gaudi 2 and NVIDIA A100 GPUs for BridgeTower training](https://huggingface.co/blog/bridgetower) for concrete examples.


## Gaudi Setup

Please refer to the Intel Gaudi AI Accelerator official [installation guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html).

> Tests should be run in a Docker container based on Intel Gaudi Docker images.
>
> The current version has been validated for SynapseAI 1.17.


## Install the library and get example scripts

### Option 1: Use the latest stable release

To install the latest stable release of this package
>```bash
>pip install --upgrade-strategy eager optimum[habana]
>```

The `--upgrade-strategy eager` option is needed to ensure `optimum-habana` is upgraded to the latest stable release.

To use the example associated with the latest stable release, run:
> ```
> git clone https://github.com/huggingface/optimum-habana
> cd optimum-habana && git checkout v1.13.1
> ```
> with `v1.13.1` the version number of this release.

### Option 2: Use the latest main branch under development

Optimum for Intel Gaudi is a fast-moving project, and you may want to install it from source and get the latest scripts :

```bash
pip install git+https://github.com/huggingface/optimum-habana.git
git clone https://github.com/huggingface/optimum-habana
```

## Install dependencies

To use DeepSpeed on HPUs, you also need to run the following command:
>```bash
>pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.17.0
>```

To install the requirements for every example:
>```bash
>cd <example-folder>
>pip install -r requirements.txt
>```


## How to use it?

### Quick Start

Optimum for Intel Gaudi was designed with one goal in mind: **to make training and inference straightforward for Transformers and Diffusers users, while fully leveraging the power of Intel Gaudi AI Accelerators**.

#### Transformers Interface

There are two main classes one needs to know:
- [GaudiTrainer](https://huggingface.co/docs/optimum/habana/package_reference/trainer): the trainer class that takes care of compiling and distributing the model to run on HPUs, and performing training and evaluation.
- [GaudiConfig](https://huggingface.co/docs/optimum/habana/package_reference/gaudi_config): the class that enables to configure Habana Mixed Precision and to decide whether optimized operators and optimizers should be used or not.

The [GaudiTrainer](https://huggingface.co/docs/optimum/habana/package_reference/trainer) is very similar to the [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), and adapting a script using the Trainer to make it work with Intel Gaudi accelerators will mostly consist in simply swapping the `Trainer` class for the `GaudiTrainer` one.
That's how most of the [example scripts](https://github.com/huggingface/optimum-habana/tree/main/examples) were adapted from their [original counterparts](https://github.com/huggingface/transformers/tree/main/examples/pytorch).

Here is an example:
```diff
- from transformers import Trainer, TrainingArguments
+ from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments

- training_args = TrainingArguments(
+ training_args = GaudiTrainingArguments(
  # training arguments...
+ use_habana=True,
+ use_lazy_mode=True,  # whether to use lazy or eager mode
+ gaudi_config_name=path_to_gaudi_config,
)

# A lot of code here

# Initialize our Trainer
- trainer = Trainer(
+ trainer = GaudiTrainer(
    model=model,
    args=training_args,  # Original training arguments.
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
```

where `gaudi_config_name` is the name of a model from the [Hub](https://huggingface.co/Habana) (Intel Gaudi configurations are stored in model repositories) or a path to a local Intel Gaudi configuration file (you can see [here](https://huggingface.co/docs/optimum/habana/package_reference/gaudi_config) how to write your own).


#### Diffusers Interface

You can generate images from prompts using Stable Diffusion on Intel Gaudi using the [`GaudiStableDiffusionPipeline`](https://huggingface.co/docs/optimum/habana/package_reference/stable_diffusion_pipeline) class and the [`GaudiDDIMScheduler`] which have been both optimized for HPUs. Here is how to use them and the differences with the Diffusers library:

```diff
- from diffusers import DDIMScheduler, StableDiffusionPipeline
+ from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline


model_name = "runwayml/stable-diffusion-v1-5"

- scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
+ scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

- pipeline = StableDiffusionPipeline.from_pretrained(
+ pipeline = GaudiStableDiffusionPipeline.from_pretrained(
    model_name,
    scheduler=scheduler,
+   use_habana=True,
+   use_hpu_graphs=True,
+   gaudi_config="Habana/stable-diffusion",
)

outputs = generator(
    ["An image of a squirrel in Picasso style"],
    num_images_per_prompt=16,
+   batch_size=4,
)
```


### Documentation

Check out [the documentation of Optimum for Intel Gaudi](https://huggingface.co/docs/optimum/habana/index) for more advanced usage.


## Validated Models

The following model architectures, tasks and device distributions have been validated for Optimum for Intel Gaudi:

> In the tables below, :heavy_check_mark: means single-card, multi-card and DeepSpeed have all been validated.

- Transformers:
<div align="center">

| Architecture | Training | Inference | <center>Tasks</center> |
|--------------|:--------:|:---------:|:-----------------------|
| BERT         | :heavy_check_mark: | :heavy_check_mark: | <li>[text classification](https://github.com/huggingface/optimum-habana/tree/main/examples/text-classification)</li><li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text feature extraction](https://github.com/huggingface/optimum-habana/tree/main/examples/text-feature-extraction)</li> |
| RoBERTa | :heavy_check_mark: | :heavy_check_mark: | <li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li> |
| ALBERT | :heavy_check_mark: | :heavy_check_mark: | <li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li> |
| DistilBERT |:heavy_check_mark: | :heavy_check_mark: | <li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li> |
| GPT2             | :heavy_check_mark: | :heavy_check_mark: | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| BLOOM(Z) |   | <div style="text-align:left"><li>DeepSpeed</li></div> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| StarCoder / StarCoder2 | :heavy_check_mark:  | <div style="text-align:left"><li>Single card</li></div> | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| GPT-J | <div style="text-align:left"><li>DeepSpeed</li></div> | <div style="text-align:left"><li>Single card</li><li>DeepSpeed</li></div> | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| GPT-NeoX | <div style="text-align:left"><li>DeepSpeed</li></div> | <div style="text-align:left"><li>DeepSpeed</li></div> | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| OPT |   | <div style="text-align:left"><li>DeepSpeed</li></div> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Llama 2 / CodeLlama / Llama 3 / Llama Guard / Granite | :heavy_check_mark: | :heavy_check_mark: | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li><li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering)</li><li>[text classification](https://github.com/huggingface/optimum-habana/tree/main/examples/text-classification) (Llama Guard)</li> |
| StableLM |   | <div style="text-align:left"><li>Single card</li></div> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Falcon | <div style="text-align:left"><li>LoRA</li></div> | :heavy_check_mark: | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| CodeGen |   | <div style="text-align:left"><li>Single card</li></div> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| MPT |   | <div style="text-align:left"><li>Single card</li></div> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Mistral |   | <div style="text-align:left"><li>Single card</li></div> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Phi | :heavy_check_mark:  | <div style="text-align:left"><li>Single card</li></div> | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Mixtral |   | <div style="text-align:left"><li>Single card</li></div> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Persimmon |   | <div style="text-align:left"><li>Single card</li></div> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Qwen2 | <div style="text-align:left"><li>Single card</li></div> | <div style="text-align:left"><li>Single card</li></div> | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Gemma | :heavy_check_mark:  | <div style="text-align:left"><li>Single card</li></div> | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| T5 / Flan T5 | :heavy_check_mark: | :heavy_check_mark: | <li>[summarization](https://github.com/huggingface/optimum-habana/tree/main/examples/summarization)</li><li>[translation](https://github.com/huggingface/optimum-habana/tree/main/examples/translation)</li><li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering#fine-tuning-t5-on-squad20)</li> |
| BART |   | <div style="text-align:left"><li>Single card</li></div> | <li>[summarization](https://github.com/huggingface/optimum-habana/tree/main/examples/summarization)</li><li>[translation](https://github.com/huggingface/optimum-habana/tree/main/examples/translation)</li><li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering#fine-tuning-t5-on-squad20)</li> |
| ViT | :heavy_check_mark: | :heavy_check_mark: | <li>[image classification](https://github.com/huggingface/optimum-habana/tree/main/examples/image-classification)</li> |
| Swin | :heavy_check_mark: | :heavy_check_mark: | <li>[image classification](https://github.com/huggingface/optimum-habana/tree/main/examples/image-classification)</li> |
| Wav2Vec2 | :heavy_check_mark: | :heavy_check_mark: | <li>[audio classification](https://github.com/huggingface/optimum-habana/tree/main/examples/audio-classification)</li><li>[speech recognition](https://github.com/huggingface/optimum-habana/tree/main/examples/speech-recognition)</li> |
| Whisper | :heavy_check_mark: | :heavy_check_mark: | <li>[speech recognition](https://github.com/huggingface/optimum-habana/tree/main/examples/speech-recognition)</li> |
| SpeechT5 |   | <div style="text-align:left"><li>Single card</li></div> | <li>[text to speech](https://github.com/huggingface/optimum-habana/tree/main/examples/text-to-speech)</li> |
| CLIP | :heavy_check_mark: | :heavy_check_mark: | <li>[contrastive image-text training](https://github.com/huggingface/optimum-habana/tree/main/examples/contrastive-image-text)</li> |
| BridgeTower | :heavy_check_mark: | :heavy_check_mark: | <li>[contrastive image-text training](https://github.com/huggingface/optimum-habana/tree/main/examples/contrastive-image-text)</li> |
| ESMFold |   | <div style="text-align:left"><li>Single card</li></div> | <li>[protein folding](https://github.com/huggingface/optimum-habana/tree/main/examples/protein-folding)</li> |
| Blip |   | <div style="text-align:left"><li>Single card</li></div> | <li>[visual question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/visual-question-answering)</li><li>[image to text](https://github.com/huggingface/optimum-habana/tree/main/examples/image-to-text)</li> |
| OWLViT |   | <div style="text-align:left"><li>Single card</li></div> | <li>[zero shot object detection](https://github.com/huggingface/optimum-habana/tree/main/examples/zero-shot-object-detection)</li> |
| ClipSeg |   | <div style="text-align:left"><li>Single card</li></div> | <li>[object segmentation](https://github.com/huggingface/optimum-habana/tree/main/examples/object-segementation)</li> |
| Llava / Llava-next |    | <div style="text-align:left"><li>Single card</li></div> | <li>[image to text](https://github.com/huggingface/optimum-habana/tree/main/examples/image-to-text)</li> |
| Segment Anything Model |   | <div style="text-align:left"><li>Single card</li></div> | <li>[object segmentation](https://github.com/huggingface/optimum-habana/tree/main/examples/object-segementation)</li> |
| VideoMAE | | <div style="text-align:left"><li>Single card</li></div> | <li>[Video classification](https://github.com/huggingface/optimum-habana/tree/main/examples/video-classification)</li> |
| TableTransformer |   | <div style="text-align:left"><li>Single card</li></div> | <li>[table object detection](https://github.com/huggingface/optimum-habana/tree/main/examples/table-detection) </li> |
| DETR |   | <div style="text-align:left"><li>Single card</li></div> | <li>[object detection](https://github.com/huggingface/optimum-habana/tree/main/examples/object-detection)</li> |

</div>

- Diffusers:

<div align="center">

| Architecture     | Training | Inference            | Tasks |
|------------------|:--------:|:--------------------:|:------|
| Stable Diffusion | <li>[textual inversion](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion/training#textual-inversion)</li><li>[ControlNet](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion/training#controlnet-training)</li> | <li>Single card</li> | <li>[text-to-image generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion)</li> |
| Stable Diffusion XL | <li>[fine-tuning](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion/training#fine-tuning-for-stable-diffusion-xl)</li> | <li>Single card</li> | <li>[text-to-image generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion)</li> |
| LDM3D            |          | <li>Single card</li> | <li>[text-to-image generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion)</li> |

</div>

- PyTorch Image Models/TIMM:

<div align="center">

| Architecture        | Training | Inference | Tasks |
|---------------------|:--------:|:---------:|:------|
| FastViT       |          | <div style="text-align:left"><li>Single card</li></div> | <li>[image classification](https://github.com/huggingface/optimum-habana/tree/main/examples/image-classification)</li> |

</div>

- TRL:

<div align="center">

| Architecture     | Training | Inference            | Tasks                                                                                          |
|------------------|:--------:|:--------------------:|:-----------------------------------------------------------------------------------------------|
| Llama 2          | :heavy_check_mark: |           | <li>[DPO Pipeline](https://github.com/huggingface/optimum-habana/tree/main/examples/trl)</li>  |
| Llama 2          | :heavy_check_mark: |           | <li>[PPO Pipeline](https://github.com/huggingface/optimum-habana/tree/main/examples/trl)</li>  |
| Stable Diffusion | :heavy_check_mark: |           | <li>[DDPO Pipeline](https://github.com/huggingface/optimum-habana/tree/main/examples/trl)</li> |

</div>

Other models and tasks supported by the Transformers and Diffusers libraries may also work. You can refer to this [section](https://github.com/huggingface/optimum-habana#how-to-use-it) for using them with Optimum for Intel Gaudi. In addition, [this page](https://github.com/huggingface/optimum-habana/tree/main/examples) explains how to modify any [example](https://github.com/huggingface/transformers/tree/main/examples/pytorch) from the Transformers library to make it work with Optimum for Intel Gaudi.

If you find any issues while using those, please open an issue or a pull request.

After training your model, feel free to submit it to the Intel [leaderboard](https://huggingface.co/spaces/Intel/powered_by_intel_llm_leaderboard) which is designed to evaluate, score, and rank open-source LLMs that have been pre-trained or fine-tuned on Intel Hardwares. Models submitted to the leaderboard will be evaluated on the Intel Developer Cloud. The evaluation platform consists of Gaudi Accelerators and Xeon CPUs running benchmarks from the Eleuther AI Language Model Evaluation Harness.

## Development

Check the [contributor guide](https://github.com/huggingface/optimum/blob/main/CONTRIBUTING.md) for instructions.
