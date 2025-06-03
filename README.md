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

Optimum for Intel Gaudi - a.k.a. `optimum-habana` - is the interface between the Transformers and Diffusers libraries and
[Intel Gaudi AI Accelerators (HPU)](https://docs.habana.ai/en/latest/index.html). It provides a set of tools enabling easy
model loading, training and inference on single- and multi-HPU settings for different downstream tasks. The list of officially
validated models and tasks is available [here](https://github.com/huggingface/optimum-habana#validated-models). Users can
try other of the thousands of Hugging Face models on Intel Gaudi accelerators and tasks with only few changes.


## What are Intel Gaudi AI Accelerators (HPUs)?

HPUs offer fast model training and inference as well as a great price-performance ratio.
Check out [this blog post about BLOOM inference](https://huggingface.co/blog/habana-gaudi-2-bloom) and
[this post benchmarking Intel Gaudi 2 and NVIDIA A100 GPUs for BridgeTower training](https://huggingface.co/blog/bridgetower)
for concrete examples.


## Gaudi Setup

Please refer to the Intel Gaudi AI Accelerator official [installation guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html).

> [!NOTE]
> Tests should be run in a Docker container based on Intel Gaudi's official images. Instructions to
> obtain the latest containers from the Intel Gaudi Vault are available
> [here](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#use-intel-gaudi-containers).
> The current Optimum for Intel Gaudi has been validated with Intel Gaudi v1.20 stack.


## Install the library and get example scripts

### Option 1: Use the latest stable release

To install the latest stable release of this package
```bash
pip install --upgrade-strategy eager optimum[habana]
```

The `--upgrade-strategy eager` option is needed to ensure `optimum-habana` is upgraded to the latest stable release.

To use the example associated with the latest stable release, run:
```bash
git clone https://github.com/huggingface/optimum-habana
cd optimum-habana && git checkout v1.17.0
```
with `v1.17.0` being the latest Optimum for Intel Gaudi release version.

### Option 2: Use the latest main branch under development

Optimum for Intel Gaudi is a fast-moving project, and you may want to install it from source and get the latest scripts :

```bash
pip install git+https://github.com/huggingface/optimum-habana.git
git clone https://github.com/huggingface/optimum-habana
```

### Option 3: Use the `transformers_future` branch to have the latest changes from Transformers

The `transformers_future` branch is regularly updated with the latest changes from the main branches of Optimum for Intel Gaudi
and Transformers. This enables you to try out new Transformers features that have not been merged into the main branch yet.

> [!WARNING]
> The `transformers_future` branch may have some regressions or bugs and may be less stable than the main branch.

```bash
pip install git+https://github.com/huggingface/optimum-habana.git@transformers_future
git clone -b transformers_future https://github.com/huggingface/optimum-habana
```

## Install Dependencies

To use DeepSpeed on HPUs, you also need to run the following command:
```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.20.0
```

To install the requirements for every example:
```bash
cd <example-folder>
pip install -r requirements.txt
```

## How to use it?

Optimum for Intel Gaudi was designed with one goal in mind: **to make training and inference straightforward for Transformers
and Diffusers users, while fully leveraging the power of Intel Gaudi AI Accelerators**.

### Transformers Interface

There are two main classes one needs to know:

- [GaudiTrainer](https://huggingface.co/docs/optimum/habana/package_reference/trainer): the trainer class that takes care of
  compiling and distributing the model to run on HPUs, and performing training and evaluation.

- [GaudiConfig](https://huggingface.co/docs/optimum/habana/package_reference/gaudi_config): the class that enables to configure
  Gaudi Mixed Precision and to decide whether optimized operators and optimizers should be used or not.

The [GaudiTrainer](https://huggingface.co/docs/optimum/habana/package_reference/trainer) is very similar to the
[Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), and adapting a script using the Trainer to
make it work with Intel Gaudi accelerators will mostly consist in simply swapping the `Trainer` class for the `GaudiTrainer` one.

That's how most of the [example scripts](https://github.com/huggingface/optimum-habana/tree/main/examples) were adapted from their
[original counterparts](https://github.com/huggingface/transformers/tree/main/examples/pytorch).

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

where `gaudi_config_name` is the name of a model from the [Hub](https://huggingface.co/Habana) (Intel Gaudi configurations
are stored in model repositories) or a path to a local Intel Gaudi configuration file (you can see
[here](https://huggingface.co/docs/optimum/habana/package_reference/gaudi_config) how to write your own).


### Diffusers Interface

You can generate images from prompts using Stable Diffusion on Intel Gaudi using the
[`GaudiStableDiffusionPipeline`](https://huggingface.co/docs/optimum/habana/package_reference/stable_diffusion_pipeline) class and the
[`GaudiDDIMScheduler`](https://huggingface.co/docs/optimum/habana/package_reference/stable_diffusion_pipeline#optimum.habana.diffusers.GaudiDDIMScheduler)
class which have been both optimized for HPUs. Here is how to use them and the differences with the Diffusers library:

```diff
- from diffusers import DDIMScheduler, StableDiffusionPipeline
+ from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline


model_name = "CompVis/stable-diffusion-v1-4"

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

outputs = pipeline(
    ["An image of a squirrel in Picasso style"],
    num_images_per_prompt=16,
+   batch_size=4,
)
```


## Important Note on Pytorch 2.5 Performance Degradation

With the upgrade to PyTorch 2.5, users may experience some performance degradation due to changes in the handling of FP16/BF16 inputs.
The note from PyTorch 2.5 states:

"A naive SDPA math backend, when using FP16/BF16 inputs, can accumulate significant numerical errors due to the usage of low-precision
intermediate buffers. To mitigate this issue, the default behavior now involves upcasting FP16/BF16 inputs to FP32. Computations are performed
in FP32/TF32, and the final FP32 results are then downcasted back to FP16/BF16. This will improve numerical accuracy of the final output for
the math backend with FP16/BF16 inputs, but increases memory usages and may cause the performance regressions in the math backend as computations
shift from FP16/BF16 BMM to FP32/TF32 BMM/Matmul."

For scenarios where reduced-precision reductions are preferred for speed, they can be enabled with the following setting:
```python
torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
```
Additionally, the next release of Optimum Habana will include a Gaudi-specific safe_softmax implementation that will also improve performance.

More info:
- https://pytorch.org/docs/stable/notes/numerical_accuracy.html


### Documentation

Check out [the documentation of Optimum for Intel Gaudi](https://huggingface.co/docs/optimum/habana/index) for more advanced usage.


## Validated Models

The following model architectures, tasks and device distributions have been validated for Optimum for Intel Gaudi:

> [!NOTE]
> In the tables below, :heavy_check_mark: means single-card, multi-card and DeepSpeed have all been validated.

### Transformers:

| Architecture | Training | Inference | Tasks |
|:-------------|:--------:|:---------:|:------|
| BERT         | :heavy_check_mark: | :heavy_check_mark: | <li>[text classification](https://github.com/huggingface/optimum-habana/tree/main/examples/text-classification)</li><li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text feature extraction](https://github.com/huggingface/optimum-habana/tree/main/examples/text-feature-extraction)</li> |
| RoBERTa | :heavy_check_mark: | :heavy_check_mark: | <li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li> |
| ALBERT | :heavy_check_mark: | :heavy_check_mark: | <li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li> |
| DistilBERT |:heavy_check_mark: | :heavy_check_mark: | <li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li> |
| GPT2             | :heavy_check_mark: | :heavy_check_mark: | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| BLOOM(Z) |   | <li>DeepSpeed</li> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| StarCoder / StarCoder2 | :heavy_check_mark:  | <li>Single-card</li> | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| GPT-J | <li>DeepSpeed</li> | <li>Single card</li><li>DeepSpeed</li> | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| GPT-Neo |      | <li>Single card</li> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| GPT-NeoX | <li>DeepSpeed</li> | <li>DeepSpeed</li> | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| OPT |   | <li>DeepSpeed</li> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Llama 2 / CodeLlama / Llama 3 / Llama Guard / Granite | :heavy_check_mark: | :heavy_check_mark: | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li><li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering)</li><li>[text classification](https://github.com/huggingface/optimum-habana/tree/main/examples/text-classification) (Llama Guard)</li> |
| StableLM |   | <li>Single card</li> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Falcon | <li>LoRA</li> | :heavy_check_mark: | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| CodeGen |   | <li>Single card</li> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| MPT |   | <li>Single card</li> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Mistral |   | <li>Single card</li> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Phi | :heavy_check_mark:  | <li>Single card</li> | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Mixtral |   | <li>Single card</li> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Persimmon |   | <li>Single card</li> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Qwen2 | <li>Single card</li> | <li>Single card</li> | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Qwen2-MoE |   | <li>Single card</li> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Gemma | :heavy_check_mark:  | <li>Single card</li> | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Gemma2 |  | :heavy_check_mark: | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| XGLM | | <li>Single card</li> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Cohere       |          | <li>Single card</li> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| T5 / Flan T5 | :heavy_check_mark: | :heavy_check_mark: | <li>[summarization](https://github.com/huggingface/optimum-habana/tree/main/examples/summarization)</li><li>[translation](https://github.com/huggingface/optimum-habana/tree/main/examples/translation)</li><li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering#fine-tuning-t5-on-squad20)</li> |
| BART |   | <li>Single card</li> | <li>[summarization](https://github.com/huggingface/optimum-habana/tree/main/examples/summarization)</li><li>[translation](https://github.com/huggingface/optimum-habana/tree/main/examples/translation)</li><li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering#fine-tuning-t5-on-squad20)</li> |
| ViT | :heavy_check_mark: | :heavy_check_mark: | <li>[image classification](https://github.com/huggingface/optimum-habana/tree/main/examples/image-classification)</li> |
| Swin | :heavy_check_mark: | :heavy_check_mark: | <li>[image classification](https://github.com/huggingface/optimum-habana/tree/main/examples/image-classification)</li> |
| Wav2Vec2 | :heavy_check_mark: | :heavy_check_mark: | <li>[audio classification](https://github.com/huggingface/optimum-habana/tree/main/examples/audio-classification)</li><li>[speech recognition](https://github.com/huggingface/optimum-habana/tree/main/examples/speech-recognition)</li> |
| Whisper | :heavy_check_mark: | :heavy_check_mark: | <li>[speech recognition](https://github.com/huggingface/optimum-habana/tree/main/examples/speech-recognition)</li> |
| SpeechT5 |   | <li>Single card</li> | <li>[text to speech](https://github.com/huggingface/optimum-habana/tree/main/examples/text-to-speech)</li> |
| CLIP | :heavy_check_mark: | :heavy_check_mark: | <li>[contrastive image-text training](https://github.com/huggingface/optimum-habana/tree/main/examples/contrastive-image-text)</li> |
| BridgeTower | :heavy_check_mark: | :heavy_check_mark: | <li>[contrastive image-text training](https://github.com/huggingface/optimum-habana/tree/main/examples/contrastive-image-text)</li> |
| ESMFold |   | <li>Single card</li> | <li>[protein folding](https://github.com/huggingface/optimum-habana/tree/main/examples/protein-folding)</li> |
| Blip |   | <li>Single card</li> | <li>[visual question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/visual-question-answering)</li><li>[image to text](https://github.com/huggingface/optimum-habana/tree/main/examples/image-to-text)</li> |
| OWLViT |   | <li>Single card</li> | <li>[zero shot object detection](https://github.com/huggingface/optimum-habana/tree/main/examples/zero-shot-object-detection)</li> |
| ClipSeg |   | <li>Single card</li> | <li>[object segmentation](https://github.com/huggingface/optimum-habana/tree/main/examples/object-segementation)</li> |
| Llava / Llava-next / Llava-onevision |    | <li>Single card</li> | <li>[image to text](https://github.com/huggingface/optimum-habana/tree/main/examples/image-to-text)</li> |
| idefics2 | <li>LoRA</li> | <li>Single card</li> | <li>[image to text](https://github.com/huggingface/optimum-habana/tree/main/examples/image-to-text)</li> |
| Paligemma | | <li>Single card</li> | <li>[image to text](https://github.com/huggingface/optimum-habana/tree/main/examples/image-to-text)</li> |
| Segment Anything Model |   | <li>Single card</li> | <li>[object segmentation](https://github.com/huggingface/optimum-habana/tree/main/examples/object-segementation)</li> |
| VideoMAE | | <li>Single card</li> | <li>[Video classification](https://github.com/huggingface/optimum-habana/tree/main/examples/video-classification)</li> |
| TableTransformer |   | <li>Single card</li> | <li>[table object detection](https://github.com/huggingface/optimum-habana/tree/main/examples/table-detection) </li> |
| DETR |   | <li>Single card</li> | <li>[object detection](https://github.com/huggingface/optimum-habana/tree/main/examples/object-detection)</li> |
| Mllama     | <li>LoRA</li> | :heavy_check_mark: | <li>[image to text](https://github.com/huggingface/optimum-habana/tree/main/examples/image-to-text)</li> |
| MiniCPM3 |   | <li>Single card</li> | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Baichuan2 | <li>DeepSpeed</li> | <li>Single card</li> | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| DeepSeek-V2 | :heavy_check_mark: | :heavy_check_mark: | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| DeepSeek-V3 / Moonlight |   | :heavy_check_mark: | <li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| ChatGLM | <li>DeepSpeed</li> | <li>Single card</li> | <li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li><li>[text generation](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation)</li> |
| Qwen2-VL |          |  <div style="text-align:left"><li>Single card</li></div> | <li>[image to text](https://github.com/huggingface/optimum-habana/tree/main/examples/image-to-text)</li> |
| VideoLLaVA | | <div style="text-align:left"><li>Single card</li></div> | <li>[Video comprehension](https://github.com/huggingface/optimum-habana/tree/main/examples/video-comprehension)</li> |
| GLM-4V | |  <div style="text-align:left"><li>Single card</li></div> | <li>[image to text](https://github.com/huggingface/optimum-habana/tree/main/examples/image-to-text)</li> 

</div>


### Diffusers:

| Architecture        | Training | Inference | Tasks |
|:--------------------|:--------:|:---------:|:------|
| Stable Diffusion    | :heavy_check_mark: | :heavy_check_mark: | <li>[text-to-image generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion#text-to-image-generation)</li><li>[image-to-image generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion#image-to-image-generation)</li> |
| Stable Diffusion XL | :heavy_check_mark: | :heavy_check_mark: | <li>[text-to-image generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion#stable-diffusion-xl-sdxl)</li><li>[image-to-image generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion#stable-diffusion-xl-refiner)</li> |
| Stable Diffusion Depth2img |         | <li>Single card</li> | <li>[depth-to-image generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion)</li> |
| Stable Diffusion 3  | :heavy_check_mark: | <li>Single card</li> | <li>[text-to-image generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion#stable-diffusion-3-and-35-sd3)</li> |
| LDM3D            |               | <li>Single card</li> | <li>[text-to-image generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion#text-to-image-generation)</li> |
| FLUX.1           | <li>LoRA</li> | <li>Single card</li> | <li>[text-to-image generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion#flux1)</li><li>[image-to-image generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion#flux1-image-to-image)</li> |
| Text to Video    |               | <li>Single card</li> | <li>[text-to-video generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion#text-to-video-generation)</li> |
| Image to Video   |               | <li>Single card</li> | <li>[image-to-video generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion#image-to-video-generation)</li> |
| i2vgen-xl   |               | <li>Single card</li> | <li>[image-to-video generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion#I2vgen-xl)</li> |

### PyTorch Image Models/TIMM:

| Architecture        | Training | Inference | Tasks |
|:--------------------|:--------:|:---------:|:------|
| FastViT       |          | <li>Single card</li> | <li>[image classification](https://github.com/huggingface/optimum-habana/tree/main/examples/image-classification)</li> |

### TRL:

| Architecture     | Training | Inference            | Tasks                                                                                          |
|:-----------------|:--------:|:--------------------:|:-----------------------------------------------------------------------------------------------|
| Llama 2          | :heavy_check_mark: |           | <li>[DPO Pipeline](https://github.com/huggingface/optimum-habana/tree/main/examples/trl#dpo-pipeline)</li>  |
| Llama 2          | :heavy_check_mark: |           | <li>[PPO Pipeline](https://github.com/huggingface/optimum-habana/tree/main/examples/trl#ppo-pipeline)</li>  |
| Stable Diffusion | :heavy_check_mark: |           | <li>[DDPO Pipeline](https://github.com/huggingface/optimum-habana/tree/main/examples/trl#ddpo-pipeline)</li> |

Other models and tasks supported by the Transformers and Diffusers libraries may also work. You can refer to this [section](https://github.com/huggingface/optimum-habana#how-to-use-it)
for using them with Optimum for Intel Gaudi. In addition, [this page](https://github.com/huggingface/optimum-habana/tree/main/examples) explains how to modify any
[example](https://github.com/huggingface/transformers/tree/main/examples/pytorch) from the Transformers library to make it work with Optimum for Intel Gaudi.

If you find any issues while using those, please open an issue or a pull request.

After training your model, feel free to submit it to the Intel [leaderboard](https://huggingface.co/spaces/Intel/powered_by_intel_llm_leaderboard) which is designed
to evaluate, score, and rank open-source LLMs that have been pre-trained or fine-tuned on Intel Hardwares. Models submitted to the leaderboard will be evaluated on
the Intel Developer Cloud. The evaluation platform consists of Gaudi Accelerators and Xeon CPUs running benchmarks from the Eleuther AI Language Model Evaluation Harness.

The list of validated models through continuous integration tests is posted [here](https://github.com/huggingface/optimum-habana/tree/main/tests/Habana_Validated_Models.md)

## Development

Check the [contributor guide](https://github.com/huggingface/optimum/blob/main/CONTRIBUTING.md) for instructions.
