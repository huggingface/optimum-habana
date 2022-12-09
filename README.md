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

![](https://github.com/huggingface/optimum-habana/blob/main/readme_logo.png)


# Optimum Habana

🤗 Optimum Habana is the interface between the 🤗 Transformers and Diffusers libraries and [Habana's Gaudi processor (HPU)](https://docs.habana.ai/en/latest/index.html).
It provides a set of tools enabling easy model loading, training and inference on single- and multi-HPU settings for different downstream tasks.
The list of officially validated models and tasks is available [here](https://github.com/huggingface/optimum-habana#validated-models). Users can try other models and tasks with only few changes.


## What is a Habana Processing Unit (HPU)?

Quote from the Hugging Face [blog post](https://huggingface.co/blog/habana):

> Habana Gaudi training solutions, which power Amazon’s EC2 DL1 instances and Supermicro’s X12 Gaudi AI Training Server, deliver price/performance up to 40% lower than comparable training solutions and enable customers to train more while spending less. The integration of ten 100 Gigabit Ethernet ports onto every Gaudi processor enables system scaling from 1 to thousands of Gaudis with ease and cost-efficiency. Habana’s SynapseAI® is optimized—at inception—to enable Gaudi performance and usability, supports TensorFlow and PyTorch frameworks, with a focus on computer vision and natural language processing applications.


## Install
To install the latest release of this package:

```bash
pip install optimum[habana]
```

> To use DeepSpeed on HPUs, you also need to run the following command:
>```bash
>pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.7.0
>```

Optimum Habana is a fast-moving project, and you may want to install it from source:

```bash
pip install git+https://github.com/huggingface/optimum-habana.git
```

> Alternatively, you can install the package without pip as follows:
> ```bash
> git clone https://github.com/huggingface/optimum-habana.git
> cd optimum-habana
> python setup.py install
> ```

Last but not least, don't forget to install requirements for every example:

```bash
cd <example-folder>
pip install -r requirements.txt
```


## How to use it?

### Quick Start

🤗 Optimum Habana was designed with one goal in mind: **make training and evaluation straightforward for any 🤗 Transformers and 🤗 Diffusers user while leveraging the complete power of Gaudi processors**.

#### Transformers Interface

There are two main classes one needs to know:
- [GaudiTrainer](https://huggingface.co/docs/optimum/habana/package_reference/trainer): the trainer class that takes care of compiling (lazy or eager mode) and distributing the model to run on HPUs, and of performing traning and evaluation.
- [GaudiConfig](https://huggingface.co/docs/optimum/habana/package_reference/gaudi_config): the class that enables to configure Habana Mixed Precision and to decide whether optimized operators and optimizers should be used or not.

The [GaudiTrainer](https://huggingface.co/docs/optimum/habana/package_reference/trainer) is very similar to the [🤗 Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), and adapting a script using the Trainer to make it work with Gaudi will mostly consist in simply swapping the `Trainer` class for the `GaudiTrainer` one.
That's how most of the [example scripts](https://github.com/huggingface/optimum-habana/tree/main/examples) were adapted from their [original counterparts](https://github.com/huggingface/transformers/tree/main/examples/pytorch).

Original script:
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
  # training arguments...
)

# A lot of code here

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,  # Original training arguments.
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
```


Transformed version that can run on Gaudi:
```python
from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments

training_args = GaudiTrainingArguments(
  # same training arguments...
  use_habana=True,
  use_lazy_mode=True,  # whether to use lazy or eager mode
  gaudi_config_name=path_to_gaudi_config,
)

# A lot of the same code as the original script here

# Initialize our Trainer
trainer = GaudiTrainer(
    model=model,
    # You can manually specify the Gaudi configuration to use with
    # gaudi_config=my_gaudi_config
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
```

where `gaudi_config_name` is the name of a model from the [Hub](https://huggingface.co/Habana) (Gaudi configurations are stored in model repositories). You can also give the path to a custom Gaudi configuration written in a JSON file such as this one:
```json
{
  "use_habana_mixed_precision": true,
  "hmp_opt_level": "O1",
  "hmp_is_verbose": false,
  "use_fused_adam": true,
  "use_fused_clip_norm": true,
  "hmp_bf16_ops": [
    "add",
    "addmm",
    "bmm",
    "div",
    "dropout",
    "gelu",
    "iadd",
    "linear",
    "layer_norm",
    "matmul",
    "mm",
    "rsub",
    "softmax",
    "truediv"
  ],
  "hmp_fp32_ops": [
    "embedding",
    "nll_loss",
    "log_softmax"
  ]
}
```

If you prefer to instantiate a Gaudi configuration to work on it before giving it to the trainer, you can do it as follows:
```python
gaudi_config = GaudiConfig.from_pretrained(
    gaudi_config_name,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
```


#### Diffusers Interface

You can generate images from prompts using Stable Diffusion on Gaudi using the [`GaudiStableDiffusionPipeline`](https://huggingface.co/docs/optimum/habana/package_reference/stable_diffusion_pipeline) class and the [`GaudiDDIMScheduler`] that have been both optimized for HPUs. Here is how to use them and the differences with the 🤗 Diffusers library:

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

outputs = generator(
    ["An image of a squirrel in Picasso style"],
    num_images_per_prompt=16,
+   batch_size=4,
)
```


### Documentation

Check [the documentation of Optimum Habana](https://huggingface.co/docs/optimum/habana/index) for more advanced usage.


## Validated Models

The following model architectures, tasks and device distributions have been validated for 🤗 Optimum Habana:
| Architecture | Single Card | Multi Card | DeepSpeed | <center>Tasks</center> |
|--------------|:-----------:|:----------:|:---------:|------------------------|
| BERT         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | <li>[text classification](https://github.com/huggingface/optimum-habana/tree/main/examples/text-classification)</li><li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li> |
| RoBERTa | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | <ul style="margin-bottom:0;"><li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li></ul> |
| ALBERT | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | <ul style="margin-bottom:0;"><li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li></ul> |
| DistilBERT |:heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | <ul style="margin-bottom:0;"><li>[question answering](https://github.com/huggingface/optimum-habana/tree/main/examples/question-answering)</li><li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li></ul> |
| GPT2             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | <ul style="margin-bottom:0;"><li>[language modeling](https://github.com/huggingface/optimum-habana/tree/main/examples/language-modeling)</li></ul> |
| T5 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | <ul style="margin-bottom:0;"><li>[summarization](https://github.com/huggingface/optimum-habana/tree/main/examples/summarization)</li><li>[translation](https://github.com/huggingface/optimum-habana/tree/main/examples/translation)</li></ul> |
| ViT | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | <ul style="margin-bottom:0;"><li>[image classification](https://github.com/huggingface/optimum-habana/tree/main/examples/image-classification)</li></ul> |
| Swin | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | <ul style="margin-bottom:0;"><li>[image classification](https://github.com/huggingface/optimum-habana/tree/main/examples/image-classification)</li></ul> |
| Wav2Vec2 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | <ul style="margin-bottom:0;"><li>[audio classification](https://github.com/huggingface/optimum-habana/tree/main/examples/audio-classification)</li><li>[speech recognition](https://github.com/huggingface/optimum-habana/tree/main/examples/speech-recognition)</li></ul> |
| Stable Diffusion | :heavy_check_mark: | ✗ | ✗ | <ul style="margin-bottom:0;"><li>[text-to-image generation](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion)</li></ul> |

Other models and tasks supported by the 🤗 Transformers library may also work. You can refer to this [section](https://github.com/huggingface/optimum-habana#how-to-use-it) for using them with 🤗 Optimum Habana. Besides, [this page](https://github.com/huggingface/optimum-habana/tree/main/examples) explains how to modify any [example](https://github.com/huggingface/transformers/tree/main/examples/pytorch) from the 🤗 Transformers library to make it work with 🤗 Optimum Habana.

If you find any issue while using those, please open an issue or a pull request.


## Gaudi Setup

Please refer to Habana Gaudi's official [installation guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html).

> Tests should be run in a Docker container based on Habana Docker images.
>
> The current version has been validated for SynapseAI 1.7.
