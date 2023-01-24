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

# Examples

This folder contains actively maintained examples of use of 🤗 Optimum Habana for various ML tasks.

Other [examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch) from the 🤗 Transformers library can be adapted the same way to enable deployment on Gaudi processors. This simply consists in:
- replacing the `Trainer` from 🤗 Transformers with the `GaudiTrainer` from 🤗 Optimum Habana,
- replacing the `TrainingArguments` from 🤗 Transformers with the `GaudiTrainingArguments` from 🤗 Optimum Habana.


## Distributed training

All the PyTorch training scripts in this repository work out of the box with distributed training.


### Single node

To launch a script on _n_ HPUs belonging to a single Gaudi server, use the following command:

```bash
python gaudi_spawn.py \
    --world_size number_of_hpu_you_have --use_mpi \
    path_to_script.py --args1 --args2 ... --argsN
```
where `--argX` is an argument of the script to run in a distributed way.


### DeepSpeed

All the PyTorch training scripts in this repository work out of the box with DeepSpeed. To launch one of them on _n_ HPUs, use the following command:

```bash
python gaudi_spawn.py \
    --world_size number_of_hpu_you_have --use_deepspeed \
    path_to_script.py --args1 --args2 ... --argsN \
    --deepspeed path_to_my_deepspeed_config
```
where `--argX` is an argument of the script to run with DeepSpeed.


### Multi node

All the PyTorch training scripts in this repository work out of the box on several Gaudi instances. To launch one of them on _n_ nodes, use the following command:

```bash
python gaudi_spawn.py \
    --hostfile path_to_my_hostfile --use_deepspeed \
    path_to_my_script.py --args1 --args2 ... --argsN \
    --deepspeed path_to_my_deepspeed_config
```
where `--argX` is an argument of the script to run with DeepSpeed and `--hostfile` is [a file specifying the addresses and the number of devices to use for each node](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) such as:
```
ip_1 slots=8
ip_2 slots=8
...
ip_n slots=8
```

You can find more information about multi-node training in the [documentation](https://huggingface.co/docs/optimum/habana/usage_guides/multi_node_training) and in the [`multi-node-training`](https://github.com/huggingface/optimum-habana/tree/main/examples/multi-node-training) folder where a Dockerfile is provided to easily set up your environment.


## Loading from a Tensorflow/Flax checkpoint file instead of a PyTorch model

If a model also has Tensorflow or Flax checkpoints, you can load them instead of a PyTorch checkpoint by specifying `from_tf=True` or `from_flax=True` in the model instantiation.

You can try it for SQuAD [here](https://github.com/huggingface/optimum-habana/blob/688a857d5308a87a502eec7657f744429125d6f1/examples/question-answering/run_qa.py#L310) or MRPC [here](https://github.com/huggingface/optimum-habana/blob/688a857d5308a87a502eec7657f744429125d6f1/examples/text-classification/run_glue.py#L338).

You can check if a model has such checkpoints on the [Hub](https://huggingface.co/models). You can also specify a URL or a path to a Tensorflow/Flax checkpoint in `model_args.model_name_or_path`.

> Resuming from a checkpoint will only work with a PyTorch checkpoint.


## Running quick tests

Most examples are equipped with a mechanism to truncate the number of dataset samples to the desired length. This is useful for debugging purposes, for example to quickly check that all stages of the programs can complete, before running the same setup on the full dataset which may take hours to complete.

For example here is how to truncate all three splits to just 50 samples each:
```
examples/pytorch/question-answering/run_squad.py \
--max_train_samples 50 \
--max_eval_samples 50 \
--max_predict_samples 50 \
[...]
```


## Resuming training

You can resume training from a previous checkpoint like this:

1. Pass `--output_dir previous_output_dir` without `--overwrite_output_dir` to resume training from the latest checkpoint in `output_dir` (what you would use if the training was interrupted, for instance).
2. Pass `--resume_from_checkpoint path_to_a_specific_checkpoint` to resume training from that checkpoint folder.

Should you want to turn an example into a notebook where you'd no longer have access to the command
line, 🤗 GaudiTrainer supports resuming from a checkpoint via `trainer.train(resume_from_checkpoint)`.

1. If `resume_from_checkpoint` is `True` it will look for the last checkpoint in the value of `output_dir` passed via `TrainingArguments`.
2. If `resume_from_checkpoint` is a path to a specific checkpoint it will use that saved checkpoint folder to resume the training.


## Uploading the trained/fine-tuned model to the Hub

All the example scripts support the automatic upload of your final model to the [Model Hub](https://huggingface.co/models) by adding a `--push_to_hub` argument. It will then create a repository with your username slash the name of the folder you are using as `output_dir`. For instance, `"sgugger/test-mrpc"` if your username is `sgugger` and you are working in the folder `~/tmp/test-mrpc`.

To specify a given repository name, use the `--hub_model_id` argument. You will need to specify the whole repository name (including your username), for instance `--hub_model_id sgugger/finetuned-bert-mrpc`. To upload to an organization you are a member of, just use the name of that organization instead of your username: `--hub_model_id huggingface/finetuned-bert-mrpc`.

A few notes on this integration:

- you will need to be logged in to the Hugging Face website locally for it to work, the easiest way to achieve this is to run `huggingface-cli login` and then type your username and password when prompted. You can also pass along your authentication token with the `--hub_token` argument.
- the `output_dir` you pick will either need to be a new folder or a local clone of the distant repository you are using.
