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

# Multi-node Training

Multi-node training can be performed easily on Gaudi with DeepSpeed for any training script as follows:
```bash
python gaudi_spawn.py \
    --hostfile path_to_my_hostfile --use_deepspeed \
    path_to_my_script.py --args1 --args2 ... --argsN \
    --deepspeed path_to_my_deepspeed_config
```
where `--argX` is an argument of the script to run.

## Setup

Check out the [documentation](https://huggingface.co/docs/optimum/habana/usage_guides/multi_node_training) to know how to set up your Gaudi instances for multi-node runs on premises or on AWS.

A `Dockerfile` is provided [here](https://github.com/huggingface/optimum-habana/tree/main/examples/multi-node-training/Dockerfile) to easily start a multi-node run.
It is based on an image compatible with Ubuntu 20.04 but you can easily adapt it to another OS.
To build the Docker image, run:
```bash
docker build -t gaudi_multi_node PATH
```
where `PATH` is the path to the folder containing the `Dockerfile`.

To run a Docker container with the image you just built, execute:
```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host gaudi_multi_node:latest
```

> For AWS DL1 instances, `--privileged` must be passed to the `docker run` command so that EFA interfaces are visible.

Finally, you will have to copy the public key of the leader node in the `~/.ssh/authorized_keys` file of all other nodes to enable password-less SSH.


## Hostfile

DeepSpeed requires a [hostfile](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) to know the addresses of and the number of devices to use on each node. You can specify its path with `--hostfile`. This file should look like this:
```
ip_1 slots=8
ip_2 slots=8
...
ip_n slots=8
```

You can find a template [here](https://github.com/huggingface/optimum-habana/tree/main/examples/multi-node-training/hostfile).


## Environment variables

If you need to set environment variables for all nodes, you can specify them in a `.deepspeed_env` file which should be located in the local path you are executing from or in your home directory. It is formatted as follows:
```
env_variable_1_name=value
env_variable_2_name=value
...
```

You can find an example for AWS instances [here](https://github.com/huggingface/optimum-habana/tree/main/examples/multi-node-training/.deepspeed_env).


## Recommendations

- It is strongly recommended to use gradient checkpointing for multi-node runs to get the highest speedups. You can enable it with `--gradient_checkpointing` in all these examples or with `gradient_checkpointing=True` in your `GaudiTrainingArguments`.
- Larger batch sizes should lead to higher speedups.
- Multi-node inference is not recommended and can provide inconsistent results.
- On AWS DL1 instances, run your Docker containers with the `--privileged` flag so that EFA devices are visible.
