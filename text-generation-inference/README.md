<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

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

# Text Generation Inference on Habana Gaudi

To use [🤗 text-generation-inference](https://github.com/huggingface/text-generation-inference) on Habana Gaudi/Gaudi2, follow these steps:

1. Build the Docker image located in this folder with:
   ```bash
   docker build -t tgi_gaudi .
   ```
2. Launch a local server instance with:
   ```bash
   model=bigscience/bloom-560m
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host tgi_gaudi --model-id $model
   ```
3. You can then send a request:
   ```bash
   curl 127.0.0.1:8080/generate \
     -X POST \
     -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}' \
     -H 'Content-Type: application/json'
   ```
   > The first call will be slower as the model is compiled.

> For gated models such as [StarCoder](https://huggingface.co/bigcode/starcoder), you will have to pass `-e HUGGING_FACE_HUB_TOKEN=<token>` to the `docker run` command above with a valid Hugging Face Hub read token.

For more information and documentation about Text Generation Inference, checkout [the README](https://github.com/huggingface/text-generation-inference#text-generation-inference) of the original repo.

Not all features of TGI are currently supported as this is still a work in progress.
It has been tested for batches of 1 sample so far.
New features will be added soon, including:
- support for DeepSpeed-inference to use sharded models
- batching strategy to have less model compilations
- Added an env var MAX_TOTAL_TOKENS for models that require it to be set during benchmark test.
  It defaults to 0. To change it please add "ENV MAX_TOTAL_TOKENS=512" (512 is an example) to Dockerfile and rebuild the docker.
  This workaround is needed as max_total_tokens is currently not being passed from Rust to Python when running launcher app.

> The license to use TGI on Habana Gaudi is the one of TGI: https://github.com/huggingface/text-generation-inference/blob/main/LICENSE
>
> Please reach out to api-enterprise@huggingface.co if you have any question.
