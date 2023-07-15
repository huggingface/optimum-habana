# Text Generation Inference on Habana Gaudi

To use [🤗 text-generation-inference](https://github.com/huggingface/text-generation-inference) on Habana Gaudi/Gaudi2, follow these steps:

- Build the Docker image located in this folder with:
  ```bash
  docker build -t tgi_gaudi .
  ```
- Launch a local server instance with:
  ```bash
  model=bigscience/bloom-560m
  volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

  docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host tgi_gaudi --model-id $model --sharded false
  ```
- You can then send a request:
  ```bash
  curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}' \
    -H 'Content-Type: application/json'
  ```

> For gated models such as [StarCoder](https://huggingface.co/bigcode/starcoder), you will have to pass `-e HUGGING_FACE_HUB_TOKEN=<token>` to the `docker run` command above with a valid Hugging Face Hub read token.

For more information and documentation about Text Generation Inference, checkout [the README](https://github.com/huggingface/text-generation-inference#text-generation-inference) of the original repo.
