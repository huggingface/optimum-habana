# Optimum-Habana model agnostic Container (Single Node)
This folder contains scripts and configuration files that can be used to build an Optimum-Habana container with support for the following models:

|HPU|Model|Number of HPUs (world_size)|
|--|--|--|
|Gaudi2|meta-llama/Llama-3.1-8B-Instruct|1|
|Gaudi2|meta-llama/Llama-3.1-70B-Instruct|2|
|Gaudi2|meta-llama/Llama-3.1-70B-Instruct|8|
|Gaudi2|meta-llama/Llama-3.3-70B-Instruct|8|
|Gaudi3|meta-llama/Llama-3.1-8B-Instruct|1|
|Gaudi3|meta-llama/Llama-3.1-70B-Instruct|2|
|Gaudi3|meta-llama/Llama-3.1-70B-Instruct|8|
|Gaudi3|meta-llama/Llama-3.3-70B-Instruct|8|


## Quick Start
To run these models on your Gaudi machine:

1) First, obtain the Dockerfile and benchmark scripts from the Optimum-Habana repository using the command below
```bash
git clone https://github.com/huggingface/optimum-habana
cd optimum-habana/examples/text-generation/docker
```

> **IMPORTANT**
>     
> **All build and run steps listed in this document need to be executed on Gaudi Hardware**
>    

2) To build the `oh-1.22.0-gaudi` image from the Dockerfile, use the command below.
```bash
## Set the next line if you are using a HTTP proxy on your build machine
BUILD_ARGS="--build-arg http_proxy --build-arg https_proxy --build-arg no_proxy"
docker build -f Dockerfile $BUILD_ARGS -t oh-1.22.0-gaudi .
```

## Single Card Models
1) Run the container and load a shell into it. For HABANA_VISIBLE_DEVICES, chose a single, available card from 0 through 7. Point hf_cache to the local copy of your HuggingFace Cache folder.
```bash
DOCKER_OPTS="-e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host -d --runtime=habana --restart always"
DOCKER_OPTS="${DOCKER_OPTS} -e HF_TOKEN=$hf_token -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy"
DOCKER_OPTS="${DOCKER_OPS} -v /mnt/hf_cache:/mnt/hf_cache"
docker run --entrypoint /bin/bash $DOCKER_OPTS -e HABANA_VISIBLE_DEVICES=1 --name oh-1.22.0 oh-1.22.0-gaudi -c "sleep infinity"
docker exec -it oh-1.22.0 bash
```

2) Build Measurement files for Single Card models - this needs to be run once per model. 
* Change the value for model_name to the one you need to use
* Set world_size to the number of HPUs recommended as per the table above

```bash
export model_name=meta-llama/Llama-3.1-8B-Instruct
export world_size=1
export PT_HPU_LAZY_MODE=1
export HF_TOKEN=<YOUR_TOKEN_HERE>
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TQDM_DISABLE=1
export QUANT_CONFIG=/root/optimum-habana/examples/text-generation/quantization_config/maxabs_measure.json

cd /root/optimum-habana/examples/text-generation/
python3 run_lm_eval.py \
  -o acc_llama_quant.json \
  --model_name_or_path ${model_name} \
  --warmup 0 \
  --flash_attention_causal_mask \
  --attn_softmax_bf16 \
  --use_hpu_graphs \
  --trim_logits \
  --use_kv_cache \
  --bf16 \
  --batch_size 1 \
  --bucket_size=128 \
  --bucket_internal \
  --trust_remote_code \
  --tasks hellaswag hellaswag lambada_openai piqa winogrande \
  --use_flash_attention \
  --flash_attention_recompute  \
  2>&1 | tee -a measurement_logs_in${input_len}_out${output_len}_bs${batch_size}_${model_tag}_tp${world_size}.txt
```

3) Run benchmark for Single card models (world_size=1)
```bash
export QUANT_CONFIG=/root/optimum-habana/examples/text-generation/quantization_config/maxabs_quant.json
export input_len=10
export output_len=10
export batch_size=2
export world_size=1
cd /root/optimum-habana/examples/text-generation/
python3 run_generation.py \
  --model_name_or_path ${model_name} \
  --attn_softmax_bf16 \
  --trim_logits \
  --warmup 2 \
  --use_kv_cache \
  --use_hpu_graphs \
  --limit_hpu_graphs \
  --bucket_size=128 \
  --bucket_internal \
  --attn_batch_split 2 \
  --bf16 \
  --flash_attention_causal_mask \
  --use_flash_attention \
  --flash_attention_recompute \
  --batch_size ${batch_size} \
  --max_new_tokens ${output_len} \
  --max_input_tokens ${input_len} \
  2>&1 | tee -a benchmark_logs_in${input_len}_out${output_len}_bs${batch_size}_${model_tag}_tp${world_size}.txt
```

## Multi-card Models

1) Run the container and load a shell into it. For HABANA_VISIBLE_DEVICES, chose multiple available cards based on the recommended world size from the table above
```bash
DOCKER_OPTS="-e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host -d --runtime=habana --restart always"
DOCKER_OPTS="${DOCKER_OPTS} -e HF_TOKEN=$hf_token -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy"
DOCKER_OPTS="${DOCKER_OPS} -v /mnt/hf_cache:/mnt/hf_cache"
docker run --entrypoint /bin/bash $DOCKER_OPTS -e HABANA_VISIBLE_DEVICES=0,1 --name oh-1.22.-multicard0 oh-1.22.0-gaudi -c "sleep infinity"
docker exec -it oh-1.22.0 bash
```

2) Build Measurement files for Single Card models - this needs to be run once per model. 
* Change the value for model_name to the one you need to use
* Set world_size to the number of HPUs recommended as per the table above
```bash
export model_name=meta-llama/Llama-3.1-70B-Instruct
export world_size=8
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export HF_TOKEN=<YOUR_TOKEN_HERE>
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TQDM_DISABLE=1
export QUANT_CONFIG=./quantization_config/maxabs_measure.json

cd /root/optimum-habana/examples/text-generation/
python3 ../gaudi_spawn.py \
  --use_deepspeed --world_size ${world_size}  run_lm_eval.py \
  -o acc_llama_quant.json \
  --model_name_or_path ${model_name} \
  --warmup 0 \
  --flash_attention_causal_mask \
  --attn_softmax_bf16 \
  --use_hpu_graphs \
  --trim_logits \
  --use_kv_cache \
  --bf16 \
  --batch_size 1 \
  --bucket_size=128 \
  --bucket_internal \
  --trust_remote_code \
  --tasks hellaswag lambada_openai piqa winogrande \
  --use_flash_attention \
  --flash_attention_recompute \
  2>&1 | tee -a measurement_logs_in${input_len}_out${output_len}_bs${batch_size}_${model_tag}_tp${world_size}.txt
```

3) Run benchmark for Multi-card models (world_size>1)
```bash
export QUANT_CONFIG=./quantization_config/maxabs_quant.json
export input_len=10
export output_len=10
export batch_size=2
cd /root/optimum-habana/examples/text-generation/
python3 ../gaudi_spawn.py \
  --use_deepspeed \
  --world_size ${world_size} \
  run_generation.py \
  --model_name_or_path ${model_name} \
  --attn_softmax_bf16 \
  --trim_logits \
  --warmup 2 \
  --use_kv_cache \
  --use_hpu_graphs \
  --limit_hpu_graphs \
  --bucket_size=128 \
  --bucket_internal \
  --attn_batch_split 2 \
  --bf16 \
  --flash_attention_causal_mask \
  --use_flash_attention \
  --flash_attention_recompute \
  --batch_size ${batch_size} \
  --max_new_tokens ${output_len} \
  --max_input_tokens ${input_len} \
  2>&1 | tee -a benchmark_logs_in${input_len}_out${output_len}_bs${batch_size}_${model_tag}_tp${world_size}.txt
```
