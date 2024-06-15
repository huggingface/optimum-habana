# Replicating Large Language Model Inference Performance on the Intel&reg; Gaudi&reg; AI Accelerators
This page provides instructions on how to run the exact configuration to replicate the Intel Gaudi Throughput Performance Numbers posted on the Intel Gaudi [Developer website](https://www.intel.com/content/www/us/en/developer/platform/gaudi/model-performance.html) for the following Transformer based models:  

<div align="left">

| Model | Precision | Intel Gaudi Accelerators |
|--------------|:---------:|:-------------:|
| text-generation inference Llama 2 7B  | FP8 | 1 card | 
| text-generation inference Llama 2 70B | FP8 | 2, 4 cards |
| text-generation inference Mistral 7B Instruct | FP8 | 1 card |

</div>

## Initial Setup

1. Get access to an Intel Gaudi node and [setup](https://docs.habana.ai/en/latest/shared/Pull_Prebuilt_Containers.html#pulling-prebuilt-container) the Intel Gaudi PyTorch docker for release 1.16.0
```bash
 docker run -itd --name 116 --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.16.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.2:latest
 docker exec -it 116 bash
```
2. Since the published performance numbers for 1.16.0 used the 1.11.1 version of Optimum Habana, you will install the 1.11.1 version of the Optimum Habana library
```bash
 pip install optimum-habana==1.11.1
```
3. Pull the 1.11.1 Version of the Optimum Habana Examples GitHub repository 
 ```
 cd ~ && git clone https://github.com/huggingface/optimum-habana
 cd optimum-habana && git checkout v1.11.1
 ```

## Install dependencies for the text-generation Task example
Go to the text-generation folder and Install the requirements:
```bash
cd examples/text-generation
pip install -r requirements.txt
```
install DeepSpeed:
```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.16.0
```
## Run the exapmles -  Measure first then run Quantization Second
Since these are using FP8 precsion, you will first run the model one time to make the measurement and quantization settings using `QUANT_CONFIG=./quantization_config/maxabs_measure.json` and then you will run the model a second time with the `QUANT_CONFIG=./quantization_config/maxabs_quant.json`, where the existing measurement files will be used from the ./hqt_output folder. These examples are taking advantage of several tecniques such as maximizing batch size, bucket size and using Flash Attention

### How to access and Use the Llama 2 and Mistral models
To use the Llama 2 and Mistral models, you will need a HuggingFace account, agree to the terms of use of the model in its model card on the HF Hub, and create a read token. You then copy that token to the `huggingface-cli login token in the instruction below.

Use of the pretrained model is subject to compliance with third party licenses, including the “Llama 2 Community License Agreement” (LLAMAV2). For guidance on the intended use of the LLAMA2 model, what will be considered misuse and out-of-scope uses, who are the intended users and additional terms please review and read the instructions in this link https://ai.meta.com/llama/license/. Users bear sole liability and responsibility to follow and comply with any third party licenses, and Habana Labs disclaims and will bear no liability with respect to users’ use or compliance with third party licenses.

```bash
huggingface-cli login --token <your_token_here>
```
Now you can select the model, token length, and number of Intel Gaudi AI cards you want to use and run the associated command.  Copy the command on the right side into your terminal 

| Model      | Input Token Length | Output Token Length  | Intel Gaudi Cards | Batch Size | Max Throughput   | Run Command (Copy and paste into the terminal)                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|------------|--------------------|----------------------|-------------------|------------|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LLaMA 2 7b   | 128                | 128                  | 1                 | 1230       | 13163 tokens/sec |`QUANT_CONFIG=./quantization_config/maxabs_quant.json   TQDM_DISABLE=1 python3  run_generation.py --model_name_or_path meta-llama/Llama-2-7b-hf --attn_softmax_bf16 --use_hpu_graphs --trim_logits --use_kv_cache --limit_hpu_graphs --bucket_size=128 --bucket_internal --max_new_tokens 128 --max_input_tokens 128 --bf16 --batch_size 1230  --use_flash_attention --flash_attention_recompute`                                                             |
|            | 128                | 2048                 | 1                 | 163        | 4777 tokens/sec  |`QUANT_CONFIG=./quantization_config/maxabs_quant.json   TQDM_DISABLE=1 python3  run_generation.py --model_name_or_path meta-llama/Llama-2-7b-hf --attn_softmax_bf16 --use_hpu_graphs --trim_logits --use_kv_cache --limit_hpu_graphs --bucket_size=128 --bucket_internal --max_new_tokens 2048 --max_input_tokens 128 --bf16 --batch_size 163  --use_flash_attention --flash_attention_recompute`                                                                |
|            | 2048               | 128                  | 1                 | 94         | 1291 tokens/sec  |`QUANT_CONFIG=./quantization_config/maxabs_quant.json   TQDM_DISABLE=1 python3  run_generation.py --model_name_or_path meta-llama/Llama-2-7b-hf --attn_softmax_bf16 --use_hpu_graphs --trim_logits --use_kv_cache --limit_hpu_graphs --bucket_size=128 --bucket_internal --max_new_tokens 128 --max_input_tokens 2048 --bf16 --batch_size 94  --use_flash_attention --flash_attention_recompute`                                                                 |
|            | 2048               | 2048                 | 1                 | 81         | 1943 tokens/sec  |`QUANT_CONFIG=./quantization_config/maxabs_quant.json   TQDM_DISABLE=1 python3  run_generation.py --model_name_or_path meta-llama/Llama-2-7b-hf --attn_softmax_bf16 --use_hpu_graphs --trim_logits --use_kv_cache --limit_hpu_graphs --bucket_size=128 --bucket_internal --max_new_tokens 2048 --max_input_tokens 2048 --bf16 --batch_size 81  --use_flash_attention --flash_attention_recompute`                                                                |
| LLaMA 2 70b  | 128                | 128                  | 2                 | 1750       | 2727 tokens/sec  |`QUANT_CONFIG=./quantization_config/maxabs_quant.json   TQDM_DISABLE=1 python3 ../gaudi_spawn.py --use_deepspeed --world_size 2 run_generation.py --model_name_or_path meta-llama/Llama-2-70b-hf --attn_softmax_bf16 --use_hpu_graphs --trim_logits --use_kv_cache --limit_hpu_graphs --bucket_size=128 --bucket_internal --max_new_tokens 128 --max_input_tokens 128 --bf16 --batch_size 1750 --disk_offload --use_flash_attention --flash_attention_recompute` |
|            | 128                | 2048                 | 4                 | 750        | 7422 tokens/sec  |`QUANT_CONFIG=./quantization_config/maxabs_quant.json   TQDM_DISABLE=1 python3 ../gaudi_spawn.py --use_deepspeed --world_size 4 run_generation.py --model_name_or_path meta-llama/Llama-2-70b-hf --attn_softmax_bf16 --use_hpu_graphs --trim_logits --use_kv_cache --limit_hpu_graphs --bucket_size=128 --bucket_internal --max_new_tokens 2048 --max_input_tokens 128 --bf16 --batch_size 750  --use_flash_attention --flash_attention_recompute`               |
|            | 2048               | 128                  | 2                 | 95         | 276 tokens/sec   |`QUANT_CONFIG=./quantization_config/maxabs_quant.json   TQDM_DISABLE=1 python3 ../gaudi_spawn.py --use_deepspeed --world_size 2 run_generation.py --model_name_or_path meta-llama/Llama-2-70b-hf --attn_softmax_bf16 --use_hpu_graphs --trim_logits --use_kv_cache --limit_hpu_graphs --bucket_size=128 --bucket_internal --max_new_tokens 128 --max_input_tokens 2048 --bf16 --batch_size 95 --disk_offload --use_flash_attention --flash_attention_recompute`  |
|            | 2048               | 2048                 | 2                 | 78         | 958 tokens/sec   |`QUANT_CONFIG=./quantization_config/maxabs_quant.json   TQDM_DISABLE=1 python3 ../gaudi_spawn.py --use_deepspeed --world_size 2 run_generation.py --model_name_or_path meta-llama/Llama-2-70b-hf --attn_softmax_bf16 --use_hpu_graphs --trim_logits --use_kv_cache --limit_hpu_graphs --bucket_size=128 --bucket_internal --max_new_tokens 2048 --max_input_tokens 2048 --bf16 --batch_size 78 --disk_offload --use_flash_attention --flash_attention_recompute` |
| Mistral 7b Instruct | 128                | 128                  | 1                 | 896        | 13112 tokens/sec |`QUANT_CONFIG=./quantization_config/maxabs_quant.json python run_generation.py   --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 --attn_softmax_bf16 --use_hpu_graphs --trim_logits --use_kv_cache --reuse_cache --bf16 --batch_size 896 --max_new_tokens 128 --max_input_tokens 128 --limit_hpu_graphs`                                                                                                                                                |
|            | 128                | 2048                 | 1                 | 120        | 7947 tokens/sec  |`QUANT_CONFIG=./quantization_config/maxabs_quant.json python run_generation.py   --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 --attn_softmax_bf16 --use_hpu_graphs --trim_logits --use_kv_cache --reuse_cache --bf16 --batch_size 120 --max_new_tokens 2048 --max_input_tokens 128 --limit_hpu_graphs  --bucket_internal --bucket_size 128`                                                                                                          |
|            | 2048               | 128                  | 1                 | 120        | 1360 tokens/sec  |`QUANT_CONFIG=./quantization_config/maxabs_quant.json python run_generation.py   --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 --attn_softmax_bf16 --use_hpu_graphs --trim_logits --use_kv_cache --reuse_cache --bf16 --batch_size 120 --max_new_tokens 128 --max_input_tokens 2048 --limit_hpu_graphs`                                                                                                                                               |
|            | 2048               | 2048                 | 1                 | 44         | 3143 tokens/sec  |`QUANT_CONFIG=./quantization_config/maxabs_quant.json python run_generation.py   --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 --attn_softmax_bf16 --use_hpu_graphs --trim_logits --use_kv_cache --reuse_cache --bf16 --batch_size 44 --max_new_tokens 2048 --max_input_tokens 2048 --limit_hpu_graphs  --bucket_internal --bucket_size 128`                                                                                                          |



