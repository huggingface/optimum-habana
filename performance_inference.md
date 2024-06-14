# Replicating Large Language Model Inference Performance on the Intel&reg; Gaudi&reg; AI Processor
This page provides instructions on how to run the exact configuration to replicate the Intel Gaudi Max Throughput Performance Numbers posted on the Intel Gaudi [Devleloper website](https://www.intel.com/content/www/us/en/developer/platform/gaudi/model-performance.html) for the following Transformer based models:  

<div align="left">

| Model | Precision | Intel Gaudi Accelerators |
|--------------|:---------:|:-------------:|
| Llama 2 7B  | FP8 | 1 card | 
| Llama 2 70B | FP8 | 2, 4 cards |
| Mistral 7B Instruct | FP8 | 1 card |

</div>

## Initial Setup: 

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
## Run the exapmles
Since these are using FP8 precsion, the existing measurement files will be used. 

these examples are taking advantage of several tecniques 
