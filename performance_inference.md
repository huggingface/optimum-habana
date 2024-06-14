# Replicating Large Language Model Inference Performance on Intel&reg; Gaudi&reg; AI Processor
This page provides instructions on how to run the exact configuration to replicate the Intel Gaudi Performance Numbers posted on the Intel Gaudi [Devleloper website](https://www.intel.com/content/www/us/en/developer/platform/gaudi/model-performance.html)

### Follow these steps: 

1. Get access to an Intel Gaudi node and [setup](https://docs.habana.ai/en/latest/shared/Pull_Prebuilt_Containers.html#pulling-prebuilt-container) the Intel Gaudi PyTorch docker for release 1.16.0
2. Since the performance numbers for 1.16.0 used the previous version of Optimum Habana, you will install the 1.11.1 
>```bash
> pip install optimum-habana==1.11.1
>```

3. Pull the 1.11.1 Version of the Optimum Habana Examples 
> ```
> cd ~ && git clone https://github.com/huggingface/optimum-habana
> cd optimum-habana && git checkout v1.11.1
> ```

4. Go to the text-generation examples and configure the setup 
> ```
> git clone https://github.com/huggingface/optimum-habana
> cd optimum-habana && git checkout v1.11.1
> ```

### Option 2: Use the latest main branch under development

Optimum for Intel Gaudi is a fast-moving project, and you may want to install it from source and get the latest scripts :

```bash
pip install git+https://github.com/huggingface/optimum-habana.git
git clone https://github.com/huggingface/optimum-habana
```

## Install dependencies

To use DeepSpeed on HPUs, you also need to run the following command:
>```bash
>pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.16.0
>```

To install the requirements for every example:
>```bash
>cd <example-folder>
>pip install -r requirements.txt
>```

i.	Setup the environment
ii.	Use the intel Gaudi docker 
iii.	Use the apprpiate Optimum habana version
iv.	Go to text-generation and 
v.	Run the following commands 
