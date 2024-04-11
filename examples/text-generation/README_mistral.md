## fp8 Mistral commands

### measurement step, done once to generate hqt\_output folder
Here is an example to measure the tensor quantization statistics on mistralai/Mistral-7B-Instruct-v0.2:

```bash
QUANT_CONFIG=./quantization_config/maxabs_measure_include_outputs.json python run_generation.py \
--model_name_or_path mistralai/Mistral-7B-v0.1 \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--reuse_cache \
--bf16 \
--batch_size 1 2>&1 | tee mistral_measurement.txt
```

### quantization step, bs 896

```bash
python run_generation.py \
--model_name_or_path  mistralai/Mistral-7B-v0.1 \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--limit_hpu_graphs \
--reuse_cache \
--max_new_tokens 128 \
--max_input_tokens 128 \
--bf16 \
--batch_size 896 --fp8 2>&1 | tee fp8_quant_mistral_bs896_outp128_in128.log
```


### quantization step, bs 120

```bash
python run_generation.py \
--model_name_or_path  mistralai/Mistral-7B-v0.1 \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--limit_hpu_graphs \
--reuse_cache \
--max_new_tokens 2048 \
--max_input_tokens 128 \
--bf16 \
--batch_size 120 --fp8 2>&1 | tee fp8_quant_mistral_bs120_outp2048_in128.log
```



### quantization step, bs 64 

```bash
python run_generation.py \
--model_name_or_path  mistralai/Mistral-7B-v0.1 \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--limit_hpu_graphs \
--reuse_cache \
--max_new_tokens 128 \
--max_input_tokens 2048 \
--bf16 \
--batch_size 64 --fp8 2>&1 | tee fp8_quant_mistral_bs64_outp128_in2048.log
```



### quantization step, bs 56 

```bash
python run_generation.py \
--model_name_or_path  mistralai/Mistral-7B-v0.1 \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--limit_hpu_graphs \
--reuse_cache \
--max_new_tokens 2048 \
--max_input_tokens 2048 \
--bf16 \
--batch_size 56 --fp8 2>&1 | tee fp8_quant_mistral_bs56_outp2048_in2048.log
```
