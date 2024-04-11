## fp8 Mistral performance commands

1. measurement step, done once to generate hqt\_output folder
Here is an example to measure the tensor quantization statistics on mistralai/Mistral-7B-Instruct-v0.2:

```bash
QUANT_CONFIG=./quantization_config/maxabs_measure_include_outputs.json python run_generation.py \
--model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--reuse_cache \
--bf16 \
--batch_size 1
```

2. quantization step, bs 896
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant.json python run_generation.py \
--model_name_or_path  mistralai/Mistral-7B-Instruct-v0.2 \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--limit_hpu_graphs \
--reuse_cache \
--max_new_tokens 128 \
--max_input_tokens 128 \
--bf16 \
--batch_size 896 \
--fp8
```


3. quantization step, bs 120
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant.json python run_generation.py \
--model_name_or_path  mistralai/Mistral-7B-Instruct-v0.2 \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--limit_hpu_graphs \
--reuse_cache \
--max_new_tokens 2048 \
--max_input_tokens 128 \
--bf16 \
--batch_size 120 \
--fp8
```



4. quantization step, bs 64 
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant.json python run_generation.py \
--model_name_or_path  mistralai/Mistral-7B-Instruct-v0.2 \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--limit_hpu_graphs \
--reuse_cache \
--max_new_tokens 128 \
--max_input_tokens 2048 \
--bf16 \
--batch_size 64 \
--fp8
```



5. quantization step, bs 56 
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant.json python run_generation.py \
--model_name_or_path  mistralai/Mistral-7B-Instruct-v0.2 \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--limit_hpu_graphs \
--reuse_cache \
--max_new_tokens 2048 \
--max_input_tokens 2048 \
--bf16 \
--batch_size 56 \
--fp8
```

## fp8 Mistral accuracy commands

1. measurement step
```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python run_lm_eval.py \
-o acc_mistral_bs1_measure_v2.txt \
--model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--reuse_cache \
--bf16 \
--batch_size 1
```

2. quantization step
```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python run_lm_eval.py \
-o acc_mistral_bs1_measure_v2.txt \
--model_name_or_path  mistralai/Mistral-7B-Instruct-v0.2 \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--reuse_cache \
--bf16 \
--batch_size 1
```
