model=path/to/model
tag=run1

python run_generation.py --model_name_or_path $model --use_hpu_graphs --use_kv_cache --bf16 --batch_size 1 --max_input_tokens 128 --max_new_tokens 128 --reuse_cache --trim_logits --attn_softmax_bf16 --limit_hpu_graph 2>&1 | tee log_$tag_128_128_bf16_1x.txt
python run_generation.py --model_name_or_path $model --use_hpu_graphs --use_kv_cache --bf16 --batch_size 1 --max_input_tokens 2048 --max_new_tokens 128 --reuse_cache --trim_logits --attn_softmax_bf16 --limit_hpu_graph 2>&1 | tee log_$tag_128_128_bf16_1x.txt
python run_generation.py --model_name_or_path $model --use_hpu_graphs --use_kv_cache --bf16 --batch_size 1 --max_input_tokens 128 --max_new_tokens 2048 --reuse_cache --trim_logits --attn_softmax_bf16 --limit_hpu_graph --bucket_size 128 --bucket_internal 2>&1 | tee log_$tag_128_128_bf16_1x.txt
python run_generation.py --model_name_or_path $model --use_hpu_graphs --use_kv_cache --bf16 --batch_size 1 --max_input_tokens 2048 --max_new_tokens 2048 --reuse_cache --trim_logits --attn_softmax_bf16 --limit_hpu_graph --bucket_size 128 --bucket_internal 2>&1 | tee log_$tag_128_128_bf16_1x.txt
