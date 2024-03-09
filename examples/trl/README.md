# Examples


## Prerequisites

Install all the dependencies in the `requirements.txt`:

```
$ pip install -U -r requirements.txt
```

## DPO pipeline

### Training

The following example is for the creation of StackLlaMa 2: a Stack exchange llama-v2-7b model.
There are two main steps to the DPO training process:
1. Supervised fine-tuning of the base llama-v2-7b model to create llama-v2-7b-se:

    ```
    python ../gaudi_spawn.py --world_size 8 --use_mpi sft.py \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --output_dir="./sft" \
        --max_steps=500 \
        --logging_steps=10 \
        --save_steps=100 \
        --per_device_train_batch_size=4 \
        --per_device_eval_batch_size=1 \
        --gradient_accumulation_steps=2 \
        --learning_rate=1e-4 \
        --lr_scheduler_type="cosine" \
        --warmup_steps=100 \
        --weight_decay=0.05 \
        --optim="paged_adamw_32bit" \
        --lora_target_modules "q_proj" "v_proj" \
        --bf16 \
        --remove_unused_columns=False \
        --run_name="sft_llama2" \
        --report_to=none \
        --use_habana \
        --use_lazy_mode
    ```
    To merge the adaptors to get the final sft merged checkpoint, we can use the `merge_peft_adapter.py` helper script that comes with TRL:
    ```
    python merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-7b-hf" --adapter_model_name="sft" --output_name="sft/final_merged_checkpoint"
    ```

2. Run the DPO trainer using the model saved by the previous step:
    ```
    python ../gaudi_spawn.py --world_size 8 --use_mpi dpo.py \
        --model_name_or_path="sft/final_merged_checkpoint" \
        --tokenizer_name_or_path=meta-llama/Llama-2-7b-hf \
        --lora_target_modules "q_proj" "v_proj" "k_proj" "out_proj" "fc_in" "fc_out" "wte" \
        --output_dir="dpo" \
        --report_to=none
    ```
For large model like Llama2-70B, we could use DeepSpeed Zero-3 to enable DPO training in multi-card.
steps like:
1. Supervised fine-tuning of the base llama-v2-70b model to create llama-v2-70b-se:

    ```
    DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1 python ../gaudi_spawn.py --world_size 8 --use_deepspeed sft.py \
        --model_name_or_path meta-llama/Llama-2-70b-hf \
        --deepspeed ../language-modeling/llama2_ds_zero3_config.json \
        --output_dir="./sft" \
        --max_steps=500 \
        --logging_steps=10 \
        --save_steps=100 \
        --per_device_train_batch_size=1 \
        --per_device_eval_batch_size=1 \
        --gradient_accumulation_steps=2 \
        --learning_rate=1e-4 \
        --lr_scheduler_type="cosine" \
        --warmup_steps=100 \
        --weight_decay=0.05 \
        --optim="paged_adamw_32bit" \
        --lora_target_modules "q_proj" "v_proj" \
        --bf16 \
        --remove_unused_columns=False \
        --run_name="sft_llama2" \
        --report_to=none \
        --use_habana \
        --use_lazy_mode
    ```
    To merge the adaptors to get the final sft merged checkpoint, we can use the `merge_peft_adapter.py` helper script that comes with TRL:
    ```
    python merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-70b-hf" --adapter_model_name="sft" --output_name="sft/final_merged_checkpoint"
    ```

2. Run the DPO trainer using the model saved by the previous step:
    ```
    DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1 python ../gaudi_spawn.py --world_size 8 --use_deepspeed dpo.py \
        --model_name_or_path="sft/final_merged_checkpoint" \
        --tokenizer_name_or_path=meta-llama/Llama-2-70b-hf \
        --deepspeed ../language-modeling/llama2_ds_zero3_config.json \
        --lora_target_modules "q_proj" "v_proj" "k_proj" "out_proj" "fc_in" "fc_out" "wte" \
        --output_dir="dpo" \
        --max_prompt_length=256 \
        --max_length=512 \
        --report_to=none
    ```


### Merging the adaptors

To merge the adaptors into the base model we can use the `merge_peft_adapter.py` helper script that comes with TRL:

```
python merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-7b-hf" --adapter_model_name="dpo" --output_name="stack-llama-2"
```

which will also push the model to your HuggingFace hub account.

### Running the model

We can load the DPO-trained LoRA adaptors which were saved by the DPO training step and run it through the [text-generation example](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation).

```
python run_generation.py \
--model_name_or_path ../trl/stack-llama-2/ \
--use_hpu_graphs --use_kv_cache --batch_size 1 --bf16 --max_new_tokens 100 \
--prompt "Here is my prompt"

```


## PPO pipeline

### Training

The following example is for the creation of StackLlaMa 2: a Stack exchange llama-v2-7b model.
There are three main steps to the PPO training process:
1. Supervised fine-tuning of the base llama-v2-7b model to create llama-v2-7b-se:
    ```
    python ../gaudi_spawn.py --world_size 8 --use_mpi sft.py \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --output_dir="./sft" \
        --max_steps=500 \
        --logging_steps=10 \
        --save_steps=100 \
        --per_device_train_batch_size=4 \
        --per_device_eval_batch_size=1 \
        --gradient_accumulation_steps=2 \
        --learning_rate=1e-4 \
        --lr_scheduler_type="cosine" \
        --warmup_steps=100 \
        --weight_decay=0.05 \
        --optim="paged_adamw_32bit" \
        --lora_target_modules "q_proj" "v_proj" \
        --bf16 \
        --remove_unused_columns=False \
        --run_name="sft_llama2" \
        --report_to=none \
        --use_habana \
        --use_lazy_mode
    ```
    To merge the adaptors to get the final sft merged checkpoint, we can use the `merge_peft_adapter.py` helper script that comes with TRL:
    ```
    python merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-7b-hf" --adapter_model_name="sft" --output_name="sft/final_merged_checkpoint"
    ```
2. Reward modeling using dialog pairs from the SE dataset on the llama-v2-7b-se to create llama-v2-7b-se-rm
    ```
    python ../gaudi_spawn.py --world_size 8 --use_mpi reward_modeling.py \
        --model_name=./sft/final_merged_checkpoint \
        --tokenizer_name=meta-llama/Llama-2-7b-hf \
        --output_dir=./rm
    ```
    To merge the adaptors into the base model we can use the `merge_peft_adapter.py` helper script that comes with TRL:

    ```
    python merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-7b-hf" --adapter_model_name="rm" --output_name="rm_merged_checkpoint"
    ```

3. RL fine-tuning of llama-v2-7b-se with the llama-v2-7b-se-rm reward model:
    ```
    python ../gaudi_spawn.py --world_size 8 --use_mpi ppo.py \
        --model_name=./sft/final_merged_checkpoint \
        --reward_model_name=./rm_merged_checkpoint \
        --tokenizer_name=meta-llama/Llama-2-7b-hf \
        --adafactor=False \
        --output_max_length=128 \
        --batch_size=8 \
        --gradient_accumulation_steps=8 \
        --batched_gen=True \
        --ppo_epochs=4 \
        --seed=0 \
        --learning_rate=1.4e-5 \
        --early_stopping=True \
        --output_dir=llama-se-rl-finetune
    ```
    To merge the adaptors into the base model we can use the `merge_peft_adapter.py` helper script that comes with TRL:

    ```
    python merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-7b-hf" --adapter_model_name="llama-se-rl-finetune" --output_name="rl_merged_checkpoint"
    ```

### Running the model
We can load the PPO-trained LoRA adaptors which were saved by the PPO training step and run it through the [text-generation example](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation).

```
python run_generation.py \
--model_name_or_path ../trl/rl_merged_checkpoint/ \
--use_hpu_graphs --use_kv_cache --batch_size 1 --bf16 --max_new_tokens 100 \
--prompt "Here is my prompt"
```

