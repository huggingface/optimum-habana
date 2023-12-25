# DPO pipeline for the creation of StackLlaMa 2: a Stack exchange llama-v2-7b model

## Prerequisites

Install all the dependencies in the `requirements.txt`:

```
$ pip install -U -r requirements.txt
```


## Training

There were two main steps to the DPO training process:
1. Supervised fine-tuning of the base llama-v2-7b model to create llama-v2-7b-se:

    ```
    python ../../gaudi_spawn.py --world_size 8 --use_mpi sft_llama2.py \
        --output_dir="./sft" \
        --max_steps=500 \
        --logging_steps=10 \
        --save_steps=10 \
        --per_device_train_batch_size=4 \
        --per_device_eval_batch_size=1 \
        --gradient_accumulation_steps=2 \
        --learning_rate=1e-4 \
        --lr_scheduler_type="cosine" \
        --warmup_steps=100 \
        --weight_decay=0.05 \
        --optim="paged_adamw_32bit" \
        --bf16 \
        --remove_unused_columns=False \
        --run_name="sft_llama2" \
        --report_to=none \
        --use_habana \
        --use_lazy_mode
    ```
2. Run the DPO trainer using the model saved by the previous step:
    ```
    python ../../gaudi_spawn.py --world_size 8 --use_mpi dpo_llama2.py \
        --model_name_or_path="sft/final_merged_checkpoint" \
        --output_dir="dpo" \
        --report_to=none
    ```


## Merging the adaptors

To merge the adaptors into the base model we can use the `merge_peft_adapter.py` helper script that comes with TRL:

```
python merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-7b-hf" --adapter_model_name="dpo" --output_name="stack-llama-2"
```

which will also push the model to your HuggingFace hub account.

## Running the model

We can load the DPO-trained LoRA adaptors which were saved by the DPO training step and load them via:

```py
from peft import AutoPeftModelForCausalLM


model = AutoPeftModelForCausalLM.from_pretrained(
    "dpo/final_checkpoint",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
)

model.generate(...)
```
