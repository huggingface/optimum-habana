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
        --training-args.output_dir="./sft" \
        --training-args.max_steps=500 \
        --training-args.logging_steps=10 \
        --training-args.save_steps=10 \
        --training-args.per_device_train_batch_size=4 \
        --training-args.per_device_eval_batch_size=1 \
        --training-args.gradient_accumulation_steps=2 \
        --training-args.learning_rate=1e-4 \
        --training-args.lr_scheduler_type="cosine" \
        --training-args.warmup_steps=100 \
        --training-args.weight_decay=0.05 \
        --training-args.optim="paged_adamw_32bit" \
        --training-args.bf16 \
        --training-args.remove_unused_columns=False \
        --training-args.run_name="sft_llama2" \
        --training-args.report_to=none
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
