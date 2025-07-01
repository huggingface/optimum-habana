# Examples


## Requirements

First, you should install the requirements:
```bash
$ pip install -U -r requirements.txt
```

## Supervised Finetuning

1. The following example is for the supervised Lora finetune with Qwen2 model for conversational format dataset.

    ```bash
    PT_HPU_LAZY_MODE=1 python sft.py \
        --model_name_or_path "Qwen/Qwen2-7B" \
        --dataset_name "philschmid/dolly-15k-oai-style" \
        --streaming False \
        --bf16 True \
        --subset '' \
        --output_dir ./model_qwen \
        --num_train_epochs 1 \
        --per_device_train_batch_size 16 \
        --eval_strategy "no" \
        --save_strategy "no" \
        --learning_rate 3e-4 \
        --warmup_ratio  0.03 \
        --lr_scheduler_type "cosine" \
        --max_grad_norm  0.3 \
        --logging_steps 1 \
        --do_train \
        --do_eval \
        --use_habana \
        --use_lazy_mode \
        --throughput_warmup_steps 3 \
        --use_peft True \
        --lora_r 4 \
        --lora_alpha=16 \
        --lora_dropout=0.05 \
        --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj" \
        --max_seq_length 512 \
        --adam_epsilon 1e-08 \
        --use_flash_attention
    ```

2. Supervised fine-tuning of the mistralai/Mixtral-8x7B-Instruct-v0.1 on 4 cards:

    ```bash
    PT_HPU_LAZY_MODE=1 DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1 PT_ENABLE_INT64_SUPPORT=1 python ../gaudi_spawn.py --world_size 4 --use_deepspeed sft.py \
        --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
        --dataset_name "philschmid/dolly-15k-oai-style" \
        --subset 'data/' \
        --streaming False \
        --deepspeed ../language-modeling/llama2_ds_zero3_config.json \
        --output_dir="./model_mixtral" \
        --do_train \
        --max_steps=500 \
        --logging_steps=10 \
        --save_steps=100 \
        --per_device_train_batch_size=2 \
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
        --max_seq_length 512 \
        --run_name="sft_mixtral" \
        --report_to=none \
        --use_habana \
        --use_lazy_mode
    ```

## DPO pipeline

### Training

#### For meta-llama/Llama-2-70b-hf

The following example is for the creation of StackLlaMa 2: a Stack exchange llama-v2-70b model. There are two main steps to the DPO training process.

For large model like Llama2-70B, we could use DeepSpeed Zero-3 to enable DPO training in multi-card.
steps like:
1. Supervised fine-tuning of the base llama-v2-70b model to create llama-v2-70b-se:

    ```bash
    PT_HPU_LAZY_MODE=1 DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1 PT_ENABLE_INT64_SUPPORT=1 python ../gaudi_spawn.py --world_size 8 --use_deepspeed sft.py \
        --model_name_or_path meta-llama/Llama-2-70b-hf \
        --dataset_name "lvwerra/stack-exchange-paired" \
        --deepspeed ../language-modeling/llama2_ds_zero3_config.json \
        --output_dir="./sft" \
        --do_train \
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
    ```bash
    python merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-70b-hf" --adapter_model_name="sft" --output_name="sft/final_merged_checkpoint"
    ```

2. Run the DPO trainer using the model saved by the previous step:
    ```bash
    PT_HPU_LAZY_MODE=1 DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1 PT_ENABLE_INT64_SUPPORT=1 python ../gaudi_spawn.py --world_size 8 --use_deepspeed dpo.py \
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

```bash
python merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-70b-hf" --adapter_model_name="dpo" --output_name="stack-llama-2"
```

which will also push the model to your HuggingFace hub account.

### Running the model

We can load the DPO-trained LoRA adaptors which were saved by the DPO training step and run it through the [text-generation example](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation).

```bash
PT_HPU_LAZY_MODE=1 PT_ENABLE_INT64_SUPPORT=1 python ../gaudi_spawn.py --world_size 8 --use_deepspeed run_generation.py \
--model_name_or_path ../trl/stack-llama-2/ \
--use_hpu_graphs \
--use_kv_cache \
--batch_size 1 \
--bf16 \
--max_new_tokens 100 \
--prompt "Here is my prompt"
```


## PPO pipeline

### Training

The following example is for the creation of StackLlaMa 2: a Stack exchange llama-v2-7b model.
There are three main steps to the PPO training process:
1. Supervised fine-tuning of the base llama-v2-7b model to create llama-v2-7b-se:
    ```bash
    PT_HPU_LAZY_MODE=1 PT_ENABLE_INT64_SUPPORT=1 python ../gaudi_spawn.py --world_size 8 --use_mpi sft.py \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --dataset_name "lvwerra/stack-exchange-paired" \
        --output_dir="./sft" \
        --do_train \
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
    ```bash
    python merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-7b-hf" --adapter_model_name="sft" --output_name="sft/final_merged_checkpoint"
    ```
2. Reward modeling using dialog pairs from the SE dataset on the llama-v2-7b-se to create llama-v2-7b-se-rm
    ```bash
    PT_HPU_LAZY_MODE=1 PT_ENABLE_INT64_SUPPORT=1 python ../gaudi_spawn.py --world_size 8 --use_mpi reward_modeling.py \
        --model_name_or_path=./sft/final_merged_checkpoint \
        --tokenizer_name_or_path=meta-llama/Llama-2-7b-hf \
        --output_dir=./rm
    ```
    To merge the adaptors into the base model we can use the `merge_peft_adapter.py` helper script that comes with TRL:

    ```bash
    python merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-7b-hf" --adapter_model_name="rm" --output_name="rm_merged_checkpoint"
    ```

3. RL fine-tuning of llama-v2-7b-se with the llama-v2-7b-se-rm reward model:
    ```bash
    PT_HPU_LAZY_MODE=1 PT_ENABLE_INT64_SUPPORT=1 python ../gaudi_spawn.py --world_size 8 --use_mpi ppo.py \
        --model_name_or_path=./sft/final_merged_checkpoint \
        --reward_model_name=./rm_merged_checkpoint \
        --tokenizer_name_or_path=meta-llama/Llama-2-7b-hf \
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

    ```bash
    python merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-7b-hf" --adapter_model_name="llama-se-rl-finetune" --output_name="rl_merged_checkpoint"
    ```

### Running the model
We can load the PPO-trained LoRA adaptors which were saved by the PPO training step and run it through the [text-generation example](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation).

```bash
PT_HPU_LAZY_MODE=1 PT_ENABLE_INT64_SUPPORT=1 python run_generation.py \
--model_name_or_path ../trl/rl_merged_checkpoint/ \
--use_hpu_graphs \
--use_kv_cache \
--batch_size 1 \
--bf16 \
--max_new_tokens 100 \
--prompt "Here is my prompt"
```

## DDPO pipeline

### Training
The following example is for fine-tuning stable diffusion using Denoising Diffusion Policy Optimization
([DDPO](https://huggingface.co/docs/trl/en/ddpo_trainer)). The implementation supports LoRA and
non-LoRA-based training. LoRA based training is faster and less finicky to converge than non-LoRA
based training. Recommendations for non-Lora based training (described [here](https://huggingface.co/blog/trl-ddpo))
are setting the learning rate relatively low (e.g., 1e-5) and disabling mixed precision training.
HPU graphs are enabled by default for better performance.

There are two main steps to the DDPO training process:

1. Fine-tuning of the base stable-diffusion model with LoRA to create ddpo-aesthetic-predictor:
```bash
PT_HPU_LAZY_MODE=1 python ddpo.py \
  --num_epochs=200 \
  --train_gradient_accumulation_steps=1 \
  --sample_num_steps=50 \
  --sample_batch_size=6 \
  --train_batch_size=3 \
  --sample_num_batches_per_epoch=4 \
  --per_prompt_stat_tracking=True \
  --per_prompt_stat_tracking_buffer_size=32 \
  --train_learning_rate=1e-05 \
  --tracker_project_name="stable_diffusion_training" \
  --log_with="tensorboard" \
  --use_habana \
  --use_hpu_graphs \
  --bf16 \
  --hf_hub_model_id="ddpo-finetuned-stable-diffusion" \
  --push_to_hub False \
  --sdp_on_bf16
```
> [!NOTE]
> Due to a known issue on Gaudi3, sample_batch_sizes should be changed to 3. The issue will be fixed in the future release.


2. Inference using the fine-tuned LoRA weights as shown in the example below:
```python
import torch

from optimum.habana import GaudiConfig
from optimum.habana.trl import GaudiDefaultDDPOStableDiffusionPipeline

gaudi_config = GaudiConfig.from_pretrained("Habana/stable-diffusion")
model_id = "CompVis/stable-diffusion-v1-4"
lora_model_id = "ddpo-finetuned-stable-diffusion"
pipeline = GaudiDefaultDDPOStableDiffusionPipeline(
    model_id,
    use_habana=True,
    use_hpu_graphs=True,
    gaudi_config=gaudi_config,
)
pipeline.sd_pipeline.load_lora_weights(lora_model_id)
device = torch.device("hpu")

# memory optimization
pipeline.vae.to(device, torch.bfloat16)
pipeline.text_encoder.to(device, torch.bfloat16)
pipeline.unet.to(device, torch.bfloat16)

prompts = ["lion", "squirrel", "crab", "starfish", "whale", "sponge", "plankton"]
results = pipeline(prompts)

for prompt, image in zip(prompts, results.images):
    image.save(f"{prompt}.png")
```
