from datasets import load_dataset
from optimum.habana.trl import GaudiGRPOTrainer, GaudiGRPOConfig
from optimum.habana import GaudiConfig
from transformers import HfArgumentParser
from trl import GRPOTrainer, GRPOConfig

NUM_WORKERS = 16
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"


# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


if __name__ == "__main__":
    parser = HfArgumentParser(GaudiGRPOConfig)
    (training_args,) = parser.parse_args_into_dataclasses()

    train_dataset = load_dataset("trl-lib/tldr",
        split="train",
        data_dir='',
        num_proc=NUM_WORKERS
    )
    
    gaudi_config = GaudiConfig()

    gaudi_config.use_fused_adam = True
    gaudi_config.use_fused_clip_norm = True

    trainer = GaudiGRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=reward_len,
        train_dataset=train_dataset,
        gaudi_config=gaudi_config,
        args=training_args
    )
    trainer.train()

    print("Done!")
