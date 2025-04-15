import os
os.environ["PT_HPU_LAZY_MODE"]="1"

from datasets import load_dataset
from optimum.habana import GaudiConfig
from optimum.habana.trl import GaudiGRPOTrainer, GaudiGRPOConfig

if __name__ == "__main__":
    #parser = HfArgumentParser(ScriptArguments)
    #script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("trl-lib/tldr", split="train")

    # Define the reward function, which rewards completions that are close to 20 characters
    def reward_len(completions, **kwargs):
        return [-abs(20 - len(completion)) for completion in completions]

    training_args = GaudiGRPOConfig(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        output_dir="Qwen2-0.5B-GRPO",
        logging_steps=10,
        use_habana=True,
        use_lazy_mode=True,
    )

    gaudi_config = GaudiConfig()
    trainer = GaudiGRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
        gaudi_config=gaudi_config,
    )
    trainer.train()
