from datasets import load_dataset
from optimum.habana.trl import GaudiGRPOTrainer, GaudiGRPOConfig
from optimum.habana import GaudiConfig, GaudiTrainer
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import ScriptArguments


NUM_WORKERS = 8
# MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
MODEL_NAME = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


if __name__ == "__main__":
    parser = HfArgumentParser(GaudiGRPOConfig)
    (training_args,) = parser.parse_args_into_dataclasses()

    # dataset_name = "philschmid/dolly-15k-oai-style"
    dataset_name = "trl-lib/tldr"

    train_dataset = load_dataset(dataset_name,
        split="train",
        data_dir='',
        num_proc=NUM_WORKERS
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )

    gaudi_config = GaudiConfig()

    gaudi_config.use_fused_adam = True
    gaudi_config.use_fused_clip_norm = True

    trainer = GaudiGRPOTrainer(
        model=MODEL_NAME,
        reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
        train_dataset=train_dataset,
        gaudi_config=gaudi_config,
        args=training_args,
        processing_class=tokenizer,
    )

    trainer.train()

    print("Done!")
