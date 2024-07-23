from .test_examples import TIME_PERF_FACTOR
import subprocess
import pytest
from pathlib import Path
import os
from datasets import load_dataset
import shutil

def test_sft_train():
    env_variables = os.environ.copy()
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"
    filename = f"{path_to_example_dir / 'trl' / 'sft.py'}"
    gaudispawn_filename = f"{path_to_example_dir / 'gaudi_spawn.py'}"

    command = ['python3', gaudispawn_filename, '--world_size', '8', '--use_deepspeed', filename, \
    '--model_name_or_path', 'Qwen/Qwen2-7B', \
    '--dataset_name', 'philschmid/dolly-15k-oai-style', '--streaming', 'False', '--bf16', \
    'True', '--output_dir', './model_qwen', '--num_train_epochs', '1', '--per_device_train_batch_size', '8', \
    '--evaluation_strategy', 'no', '--save_strategy', 'no', '--learning_rate', '3e-4', \
    '--warmup_ratio', '0.03', '--lr_scheduler_type', 'cosine', '--max_grad_norm', '0.3', \
    '--logging_steps', '1', '--do_train', '--do_eval', '--use_habana', '--use_lazy_mode', \
    '--throughput_warmup_steps', '3', '--lora_r', '4', '--lora_alpha=16', '--lora_dropout=0.05', \
    '--lora_target_modules', 'q_proj', 'v_proj', 'k_proj', 'o_proj', '--max_seq_length', \
    '512', '--adam_epsilon', '1e-08', '--packing', 'False', '--num_bucket', '8', '--subset', "''", '--max_steps', '100']
    env_variables["DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED"] = "1"
    print(f"\n\nCommand to test: {' '.join(command)}\n")
    proc = subprocess.run(command, env=env_variables, stdout = subprocess.PIPE, stderr = subprocess.PIPE, universal_newlines = True)

    try:
        assert proc.returncode == 0
    except AssertionError as e:
        if "'--token', 'hf_" in e.args[0]:
            e.args = (f"The following command failed:\n{' '.join(command)}",)
        raise

    alllines = proc.stdout.split('\n')
    train_samples_per_second = float([line for line in alllines if 'train_samples_per_second' in line][-1].split('=')[-1])
    perplexity = float([line for line in alllines if 'perplexity' in line][-1].split('=')[-1])
    assert train_samples_per_second > 0.9 * 30.12
    assert perplexity < 1.05 * 4.8347


