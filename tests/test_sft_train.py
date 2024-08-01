import os
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.parametrize("model_name,expected", [("Qwen/Qwen2-7B", (30.12, 4.8347)), ("Qwen/Qwen2-72B", (6.969, 3.6))])
def test_sft_train(model_name, expected):
    ds_config = """{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu" :"auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "stage3_gather_16bit_weights_on_model_save": true
    },
    "flops_profiler": {
        "enabled": false,
        "profile_step": 1,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": true,
        "output_file": null
    }
}
    """
    env_variables = os.environ.copy()
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"
    filename = f"{path_to_example_dir / 'trl' / 'sft.py'}"
    gaudispawn_filename = f"{path_to_example_dir / 'gaudi_spawn.py'}"

    command = [
        "python3",
        gaudispawn_filename,
        "--world_size",
        "8",
        "--use_deepspeed",
        filename,
        "--model_name_or_path",
        model_name,
        "--dataset_name",
        "philschmid/dolly-15k-oai-style",
        "--streaming",
        "False",
        "--bf16",
        "True",
        "--output_dir",
        "./model_qwen",
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        "8",
        "--evaluation_strategy",
        "no",
        "--save_strategy",
        "no",
        "--learning_rate",
        "3e-4",
        "--warmup_ratio",
        "0.03",
        "--lr_scheduler_type",
        "cosine",
        "--max_grad_norm",
        "0.3",
        "--logging_steps",
        "1",
        "--do_train",
        "--do_eval",
        "--use_habana",
        "--use_lazy_mode",
        "--throughput_warmup_steps",
        "3",
        "--lora_r",
        "4",
        "--lora_alpha=16",
        "--lora_dropout=0.05",
        "--lora_target_modules",
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "--max_seq_length",
        "512",
        "--adam_epsilon",
        "1e-08",
        "--packing",
        "False",
        "--num_bucket",
        "8",
        "--subset",
        "''",
    ]
    if "72" in model_name:
        command += [
            "--max_steps",
            "50",
            "--gradient_checkpointing",
            "True",
            "--pipelining_fwd_bwd",
            "True",
        ]
    else:
        command += ["--max_steps", "100"]
    env_variables["DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED"] = "1"
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(str.encode(ds_config))
        fp.flush()
        if "72" in model_name:
            command += ["--deepspeed", fp.name]
        proc = subprocess.run(
            command, env=env_variables, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )

    assert proc.returncode == 0, f"Got these from process: stderr={proc.stderr}, stdout={proc.stdout}"
    alllines = proc.stdout.split("\n")
    train_samples_per_second = float(
        [line for line in alllines if "train_samples_per_second" in line][-1].split("=")[-1]
    )
    perplexity = float([line for line in alllines if "perplexity" in line][-1].split("=")[-1])
    train_samples_per_second_expected, perplexity_expected = expected
    assert (
        train_samples_per_second > 0.9 * train_samples_per_second_expected
    ), f"Got {train_samples_per_second}, expected 0.9*{train_samples_per_second_expected}"
    assert perplexity < 1.05 * perplexity_expected, f"Got {perplexity}, expected 1.05*{perplexity_expected}"
