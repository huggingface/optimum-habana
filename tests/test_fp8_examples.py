import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from .test_examples import ACCURACY_PERF_FACTOR, TIME_PERF_FACTOR


if os.environ.get("GAUDI2_CI", "0") == "1":
    # Gaudi2 CI baselines
    MODELS_TO_TEST = {
        "fp8": [
            (
                "mistralai/Mistral-7B-Instruct-v0.2",
                "tatsu-lab/alpaca",
                "",
                12.373,
                0.7538,
                "language-modeling",
                8,
                8,
                "run_lora_clm.py",
            ),
        ],
    }
else:
    # FP8 is not supported on Gaudi1
    MODELS_TO_TEST = {"fp8": []}


def _test_fp8_train(
    model_name: str,
    dataset_name: str,
    gaudi_config: str,
    baseline: float,
    baseline_acc: float,
    task: str,
    batch_size_train: int,
    batch_size_eval: int,
    script: str,
    token: str,
    world_size: int = 8,
):
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"

    # Install question-answering example requirements
    cmd_line = f"pip install -r {path_to_example_dir / task / 'requirements.txt'}".split()
    p = subprocess.Popen(cmd_line)
    return_code = p.wait()
    assert return_code == 0

    command = ["python3"]

    command += [
        f"{path_to_example_dir / task / script}",
        f"--model_name_or_path {model_name}",
        f"--dataset_name {dataset_name}",
        "--do_train",
        "--do_eval",
        f"--per_device_eval_batch_size {batch_size_eval}",
        f"--per_device_train_batch_size {batch_size_train}",
        "--use_habana",
        "--use_lazy_mode",
        "--fp8 True",
    ]

    if model_name == "mistralai/Mistral-7B-Instruct-v0.2":
        command += [
            "--num_train_epochs 3",
            "--eval_strategy no",
            "--save_strategy no",
            "--learning_rate 4e-4",
            "--warmup_ratio 0.03",
            "--lr_scheduler_type constant",
            "--max_grad_norm 0.3",
            "--logging_steps 1",
            "--throughput_warmup_steps 5",
            "--lora_rank 8",
            "--lora_target_modules v_proj q_proj",
            "--lora_alpha 16",
            "--lora_dropout 0.05",
            "--dataset_concatenation",
            "--max_seq_length 512",
            "--low_cpu_mem_usage True",
            "--validation_split_percentage 4",
            "--adam_epsilon 1e-08",
            f"--token {token.value}",
        ]

    with TemporaryDirectory() as tmp_dir:
        command.append(f"--output_dir {tmp_dir}")
        print(f"\n\nCommand to test: {' '.join(command)}\n")

        pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
        command = [x for y in command for x in re.split(pattern, y) if x]

        proc = subprocess.run(command)

        # Ensure the run finished without any issue
        # Use try-except to avoid logging the token if used
        try:
            assert proc.returncode == 0
        except AssertionError as e:
            if "'--token', 'hf_" in e.args[0]:
                e.args = (f"The following command failed:\n{' '.join(command[:-2])}",)
            raise

        with open(Path(tmp_dir) / "all_results.json") as fp:
            results = json.load(fp)

        # Ensure performance requirements (throughput) are met
        assert results["train_samples_per_second"] >= (2 - TIME_PERF_FACTOR) * baseline
        assert results["eval_accuracy"] >= ACCURACY_PERF_FACTOR * baseline_acc


@pytest.mark.parametrize(
    "model_name, dataset_name, gaudi_config, baseline, baseline_acc, task, bs_train, bs_eval, script",
    MODELS_TO_TEST["fp8"],
)
def test_fp8_train(
    model_name: str,
    dataset_name: str,
    gaudi_config: str,
    baseline: float,
    baseline_acc: float,
    task: str,
    bs_train: int,
    bs_eval: int,
    script: str,
    token: str,
):
    _test_fp8_train(
        model_name, dataset_name, gaudi_config, baseline, baseline_acc, task, bs_train, bs_eval, script, token
    )
