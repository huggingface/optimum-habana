import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from .test_examples import TIME_PERF_FACTOR


prev_quant_model_name = None
prev_quant_rank = 0

if os.environ.get("GAUDI2_CI", "0") == "1":
    # Gaudi2 CI baselines
    MODELS_TO_TEST = {
        "bf16_1x": [
            ("google/gemma-2b-it", "mamamiya405/finred", "Habana/gpt2", True, 9.5, 31.5),
            ("google/gemma-2b-it", "mamamiya405/finred", "Habana/gpt2", False, 6.5, 24.01),
        ],
    }

def _test_language_modeling(
    model_name: str,
    baseline_train: float,
    baseline_eval: float,
    token: str,
    dataset_name: str,
    gaudi_config_name: str,
    use_lazy_mode: bool = False,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    num_train_epochs: int = 1,
):
    command = ["python3"]
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"
    env_variables = os.environ.copy()

    if not use_lazy_mode:
        env_variables["PT_HPU_LAZY_MODE"] = "0"

    command += [
        f"{path_to_example_dir / 'language-modeling' / 'run_clm.py'}",
        f"--model_name_or_path {model_name}",
        f"--per_device_train_batch_size {per_device_train_batch_size}",
        f"--per_device_eval_batch_size {per_device_eval_batch_size}",
        f"--dataset_name {dataset_name}",
        "--use_habana",
        "--do_train",
        "--do_eval",
        f"--gaudi_config_name {gaudi_config_name}",
        "--gradient_checkpointing",
        "--bf16",
        f"--num_train_epochs {num_train_epochs}",
        f"--use_lazy_mode {use_lazy_mode}",
    ]

    with TemporaryDirectory() as tmp_dir:
        command.append(f"--output_dir {tmp_dir} --overwrite_output_dir")
        command.append(f"--token {token.value}")
        pattern = re.compile(r"([\"\'].+?[\"\'])|\s")

        command = [x for y in command for x in re.split(pattern, y) if x]
        print(f"\n\nCommand to test: {' '.join(command[:-2])}\n")
        proc = subprocess.run(command, env=env_variables)

        # Ensure the run finished without any issue
        # Use try-except to avoid logging the token if used
        try:
            assert proc.returncode == 0
        except AssertionError as e:
            if "'--token', 'hf_" in e.args[0]:
                e.args = (f"The following command failed:\n{' '.join(command[:-2])}",)
            raise

        with open(Path(tmp_dir) / "train_results.json") as fp:
            results = json.load(fp)

        # Ensure performance requirements (throughput) are met
        assert results["train_samples_per_second"] >= (2 - TIME_PERF_FACTOR) * baseline_train

        with open(Path(tmp_dir) / "eval_results.json") as fp:
            results = json.load(fp)

        assert results["eval_samples_per_second"] >= (2 - TIME_PERF_FACTOR) * baseline_eval


@pytest.mark.parametrize("model_name, dataset_name, gaudi_config_name, use_lazy_mode, baseline_train, baseline_eval", MODELS_TO_TEST["bf16_1x"])
def test_language_modeling_bf16_1x(model_name: str, baseline_train: float, baseline_eval: float, dataset_name: str, gaudi_config_name: str, use_lazy_mode: bool, token: str):
    _test_language_modeling(model_name, baseline_train, baseline_eval, token, dataset_name, gaudi_config_name, use_lazy_mode)
