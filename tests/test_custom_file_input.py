import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from transformers.testing_utils import slow

from .utils import OH_DEVICE_CONTEXT


PATH_TO_RESOURCES = Path(__file__).resolve().parent.parent / "tests/resource"


if OH_DEVICE_CONTEXT not in ["gaudi1"]:
    # gaudi2+
    MODEL_FILE_OPTIONS_TO_TEST = {
        "bf16": [
            (
                "bigcode/starcoder",
                [
                    "--do_train",
                    f"--train_file {PATH_TO_RESOURCES}/custom_dataset.jsonl",
                    "--validation_split_percentage 10",
                ],
            ),
            (
                "bigcode/starcoder",
                [
                    "--do_train",
                    f"--train_file {PATH_TO_RESOURCES}/custom_dataset.txt",
                    "--validation_split_percentage 10",
                ],
            ),
            (
                "bigcode/starcoder",
                [
                    "--do_train",
                    f"--train_file {PATH_TO_RESOURCES}/custom_dataset.jsonl",
                    "--do_eval",
                    f"--validation_file {PATH_TO_RESOURCES}/custom_dataset.jsonl",
                    "--validation_split_percentage 20",
                ],
            ),
            (
                "bigcode/starcoder",
                [
                    "--do_train",
                    f"--train_file {PATH_TO_RESOURCES}/custom_dataset.txt",
                    "--do_eval",
                    f"--validation_file {PATH_TO_RESOURCES}/custom_dataset.txt",
                    "--validation_split_percentage 20",
                ],
            ),
            (
                "bigcode/starcoder",
                [
                    "--do_train",
                    "--dataset_name timdettmers/openassistant-guanaco",
                    "--do_eval",
                    f"--validation_file {PATH_TO_RESOURCES}/custom_dataset.jsonl",
                    "--validation_split_percentage 20",
                ],
            ),
        ],
    }
else:
    MODEL_FILE_OPTIONS_TO_TEST = {
        "bf16": [
            (
                "meta-llama/Llama-2-7b-hf",
                [
                    "--do_train",
                    f"--train_file {PATH_TO_RESOURCES}/custom_dataset.jsonl",
                    "--validation_split_percentage 10",
                ],
            ),
            (
                "meta-llama/Llama-2-7b-hf",
                [
                    "--do_train",
                    f"--train_file {PATH_TO_RESOURCES}/custom_dataset.txt",
                    "--validation_split_percentage 10",
                ],
            ),
            (
                "meta-llama/Llama-2-7b-hf",
                [
                    "--do_train",
                    f"--train_file {PATH_TO_RESOURCES}/custom_dataset.jsonl",
                    "--do_eval",
                    f"--validation_file {PATH_TO_RESOURCES}/custom_dataset.jsonl",
                    "--validation_split_percentage 20",
                ],
            ),
            (
                "meta-llama/Llama-2-7b-hf",
                [
                    "--do_train",
                    f"--train_file {PATH_TO_RESOURCES}/custom_dataset.txt",
                    "--do_eval",
                    f"--validation_file {PATH_TO_RESOURCES}/custom_dataset.txt",
                    "--validation_split_percentage 20",
                ],
            ),
            (
                "meta-llama/Llama-2-7b-hf",
                [
                    "--do_train",
                    "--dataset_name timdettmers/openassistant-guanaco",
                    "--do_eval",
                    f"--validation_file {PATH_TO_RESOURCES}/custom_dataset.jsonl",
                    "--validation_split_percentage 20",
                ],
            ),
        ],
    }


def _install_requirements():
    PATH_TO_EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples"
    cmd_line = f"pip install -r {PATH_TO_EXAMPLE_DIR / 'language-modeling' / 'requirements.txt'}".split()
    p = subprocess.Popen(cmd_line)
    return_code = p.wait()
    assert return_code == 0


def _test_custom_file_inputs(model_name: str, test_commands: list):
    _install_requirements()
    command = ["python3"]
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"
    env_variables = os.environ.copy()

    command += [
        f"{path_to_example_dir / 'language-modeling' / 'run_lora_clm.py'}",
        f"--model_name_or_path {model_name}",
        "--bf16",
        "--use_hpu_graphs",
        "--use_habana",
        "--num_train_epochs 3",
        "--per_device_train_batch_size 2",
        "--per_device_eval_batch_size 2",
        "--gradient_accumulation_steps 4",
        "--evaluation_strategy no",
        "--save_strategy steps ",
        "--save_steps 2000",
        "--save_total_limit 1",
        "--learning_rate 1e-4",
        "--logging_steps 1",
        "--use_lazy_mode",
        "--dataset_concatenation",
        "--throughput_warmup_steps 3",
    ]
    command.extend(test_commands)

    with TemporaryDirectory() as tmp_dir:
        command.append(f"--output_dir {tmp_dir}")
        print(f"\n\nCommand to test: {' '.join(command)}\n")

        pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
        command = [x for y in command for x in re.split(pattern, y) if x]

        proc = subprocess.run(command, env=env_variables)

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

        # Ensure model ran successfully
        assert "epoch" in results
        if "train_samples_per_second" in results:
            assert results["train_samples_per_second"] > 0
        if "eval_samples_per_second" in results:
            assert results["eval_samples_per_second"] > 0


@slow
@pytest.mark.parametrize("model_name, test_commands", MODEL_FILE_OPTIONS_TO_TEST["bf16"])
def test_custom_file_inputs_bf16(model_name: str, test_commands: list):
    _test_custom_file_inputs(model_name, test_commands)
