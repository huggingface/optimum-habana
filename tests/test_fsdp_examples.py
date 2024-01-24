import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from .test_examples import ACCURACY_PERF_FACTOR, TIME_PERF_FACTOR


# Gaudi2 CI baselines
MODELS_TO_TEST = {
    "bf16": [
        (
            "bert-base-uncased",
            "Habana/bert-base-uncased",
            2807,
            85.4688,
            "question-answering",
            24,
            8,
            "run_qa.py",
            "full_shard",
        ),
    ],
}


def _test_fsdp(
    model_name: str,
    gaudi_config: str,
    baseline: float,
    baseline_acc: float,
    task: str,
    batch_size_train: int,
    batch_size_eval: int,
    script: str,
    policy: str,
    world_size: int = 8,
):
    os.environ["PT_HPU_LAZY_MODE"] = "0"
    os.environ["PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE"] = "0"  # To be removed later
    os.environ["PT_HPU_EAGER_PIPELINE_ENABLE"] = "0"  # To be removed later
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"

    # Install question-answering example requirements
    cmd_line = f"pip install -r {path_to_example_dir / task / 'requirements.txt'}".split()
    p = subprocess.Popen(cmd_line)
    return_code = p.wait()
    assert return_code == 0

    command = ["python3"]

    command += [
        f"{path_to_example_dir / 'gaudi_spawn.py'}",
        "--use_mpi",
        f"--world_size {world_size}",
    ]

    command += [
        f"{path_to_example_dir / task / script}",
        f"--model_name_or_path {model_name}",
        "--do_train",
        "--dataset_name squad",
        "--max_seq_length 384",
        f"--per_device_eval_batch_size {batch_size_eval}",
        f"--per_device_train_batch_size {batch_size_train}",
        "--learning_rate 3e-05",
        "--num_train_epochs 2.0",
        "--logging_steps 20",
        "--save_steps 5000",
        "--seed 42",
        "--doc_stride 128",
        "--use_habana",
        "--overwrite_output_dir",
        f"--gaudi_config_name {gaudi_config}",
        "--throughput_warmup_steps 100",
        f"--fsdp_config {path_to_example_dir / task / 'fsdp_config.json'}",
        f"--fsdp '{policy}'",
        "--do_eval",
        "--torch_compile_backend aot_hpu_training_backend",
        "--torch_compile",
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
        assert results["eval_f1"] >= ACCURACY_PERF_FACTOR * baseline_acc


@pytest.mark.parametrize(
    "model_name, gaudi_config, baseline, baseline_acc, task, bs_train, bs_eval, script, policy", MODELS_TO_TEST["bf16"]
)
def test_fsdp_bf16(
    model_name: str,
    gaudi_config: str,
    baseline: float,
    baseline_acc: float,
    task: str,
    bs_train: int,
    bs_eval: int,
    script: str,
    policy: str,
):
    _test_fsdp(model_name, gaudi_config, baseline, baseline_acc, task, bs_train, bs_eval, script, policy)
