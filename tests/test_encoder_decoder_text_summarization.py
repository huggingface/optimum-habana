import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from .test_examples import ACCURACY_PERF_FACTOR, TIME_PERF_FACTOR


if os.environ.get("GAUDI2_CI", "false") == "true":
    # Gaudi2 CI baselines
    MODELS_TO_TEST = {
        "bf16": [
            ("facebook/bart-large-cnn", "Habana/bart", 5.568, 26.0688, 2, 1),
        ],
    }
else:
    # Gaudi1 CI baselines
    MODELS_TO_TEST = {
        "bf16": [
            ("facebook/bart-large-cnn", "Habana/bart", 2.612, 26.3777, 2, 1),
        ],
    }


def _test_text_summarization(
    model_name: str,
    gaudi_config: str,
    baseline: float,
    baseline_acc: float,
    batch_size: int,
    num_beams: int,
    token: str,
    deepspeed: bool = False,
    world_size: int = 8,
):
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"

    # Install summarization example requirements
    cmd_line = f"pip install -r {path_to_example_dir / 'summarization' / 'requirements.txt'}".split()
    p = subprocess.Popen(cmd_line)
    return_code = p.wait()
    assert return_code == 0

    command = ["python3"]

    if deepspeed:
        command += [
            f"{path_to_example_dir / 'gaudi_spawn.py'}",
            "--use_deepspeed",
            f"--world_size {world_size}",
        ]

    command += [
        f"{path_to_example_dir / 'summarization' / 'run_summarization.py'}",
        f"--model_name_or_path {model_name}",
        "--do_predict",
        "--predict_with_generate",
        "--dataset_name cnn_dailymail",
        "--dataset_config 3.0.0",
        "--use_habana",
        f"--per_device_eval_batch_size {batch_size}",
        f"--gaudi_config_name {gaudi_config}",
        f"--generation_num_beams {num_beams}",
        "--ignore_pad_token_for_loss False",
        "--pad_to_max_length",
        "--use_hpu_graphs_for_inference",
        "--use_lazy_mode",
        "--max_predict_samples 200",
    ]

    if not deepspeed:
        command.append("--bf16")

    with TemporaryDirectory() as tmp_dir:
        command.append(f"--output_dir {tmp_dir}")
        print(f"\n\nCommand to test: {' '.join(command)}\n")

        # command.append(f"--token {token.value}")

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

        with open(Path(tmp_dir) / "predict_results.json") as fp:
            results = json.load(fp)

        # Ensure performance requirements (throughput) are met
        assert results["predict_samples_per_second"] >= (2 - TIME_PERF_FACTOR) * baseline
        assert results["predict_rougeLsum"] >= ACCURACY_PERF_FACTOR * baseline_acc


@pytest.mark.parametrize(
    "model_name, gaudi_config, baseline, baseline_acc, batch_size, num_beams", MODELS_TO_TEST["bf16"]
)
def test_text_summarization_bf16(
    model_name: str,
    gaudi_config: str,
    baseline: float,
    baseline_acc: float,
    batch_size: int,
    num_beams: int,
    token: str,
):
    _test_text_summarization(model_name, gaudi_config, baseline, baseline_acc, batch_size, num_beams, token)
