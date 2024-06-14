import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from .test_examples import TIME_PERF_FACTOR


if os.environ.get("GAUDI2_CI", "0") == "1":
    # Gaudi2 CI baselines
    MODELS_TO_TEST = {
        "bf16": [
            ("llava-hf/llava-1.5-7b-hf", 1, 87.2901500056982),
            ("llava-hf/llava-1.5-13b-hf", 1, 54.41252589197953),
            ("llava-hf/llava-v1.6-mistral-7b-hf", 1, 33.17984878151546),
            ("llava-hf/llava-v1.6-vicuna-13b-hf", 1, 23.527610042925),
        ],
        "fp8": [
            ("llava-hf/llava-1.5-7b-hf", 1, 123.00953973789325),
            ("llava-hf/llava-1.5-13b-hf", 1, 82.81132373492122),
        ],
    }
else:
    # Gaudi1 CI baselines
    MODELS_TO_TEST = {
        "bf16": [
            ("llava-hf/llava-1.5-7b-hf", 1, 28.04096918512148),
            ("llava-hf/llava-1.5-13b-hf", 1, 16.704731010481538),
            ("llava-hf/llava-v1.6-mistral-7b-hf", 1, 10.759228696741),
            ("llava-hf/llava-v1.6-vicuna-13b-hf", 1, 6.96732060769783),
        ],
        "fp8": [],
    }


def _test_image_to_text(
    model_name: str,
    baseline: float,
    token: str,
    batch_size: int = 1,
    fp8: bool = False,
):
    command = ["python3"]
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"
    env_variables = os.environ.copy()

    command += [
        f"{path_to_example_dir / 'image-to-text' / 'run_pipeline.py'}",
        f"--model_name_or_path {model_name}",
        f"--batch_size {batch_size}",
        "--max_new_tokens 20",
    ]

    command += [
        "--use_hpu_graphs",
    ]

    command.append("--bf16")

    with TemporaryDirectory() as tmp_dir:
        command.append(f"--output_dir {tmp_dir}")
        print(f"\n\nCommand to test: {' '.join(command)}\n")

        command.append(f"--token {token.value}")

        pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
        command = [x for y in command for x in re.split(pattern, y) if x]

        if fp8:
            print(f"\n\nCommand to test: {' '.join(command)}\n")
            env_variables["QUANT_CONFIG"] = os.path.join(
                path_to_example_dir, "image-to-text/quantization_config/maxabs_measure_include_outputs.json"
            )
            subprocess.run(command, env=env_variables)
            env_variables["QUANT_CONFIG"] = os.path.join(
                path_to_example_dir, "image-to-text/quantization_config/maxabs_quant.json"
            )

        proc = subprocess.run(command, env=env_variables)

        # Ensure the run finished without any issue
        # Use try-except to avoid logging the token if used
        try:
            assert proc.returncode == 0
        except AssertionError as e:
            if "'--token', 'hf_" in e.args[0]:
                e.args = (f"The following command failed:\n{' '.join(command[:-2])}",)
            raise

        with open(Path(tmp_dir) / "results.json") as fp:
            results = json.load(fp)

        # Ensure performance requirements (throughput) are met
        assert results["throughput"] >= (2 - TIME_PERF_FACTOR) * baseline


@pytest.mark.parametrize("model_name, batch_size, baseline", MODELS_TO_TEST["bf16"])
def test_image_to_text_bf16(model_name: str, baseline: float, batch_size: int, token: str):
    _test_image_to_text(model_name, baseline, token, batch_size)


@pytest.mark.parametrize("model_name, batch_size, baseline", MODELS_TO_TEST["fp8"])
def test_image_to_text_fp8(model_name: str, baseline: float, batch_size: int, token: str):
    _test_image_to_text(model_name, baseline, token, batch_size, fp8=True)
