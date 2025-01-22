import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Union

import pytest

from .test_examples import TIME_PERF_FACTOR


if os.environ.get("GAUDI2_CI", "0") == "1":
    # Gaudi2 CI baselines
    MODELS_TO_TEST = {
        "bf16": [
            # ("llava-hf/llava-1.5-7b-hf", 1, 77.98733740859008),
            # ("llava-hf/llava-1.5-13b-hf", 1, 48.54364937033955),
            ("llava-hf/llava-v1.6-mistral-7b-hf", 1, 33.17984878151546),
            ("llava-hf/llava-v1.6-vicuna-7b-hf", 1, 35.00608681379742),
            ("llava-hf/llava-v1.6-vicuna-13b-hf", 1, 23.527610042925),
            ("google/paligemma-3b-mix-224", 1, 132.8949150246155),
            ("HuggingFaceM4/idefics2-8b", 1, 21.89944593215077),
            ("meta-llama/Llama-3.2-11B-Vision-Instruct", 1, 18.974541922240313),
            ("tiiuae/falcon-11B-vlm", 1, 23.69260849957278),
            ("Qwen/Qwen2-VL-2B-Instruct", 1, 28.755882208438422),
            ("Qwen/Qwen2-VL-7B-Instruct", 1, 19.32562189532818),
        ],
        "fp8": [
            # ("llava-hf/llava-1.5-7b-hf", 1, 98.72578382705062),
            # ("llava-hf/llava-1.5-13b-hf", 1, 67.20488222876344),
            ("llava-hf/llava-v1.6-mistral-7b-hf", 1, 45.011551008367084),
            ("llava-hf/llava-v1.6-vicuna-7b-hf", 1, 45.18544502949674),
            ("llava-hf/llava-v1.6-vicuna-13b-hf", 1, 30.9535718774675),
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

def install_requirements(requirements_filename: Union[str, os.PathLike]):
    """
    Installs the necessary requirements to run the example if the provided file exists, otherwise does nothing.
    """

    if not Path(requirements_filename).exists():
        return

    cmd_line = f"pip install -r {requirements_filename}".split()
    p = subprocess.Popen(cmd_line)
    return_code = p.wait()
    assert return_code == 0

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
    install_requirements(path_to_example_dir / 'image-to-text/requirements.txt')

    command += [
        f"{path_to_example_dir / 'image-to-text' / 'run_pipeline.py'}",
        f"--model_name_or_path {model_name}",
        f"--batch_size {batch_size}",
        "--max_new_tokens 20",
    ]

    command += [
        "--use_hpu_graphs",
    ]

    if "meta-llama/Llama-3.2-11B-Vision-Instruct" in model_name or "tiiuae/falcon-11B-vlm" in model_name:
        command += [
            "--sdp_on_bf16",
        ]

    command.append("--bf16")
    command.append("--sdp_on_bf16")

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
            quant_file_path = "image-to-text/quantization_config/maxabs_quant.json"
            if model_name in [
                "llava-hf/llava-v1.6-mistral-7b-hf",
                "llava-hf/llava-v1.6-vicuna-7b-hf",
                "llava-hf/llava-v1.6-vicuna-13b-hf",
                "llava-hf/llava-1.5-7b-hf",
                "llava-hf/llava-1.5-13b-hf",
            ]:
                quant_file_path = "image-to-text/quantization_config/maxabs_quant_scale_format_const.json"

            env_variables["QUANT_CONFIG"] = os.path.join(path_to_example_dir, quant_file_path)

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
