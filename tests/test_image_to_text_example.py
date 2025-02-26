import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from .test_examples import TIME_PERF_FACTOR
from .utils import OH_DEVICE_CONTEXT


if OH_DEVICE_CONTEXT in ["gaudi2"]:
    # Gaudi2 CI baselines
    MODELS_TO_TEST = {
        "bf16": [
            # ("llava-hf/llava-1.5-7b-hf", 1),
            # ("llava-hf/llava-1.5-13b-hf", 1),
            ("llava-hf/llava-v1.6-mistral-7b-hf", 1),
            ("llava-hf/llava-v1.6-vicuna-7b-hf", 1),
            ("llava-hf/llava-v1.6-vicuna-13b-hf", 1),
            ("google/paligemma-3b-mix-224", 1),
            ("HuggingFaceM4/idefics2-8b", 1),
            ("meta-llama/Llama-3.2-11B-Vision-Instruct", 1),
            ("tiiuae/falcon-11B-vlm", 1),
            ("Qwen/Qwen2-VL-2B-Instruct", 1),
            ("Qwen/Qwen2-VL-7B-Instruct", 1),
        ],
        "fp8": [
            # ("llava-hf/llava-1.5-7b-hf", 1),
            # ("llava-hf/llava-1.5-13b-hf", 1),
            ("llava-hf/llava-v1.6-mistral-7b-hf", 1),
            ("llava-hf/llava-v1.6-vicuna-7b-hf", 1),
            ("llava-hf/llava-v1.6-vicuna-13b-hf", 1),
        ],
    }
else:
    # Gaudi1 CI baselines
    MODELS_TO_TEST = {
        "bf16": [
            ("llava-hf/llava-1.5-7b-hf", 1),
            ("llava-hf/llava-1.5-13b-hf", 1),
            ("llava-hf/llava-v1.6-mistral-7b-hf", 1),
            ("llava-hf/llava-v1.6-vicuna-13b-hf", 1),
        ],
        "fp8": [],
    }


def _test_image_to_text(
    model_name: str,
    baseline,
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
        "--ignore_eos",
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
        baseline.assertRef(
            compare=lambda actual, ref: actual >= (2 - TIME_PERF_FACTOR) * ref,
            context=[OH_DEVICE_CONTEXT],
            throughput=results["throughput"],
        )


@pytest.mark.parametrize("model_name, batch_size", MODELS_TO_TEST["bf16"])
def test_image_to_text_bf16(model_name: str, batch_size: int, baseline, token):
    _test_image_to_text(model_name, baseline, token, batch_size)


@pytest.mark.parametrize("model_name, batch_size", MODELS_TO_TEST["fp8"])
def test_image_to_text_fp8(model_name: str, batch_size: int, baseline, token):
    _test_image_to_text(model_name, baseline, token, batch_size, fp8=True)
