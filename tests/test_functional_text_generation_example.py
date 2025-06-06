import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from optimum.habana.utils import set_seed

from .utils import OH_DEVICE_CONTEXT


MODELS_TO_TEST = [
    "bigcode/starcoder",
    "bigcode/starcoder2-3b",
    "google/gemma-7b",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mixtral-8x7B-v0.1",
    "Qwen/Qwen2-7B",
]


def _test_text_generation(
    model_name: str,
    token: str,
    baseline,
):
    set_seed(42)
    command = ["python3"]
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"
    env_variables = os.environ.copy()

    command += [
        f"{path_to_example_dir}/text-generation/run_generation.py ",
        f"--model_name_or_path {model_name}",
        "--use_kv_cache",
        "--use_hpu_graphs",
        "--bf16",
    ]

    with TemporaryDirectory() as tmp_dir:
        command.append(f"--output_dir {tmp_dir}")
        command.append(f"--token {token.value}")

        pattern = re.compile(r"([\"\"].+?[\"\"])|\s")

        command = [x for y in command for x in re.split(pattern, y) if x]
        if "starcoder" in model_name:
            command.append("--prompt")
            command.append("def print_hello_world():")
        print(f"\n\nCommand to test: {' '.join(command)}\n")
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

        baseline.assertEqual(output=results["output"][0][0])


@pytest.mark.skipif("gaudi1" == OH_DEVICE_CONTEXT, reason="execution not supported on gaudi1")
@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_text_generation_bf16_1x(model_name: str, token: str, baseline):
    _test_text_generation(model_name, token, baseline)
