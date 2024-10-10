import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from optimum.habana.utils import set_seed


MODELS_TO_TEST = [
    # ("bigcode/starcoder",""),
    # ("bigcode/starcoder2-15b", ""),
    # ("google/gemma-2b", ""),
    (
        "meta-llama/Llama-2-7b-hf",
        "DeepSpeed is a machine learning framework for deep learning. It is designed to be fast and efficient, while also being easy to use. DeepSpeed is based on the TensorFlow framework, and it uses the TensorFlow library to perform computations.\nDeepSpeed is a deep learning framework that is designed to be fast and efficient. It is based on the TensorFlow library and uses the TensorFlow library to perform computations. DeepSpeed is designed to be easy to use and to provide a high level of flex",
    ),
    # ("mistralai/Mistral-7B-Instruct-v0.2",""),
    # ("mistralai/Mixtral-8x7B-v0.1", "" )
    # ("Qwen/Qwen2-7B", "")
]


def _test_text_generation(
    model_name: str,
    expected_output: str,
    token: str,
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

        with open(Path(tmp_dir) / "results.json") as fp:
            results = json.load(fp)

        assert results["output"][0] == expected_output


@pytest.mark.parametrize("model_name, expected_output", MODELS_TO_TEST)
def test_text_generation_bf16_1x(model_name: str, expected_output: str, token: str):
    _test_text_generation(model_name, expected_output, token)
