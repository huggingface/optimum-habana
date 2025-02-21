import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from optimum.habana.utils import set_seed

from .utils import OH_DEVICE_CONTEXT


if OH_DEVICE_CONTEXT in ["gaudi2"]:
    MODEL_OUTPUTS = {
        "bigcode/starcoder": 'def print_hello_world():\n    print("Hello World")\n\ndef print_hello_world_twice():\n    print_hello_world()\n    print_hello_world()\n\ndef print_hello_world_thrice():\n    print_hello_world()\n    print_hello_world()\n    print_hello_world()\n\ndef print_hello_world_four_times():\n    print_hello_world()\n    print_hello_world()\n    print_hello_world()\n   ',
        "bigcode/starcoder2-3b": 'def print_hello_world():\n    print("Hello World")\n\ndef print_hello_world_with_name(name):\n    print("Hello World, " + name)\n\ndef print_hello_world_with_name_and_age(name, age):\n    print("Hello World, " + name + ", " + str(age))\n\ndef print_hello_world_with_name_and_age_and_gender(name, age, gender):\n    print("Hello',
        "google/gemma-7b": "DeepSpeed is a machine learning framework that enables training of large-scale models on commodity hardware. It is designed to be a drop-in replacement for PyTorch, and it is compatible with the existing PyTorch ecosystem. DeepSpeed is designed to be easy to use, and it provides a number of features that make it easy to train large-scale models.\n\nDeepSpeed is a machine learning framework that enables training of large-scale models on commodity hardware. It is designed to be a drop-in replacement for PyTorch, and",
        "meta-llama/Llama-2-7b-hf": "DeepSpeed is a machine learning framework for deep learning. It is designed to be fast and efficient, while also being easy to use. DeepSpeed is based on the TensorFlow framework, and it uses the TensorFlow library to perform computations.\nDeepSpeed is a deep learning framework that is designed to be fast and efficient. It is based on the TensorFlow library and uses the TensorFlow library to perform computations. DeepSpeed is designed to be easy to use and to provide a high level of flex",
        "mistralai/Mistral-7B-v0.1": "DeepSpeed is a machine learning framework that accelerates training of large models on a single machine or distributed systems. It is designed to be compatible with PyTorch and TensorFlow, and can be used to train models on a single machine or on a distributed system.\n\nDeepSpeed is a machine learning framework that accelerates training of large models on a single machine or distributed systems. It is designed to be compatible with PyTorch and TensorFlow, and can be used to train models on a single machine or on a distributed system",
        "mistralai/Mixtral-8x7B-v0.1": "DeepSpeed is a machine learning framework that enables training of large models on a single machine with a single GPU. It is designed to be easy to use and efficient, and it can be used to train models on a variety of tasks.\n\n## Introduction\n\nDeepSpeed is a machine learning framework that enables training of large models on a single machine with a single GPU. It is designed to be easy to use and efficient, and it can be used to train models on a variety of tasks.\n\n## What is DeepSpeed",
        "Qwen/Qwen2-7B": "DeepSpeed is a machine learning framework that provides a unified interface for training deep learning models. It is designed to be easy to use and to provide high performance on a variety of hardware platforms. DeepSpeed is built on top of PyTorch and TensorFlow, and it supports a wide range of models architectures, including transformer models, convolutional neural networks, and recurrent neural networks.\nDeepSpeed is designed to be easy to use, and it provides a unified interface for training deep learning models. It supports a wide range of model architectures, including",
    }
else:
    # Functional testing only on G2 onwards
    MODEL_OUTPUTS = {}


def _test_text_generation(
    model_name: str,
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

        assert results["output"][0][0] == MODEL_OUTPUTS[model_name]


@pytest.mark.parametrize("model_name", MODEL_OUTPUTS.keys())
def test_text_generation_bf16_1x(model_name: str, token: str):
    _test_text_generation(model_name, token)
