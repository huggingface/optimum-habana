import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from optimum.habana.utils import set_seed


if os.environ.get("GAUDI2_CI", "0") == "1":
    MODELS_TO_TEST = [
        (
            "bigcode/starcoder",
            "DeepSpeed is a machine learning framework for fast and accurate deep learning.\n\nDeepSpeed is a deep learning optimization library that makes distributed training easy, efficient, and effective.\n\nDeepSpeed is the engine that powers [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples).\n\nDeepSpeed is built on PyTorch and TorchScript, and can run models on CPUs, GPUs",
        ),
        (
            "bigcode/starcoder2-15b",
            "DeepSpeed is a machine learning framework that optimizes transformer models using model parallelism and low-precision arithmetic. It is built to be easy to use and efficient to run.\n\nDeepSpeed is a research framework that provides both researchers and engineers with the ability to explore new ideas in distributed training.\n\nDeepSpeed is a community-driven project. We welcome your contributions and feedback.\n\n## Installation\n\nDeepSpeed can be installed from PyPI:\n\n```\npip install deepspeed\n```\n\n",
        ),
        (
            "google/gemma-2b",
            "DeepSpeed is a machine learning framework that uses a novel approach to accelerate training of deep neural networks. It is based on a novel technique called <em>gradient-based quantization</em>, which is a technique that quantizes the weights and activations of a neural network in a way that preserves the gradient flow. This allows for faster training and better accuracy.\n\nDeepSpeed is a framework that uses a novel approach to accelerate training of deep neural networks. It is based on a novel technique called <em>gradient-based quantization</em>, which is a",
        ),
        (
            "meta-llama/Llama-2-7b-hf",
            "DeepSpeed is a machine learning framework for deep learning. It is designed to be fast and efficient, while also being easy to use. DeepSpeed is based on the TensorFlow framework, and it uses the TensorFlow library to perform computations.\nDeepSpeed is a deep learning framework that is designed to be fast and efficient. It is based on the TensorFlow library and uses the TensorFlow library to perform computations. DeepSpeed is designed to be easy to use and to provide a high level of flex",
        ),
        (
            "mistralai/Mistral-7B-Instruct-v0.2",
            "DeepSpeed is a machine learning framework for distributed training and inference, developed by Meta. It is designed to be efficient on large-scale systems, and supports both PyTorch and TensorFlow models. DeepSpeed includes several components:\n\n* **DeepSpeed Torch**: A PyTorch backend for distributed training and inference, which includes optimizations for model parallelism, data parallelism, and pipeline parallelism.\n* **DeepSpeed TensorFlow**: A TensorFlow backend for distributed training and inference, which includes",
        ),
        (
            "mistralai/Mixtral-8x7B-v0.1",
            "DeepSpeed is a machine learning framework that enables training of large models on a single machine with a single GPU. It is designed to be easy to use and efficient, and it can be used to train models on a variety of tasks.\n\n## Introduction\n\nDeepSpeed is a machine learning framework that enables training of large models on a single machine with a single GPU. It is designed to be easy to use and efficient, and it can be used to train models on a variety of tasks.\n\n## What is DeepSpeed",
        ),
        (
            "Qwen/Qwen2-7B",
            "DeepSpeed is a machine learning framework that provides a unified interface for training deep learning models. It is designed to be easy to use and to provide high performance on a variety of hardware platforms. DeepSpeed is built on top of PyTorch and TensorFlow, and it supports a wide range of models architectures, including transformer models, convolutional neural networks, and recurrent neural networks.\nDeepSpeed is designed to be easy to use, and it provides a unified interface for training deep learning models. It supports a wide range of model architectures, including",
        ),
    ]
else:
    # Functional testing only on G2 onwards
    MODELS_TO_TEST = []


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
