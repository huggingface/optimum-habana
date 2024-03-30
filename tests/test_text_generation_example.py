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
            ("bigscience/bloomz-7b1", 130.0472971205316),
            ("gpt2-xl", 281.8734689674413),
            ("EleutherAI/gpt-j-6b", 160.5823842101192),
            ("EleutherAI/gpt-neox-20b", 50.67672679310354),
            ("meta-llama/Llama-2-7b-hf", 141.25776956002076),
            ("tiiuae/falcon-40b", 25.202450111088346),
            ("bigcode/starcoder", 65.58632640700114),
            ("Salesforce/codegen2-1B", 446.4029486883532),
            ("mosaicml/mpt-30b", 36.06464336116623),
            ("mistralai/Mistral-7B-v0.1", 130.2172236767782),
            ("mistralai/Mixtral-8x7B-v0.1", 23.7931001677926),
            ("microsoft/phi-2", 224.72307766211117),
        ],
        "fp8": [
            ("tiiuae/falcon-180B", 52.85086442722326),
        ],
        "deepspeed": [
            ("bigscience/bloomz", 36.77314954096159),
            ("meta-llama/Llama-2-70b-hf", 64.10514998902435),
            ("facebook/opt-66b", 28.48069266504111),
        ],
        "torch_compile": [
            ("meta-llama/Llama-2-7b-hf", 102.27823420713148),
        ],
        "torch_compile_distributed": [
            ("meta-llama/Llama-2-7b-hf", 39.72973199515235),
        ],
    }
else:
    # Gaudi1 CI baselines
    MODELS_TO_TEST = {
        "bf16": [
            ("bigscience/bloomz-7b1", 41.7555095197846),
            ("gpt2-xl", 142.11481820425706),
            # TODO: fix OPT 6.7B
            # ("facebook/opt-6.7b", 0.0),
            ("EleutherAI/gpt-j-6b", 50.79545107991805),
            ("meta-llama/Llama-2-7b-hf", 44.39616259946937),
            ("tiiuae/falcon-7b", 44.82870145718665),
            ("bigcode/starcoder", 15.945023767901013),
            ("Salesforce/codegen2-1B", 155.32071248826423),
            ("mosaicml/mpt-7b", 45.45168927038262),
            ("mistralai/Mistral-7B-v0.1", 41.21906841459711),
            ("microsoft/phi-2", 92.53083167241344),
        ],
        "fp8": [],
        "deepspeed": [
            ("bigscience/bloomz-7b1", 31.994268212011505),
        ],
        "torch_compile": [],
        "torch_compile_distributed": [],
    }


def _test_text_generation(
    model_name: str,
    baseline: float,
    token: str,
    deepspeed: bool = False,
    world_size: int = 8,
    torch_compile: bool = False,
    fp8: bool = False,
):
    command = ["python3"]
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"
    env_variables = os.environ.copy()

    if deepspeed:
        command += [
            f"{path_to_example_dir / 'gaudi_spawn.py'}",
            "--use_deepspeed",
            f"--world_size {world_size}",
        ]

    command += [
        f"{path_to_example_dir / 'text-generation' / 'run_generation.py'}",
        f"--model_name_or_path {model_name}",
        "--batch_size 1",
        "--use_kv_cache",
        "--max_new_tokens 100",
    ]

    if torch_compile:
        command += [
            "--attn_softmax_bf16",
            "--reuse_cache",
            "--trim_logits",
            "--torch_compile",
        ]
        env_variables["PT_ENABLE_INT64_SUPPORT"] = "1"
        env_variables["PT_HPU_LAZY_MODE"] = "0"
    else:
        command += [
            "--use_hpu_graphs",
        ]

    if not deepspeed:
        command.append("--bf16")

    if fp8:
        command += [
            "--reuse_cache",
            "--trim_logits",
        ]

    with TemporaryDirectory() as tmp_dir:
        command.append(f"--output_dir {tmp_dir}")
        print(f"\n\nCommand to test: {' '.join(command)}\n")

        command.append(f"--token {token.value}")

        pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
        command = [x for y in command for x in re.split(pattern, y) if x]

        if fp8:
            env_variables["QUANT_CONFIG"] = os.path.join(
                path_to_example_dir, "text-generation/quantization_config/maxabs_measure_include_outputs.json"
            )
            subprocess.run(command, env=env_variables)
            env_variables["QUANT_CONFIG"] = os.path.join(
                path_to_example_dir, "text-generation/quantization_config/maxabs_quant.json"
            )
            command.insert(-2, "--fp8")

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


@pytest.mark.parametrize("model_name, baseline", MODELS_TO_TEST["bf16"])
def test_text_generation_bf16(model_name: str, baseline: float, token: str):
    _test_text_generation(model_name, baseline, token)


@pytest.mark.parametrize("model_name, baseline", MODELS_TO_TEST["fp8"])
def test_text_generation_fp8(model_name: str, baseline: float, token: str):
    deepspeed = True if "falcon-180B" in model_name else False
    world_size = 8 if "falcon-180B" in model_name else None
    _test_text_generation(model_name, baseline, token, deepspeed=deepspeed, world_size=world_size, fp8=True)


@pytest.mark.parametrize("model_name, baseline", MODELS_TO_TEST["deepspeed"])
def test_text_generation_deepspeed(model_name: str, baseline: float, token: str):
    world_size = 2 if "opt-66b" in model_name else 8
    _test_text_generation(model_name, baseline, token, deepspeed=True, world_size=world_size)


@pytest.mark.parametrize("model_name, baseline", MODELS_TO_TEST["torch_compile"])
def test_text_generation_torch_compile(model_name: str, baseline: float, token: str):
    _test_text_generation(model_name, baseline, token, torch_compile=True)


@pytest.mark.parametrize("model_name, baseline", MODELS_TO_TEST["torch_compile_distributed"])
def test_text_generation_torch_compile_distributed(model_name: str, baseline: float, token: str):
    world_size = 8
    _test_text_generation(model_name, baseline, token, deepspeed=True, world_size=world_size, torch_compile=True)
