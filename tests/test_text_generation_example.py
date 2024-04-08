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
            ("bigscience/bloomz-7b1", 130.10463607610703),
            ("gpt2-xl", 293.2967921508155),
            ("EleutherAI/gpt-j-6b", 157.39646612198123),
            ("EleutherAI/gpt-neox-20b", 49.65827341338015),
            ("meta-llama/Llama-2-7b-hf", 142.00624811267403),
            ("tiiuae/falcon-40b", 25.065388035178792),
            ("bigcode/starcoder", 65.50236665863024),
            ("Salesforce/codegen2-1B", 456.7740998156863),
            ("mosaicml/mpt-30b", 35.64501131267502),
            ("mistralai/Mistral-7B-v0.1", 125.26115369093216),
            ("mistralai/Mixtral-8x7B-v0.1", 23.78652574031883),
        ],
        "fp8": [
            ("tiiuae/falcon-180B", 47.67900945905787),
        ],
        "deepspeed": [
            ("bigscience/bloomz", 36.34664210641816),
            ("meta-llama/Llama-2-70b-hf", 61.973950428647164),
            ("facebook/opt-66b", 28.16154122335556),
        ],
        "torch_compile": [
            ("meta-llama/Llama-2-7b-hf", 12.468247401430999),
        ],
        "torch_compile_distributed": [
            ("meta-llama/Llama-2-7b-hf", 20.178927030275947),
        ],
        "pt2e_quant": [
            ("meta-llama/Llama-2-7b-hf", 12.468247401430999),
        ],
    }
else:
    # Gaudi1 CI baselines
    MODELS_TO_TEST = {
        "bf16": [
            ("bigscience/bloomz-7b1", 41.51855420676164),
            ("gpt2-xl", 137.159223188195),
            # TODO: fix OPT 6.7B
            # ("facebook/opt-6.7b", 0.0),
            ("EleutherAI/gpt-j-6b", 50.66146537939035),
            ("meta-llama/Llama-2-7b-hf", 44.29688546702468),
            ("tiiuae/falcon-7b", 44.217408724737744),
            ("bigcode/starcoder", 15.948143541091655),
            ("Salesforce/codegen2-1B", 153.79670508220687),
            ("mosaicml/mpt-7b", 44.80241777760578),
            ("mistralai/Mistral-7B-v0.1", 40.00435417311187),
        ],
        "deepspeed": [
            ("bigscience/bloomz-7b1", 31.044523676681507),
        ],
        "torch_compile": [],
        "torch_compile_distributed": [],
        "pt2e_quant": [],
    }


def _test_text_generation(
    model_name: str,
    baseline: float,
    token: str,
    deepspeed: bool = False,
    world_size: int = 8,
    torch_compile: bool = False,
    fp8: bool = False,
    pt2e_quant: bool = False,
):
    command = ["python3"]
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"

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
    elif pt2e_quant:
        command += [
            "--attn_softmax_bf16",
            "--reuse_cache",
            "--trim_logits",
            "--pt2e_quant",
        ]
    else:
        command += [
            "--use_hpu_graphs",
        ]

    if not deepspeed:
        command.append("--bf16")

    if fp8:
        command += [
            "--fp8",
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
            os.environ["QUANT_CONFIG"] = os.path.join(
                path_to_example_dir, "text-generation/quantization_config/maxabs_measure_include_outputs.json"
            )
            subprocess.run(command)
            os.environ["QUANT_CONFIG"] = os.path.join(
                path_to_example_dir, "text-generation/quantization_config/maxabs_quant.json"
            )

        proc = subprocess.run(command)

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
    os.environ["PT_ENABLE_INT64_SUPPORT"] = "1"
    os.environ["PT_HPU_LAZY_MODE"] = "0"
    os.environ["WORLD_SIZE"] = "0"
    _test_text_generation(model_name, baseline, token, torch_compile=True)


@pytest.mark.parametrize("model_name, baseline", MODELS_TO_TEST["torch_compile_distributed"])
def test_text_generation_torch_compile_distributed(model_name: str, baseline: float, token: str):
    world_size = 8
    os.environ["PT_ENABLE_INT64_SUPPORT"] = "1"
    os.environ["PT_HPU_LAZY_MODE"] = "0"
    _test_text_generation(model_name, baseline, token, deepspeed=True, world_size=world_size, torch_compile=True)


@pytest.mark.parametrize("model_name, baseline", MODELS_TO_TEST["pt2e_quant"])
def test_text_generation_pt2e_quant(model_name: str, baseline: float, token: str):
    os.environ["PT_ENABLE_INT64_SUPPORT"] = "1"
    os.environ["PT_HPU_LAZY_MODE"] = "0"
    os.environ["WORLD_SIZE"] = "0"
    _test_text_generation(model_name, baseline, token, pt2e_quant=True)
