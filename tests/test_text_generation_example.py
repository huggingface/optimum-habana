import json
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from .test_examples import TIME_PERF_FACTOR


MODELS_TO_TEST = {
    "bf16": [
        ("bigscience/bloomz-7b1", 41.93942748147396),
        ("gpt2-xl", 126.6292071377241),
        # TODO: fix OPT 6.7B
        # ("facebook/opt-6.7b", 0.0),
        ("EleutherAI/gpt-j-6b", 37.14562499113717),
        # TODO: GPT-NeoX doesn't fit on 1 Gaudi1 card
        # ("EleutherAI/gpt-neox-20b", 0.0),
        ("meta-llama/Llama-2-7b-hf", 43.951804139391925),
        ("tiiuae/falcon-7b", 44.288602257903726),
        ("bigcode/starcoder", 15.955986010526113),
        ("Salesforce/codegen2-1B", 109.03016111561857),
        ("mosaicml/mpt-7b", 44.888696119070424),
    ],
    "deepspeed": [
        ("bigscience/bloomz-7b1", 27.34439410425298),
    ],
}


def _test_text_generation(model_name: str, baseline: float, token: str, deepspeed: bool = False, world_size: int = 8):
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
        "--use_hpu_graphs",
        "--use_kv_cache",
        "--max_new_tokens 100",
    ]

    if not deepspeed:
        command.append("--bf16")

    with TemporaryDirectory() as tmp_dir:
        command.append(f"--output_dir {tmp_dir}")
        print(f"\n\nCommand to test: {' '.join(command)}\n")

        command.append(f"--token {token.value}")

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

        with open(Path(tmp_dir) / "results.json") as fp:
            results = json.load(fp)

        # Ensure performance requirements (throughput) are met
        assert results["throughput"] >= (2 - TIME_PERF_FACTOR) * baseline


@pytest.mark.parametrize("model_name, baseline", MODELS_TO_TEST["bf16"])
def test_text_generation_bf16(model_name: str, baseline: float, token: str):
    _test_text_generation(model_name, baseline, token)


@pytest.mark.parametrize("model_name, baseline", MODELS_TO_TEST["deepspeed"])
def test_text_generation_deepspeed(model_name: str, baseline: float, token: str):
    _test_text_generation(model_name, baseline, token, deepspeed=True)
