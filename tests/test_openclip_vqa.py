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
            ("laion/CLIP-ViT-g-14-laion2B-s12B-b42K", 1472),
            ("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", 1816),
        ],
    }
else:
    # Gaudi1 CI baselines
    MODELS_TO_TEST = {
        "bf16": [
            ("laion/CLIP-ViT-g-14-laion2B-s12B-b42K", 550),
            ("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", 1200),
        ],
    }


def _install_requirements():
    PATH_TO_EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples"

    cmd_line = (
        f"pip install -r {PATH_TO_EXAMPLE_DIR / 'visual-question-answering' / 'requirements.txt'}".split()
    )
    p = subprocess.Popen(cmd_line)
    return_code = p.wait()
    assert return_code == 0

    cmd_line = (
        f"pip install --no-deps -r {PATH_TO_EXAMPLE_DIR / 'visual-question-answering' / 'openclip_requirements.txt'}".split()
    )
    p = subprocess.Popen(cmd_line)
    return_code = p.wait()
    assert return_code == 0


def _test_openclip_vqa(model_name: str, baseline: float):
    _install_requirements()
    command = ["python3"]
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"
    env_variables = os.environ.copy()

    command += [
        f"{path_to_example_dir / 'visual-question-answering' / 'run_openclip_vqa.py'}",
        f"--model_name_or_path {model_name}",
        "--bf16",
        "--use_hpu_graphs",
    ]

    with TemporaryDirectory() as tmp_dir:
        command.append(f"--output_dir {tmp_dir}")
        print(f"\n\nCommand to test: {' '.join(command)}\n")

        pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
        command = [x for y in command for x in re.split(pattern, y) if x]

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
def test_openclip_vqa_bf16(model_name: str, baseline: float):
    _test_openclip_vqa(model_name, baseline)
