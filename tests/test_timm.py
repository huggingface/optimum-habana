import os
import re
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("test_timm_type", ["inference", "train_hpu_lazy", "train_hpu_graph"])
def test_timm(test_timm_type: str) -> None:
    task = "pytorch-image-models"

    os.environ["PT_HPU_LAZY_MODE"] = "1"
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"
    env_variables = os.environ.copy()

    # Install question-answering example requirements
    cmd_line = f"pip install -r {path_to_example_dir / task / 'requirements.txt'}".split()

    p = subprocess.Popen(cmd_line)
    return_code = p.wait()
    assert return_code == 0

    command = ["python3"]

    if test_timm_type == "inference":
        command += [
            f"{path_to_example_dir / task / 'inference.py'}",
            "--data-dir='./'",
            "--dataset hfds/johnowhitaker/imagenette2-320",
            "--device hpu",
            "--model resnet50.a1_in1k",
            "--split train",
            "--graph_mode",
        ]
    elif test_timm_type == "train_hpu_lazy":
        command += [
            f"{path_to_example_dir / task / 'train_hpu_lazy.py'}",
            "--data-dir='./'",
            "--dataset hfds/johnowhitaker/imagenette2-320",
            "--device hpu",
            "--model resnet50.a1_in1k",
            "--train-split train",
            "--val-split train",
            "--dataset-download",
            "--epochs 3",
        ]
    else:
        command += [
            f"{path_to_example_dir / task / 'train_hpu_graph.py'}",
            "--data-dir='./'",
            "--dataset hfds/johnowhitaker/imagenette2-320",
            "--device hpu",
            "--model resnet50.a1_in1k",
            "--train-split train",
            "--val-split train",
            "--dataset-download",
            "--epochs 3",
        ]

    print(f" {test_timm_type} for timm model is under testing ... ")

    print(f"\n\nCommand to test: {' '.join(command)}\n")

    pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
    command = [x for y in command for x in re.split(pattern, y) if x]

    proc = subprocess.run(command, env=env_variables)

    # Ensure the run finished without any issue
    # Use try-except to avoid logging the token if used
    try:
        assert proc.returncode == 0
    except AssertionError as e:
        e.args = (f"The following command failed:\n{' '.join(command[:-2])}",)
        raise
