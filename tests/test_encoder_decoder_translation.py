import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from .test_examples import ACCURACY_PERF_FACTOR, TIME_PERF_FACTOR


if os.environ.get("GAUDI2_CI", "0") == "1":
    # Gaudi2 CI baselines
    MODELS_TO_TEST = {
        "bf16": [
            ("facebook/mbart-large-50-many-to-many-mmt", "Habana/t5", 1.67, 1.05, 2, 1),
            # ("Babelscape/mrebel-large", "Habana/t5", 4.691, 26.0688, 2, 1),
            # ("Helsinki-NLP/opus-mt-zh-en", "Habana/t5", 4.691, 26.0688, 2, 1),
            # ("Helsinki-NLP/opus-mt-en-zh", "Habana/t5", 4.691, 26.0688, 2, 1),
            # ("Helsinki-NLP/opus-mt-en-ar", "Habana/t5", 4.691, 26.0688, 2, 1),
            # ("facebook/nllb-200-3.3B", "Habana/t5", 4.691, 26.0688, 2, 1),
            # ("facebook/nllb-200-distilled-600M", "Habana/t5", 4.691, 26.0688, 2, 1),
        ],
    }


def _test_text_translation(
    model_name: str,
    gaudi_config: str,
    baseline: float,
    baseline_acc: float,
    batch_size: int,
    num_beams: int,
    token: str,
    deepspeed: bool = False,
    world_size: int = 8,
):
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"

    # Install translation example requirements
    cmd_line = f"pip install -r {path_to_example_dir / 'translation' / 'requirements.txt'}".split()
    p = subprocess.Popen(cmd_line)
    return_code = p.wait()
    assert return_code == 0

    command = ["python3"]

    if deepspeed:
        command += [
            f"{path_to_example_dir / 'gaudi_spawn.py'}",
            "--use_deepspeed",
            f"--world_size {world_size}",
        ]

    command += [
        f"{path_to_example_dir / 'translation' / 'run_translation.py'}",
        f"--model_name_or_path {model_name}",
        "--do_predict",
        "--source_lang en",
        "--target_lang ro",
        '--source_prefix "translate English to Romanian: "' "--dataset_name wmt16",
        "--dataset_config_name ro-en",
        f"--per_device_eval_batch_size {batch_size}",
        "--predict_with_generate",
        "--use_habana",
        "--use_lazy_mode",
        "--use_hpu_graphs_for_inference",
        f"--gaudi_config_name {gaudi_config}",
        "--ignore_pad_token_for_loss False",
        "--pad_to_max_length",
        "--max_predict_samples 200",
    ]

    if not deepspeed:
        command.append("--bf16")

    with TemporaryDirectory() as tmp_dir:
        command.append(f"--output_dir {tmp_dir}")
        print(f"\n\nCommand to test: {' '.join(command)}\n")

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

        with open(Path(tmp_dir) / "predict_results.json") as fp:
            results = json.load(fp)

        # Ensure performance and accuracy requirements are met
        assert results["predict_samples_per_second"] >= (2 - TIME_PERF_FACTOR) * baseline
        assert results["predict_bleu"] >= ACCURACY_PERF_FACTOR * baseline_acc


@pytest.mark.parametrize(
    "model_name, gaudi_config, baseline, baseline_acc, batch_size, num_beams", MODELS_TO_TEST["bf16"]
)
def test_text_translation_bf16(
    model_name: str,
    gaudi_config: str,
    baseline: float,
    baseline_acc: float,
    batch_size: int,
    num_beams: int,
    token: str,
):
    _test_text_translation(model_name, gaudi_config, baseline, baseline_acc, batch_size, num_beams, token)
