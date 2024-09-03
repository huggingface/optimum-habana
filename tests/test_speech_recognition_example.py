import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


MODELS_TO_TEST = {
    "bf16": [
        ("openai/whisper-small", 32, 0.1), # TODO check baseline
    ],
}


def _test_speech_recognition(
    model_name: str,
    baseline: float,
    batch_size: int = 1,
):
    command = ["python3"]
    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"
    env_variables = os.environ.copy()

    command += [
        f"{path_to_example_dir / 'speech-recognition' / 'run_speech_recognition_seq2seq.py'}",
        f"--model_name_or_path {model_name}",
        "--dataset_name mozilla-foundation/common_voice_11_0",
        '--dataset_config_name hi',
        '--language hindi',
        '--task transcribe',
        '--eval_split_name test',
        '--gaudi_config_name Habana/whisper',
        # f'--output_dir="./results/whisper-small-clean"',
        f'--per_device_eval_batch_size {batch_size}',
        '--generation_max_length 225',
        '--preprocessing_num_workers 1',
        '--max_duration_in_seconds 30',
        '--text_column_name sentence',
        '--freeze_feature_encoder False',
        '--bf16',
        '--overwrite_output_dir',
        '--do_eval',
        '--predict_with_generate',
        '--use_habana',
        '--use_hpu_graphs_for_inference',
        '--label_features_max_length 128',
        '--dataloader_num_workers 8'
    ]

    with TemporaryDirectory() as tmp_dir:
        command.append(f"--output_dir {tmp_dir}")
        print(f"\n\nCommand to test: {' '.join(command)}\n")

        pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
        command = [x for y in command for x in re.split(pattern, y) if x]

        proc = subprocess.run(command, env=env_variables)

        # Ensure the run finished without any issue
        assert proc.returncode == 0

        with open(Path(tmp_dir) / "all_results.json") as fp:
            results = json.load(fp)

        # Ensure performance requirements (throughput) are met
        assert results["eval_samples_per_second"] >= 2 * baseline

@pytest.mark.parametrize("model_name, batch_size, baseline", MODELS_TO_TEST["bf16"])
def test_speech_recognition_bf16(model_name: str, baseline: float, batch_size: int):
    _test_speech_recognition(model_name, baseline, batch_size)
