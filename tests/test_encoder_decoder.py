import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import pytest

from .test_examples import ACCURACY_PERF_FACTOR, TIME_PERF_FACTOR


if os.environ.get("GAUDI2_CI", "0") == "1":
    # Gaudi2 CI baselines
    MODELS_TO_TEST = {
        "summarization": {
            "bf16": [
                ("facebook/bart-large-cnn", "Habana/bart", 3.9, 28.9801, 2, 2),
                ("t5-3b", "Habana/t5", 2.955, 21.8877, 2, 1),
            ],
        },
        "translation": {
            "bf16": [
                ("Babelscape/mrebel-large", "Habana/t5", 1.323, 0.1618, 2, 1),
                ("Helsinki-NLP/opus-mt-zh-en", "Habana/t5", 2.815, 0.8132, 2, 1),
                ("facebook/nllb-200-distilled-600M", "Habana/t5", 1.401, 1.2599, 2, 1),
                ("t5-small", "Habana/t5", 14.482, 11.7277, 2, 1),
            ],
        },
    }
else:
    # Gaudi1 CI baselines
    MODELS_TO_TEST = {
        "summarization": {
            "bf16": [
                ("facebook/bart-large-cnn", "Habana/bart", 2.304, 29.174, 2, 2),
                ("t5-3b", "Habana/t5", 1.005, 21.7286, 2, 1),
            ],
        },
        "translation": {
            "bf16": [
                ("Babelscape/mrebel-large", "Habana/t5", 0.995, 0.1784, 2, 1),
                ("Helsinki-NLP/opus-mt-zh-en", "Habana/t5", 2.409, 0.7995, 2, 1),
                ("facebook/nllb-200-distilled-600M", "Habana/t5", 0.998, 1.2457, 2, 1),
                ("t5-small", "Habana/t5", 9.188, 11.6126, 2, 1),
            ],
        },
    }


class TestEncoderDecoderModels:
    PATH_TO_EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples"

    def _install_requirements(self, task: str):
        cmd_line = f"pip install -r {self.PATH_TO_EXAMPLE_DIR / task / 'requirements.txt'}".split()
        p = subprocess.Popen(cmd_line)
        return_code = p.wait()
        assert return_code == 0

    def _build_command(
        self,
        task: str,
        deepspeed: bool = False,
        world_size: int = 8,
        command_args: List[str] = None,
    ):
        command = ["python3"]

        if deepspeed:
            command += [
                f"{self.PATH_TO_EXAMPLE_DIR / 'gaudi_spawn.py'}",
                "--use_deepspeed",
                f"--world_size {world_size}",
            ]

        if command_args is not None:
            command += command_args

        if not deepspeed:
            command.append("--bf16")

        return command

    def _run_test(
        self,
        command: List[str],
        task: str,
        baseline: float,
        baseline_acc: float,
    ):
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

        # Ensure performance requirements (throughput) are met
        assert results["predict_samples_per_second"] >= (2 - TIME_PERF_FACTOR) * baseline

        if task == "summarization":
            accuracy_metric = "predict_rougeLsum"
        elif task == "translation":
            accuracy_metric = "predict_bleu"
        assert results[accuracy_metric] >= ACCURACY_PERF_FACTOR * baseline_acc

    def _test_text_summarization(
        self,
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
        task = "summarization"

        # Install summarization example requirements
        self._install_requirements(task)

        command_args = [
            str(self.PATH_TO_EXAMPLE_DIR / task / f"run_{task}.py"),
            f"--model_name_or_path {model_name}",
            "--do_predict",
            "--predict_with_generate",
            "--dataset_name cnn_dailymail",
            "--dataset_config 3.0.0",
            "--use_habana",
            f"--per_device_eval_batch_size {batch_size}",
            f"--gaudi_config_name {gaudi_config}",
            f"--num_beams {num_beams}",
            "--ignore_pad_token_for_loss False",
            "--pad_to_max_length",
            "--use_hpu_graphs_for_inference",
            "--use_lazy_mode",
            "--max_predict_samples 200",
        ]

        command = self._build_command(
            task=task,
            deepspeed=deepspeed,
            world_size=world_size,
            command_args=command_args,
        )

        if not deepspeed and model_name == "t5-3b":
            command.append("--bf16_full_eval")

        self._run_test(command, task, baseline, baseline_acc)

    def _test_text_translation(
        self,
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
        task = "translation"

        # Install summarization example requirements
        self._install_requirements(task)

        command_args = [
            str(self.PATH_TO_EXAMPLE_DIR / task / f"run_{task}.py"),
            f"--model_name_or_path {model_name}",
            "--do_predict",
            "--source_lang en",
            "--target_lang ro",
            '--source_prefix "translate English to Romanian: "' "--dataset_name wmt16",
            "--dataset_config_name ro-en",
            f"--per_device_eval_batch_size {batch_size}",
            f"--generation_num_beams {num_beams}",
            "--predict_with_generate",
            "--use_habana",
            "--use_lazy_mode",
            "--use_hpu_graphs_for_inference",
            f"--gaudi_config_name {gaudi_config}",
            "--ignore_pad_token_for_loss False",
            "--pad_to_max_length",
            "--max_predict_samples 200",
        ]

        if "opus-mt-zh-en" in model_name:
            command_args.append("--max_source_length 512")

        command = self._build_command(
            task=task,
            deepspeed=deepspeed,
            world_size=world_size,
            command_args=command_args,
        )

        self._run_test(command, task, baseline, baseline_acc)

    @pytest.mark.parametrize(
        "model_name, gaudi_config, baseline, baseline_acc, batch_size, num_beams",
        MODELS_TO_TEST["summarization"]["bf16"],
    )
    def test_text_summarization_bf16(
        self,
        model_name: str,
        gaudi_config: str,
        baseline: float,
        baseline_acc: float,
        batch_size: int,
        num_beams: int,
        token: str,
    ):
        self._test_text_summarization(model_name, gaudi_config, baseline, baseline_acc, batch_size, num_beams, token)

    @pytest.mark.parametrize(
        "model_name, gaudi_config, baseline, baseline_acc, batch_size, num_beams",
        MODELS_TO_TEST["translation"]["bf16"],
    )
    def test_text_translation_bf16(
        self,
        model_name: str,
        gaudi_config: str,
        baseline: float,
        baseline_acc: float,
        batch_size: int,
        num_beams: int,
        token: str,
    ):
        self._test_text_translation(model_name, gaudi_config, baseline, baseline_acc, batch_size, num_beams, token)
