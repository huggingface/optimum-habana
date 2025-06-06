import json
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import pytest

from .test_examples import ACCURACY_PERF_FACTOR, TIME_PERF_FACTOR
from .utils import OH_DEVICE_CONTEXT


MODELS_TO_TEST = {
    "summarization": {
        "bf16": [
            ("facebook/bart-large-cnn", "Habana/bart", 2, 2),
            ("t5-3b", "Habana/t5", 2, 1),
        ],
    },
    "translation": {
        "bf16": [
            ("t5-small", "Habana/t5", 2, 1),
        ],
    },
}


class TestEncoderDecoderModels:
    PATH_TO_EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples"

    @pytest.fixture(autouse=True)
    def _pretest(self, baseline):
        """
        This is automatically called before each test function is executed.

        Collect custom fixtures (from conftest.py).
        """
        self.baseline = baseline

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
    ):
        with TemporaryDirectory() as tmp_dir:
            command.append(f"--output_dir {tmp_dir}")
            print(f"\n\nCommand to test: {' '.join(command)}\n")

            pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
            command = [x for y in command for x in re.split(pattern, y) if x]

            proc = subprocess.run(command)

            # Ensure the run finished without any issue
            assert proc.returncode == 0

            with open(Path(tmp_dir) / "predict_results.json") as fp:
                results = json.load(fp)

        # Ensure performance requirements (throughput) are met
        self.baseline.assertRef(
            compare=lambda actual, ref: actual >= (2 - TIME_PERF_FACTOR) * ref,
            context=[OH_DEVICE_CONTEXT],
            predict_samples_per_second=results["predict_samples_per_second"],
        )

        if task == "summarization":
            accuracy_metric = "predict_rougeLsum"
        elif task == "translation":
            accuracy_metric = "predict_bleu"
        self.baseline.assertRef(
            compare=lambda actual, ref: actual >= ACCURACY_PERF_FACTOR * ref,
            context=[OH_DEVICE_CONTEXT],
            **{accuracy_metric: results[accuracy_metric]},
        )

    def _test_text_summarization(
        self,
        model_name: str,
        gaudi_config: str,
        batch_size: int,
        num_beams: int,
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
            "--throughput_warmup_steps 3",
        ]

        command = self._build_command(
            task=task,
            deepspeed=deepspeed,
            world_size=world_size,
            command_args=command_args,
        )

        if not deepspeed and model_name == "t5-3b":
            command.append("--bf16_full_eval")

        self._run_test(command, task)

    def _test_text_translation(
        self,
        model_name: str,
        gaudi_config: str,
        batch_size: int,
        num_beams: int,
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
            '--source_prefix "translate English to Romanian: "',
            "--dataset_name wmt16",
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
            "--throughput_warmup_steps 3",
        ]

        if "opus-mt-zh-en" in model_name:
            command_args.append("--max_source_length 512")

        if "Babelscape/mrebel-large" in model_name or "nllb-200-distilled-600M" in model_name:
            command_args.append("--sdp_on_bf16")

        command = self._build_command(
            task=task,
            deepspeed=deepspeed,
            world_size=world_size,
            command_args=command_args,
        )

        self._run_test(command, task)

    @pytest.mark.parametrize(
        "model_name, gaudi_config, batch_size, num_beams",
        MODELS_TO_TEST["summarization"]["bf16"],
    )
    def test_text_summarization_bf16(
        self,
        model_name: str,
        gaudi_config: str,
        batch_size: int,
        num_beams: int,
    ):
        self._test_text_summarization(model_name, gaudi_config, batch_size, num_beams)

    @pytest.mark.parametrize(
        "model_name, gaudi_config, batch_size, num_beams",
        MODELS_TO_TEST["translation"]["bf16"],
    )
    def test_text_translation_bf16(
        self,
        model_name: str,
        gaudi_config: str,
        batch_size: int,
        num_beams: int,
    ):
        self._test_text_translation(model_name, gaudi_config, batch_size, num_beams)
