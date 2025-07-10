# coding=utf-8
# Copyright 2025 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture
def oh_path():
    cwd = Path.cwd()
    if cwd.name.startswith("optimum-habana"):
        oh_path = cwd
    for parent in cwd.parents:
        if parent.name.startswith("optimum-habana"):
            oh_path = parent
    return oh_path.resolve()


@pytest.fixture
def profiling_dir(oh_path):
    p = oh_path / "hpu_profile"
    yield p
    if p.exists():
        shutil.rmtree(p)


@pytest.fixture
def temp_dir():
    td = TemporaryDirectory()
    yield td.name
    td.cleanup()


def install_requirements(requirements_file_path):
    print(f"Installing {requirements_file_path}")
    p = subprocess.run(f"pip install -r {requirements_file_path}", shell=True)
    assert p.returncode == 0, f"Failed to install {requirements_file_path}"


def run_command_and_check_profiler_output(command, expected_directories, expected_num_files):
    print(f"\nRunning command: {command}")
    p = subprocess.run(command, shell=True)
    rc = p.returncode
    stdout = "" if p.stdout is None else p.stdout.decode()
    stderr = "" if p.stderr is None else p.stderr.decode()
    if rc != 0:
        msg = f"Command failed with return code {rc}\nstdout: {stdout}\nstderr: {stderr}"
    assert rc == 0, msg

    for expected_dir in expected_directories:
        assert expected_dir.exists(), f"No profiling directory {expected_dir}"
        assert len(list(expected_dir.glob("*.json"))) == expected_num_files


def test_integration_train_and_eval(oh_path, profiling_dir, temp_dir):
    command = (
        f"python3 {oh_path}/examples/text-classification/run_glue.py "
        "--model_name_or_path bert-large-uncased-whole-word-masking "
        "--gaudi_config_name Habana/bert-large-uncased-whole-word-masking "
        f"--task_name mrpc --do_train --output_dir {temp_dir} "
        "--overwrite_output_dir --learning_rate 3e-05 "
        "--per_device_train_batch_size 1 --per_device_eval_batch_size 1 "
        "--num_train_epochs 1 --use_habana --throughput_warmup_steps 1 "
        "--save_strategy no --use_lazy_mode --do_eval --max_seq_length 128 "
        "--use_hpu_graphs_for_inference --sdp_on_bf16 --profiling_steps 1 "
        "--profiling_warmup_steps 1 --profiling_steps_eval 1 "
        "--profiling_warmup_steps_eval 1"
    )
    install_requirements(f"{oh_path}/examples/text-classification/requirements.txt")
    expected_dirs = [
        profiling_dir / "train",
        profiling_dir / "evaluation",
    ]
    run_command_and_check_profiler_output(command, expected_dirs, expected_num_files=1)


def test_integration_text_generation(oh_path, profiling_dir, temp_dir):
    command = (
        f"python3 {oh_path}/examples/text-generation/run_generation.py "
        "--model_name_or_path bigscience/bloomz-7b1 --batch_size 1 --use_kv_cache "
        f"--max_new_tokens 100 --use_hpu_graphs --bf16 --output_dir {temp_dir} "
        "--profiling_steps 1 --profiling_warmup_steps 1"
    )
    install_requirements(f"{oh_path}/examples/text-generation/requirements.txt")
    expected_dirs = [profiling_dir / "generate"]
    run_command_and_check_profiler_output(command, expected_dirs, expected_num_files=1)


@pytest.mark.x8
def test_integration_stable_diffusion(oh_path, profiling_dir, temp_dir):
    world_size = 8
    command = (
        f"python {oh_path}/examples/gaudi_spawn.py --world_size {world_size} "
        f"{oh_path}/examples/stable-diffusion/text_to_image_generation.py "
        "--model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 "
        '--prompts "Sailing ship painting by Van Gogh" --num_images_per_prompt 1 '
        f"--batch_size 1 --image_save_dir {temp_dir} --scheduler euler_discrete "
        "--use_habana --use_hpu_graphs --gaudi_config Habana/stable-diffusion --bf16 "
        "--num_inference_steps 10 --optimize --sdp_on_bf16 "
        "--profiling_steps 1 --profiling_warmup_steps 1 --distributed"
    )
    install_requirements(f"{oh_path}/examples/stable-diffusion/requirements.txt")
    expected_dirs = [profiling_dir / "stable_diffusion"]
    run_command_and_check_profiler_output(command, expected_dirs, expected_num_files=world_size)
