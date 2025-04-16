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

import os
import shutil
from unittest.mock import MagicMock

import pytest

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
from optimum.habana.utils import HabanaProfile


adapt_transformers_to_gaudi()


PROFILER_OUTPUT_DIR = "./hpu_profile"


@pytest.fixture
def patched_profiler(monkeypatch):
    p = HabanaProfile(warmup=1, active=1)
    mock_start = MagicMock()
    mock_stop = MagicMock()
    mock_step = MagicMock()
    monkeypatch.setattr(p._profiler, "start", mock_start)
    monkeypatch.setattr(p._profiler, "stop", mock_stop)
    monkeypatch.setattr(p._profiler, "step", mock_step)
    yield p


@pytest.fixture(autouse=True)
def cleanup():
    shutil.rmtree(PROFILER_OUTPUT_DIR, ignore_errors=True)
    HabanaProfile._profilers = []


def run_profiling(profiler):
    profiler.start()
    for _ in range(2):
        profiler.step()
    profiler.stop()


def test_init_profiler_with_no_steps():
    profiler = HabanaProfile()
    assert profiler._profiler is None
    assert profiler.start() is None
    assert not profiler._running
    assert profiler.step() is None
    assert profiler.stop() is None


def test_init_profiler_with_steps(patched_profiler):
    assert not patched_profiler._running
    assert patched_profiler._profiler is not None


def test_start_profiling(patched_profiler):
    patched_profiler.start()
    assert patched_profiler._running
    patched_profiler._profiler.start.assert_called_once()


def test_call_step_on_profiler(patched_profiler):
    patched_profiler.start()
    patched_profiler.step()
    assert patched_profiler._running
    patched_profiler._profiler.step.assert_called_once()


def test_stop_profiling(patched_profiler):
    patched_profiler.start()
    patched_profiler.stop()
    assert not patched_profiler._running
    patched_profiler._profiler.stop.assert_called_once()


def test_profiler_files():
    profiler = HabanaProfile(warmup=1, active=1)
    run_profiling(profiler)
    assert os.path.exists(PROFILER_OUTPUT_DIR)
    assert len(os.listdir(PROFILER_OUTPUT_DIR)) == 1


def test_profiler_with_name():
    profiler = HabanaProfile(warmup=1, active=1, name="test")
    run_profiling(profiler)
    expected_dir = os.path.join(PROFILER_OUTPUT_DIR, "test")
    assert os.path.exists(expected_dir)
    assert len(os.listdir(expected_dir)) == 1


def test_profiler_with_no_steps_doesnt_run():
    profiler = HabanaProfile()
    run_profiling(profiler)
    assert not os.path.exists(PROFILER_OUTPUT_DIR)


def test_two_profilers_can_run_sequentially():
    profiler_0 = HabanaProfile(warmup=1, active=1)
    run_profiling(profiler_0)
    profiler_1 = HabanaProfile(warmup=1, active=1)
    run_profiling(profiler_1)
    assert os.path.exists(PROFILER_OUTPUT_DIR)
    assert len(os.listdir(PROFILER_OUTPUT_DIR)) == 2


def test_cannot_start_profiler_when_another_is_running(patched_profiler):
    another_profiler = HabanaProfile(warmup=1, active=1)
    patched_profiler.start()
    with pytest.raises(RuntimeError):
        another_profiler.start()
