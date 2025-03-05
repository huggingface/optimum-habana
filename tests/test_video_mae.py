# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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


import time
from unittest import TestCase

import habana_frameworks.torch as ht
import numpy as np
import pytest
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from .utils import OH_DEVICE_CONTEXT


MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"


@pytest.fixture(scope="module")
def frame_buf():
    return list(np.random.default_rng(123).random((16, 3, 224, 224)))


@pytest.fixture(scope="module")
def processor():
    return VideoMAEImageProcessor.from_pretrained(MODEL_NAME)


@pytest.fixture(autouse=True, scope="class")
def inputs(request, frame_buf, processor):
    request.cls.inputs = processor(frame_buf, return_tensors="pt")
    request.cls.inputs_hpu = request.cls.inputs.copy().to("hpu")


@pytest.fixture(autouse=True, scope="class")
def outputs_cpu(request):
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME)
    model.eval()

    with torch.no_grad():
        output = model(**request.cls.inputs)
    request.cls.outputs_cpu = output


@pytest.fixture(autouse=True, scope="class")
def model_hpu(request):
    request.cls.model_hpu = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME).to("hpu")
    request.cls.model_hpu_graph = ht.hpu.wrap_in_hpu_graph(request.cls.model_hpu)


@pytest.fixture(autouse=True, scope="class")
def outputs_hpu_default(request):
    with torch.no_grad():
        output = request.cls.model_hpu(**request.cls.inputs_hpu)
    request.cls.outputs_hpu_default = output


class GaudiVideoMAETester(TestCase):
    """
    Tests for VideoMAE on Gaudi
    """

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

    def test_inference_default(self):
        """
        Tests for equivalent cpu and hpu runs
        """
        self.assertTrue(
            torch.equal(
                self.outputs_cpu.logits.topk(10).indices,
                self.outputs_hpu_default.logits.cpu().topk(10).indices,
            )
        )
        self.assertTrue(torch.allclose(self.outputs_cpu.logits, self.outputs_hpu_default.logits, atol=5e-3))

    def test_inference_bf16(self):
        """
        Tests for similar bf16 to regular inference
        """
        with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            outputs = self.model_hpu(**self.inputs_hpu)
        self.assertTrue(
            torch.equal(
                self.outputs_hpu_default.logits.topk(5).indices,
                outputs.logits.topk(5).indices,
            )
        )

    def test_inference_graph_bf16(self):
        """
        Test for similar bf16 to regular inference in graph mode
        """
        with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            outputs = self.model_hpu_graph(**self.inputs_hpu)
        self.assertTrue(
            torch.equal(
                self.outputs_hpu_default.logits.topk(5).indices,
                outputs.logits.topk(5).indices,
            )
        )

    def test_latency_graph_bf16(self):
        """
        Tests for performance degredations by up to 5%
        """
        warm_up_iters = 5
        test_iters = 10
        with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            for _ in range(warm_up_iters):
                self.model_hpu_graph(**self.inputs_hpu)
        torch.hpu.synchronize()
        start_time = time.time()
        with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            for _ in range(test_iters):
                self.model_hpu_graph(**self.inputs_hpu)
                torch.hpu.synchronize()
        time_per_iter = (time.time() - start_time) * 1000 / test_iters  # Time in ms
        self.baseline.assertRef(
            compare=lambda latency, expect: latency < (1.05 * expect),
            context=[OH_DEVICE_CONTEXT],
            latency=time_per_iter,
        )
