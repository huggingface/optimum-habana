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
import pytest
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

from .utils import OH_DEVICE_CONTEXT


adapt_transformers_to_gaudi()

MODEL_NAME = "microsoft/table-transformer-detection"
if OH_DEVICE_CONTEXT in ["gaudi2"]:
    LATENCY_TABLE_TRANSFORMER_BF16_GRAPH_BASELINE = 2.2
else:
    LATENCY_TABLE_TRANSFORMER_BF16_GRAPH_BASELINE = 6.6


@pytest.fixture(scope="module")
def processor():
    return AutoImageProcessor.from_pretrained(MODEL_NAME)


@pytest.fixture(autouse=True, scope="class")
def inputs(request, processor):
    file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
    image = Image.open(file_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    request.cls.processor = processor
    request.cls.inputs = inputs
    request.cls.inputs_hpu = inputs.copy().to("hpu")
    request.cls.target_sizes = torch.tensor([image.size[::-1]])


@pytest.fixture(autouse=True, scope="class")
def outputs_cpu(request):
    model = TableTransformerForObjectDetection.from_pretrained(MODEL_NAME)
    model.eval()

    with torch.no_grad():
        output = model(**request.cls.inputs)

    request.cls.outputs_cpu = output
    request.cls.results_cpu = request.cls.processor.post_process_object_detection(
        output, threshold=0.9, target_sizes=request.cls.target_sizes
    )[0]


@pytest.fixture(autouse=True, scope="class")
def model_hpu(request):
    model = TableTransformerForObjectDetection.from_pretrained(MODEL_NAME).to("hpu")
    model.eval()
    request.cls.model_hpu = model
    request.cls.model_hpu_graph = ht.hpu.wrap_in_hpu_graph(model)


@pytest.fixture(autouse=True, scope="class")
def outputs_hpu_default(request):
    with torch.no_grad():
        output = request.cls.model_hpu(**request.cls.inputs_hpu)
    request.cls.outputs_hpu_default = output
    request.cls.results_hpu_default = request.cls.processor.post_process_object_detection(
        output, threshold=0.9, target_sizes=request.cls.target_sizes
    )[0]


class GaudiTableTransformerTester(TestCase):
    """
    Tests for Table Transformer Detection on Gaudi
    """

    def test_inference_default(self):
        """
        Tests for equivalent cpu and hpu runs
        """
        print(self.results_cpu)
        print(self.results_hpu_default)
        self.assertTrue(
            all(
                torch.allclose(v_cpu, v_hpu)
                for v_cpu, v_hpu in zip(self.results_cpu.values(), self.results_hpu_default.values())
            )
        )

    def test_inference_bf16(self):
        """
        Tests for similar bf16 to regular inference
        """
        with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            output = self.model_hpu(**self.inputs_hpu)
        results = self.processor.post_process_object_detection(output, threshold=0.9, target_sizes=self.target_sizes)[
            0
        ]
        self.assertTrue(
            all(
                torch.allclose(v, v_bf16, atol=1e-5)
                for v, v_bf16 in zip(self.results_hpu_default.values(), results.values())
            )
        )

    def test_inference_graph_bf16(self):
        with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16):
            output = self.model_hpu_graph(**self.inputs_hpu)
        results = self.processor.post_process_object_detection(output, threshold=0.9, target_sizes=self.target_sizes)[
            0
        ]
        self.assertTrue(
            all(
                torch.allclose(v, v_bf16, atol=1e-5)
                for v, v_bf16 in zip(self.results_hpu_default.values(), results.values())
            )
        )

    def test_latency_graph_bf16(self):
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
        print(time_per_iter)
        self.assertLess(time_per_iter, 1.05 * LATENCY_TABLE_TRANSFORMER_BF16_GRAPH_BASELINE)
