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
import requests
import timm
import torch
from PIL import Image

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

from .utils import OH_DEVICE_CONTEXT


adapt_transformers_to_gaudi()


class GaudiFastViTTester(TestCase):
    """
    Tests for FastViT model
    """

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

    def prepare_model_and_processor(self):
        model = timm.create_model("timm/fastvit_t8.apple_in1k", pretrained=True)
        model.to("hpu")
        model = model.eval()
        data_config = timm.data.resolve_model_data_config(model)
        processor = timm.data.create_transform(**data_config, is_training=False)
        return model, processor

    def prepare_data(self):
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        image = Image.open(requests.get(url, stream=True).raw)
        return image

    def test_inference_default(self):
        model, processor = self.prepare_model_and_processor()
        image = self.prepare_data()
        inputs = processor(image).unsqueeze(0).to("hpu")
        outputs = model(inputs)
        top1_probabilities, top1_class_indices = torch.topk(outputs.softmax(dim=1) * 100, k=1)
        top1_probabilities = top1_probabilities.to("cpu").detach().numpy()
        top1_class_indices = top1_class_indices.to("cpu").numpy()
        expected_scores = np.array([21.406523])  # from CPU
        expected_class = np.array([960])
        self.assertEqual(top1_class_indices, expected_class)
        self.assertLess(np.abs(top1_probabilities - expected_scores).max(), 1)

    def test_inference_autocast(self):
        model, processor = self.prepare_model_and_processor()
        image = self.prepare_data()
        inputs = processor(image).unsqueeze(0).to("hpu")

        with torch.autocast(device_type="hpu", dtype=torch.bfloat16):  # Autocast BF16
            outputs = model(inputs)
            top1_probabilities, top1_class_indices = torch.topk(outputs.softmax(dim=1) * 100, k=1)
            top1_probabilities = top1_probabilities.to("cpu").detach().numpy()
            top1_class_indices = top1_class_indices.to("cpu").numpy()
            expected_scores = np.array([21.406523])  # from CPU
            expected_class = np.array([960])
            self.assertEqual(top1_class_indices, expected_class)
            self.assertLess(np.abs(top1_probabilities - expected_scores).max(), 1)

    def test_inference_hpu_graphs(self):
        model, processor = self.prepare_model_and_processor()
        image = self.prepare_data()
        inputs = processor(image).unsqueeze(0).to("hpu")

        model = ht.hpu.wrap_in_hpu_graph(model)  # Apply graph

        outputs = model(inputs)
        top1_probabilities, top1_class_indices = torch.topk(outputs.softmax(dim=1) * 100, k=1)
        top1_probabilities = top1_probabilities.to("cpu").detach().numpy()
        top1_class_indices = top1_class_indices.to("cpu").numpy()
        expected_scores = np.array([21.406523])  # from CPU
        expected_class = np.array([960])
        self.assertEqual(top1_class_indices, expected_class)
        self.assertLess(np.abs(top1_probabilities - expected_scores).max(), 1)

    def test_no_latency_regression_autocast(self):
        warmup = 3
        iterations = 20

        model, processor = self.prepare_model_and_processor()
        image = self.prepare_data()

        model = ht.hpu.wrap_in_hpu_graph(model)

        with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
            for i in range(warmup):
                inputs = processor(image).unsqueeze(0).to("hpu")
                _ = model(inputs)
                torch.hpu.synchronize()

            total_model_time = 0
            for i in range(iterations):
                inputs = processor(image).unsqueeze(0).to("hpu")
                model_start_time = time.time()
                _ = model(inputs)
                torch.hpu.synchronize()
                model_end_time = time.time()
                total_model_time = total_model_time + (model_end_time - model_start_time)

        self.baseline.assertRef(
            compare=lambda latency, expect: latency <= (1.05 * expect),
            context=[OH_DEVICE_CONTEXT],
            latency=total_model_time * 1000 / iterations,  # in terms of ms
        )
