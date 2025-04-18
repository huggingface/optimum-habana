# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

from .utils import OH_DEVICE_CONTEXT


adapt_transformers_to_gaudi()


class GaudiSAMTester(TestCase):
    """
    Tests for Segment Anything Model - SAM
    """

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

    def prepare_model_and_processor(self):
        model = AutoModel.from_pretrained("facebook/sam-vit-huge").to("hpu")
        processor = AutoProcessor.from_pretrained("facebook/sam-vit-huge")
        model = model.eval()
        return model, processor

    def prepare_data(self):
        image = Image.open(
            requests.get(
                "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png", stream=True
            ).raw
        ).convert("RGB")
        input_points = [[[450, 600]]]
        return input_points, image

    def test_inference_default(self):
        model, processor = self.prepare_model_and_processor()
        input_points, image = self.prepare_data()
        inputs = processor(image, input_points=input_points, return_tensors="pt").to("hpu")
        outputs = model(**inputs)
        scores = outputs.iou_scores
        scores = scores[0][0]
        expected_scores = np.array([0.9912, 0.9818, 0.9666])
        self.assertEqual(len(scores), 3)
        self.assertLess(np.abs(scores.cpu().detach().numpy() - expected_scores).max(), 0.02)

    def test_inference_bf16(self):
        model, processor = self.prepare_model_and_processor()
        input_points, image = self.prepare_data()
        inputs = processor(image, input_points=input_points, return_tensors="pt").to("hpu")

        with torch.autocast(device_type="hpu", dtype=torch.bfloat16):  # Autocast BF16
            outputs = model(**inputs)
            scores = outputs.iou_scores
            scores = scores[0][0]
            expected_scores = np.array([0.9912, 0.9818, 0.9666])
            self.assertEqual(len(scores), 3)
            self.assertLess(np.abs(scores.to(torch.float32).cpu().detach().numpy() - expected_scores).max(), 0.02)

    def test_inference_hpu_graphs(self):
        model, processor = self.prepare_model_and_processor()
        input_points, image = self.prepare_data()
        inputs = processor(image, input_points=input_points, return_tensors="pt").to("hpu")

        model = ht.hpu.wrap_in_hpu_graph(model)  # Apply graph

        outputs = model(**inputs)
        scores = outputs.iou_scores
        scores = scores[0][0]
        expected_scores = np.array([0.9912, 0.9818, 0.9666])
        self.assertEqual(len(scores), 3)
        self.assertLess(np.abs(scores.to(torch.float32).cpu().detach().numpy() - expected_scores).max(), 0.02)

    def test_no_latency_regression_bf16(self):
        warmup = 3
        iterations = 10

        model, processor = self.prepare_model_and_processor()
        input_points, image = self.prepare_data()

        model = ht.hpu.wrap_in_hpu_graph(model)

        with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
            for i in range(warmup):
                inputs = processor(image, input_points=input_points, return_tensors="pt").to("hpu")
                _ = model(**inputs)
                torch.hpu.synchronize()

            total_model_time = 0
            for i in range(iterations):
                inputs = processor(image, input_points=input_points, return_tensors="pt").to("hpu")
                model_start_time = time.time()
                _ = model(**inputs)
                torch.hpu.synchronize()
                model_end_time = time.time()
                total_model_time = total_model_time + (model_end_time - model_start_time)

        self.baseline.assertRef(
            compare=lambda latency, expect: latency <= (1.05 * expect),
            context=[OH_DEVICE_CONTEXT],
            latency=total_model_time * 1000 / iterations,  # in terms of ms
        )
