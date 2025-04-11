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
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

from .utils import OH_DEVICE_CONTEXT


adapt_transformers_to_gaudi()


class GaudiOWlVITTester(TestCase):
    """
    Tests for Zero Shot Object Detection - OWLVIT
    """

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

    def prepare_model_and_processor(self):
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to("hpu")
        model = model.eval()
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        return model, processor

    def prepare_data(self):
        texts = "a photo of a cat, a photo of a dog"
        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
        return texts, image

    def test_inference_default(self):
        model, processor = self.prepare_model_and_processor()
        texts, image = self.prepare_data()
        inputs = processor(text=texts, images=image, return_tensors="pt").to("hpu")
        outputs = model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        boxes = results[0]["boxes"]
        self.assertEqual(len(boxes), 2)
        expected_location = np.array([324.9933, 20.4362, 640.6164, 373.2621])
        self.assertLess(np.abs(boxes[0].cpu().detach().numpy() - expected_location).max(), 1)

    def test_inference_bf16(self):
        model, processor = self.prepare_model_and_processor()
        texts, image = self.prepare_data()
        inputs = processor(text=texts, images=image, return_tensors="pt").to("hpu")

        with torch.autocast(device_type="hpu", dtype=torch.bfloat16):  # Autocast BF16
            outputs = model(**inputs)
            target_sizes = torch.Tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.1
            )
            boxes = results[0]["boxes"]
            expected_location = np.array([324.9933, 20.4362, 640.6164, 373.2621])
            self.assertLess(np.abs(boxes[0].to(torch.float32).cpu().detach().numpy() - expected_location).max(), 2)

    def test_inference_hpu_graphs(self):
        model, processor = self.prepare_model_and_processor()
        texts, image = self.prepare_data()
        inputs = processor(text=texts, images=image, return_tensors="pt").to("hpu")

        model = ht.hpu.wrap_in_hpu_graph(model)  # Apply graph

        outputs = model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        boxes = results[0]["boxes"]
        self.assertEqual(len(boxes), 2)
        expected_location = np.array([324.9933, 20.4362, 640.6164, 373.2621])
        self.assertLess(np.abs(boxes[0].to(torch.float32).cpu().detach().numpy() - expected_location).max(), 1)

    def test_no_latency_regression_bf16(self):
        warmup = 3
        iterations = 10

        model, processor = self.prepare_model_and_processor()
        texts, image = self.prepare_data()

        model = ht.hpu.wrap_in_hpu_graph(model)

        with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
            for i in range(warmup):
                inputs = processor(text=texts, images=image, return_tensors="pt").to("hpu")
                _ = model(**inputs)
                torch.hpu.synchronize()

            total_model_time = 0
            for i in range(iterations):
                inputs = processor(text=texts, images=image, return_tensors="pt").to("hpu")
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
