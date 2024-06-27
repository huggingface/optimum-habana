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

import os
import time
from unittest import TestCase

import habana_frameworks.torch as ht
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, DetrForObjectDetection

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

from .test_examples import TIME_PERF_FACTOR


adapt_transformers_to_gaudi()

if os.environ.get("GAUDI2_CI", "0") == "1":
    # Gaudi2 CI baselines
    LATENCY_DETR_BF16_GRAPH_BASELINE = 7.0
else:
    # Gaudi1 CI baselines
    LATENCY_DETR_BF16_GRAPH_BASELINE = 14.5


class GaudiDETRTester(TestCase):
    """
    Tests for Object Detection - DETR
    """

    def prepare_model_and_processor(self):
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101").to("hpu")
        model = model.eval()
        processor = AutoProcessor.from_pretrained("facebook/detr-resnet-101")
        return model, processor

    def prepare_data(self):
        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
        return image

    def test_inference_default(self):
        model, processor = self.prepare_model_and_processor()
        image = self.prepare_data()
        inputs = processor(images=image, return_tensors="pt").to("hpu")
        outputs = model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        boxes = results["boxes"]
        self.assertEqual(len(boxes), 5)
        expected_location = np.array([344.0622, 24.8543, 640.3398, 373.7401])
        self.assertLess(np.abs(boxes[0].cpu().detach().numpy() - expected_location).max(), 1)

    def test_inference_autocast(self):
        model, processor = self.prepare_model_and_processor()
        image = self.prepare_data()
        inputs = processor(images=image, return_tensors="pt").to("hpu")

        with torch.autocast(device_type="hpu", dtype=torch.bfloat16):  # Autocast BF16
            outputs = model(**inputs)
            target_sizes = torch.Tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
            boxes = results["boxes"]
            self.assertEqual(len(boxes), 5)
            expected_location = np.array([342, 25.25, 636, 376])
            self.assertLess(np.abs(boxes[0].to(torch.float32).cpu().detach().numpy() - expected_location).max(), 5)

    def test_inference_hpu_graphs(self):
        model, processor = self.prepare_model_and_processor()
        image = self.prepare_data()
        inputs = processor(images=image, return_tensors="pt").to("hpu")

        model = ht.hpu.wrap_in_hpu_graph(model)  # Apply graph

        outputs = model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        boxes = results[0]["boxes"]
        self.assertEqual(len(boxes), 5)
        expected_location = np.array([344.0622, 24.8543, 640.3398, 373.7401])
        self.assertLess(np.abs(boxes[0].to(torch.float32).cpu().detach().numpy() - expected_location).max(), 1)

    def test_no_latency_regression_autocast(self):
        warmup = 3
        iterations = 10

        model, processor = self.prepare_model_and_processor()
        image = self.prepare_data()

        model = ht.hpu.wrap_in_hpu_graph(model)

        with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
            for i in range(warmup):
                inputs = processor(images=image, return_tensors="pt").to("hpu")
                _ = model(**inputs)
                torch.hpu.synchronize()

            total_model_time = 0
            for i in range(iterations):
                inputs = processor(images=image, return_tensors="pt").to("hpu")
                model_start_time = time.time()
                _ = model(**inputs)
                torch.hpu.synchronize()
                model_end_time = time.time()
                total_model_time = total_model_time + (model_end_time - model_start_time)

        latency = total_model_time * 1000 / iterations  # in terms of ms
        self.assertLessEqual(latency, TIME_PERF_FACTOR * LATENCY_DETR_BF16_GRAPH_BASELINE)
