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
from unittest import TestCase, skipIf

import habana_frameworks.torch as ht
import numpy as np
import pytest
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, DetrForObjectDetection

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

from .test_examples import TIME_PERF_FACTOR
from .utils import OH_DEVICE_CONTEXT


adapt_transformers_to_gaudi()


def is_eager_mode():
    return os.environ.get("PT_HPU_LAZY_MODE", "1") == "0"


class GaudiDETRTester(TestCase):
    """
    Tests for Object Detection - DETR
    """

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

    def get_expected_loc(self, mode="default"):
        expected_location_def = np.array([344.0622, 24.8543, 640.3398, 373.7401])
        expected_location_ac = np.array([342, 25.25, 636, 376])
        modeCheck = True if mode in ["default", "autocast"] else False
        self.assertEqual(modeCheck, True)
        if mode == "default":
            return expected_location_def
        else:
            return expected_location_ac

    def get_expected_num_boxes(self):
        # For image http://images.cocodataset.org/val2017/000000039769.jpg
        # the expected result is 5
        return 5

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
        self.assertEqual(len(boxes), self.get_expected_num_boxes())
        expected_loc = self.get_expected_loc()
        self.assertLess(np.abs(boxes[0].cpu().detach().numpy() - expected_loc).max(), 1)

    def test_inference_autocast(self):
        model, processor = self.prepare_model_and_processor()
        image = self.prepare_data()
        inputs = processor(images=image, return_tensors="pt").to("hpu")

        with torch.autocast(device_type="hpu", dtype=torch.bfloat16):  # Autocast BF16
            outputs = model(**inputs)
            target_sizes = torch.Tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
            boxes = results["boxes"]
            self.assertEqual(len(boxes), self.get_expected_num_boxes())
            expected_location = self.get_expected_loc(mode="autocast")
            self.assertLess(np.abs(boxes[0].to(torch.float32).cpu().detach().numpy() - expected_location).max(), 5)

    @skipIf(is_eager_mode(), reason="ht.hpu.wrap_in_hpu_graph is supported only in lazy mode")
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
        expected_location = self.get_expected_loc()
        self.assertLess(np.abs(boxes[0].to(torch.float32).cpu().detach().numpy() - expected_location).max(), 1)

    @skipIf(is_eager_mode(), reason="ht.hpu.wrap_in_hpu_graph is supported only in lazy mode")
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

        self.baseline.assertRef(
            compare=lambda latency, expect: latency <= (TIME_PERF_FACTOR * expect),
            context=[OH_DEVICE_CONTEXT],
            latency=total_model_time * 1000 / iterations,  # in terms of ms
        )


class GaudiDetrResnet50_Tester(GaudiDETRTester):
    """
    Tests for Custom Configuration of Detr-Resnet50 model
    """

    def get_num_labels(self):
        # COCO-2017, the dataset the test is based on uses 91 labels
        # TODO: automate this from dataset info. For now, just return this number
        return 91

    def prepare_model_and_processor(self):
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            revision="no_timm",
            num_labels=self.get_num_labels(),
            ignore_mismatched_sizes=True,
        )
        model = model.to("hpu")
        model = model.eval()
        processor = AutoProcessor.from_pretrained("facebook/detr-resnet-50")
        return model, processor

    def get_expected_loc(self, mode="default"):
        # Reference: first box co-ordinates listed in model card:
        # https://huggingface.co/facebook/detr-resnet-50#how-to-use
        expected_location = np.array([40.16, 70.81, 175.55, 117.98])
        modeCheck = True if mode in ["default", "autocast"] else False
        self.assertEqual(modeCheck, True)
        return expected_location

    def get_expected_num_boxes(self):
        # For image http://images.cocodataset.org/val2017/000000039769.jpg
        # the expected result is 5
        return 5
