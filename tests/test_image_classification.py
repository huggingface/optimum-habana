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

from unittest import TestCase

import habana_frameworks.torch as ht
import numpy as np
import pytest
import requests
import torch
from PIL import Image

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
from optimum.habana.utils import HabanaGenerationTime

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
        import timm

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
                with HabanaGenerationTime() as timer:
                    _ = model(inputs)
                    torch.hpu.synchronize()
                total_model_time += timer.last_duration

        self.baseline.assertRef(
            compare=lambda latency, expect: latency <= (1.05 * expect),
            context=[OH_DEVICE_CONTEXT],
            latency=total_model_time * 1000 / iterations,  # in terms of ms
        )


class GaudiSiglipTester(TestCase):
    """
    Tests for Sigclip
    """

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

    def prepare_model_and_processor(self):
        from transformers import AutoImageProcessor, SiglipForImageClassification

        torch.manual_seed(3)
        # note: we are loading a `SiglipModel` from the hub here,
        # so the head will be randomly initialized, hence the predictions will be random if seed is not set above.
        image_processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
        model_class = SiglipForImageClassification.from_pretrained("google/siglip-base-patch16-224").to("hpu")

        return model_class, image_processor

    def prepare_model_and_processor_prob(self):
        from transformers import SiglipModel, SiglipProcessor

        torch.manual_seed(3)
        # note: we are loading a `SiglipModel` from the hub here,
        # so the head will be randomly initialized, hence the predictions will be random if seed is not set above.
        image_processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        model = SiglipModel.from_pretrained(
            "google/siglip-so400m-patch14-384",
            torch_dtype=torch.bfloat16,
            device_map="hpu",
        )
        return model, image_processor

    def prepare_data(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        return image

    def test_inference_default(self):
        # test classification
        model_class, processor = self.prepare_model_and_processor()
        image = self.prepare_data()
        inputs = processor(images=image, return_tensors="pt").to("hpu")
        outputs = model_class(**inputs)
        logits = outputs.logits
        # model predicts one of the two classes
        predicted_class_idx = logits.argmax(-1).item()
        self.assertEqual(model_class.config.id2label[predicted_class_idx], "LABEL_1")

    def test_inference_prob(self):
        # test probs
        device = "hpu"
        model_inf, processor = self.prepare_model_and_processor_prob()
        image = self.prepare_data()
        candidate_labels = ["2 cats", "2 dogs"]
        texts = [f"This is a photo of {label}." for label in candidate_labels]
        inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt").to("hpu")

        with torch.no_grad():
            with torch.autocast(device):
                outputs = model_inf(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image).to("cpu").float()  # these are the probabilities
        expected_scores = np.array([0.586])
        self.assertLess(np.abs(probs[0][0] - expected_scores), 0.05)
