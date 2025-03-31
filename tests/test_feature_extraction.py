# coding=utf-8
# Copyright 2022 the HuggingFace Inc. team.
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
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

from .utils import OH_DEVICE_CONTEXT


adapt_transformers_to_gaudi()

MODEL_NAME = "Supabase/gte-small"

INPUT_TEXTS = [
    "what is the capital of China?",
    "how to implement quick sort in Python?",
    "Beijing",
    "sorting algorithms",
]

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embeddings(outputs, batch_dict):
    return F.normalize(average_pool(outputs.last_hidden_state, batch_dict["attention_mask"]))


def scores(embeddings):
    return (embeddings[:1] @ embeddings[1:].T) * 100


def get_batch_dict():
    return TOKENIZER(INPUT_TEXTS, max_length=512, padding=True, truncation=True, return_tensors="pt")


@pytest.fixture(scope="module")
def model():
    return AutoModel.from_pretrained(MODEL_NAME)


@pytest.fixture(autouse=True, scope="class")
def cpu_results(request, model):
    batch_dict = get_batch_dict()
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings_cpu = embeddings(outputs, batch_dict)
    request.cls.scores_cpu = scores(embeddings_cpu)


@pytest.fixture(autouse=True, scope="class")
def default_hpu_results(request, model):
    request.cls.model_hpu = model.to("hpu")
    request.cls.model_hpu_graph = ht.hpu.wrap_in_hpu_graph(model.to("hpu"))
    batch_dict = get_batch_dict().to("hpu")
    with torch.no_grad():
        outputs = request.cls.model_hpu(**batch_dict)
        embeddings_hpu_default = embeddings(outputs, batch_dict)
    request.cls.scores_hpu_default = scores(embeddings_hpu_default)


class GaudiFeatureExtractionTester(TestCase):
    """
    Tests for Supabase/gte-small feature extraction on Gaudi
    """

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

    def test_inference_default(self):
        """
        Tests for equivalent CPU and HPU outputs
        """
        self.assertTrue(torch.allclose(self.scores_cpu, self.scores_hpu_default, rtol=1e-3))

    def test_inference_bf16(self):
        """
        Test for similar bf16 and regular outputs
        """
        batch_dict = get_batch_dict()
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16), torch.no_grad():
            outputs = self.model_hpu(**batch_dict)
            embeddings_hpu_bf16 = embeddings(outputs, batch_dict)
        scores_hpu_bf16 = scores(embeddings_hpu_bf16)
        self.assertTrue(torch.allclose(scores_hpu_bf16, self.scores_hpu_default, rtol=1e-2))

    def test_inference_graph_bf16(self):
        batch_dict = get_batch_dict().to("hpu")
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16), torch.no_grad():
            outputs = self.model_hpu_graph(**batch_dict)
            embeddings_hpu_graph_bf16 = embeddings(outputs, batch_dict)
        scores_hpu_graph_bf16 = scores(embeddings_hpu_graph_bf16)
        self.assertTrue(torch.allclose(scores_hpu_graph_bf16, self.scores_hpu_default, rtol=1e-2))

    def test_latency_graph_bf16(self):
        batch_dict = get_batch_dict().to("hpu")
        warm_up_iters = 5
        test_iters = 50
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16), torch.no_grad():
            for _ in range(warm_up_iters):
                self.model_hpu_graph(**batch_dict)
        torch.hpu.synchronize()
        start_time = time.time()
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16), torch.no_grad():
            for _ in range(test_iters):
                outputs = self.model_hpu_graph(**batch_dict)
                embeddings(outputs, batch_dict)
        torch.hpu.synchronize()
        end_time = time.time()
        time_per_iter = (end_time - start_time) * 1000 / test_iters  # time in ms
        self.baseline.assertRef(
            compare=lambda actual, ref: actual < (1.05 * ref),
            context=[OH_DEVICE_CONTEXT],
            time_per_iter=time_per_iter,
        )
