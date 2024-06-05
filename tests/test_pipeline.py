# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
from unittest import TestCase

import torch
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from transformers import pipeline

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


class GaudiPipelineTester(TestCase):
    def _test_image_to_text(self, model, expected_result):
        adapt_transformers_to_gaudi()
        MODEL_DTYPE_LIST = [torch.bfloat16, torch.float32]
        generate_kwargs = {
            "lazy_mode": True,
            "hpu_graphs": True,
            "max_new_tokens": 128,
            "ignore_eos": False,
        }
        image = os.path.dirname(__file__) + "/resource/img/image-captioning-example.png"
        for model_dtype in MODEL_DTYPE_LIST:
            generator = pipeline(
                "image-to-text",
                model=model,
                torch_dtype=model_dtype,
                device="hpu",
            )
            generator.model = wrap_in_hpu_graph(generator.model)
            for i in range(3):
                output = generator(image, generate_kwargs=generate_kwargs)
            self.assertTrue(output[0]["generated_text"].startswith(expected_result))

    def test_image_to_text_blip(self):
        model = "Salesforce/blip-image-captioning-base"
        expected_result = "a soccer player is playing a game on the app"
        self._test_image_to_text(model, expected_result)

    def test_image_to_text_vit(self):
        model = "nlpconnect/vit-gpt2-image-captioning"
        expected_result = "a soccer game with a player jumping to catch"
        self._test_image_to_text(model, expected_result)
