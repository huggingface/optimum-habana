# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import unittest

import torch
from transformers import AutoModelForCausalLM
from transformers.activations import GELUActivation, NewGELUActivation

from optimum.habana.fx import GeluToFusedGelu, symbolic_trace


def get_models(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)

    traced_model = symbolic_trace(
        model,
        input_names=["input_ids", "attention_mask", "labels"],
    )
    transformation = GeluToFusedGelu()
    transformed_model = transformation(traced_model)

    return model, transformed_model


def flatten_output(output):
    flatten = []
    for x in output:
        if isinstance(x, (tuple, list)):
            flatten += flatten_output(x)
        elif not isinstance(x, torch.Tensor):
            continue
        else:
            flatten.append(x)
    return flatten


class FXTransformationTester(unittest.TestCase):
    def test_gelu_to_fused_gelu_modules(self):
        model, transformed_model = get_models("hf-internal-testing/tiny-random-gpt2")

        num_gelu_new_original = sum((1 if isinstance(module, NewGELUActivation) else 0 for module in model.modules()))
        num_gelu_new_transformed = sum(
            (1 if isinstance(module, NewGELUActivation) else 0 for module in transformed_model.modules())
        )
        num_gelu_transformed = sum(
            (1 if isinstance(module, GELUActivation) else 0 for module in transformed_model.modules())
        )

        self.assertEqual(
            num_gelu_new_original,
            num_gelu_transformed,
            "There should be as many `NewGELUActivation` modules in the original model as `GELUActivation` in the transformed model.",
        )
        self.assertEqual(
            num_gelu_new_transformed,
            0,
            "There shouldn't be any `NewGELUActivation` module in the transformed model.",
        )

    def test_gelu_to_fused_gelu_output(self):
        model, transformed_model = get_models("hf-internal-testing/tiny-random-gpt2")

        inputs = {
            "input_ids": torch.randint(2, (4, 512)),
            "attention_mask": torch.randint(2, (4, 512)),
            "labels": torch.randint(2, (4, 512)),
        }

        original_output = flatten_output(model(**inputs))
        transformed_output = flatten_output(transformed_model(**inputs))

        num_outputs = len(original_output)
        for i in range(num_outputs):
            self.assertTrue(
                torch.allclose(original_output[i], transformed_output[i]),
                "The output of the transformed model doesn't match with the output of the original model.",
            )
