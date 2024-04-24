# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import packaging.version
import torch
import transformers
from datasets import load_dataset
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
        image = "./tests/resource/image-captioning-example.png"
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

    def _test_text_to_speech(self, model, expected_sample_rate):
        adapt_transformers_to_gaudi()
        MODEL_DTYPE_LIST = [torch.bfloat16, torch.float32]
        text = "hello, the dog is cooler"
        for model_dtype in MODEL_DTYPE_LIST:
            generator = pipeline(
                "text-to-speech",
                model=model,
                torch_dtype=model_dtype,
                device="hpu",
            )
            forward_params = None
            if generator.model.config.model_type == "speecht5":
                embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to("hpu")
                forward_params = {"speaker_embeddings": speaker_embedding}
            if generator.model.config.model_type == "seamless_m4t":
                forward_params = {"tgt_lang": "eng"}

            generate_kwargs = None
            if generator.model.can_generate():
                generate_kwargs = {"lazy_mode": True, "ignore_eos": False, "hpu_graphs": True}

            generator.model = wrap_in_hpu_graph(generator.model)
            with torch.autocast(
                "hpu", torch.bfloat16, enabled=(model_dtype == torch.bfloat16)
            ), torch.no_grad(), torch.inference_mode():
                for i in range(3):
                    output = generator(text, forward_params=forward_params, generate_kwargs=generate_kwargs)
            self.assertTrue(isinstance(output["audio"], np.ndarray))
            self.assertEqual(output["sampling_rate"], expected_sample_rate)

    def test_image_to_text_blip(self):
        model = "Salesforce/blip-image-captioning-base"
        expected_result = "a soccer player is playing a game on the app"
        self._test_image_to_text(model, expected_result)

    def test_image_to_text_vit(self):
        model = "nlpconnect/vit-gpt2-image-captioning"
        expected_result = "a soccer game with a player jumping to catch"
        self._test_image_to_text(model, expected_result)

    def test_text_to_speech_speecht5(self):
        model = "microsoft/speecht5_tts"
        expected_result = 16000
        self._test_text_to_speech(model, expected_result)

    def test_text_to_speech_m4t(self):
        model = "facebook/hf-seamless-m4t-medium"
        expected_result = 16000
        if packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.40.0"):
            self._test_text_to_speech(model, expected_result)

    def test_text_to_speech_mms(self):
        model = "facebook/mms-tts-eng"
        expected_result = 16000
        self._test_text_to_speech(model, expected_result)
