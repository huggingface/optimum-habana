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

import operator
import os

import numpy as np
import pytest
import torch
from datasets import load_dataset
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from transformers import pipeline

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


MODELS_TO_TEST = {
    "text-to-speech": [
        "microsoft/speecht5_tts",
        "facebook/hf-seamless-m4t-medium",
        "facebook/mms-tts-eng",
    ],
    "image-to-text": [
        ("Salesforce/blip-image-captioning-base", 44),
        ("nlpconnect/vit-gpt2-image-captioning", 44),
    ],
}


class TestGaudiPipeline:
    @pytest.mark.parametrize("model, validate_length", MODELS_TO_TEST["image-to-text"])
    def test_image_to_text(self, model, validate_length, baseline):
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

            result = output[0]["generated_text"][:validate_length]
            baseline.assertRef(compare=operator.eq, generated_text=result)

    @pytest.mark.parametrize("model", MODELS_TO_TEST["text-to-speech"])
    def test_text_to_speech(self, model, baseline):
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
            with (
                torch.autocast("hpu", torch.bfloat16, enabled=(model_dtype == torch.bfloat16)),
                torch.no_grad(),
                torch.inference_mode(),
            ):
                for i in range(3):
                    output = generator(text, forward_params=forward_params, generate_kwargs=generate_kwargs)
            assert isinstance(output["audio"], np.ndarray)

            baseline.assertRef(compare=operator.eq, sampling_rate=output["sampling_rate"])
