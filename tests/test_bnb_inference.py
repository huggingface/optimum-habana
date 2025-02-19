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

import copy

import pytest
import torch
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from optimum.habana.transformers import modeling_utils

from .utils import OH_DEVICE_CONTEXT


modeling_utils.adapt_transformers_to_gaudi()


MODEL_ID = "meta-llama/Llama-3.2-1B"


def get_model(token: str):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=nf4_config, device_map={"": "hpu"}, torch_dtype=torch.bfloat16, token=token.value
    )

    return model


@pytest.mark.skipif("gaudi1" == OH_DEVICE_CONTEXT, reason="execution not supported on gaudi1")
def test_nf4_quantization_inference(token: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token.value)

    model = get_model(token)

    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = 20
    generation_config.use_cache = True
    generation_config.use_flash_attention = True

    model = wrap_in_hpu_graph(model)

    input_text = "Hello my name is"
    inputs = tokenizer(input_text, return_tensors="pt").to(device="hpu")

    torch.manual_seed(42)
    outputs = model.generate(**inputs, generation_config=generation_config, hpu_graphs=True, lazy_mode=True)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    assert decoded_output == "Hello my name is Marlene and I am 36 years old. I am a very happy person, I love to"
