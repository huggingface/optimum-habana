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
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


assert os.environ.get("GAUDI2_CI", "0") == "1", "Execution does not support on Gaudi1"

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


def test_nf4_quantization_inference(token: str):
    os.environ["PT_HPU_LAZY_MODE"] = "0"
    from optimum.habana.transformers import modeling_utils

    modeling_utils.adapt_transformers_to_gaudi()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token.value)

    model = get_model(token)

    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = 20
    generation_config.use_cache = True
    generation_config.use_flash_attention = True

    # TODO: Uncomment after SW-224176 (and related torch.compile issues) is fixed
    # model.model = torch.compile(model.model, backend="hpu_backend")

    input_text = "Hello my name is"
    inputs = tokenizer(input_text, return_tensors="pt").to(device="hpu")

    torch.manual_seed(42)
    outputs = model.generate(**inputs, generation_config=generation_config, lazy_mode=False)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert decoded_output == "Hello my name is Kelsey and I am a 16 year old girl who loves to draw and paint. I have"
