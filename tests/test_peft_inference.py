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

import pytest
import torch
from peft import (
    AdaptionPromptConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
    get_peft_model,
    tuners,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from optimum.habana.peft.peft_model import gaudi_generate, gaudi_prepare_inputs_for_generation
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


TEST_CASES = [
    ("huggyllama/llama-7b", "prompt-tuning"),
    ("huggyllama/llama-7b", "prefix-tuning"),
    ("huggyllama/llama-7b", "p-tuning"),
    ("huggyllama/llama-7b", "llama-adapter"),
]


class TestGaudiPeftTextGeneration:
    def _text_generation(self, model, tokenizer, extra_kwargs=None):
        generate_kwargs = {
            "lazy_mode": True,
            "hpu_graphs": True,
            "max_new_tokens": 128,
            "ignore_eos": True,
        }
        if extra_kwargs:
            generate_kwargs.update(extra_kwargs)
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device="hpu",
        )
        output = generator("Hello, Boy", **generate_kwargs)
        return output[0]["generated_text"]

    def _test_text_generation(self, model_name_or_path, peft_method):
        adapt_transformers_to_gaudi()
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if peft_method == "prompt-tuning":
            config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=8,
            )
        elif peft_method == "prefix-tuning":
            config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=8,
            )
        elif peft_method == "p-tuning":
            config = PromptEncoderConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=8,
            )
        elif peft_method == "llama-adapter":
            from optimum.habana.peft.layer import (
                GaudiAdaptedAttention_getattr,
                GaudiAdaptedAttentionPreAttnForward,
            )

            tuners.adaption_prompt.layer.AdaptedAttention.pre_attn_forward = GaudiAdaptedAttentionPreAttnForward
            tuners.adaption_prompt.layer.AdaptedAttention.__getattr__ = GaudiAdaptedAttention_getattr
            config = AdaptionPromptConfig(
                adapter_layers=2,
                adapter_len=4,
                task_type=TaskType.CAUSAL_LM,
            )

        result = self._text_generation(model, tokenizer)
        model = get_peft_model(model, config)
        model.__class__.generate = gaudi_generate
        model.__class__.prepare_inputs_for_generation = gaudi_prepare_inputs_for_generation

        result1 = self._text_generation(model, tokenizer)
        if peft_method != "llama-adapter":
            assert result != result1

        result2 = self._text_generation(model, tokenizer, extra_kwargs={"reuse_cache": True})
        assert result1 == result2

        result3 = self._text_generation(model, tokenizer, extra_kwargs={"bucket_size": 10})
        assert result1 == result3

    @pytest.mark.parametrize("model, method", TEST_CASES)
    def test_text_generation_llama(self, model, method):
        self._test_text_generation(model, method)
