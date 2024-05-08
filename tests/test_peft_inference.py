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

from unittest import TestCase

from peft import (
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
    get_peft_model,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from optimum.habana.peft.peft_model import gaudi_generate, gaudi_prepare_inputs_for_generation
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


class GaudiPeftTester(TestCase):
    def __init__(self, *args, **kwargs):
        adapt_transformers_to_gaudi()
        super().__init__(*args, **kwargs)

    def _text_generation(self, model, tokenizer, extra_kwargs=None):
        generate_kwargs = {
            "lazy_mode": True,
            "hpu_graphs": True,
            "max_new_tokens": 128,
            "ignore_eos": True,
        }
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device="hpu",
        )
        output = generator("Hello, Boy", **generate_kwargs)
        return output[0]["generated_text"]

    def _test_text_generation(self, model_name_or_path, peft_method):
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
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

        result = self._text_generation(model, tokenizer)
        model = get_peft_model(model, config)
        model.__class__.generate = gaudi_generate
        model.__class__.prepare_inputs_for_generation = gaudi_prepare_inputs_for_generation

        result1 = self._text_generation(model, tokenizer)
        self.assertNotEqual(result, result1)

        result2 = self._text_generation(model, tokenizer, extra_kwargs={"reuse_cache": True})
        self.assertEqual(result1, result2)

        result3 = self._text_generation(model, tokenizer, extra_kwargs={"bucket_size": 10})
        self.assertEqual(result1, result3)

    def test_text_generation_llama(self):
        self._test_text_generation("huggyllama/llama-7b", "prompt-tuning")
        self._test_text_generation("huggyllama/llama-7b", "p-tuning")
        self._test_text_generation("huggyllama/llama-7b", "prefix-tuning")
