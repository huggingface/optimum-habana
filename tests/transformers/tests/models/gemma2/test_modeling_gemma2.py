# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Gemma2 model."""

import unittest

import pytest
from packaging import version
from parameterized import parameterized
from pytest import mark
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma2Config, is_torch_available, pipeline
from transformers.generation.configuration_utils import GenerationConfig
from transformers.testing_utils import (
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_gpu,
    slow,
    tooslow,
)

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

from ...models.gemma.test_modeling_gemma import GemmaModelTest, GemmaModelTester
from ...test_configuration_common import ConfigTester


torch_device = "hpu"
adapt_transformers_to_gaudi()


if is_torch_available():
    import torch
    from transformers import (
        Gemma2ForCausalLM,
        Gemma2ForSequenceClassification,
        Gemma2ForTokenClassification,
        Gemma2Model,
    )

# @require_torch
# class GemmaModelTester:
#     config_class = GemmaConfig
#     if is_torch_available():
#         model_class = GemmaModel
#         for_causal_lm_class = GemmaForCausalLM
#         for_sequence_class = GemmaForSequenceClassification
#         for_token_class = GemmaForTokenClassification

#     def __init__(
#         self,
#         parent,
#         batch_size=13,
#         seq_length=7,
#         is_training=True,
#         use_input_mask=True,
#         use_token_type_ids=False,
#         use_labels=True,
#         vocab_size=99,
#         hidden_size=32,
#         num_hidden_layers=2,
#         num_attention_heads=4,
#         num_key_value_heads=2,
#         intermediate_size=37,
#         hidden_act="gelu",
#         hidden_dropout_prob=0.1,
#         attention_probs_dropout_prob=0.1,
#         max_position_embeddings=512,
#         type_vocab_size=16,
#         type_sequence_label_size=2,
#         initializer_range=0.02,
#         num_labels=3,
#         num_choices=4,
#         pad_token_id=0,
#         scope=None,
#     ):
#         self.parent = parent
#         self.batch_size = batch_size
#         self.seq_length = seq_length
#         self.is_training = is_training
#         self.use_input_mask = use_input_mask
#         self.use_token_type_ids = use_token_type_ids
#         self.use_labels = use_labels
#         self.vocab_size = vocab_size
#         self.hidden_size = hidden_size
#         self.num_hidden_layers = num_hidden_layers
#         self.num_attention_heads = num_attention_heads
#         self.num_key_value_heads = num_key_value_heads
#         self.intermediate_size = intermediate_size
#         self.hidden_act = hidden_act
#         self.hidden_dropout_prob = hidden_dropout_prob
#         self.attention_probs_dropout_prob = attention_probs_dropout_prob
#         self.max_position_embeddings = max_position_embeddings
#         self.type_vocab_size = type_vocab_size
#         self.type_sequence_label_size = type_sequence_label_size
#         self.initializer_range = initializer_range
#         self.num_labels = num_labels
#         self.num_choices = num_choices
#         self.pad_token_id = pad_token_id
#         self.scope = scope
#         self.head_dim = self.hidden_size // self.num_attention_heads

#     # Copied from tests.models.mistral.test_modeling_mistral.MistralModelTester.prepare_config_and_inputs
#     def prepare_config_and_inputs(self):
#         input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

#         input_mask = None
#         if self.use_input_mask:
#             input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

#         token_type_ids = None
#         if self.use_token_type_ids:
#             token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

#         sequence_labels = None
#         token_labels = None
#         choice_labels = None
#         if self.use_labels:
#             sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
#             token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
#             choice_labels = ids_tensor([self.batch_size], self.num_choices)

#         config = self.get_config()

#         return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

#     def get_config(self):
#         return self.config_class(
#             vocab_size=self.vocab_size,
#             hidden_size=self.hidden_size,
#             num_hidden_layers=self.num_hidden_layers,
#             num_attention_heads=self.num_attention_heads,
#             num_key_value_heads=self.num_key_value_heads,
#             intermediate_size=self.intermediate_size,
#             hidden_act=self.hidden_act,
#             hidden_dropout_prob=self.hidden_dropout_prob,
#             attention_probs_dropout_prob=self.attention_probs_dropout_prob,
#             max_position_embeddings=self.max_position_embeddings,
#             type_vocab_size=self.type_vocab_size,
#             is_decoder=False,
#             initializer_range=self.initializer_range,
#             pad_token_id=self.pad_token_id,
#             head_dim=self.head_dim,
#         )

#     def create_and_check_model(
#         self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
#     ):
#         model = self.model_class(config=config)
#         model.to(torch_device)
#         model.eval()
#         result = model(input_ids, attention_mask=input_mask)
#         result = model(input_ids)
#         self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

#     # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs_for_common with Llama->Gemma
#     def prepare_config_and_inputs_for_common(self):
#         config_and_inputs = self.prepare_config_and_inputs()
#         (
#             config,
#             input_ids,
#             token_type_ids,
#             input_mask,
#             sequence_labels,
#             token_labels,
#             choice_labels,
#         ) = config_and_inputs
#         inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
#         return config, inputs_dict


# @require_torch
# class GemmaModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
#     all_model_classes = (
#         (GemmaModel, GemmaForCausalLM, GemmaForSequenceClassification, GemmaForTokenClassification)
#         if is_torch_available()
#         else ()
#     )
#     pipeline_model_mapping = (
#         {
#             "feature-extraction": GemmaModel,
#             "text-classification": GemmaForSequenceClassification,
#             "token-classification": GemmaForTokenClassification,
#             "text-generation": GemmaForCausalLM,
#             "zero-shot": GemmaForSequenceClassification,
#         }
#         if is_torch_available()
#         else {}
#     )
#     test_headmasking = False
#     test_pruning = False

#     # Need to remove 0.9 in `test_cpu_offload`
#     # This is because we are hitting edge cases with the causal_mask buffer
#     model_split_percents = [0.5, 0.6]

#     # used in `test_torch_compile_for_training`
#     _torch_compile_train_cls = GemmaForCausalLM if is_torch_available() else None

#     # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79245/workflows/9490ef58-79c2-410d-8f51-e3495156cf9c/jobs/1012146
#     def is_pipeline_test_to_skip(
#         self,
#         pipeline_test_case_name,
#         config_class,
#         model_architecture,
#         tokenizer_name,
#         image_processor_name,
#         feature_extractor_name,
#         processor_name,
#     ):
#         return True

#     def setUp(self):
#         self.model_tester = GemmaModelTester(self)
#         self.config_tester = ConfigTester(self, config_class=GemmaConfig, hidden_size=37)

#     def test_config(self):
#         self.config_tester.run_common_tests()

#     def test_model(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs()
#         self.model_tester.create_and_check_model(*config_and_inputs)

#     def test_model_various_embeddings(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs()
#         for type in ["absolute", "relative_key", "relative_key_query"]:
#             config_and_inputs[0].position_embedding_type = type
#             self.model_tester.create_and_check_model(*config_and_inputs)

#     def test_Gemma_sequence_classification_model(self):
#         config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
#         config.num_labels = 3
#         input_ids = input_dict["input_ids"]
#         attention_mask = input_ids.ne(1).to(torch_device)
#         sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
#         model = self.model_tester.for_sequence_class(config)
#         model.to(torch_device)
#         model.eval()
#         result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
#         self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

#     def test_Gemma_sequence_classification_model_for_single_label(self):
#         config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
#         config.num_labels = 3
#         config.problem_type = "single_label_classification"
#         input_ids = input_dict["input_ids"]
#         attention_mask = input_ids.ne(1).to(torch_device)
#         sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
#         model = self.model_tester.for_sequence_class(config)
#         model.to(torch_device)
#         model.eval()
#         result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
#         self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

#     def test_Gemma_sequence_classification_model_for_multi_label(self):
#         config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
#         config.num_labels = 3
#         config.problem_type = "multi_label_classification"
#         input_ids = input_dict["input_ids"]
#         attention_mask = input_ids.ne(1).to(torch_device)
#         sequence_labels = ids_tensor(
#             [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
#         ).to(torch.float)
#         model = self.model_tester.for_sequence_class(config)
#         model.to(torch_device)
#         model.eval()
#         result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
#         self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

#     def test_Gemma_token_classification_model(self):
#         config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
#         config.num_labels = 3
#         input_ids = input_dict["input_ids"]
#         attention_mask = input_ids.ne(1).to(torch_device)
#         token_labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels)
#         model = self.model_tester.for_token_class(config=config)
#         model.to(torch_device)
#         model.eval()
#         result = model(input_ids, attention_mask=attention_mask, labels=token_labels)
#         self.assertEqual(
#             result.logits.shape,
#             (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_labels),
#         )

#     @unittest.skip(reason="Gemma uses GQA on all models so the KV cache is a non standard format")
#     def test_past_key_values_format(self):
#         pass

#     @require_flash_attn
#     @require_torch_gpu
#     @pytest.mark.flash_attn_test
#     @slow
#     def test_flash_attn_2_inference_equivalence_right_padding(self):
#         self.skipTest(reason="Gemma flash attention does not support right padding")

#     @require_torch_sdpa
#     @require_torch_accelerator
#     @slow
#     def test_sdpa_equivalence(self):
#         for model_class in self.all_model_classes:
#             if not model_class._supports_sdpa:
#                 self.skipTest(reason="Model does not support SDPA")

#             config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
#             model = model_class(config)

#             with tempfile.TemporaryDirectory() as tmpdirname:
#                 model.save_pretrained(tmpdirname)
#                 model_sdpa = model_class.from_pretrained(
#                     tmpdirname, torch_dtype=torch.float16, attn_implementation="sdpa"
#                 )
#                 model_sdpa.to(torch_device)

#                 model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, attn_implementation="eager")
#                 model.to(torch_device)

#                 dummy_input = inputs_dict[model_class.main_input_name]
#                 dummy_input = dummy_input.to(torch_device)
#                 outputs = model(dummy_input, output_hidden_states=True)
#                 outputs_sdpa = model_sdpa(dummy_input, output_hidden_states=True)

#                 logits = outputs.hidden_states[-1]
#                 logits_sdpa = outputs_sdpa.hidden_states[-1]

#                 # gemma sdpa needs a high tolerance
#                 assert torch.allclose(logits_sdpa, logits, atol=3e-3)

#     @require_flash_attn
#     @require_torch_gpu
#     @pytest.mark.flash_attn_test
#     @is_flaky()
#     @slow
#     def test_flash_attn_2_equivalence(self):
#         for model_class in self.all_model_classes:
#             if not model_class._supports_flash_attn_2:
#                 self.skipTest(reason="Model does not support Flash Attention 2")

#             config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
#             model = model_class(config)

#             with tempfile.TemporaryDirectory() as tmpdirname:
#                 model.save_pretrained(tmpdirname)
#                 model_fa = model_class.from_pretrained(
#                     tmpdirname, torch_dtype=torch.float16, attn_implementation="flash_attention_2"
#                 )
#                 model_fa.to(torch_device)

#                 model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, attn_implementation="eager")
#                 model.to(torch_device)

#                 dummy_input = inputs_dict[model_class.main_input_name]
#                 dummy_input = dummy_input.to(torch_device)
#                 outputs = model(dummy_input, output_hidden_states=True)
#                 outputs_fa = model_fa(dummy_input, output_hidden_states=True)

#                 logits = outputs.hidden_states[-1]
#                 logits_fa = outputs_fa.hidden_states[-1]

#                 # gemma flash attention 2 needs a high tolerance
#                 assert torch.allclose(logits_fa, logits, atol=3e-3)


class Gemma2ModelTester(GemmaModelTester):
    if is_torch_available():
        config_class = Gemma2Config
        model_class = Gemma2Model
        for_causal_lm_class = Gemma2ForCausalLM
        for_sequence_class = Gemma2ForSequenceClassification
        for_token_class = Gemma2ForTokenClassification


@require_torch
class Gemma2ModelTest(GemmaModelTest, unittest.TestCase):
    all_model_classes = (
        (Gemma2Model, Gemma2ForCausalLM, Gemma2ForSequenceClassification, Gemma2ForTokenClassification)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Gemma2Model,
            "text-classification": Gemma2ForSequenceClassification,
            "token-classification": Gemma2ForTokenClassification,
            "text-generation": Gemma2ForCausalLM,
            "zero-shot": Gemma2ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]

    def setUp(self):
        self.model_tester = Gemma2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gemma2Config, hidden_size=37)

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @unittest.skip("Gemma2's forcefully disables sdpa due to softcapping")
    def test_sdpa_can_dispatch_non_composite_models(self):
        pass

    @unittest.skip("Gemma2's eager attn/sdpa attn outputs are expected to be different")
    def test_eager_matches_sdpa_generate(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip("Gemma2 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("Gemma2 has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip("Gemma2 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("Gemma2 has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("Gemma2 has HybridCache and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("Gemma2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("Gemma2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Gemma2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip("Gemma2 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("Gemma2 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("Gemma2 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_continue_from_inputs_embeds(self):
        pass

    # @unittest.skip("Gemma2's eager attn/sdpa attn outputs are expected to be different")
    # def test_sdpa_equivalence(self):
    #     pass

    @unittest.skip(reason="SDPA can't dispatch on flash due to unsupported head dims")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(
        reason="HybridCache can't be gathered because it is not iterable. Adding a simple iter and dumping `distributed_iterator`"
        " as in Dynamic Cache doesnt work. NOTE: @gante all cache objects would need better compatibility with multi gpu setting"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass


@slow
@require_torch_gpu
class Gemma2IntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]
    # # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # # Depending on the hardware we get different logits / generations
    # cuda_compute_capability_major_version = None

    # @classmethod
    # def setUpClass(cls):
    #     if is_torch_available() and torch.cuda.is_available():
    #         # 8 is for A100 / A10 and 7 for T4
    #         cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    @tooslow
    @require_read_token
    def test_model_9b_bf16(self):
        model_id = "google/gemma-2-9b"
        EXPECTED_TEXTS = [
            "<bos>Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
            "<pad><pad><bos>Hi today I'm going to be talking about the history of the United States. The United States of America",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @tooslow
    @require_read_token
    def test_model_9b_fp16(self):
        model_id = "google/gemma-2-9b"
        EXPECTED_TEXTS = [
            "<bos>Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
            "<pad><pad><bos>Hi today I'm going to be talking about the history of the United States. The United States of America",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16, attn_implementation="eager"
        ).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    @tooslow
    def test_model_9b_pipeline_bf16(self):
        # See https://github.com/huggingface/transformers/pull/31747 -- pipeline was broken for Gemma2 before this PR
        model_id = "google/gemma-2-9b"
        # EXPECTED_TEXTS should match the same non-pipeline test, minus the special tokens
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
            "Hi today I'm going to be talking about the history of the United States. The United States of America",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="flex_attention"
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        output = pipe(self.input_text, max_new_tokens=20, do_sample=False, padding=True)

        self.assertEqual(output[0][0]["generated_text"], EXPECTED_TEXTS[0])
        self.assertEqual(output[1][0]["generated_text"], EXPECTED_TEXTS[1])

    @require_read_token
    def test_model_2b_pipeline_bf16_flex_attention(self):
        # See https://github.com/huggingface/transformers/pull/31747 -- pipeline was broken for Gemma2 before this PR
        model_id = "google/gemma-2-2b"
        # EXPECTED_TEXTS should match the same non-pipeline test, minus the special tokens
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1960s and I am trying to find out what the average",
            "Hi today I'm going to be talking about the 10 best anime of all time.\n\n1",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="flex_attention"
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        output = pipe(self.input_text, max_new_tokens=20, do_sample=False, padding=True)

        self.assertEqual(output[0][0]["generated_text"], EXPECTED_TEXTS[0])
        self.assertEqual(output[1][0]["generated_text"], EXPECTED_TEXTS[1])

    @require_read_token
    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    @tooslow
    def test_model_9b_flash_attn(self):
        # See https://github.com/huggingface/transformers/issues/31953 --- flash attn was generating garbage for gemma2, especially in long context
        model_id = "google/gemma-2-9b"
        EXPECTED_TEXTS = [
            '<bos>Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many people died in the United States. I have found a few sites that say 500,000 but I am not sure if that is correct. I have also found a site that says 675,000 but I am not sure if that is correct either. I am trying to find out how many people died in the United States. I have found a few',
            "<pad><pad><bos>Hi today I'm going to be talking about the history of the United States. The United States of America is a country in North America. It is the third largest country in the world by total area and the third most populous country with over 320 million people. The United States is a federal republic consisting of 50 states and a federal district. The 48 contiguous states and the district of Columbia are in central North America between Canada and Mexico. The state of Alaska is in the"
        ]  # fmt: skip

        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="flash_attention_2", torch_dtype="float16"
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @slow
    @require_read_token
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.5.0"):
            self.skipTest(reason="This test requires torch >= 2.5 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
            convert_and_export_with_cache,
        )

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", pad_token="</s>", padding_side="right")
        EXPECTED_TEXT_COMPLETION = [
            "Hello I am doing a project for my school and I need to know how to make a program that will take a number",
        ]
        max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
            "input_ids"
        ].shape[-1]

        # Load model
        device = "cpu"
        dtype = torch.bfloat16
        cache_implementation = "static"
        attn_implementation = "sdpa"
        batch_size = 1
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b",
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            generation_config=GenerationConfig(
                use_cache=True,
                cache_implementation=cache_implementation,
                max_length=max_generation_length,
                cache_config={
                    "batch_size": batch_size,
                    "max_cache_len": max_generation_length,
                },
            ),
        )

        prompts = ["Hello I am doing"]
        prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

        # Static Cache + export
        exported_program = convert_and_export_with_cache(model)
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)

    @require_read_token
    @tooslow
    def test_model_9b_bf16_flex_attention(self):
        model_id = "google/gemma-2-9b"
        EXPECTED_TEXTS = [
            "<bos>Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
            "<pad><pad><bos>Hi today I'm going to be talking about the history of the United States. The United States of America",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="flex_attention"
        ).to(torch_device)
        assert model.config._attn_implementation == "flex_attention"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @parameterized.expand([("flash_attention_2",), ("sdpa",), ("flex_attention",), ("eager",)])
    @require_read_token
    def test_generation_beyond_sliding_window(self, attn_implementation: str):
        """Test that we can correctly generate beyond the sliding window. This is non trivial as
        we need to correctly slice the attention mask in all cases (because we use a HybridCache).
        Outputs for every attention functions should be coherent and identical.
        """
        model_id = "google/gemma-2-2b"
        EXPECTED_COMPLETIONS = [
            " the people, the food, the culture, the history, the music, the art, the architecture",
            ", green, yellow, orange, purple, pink, brown, black, white, gray, silver",
        ]

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding="left")
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation=attn_implementation, torch_dtype=torch.float16
        ).to(torch_device)

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.sliding_window)

        out = model.generate(**inputs, max_new_tokens=20)[:, input_size:]
        output_text = tokenizer.batch_decode(out)

        self.assertEqual(output_text, EXPECTED_COMPLETIONS)
