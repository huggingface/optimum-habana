# coding=utf-8
# Copyright 2020 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
import unittest
import warnings

import numpy as np
import pytest
from transformers import is_torch_available, pipeline
from transformers.testing_utils import require_torch, slow

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

from ..test_modeling_common import floats_tensor, ids_tensor
from .test_framework_agnostic import GenerationIntegrationTestsMixin


if is_torch_available():
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoModelForSpeechSeq2Seq,
        AutoModelForVision2Seq,
        AutoTokenizer,
        BartForConditionalGeneration,
        BartTokenizer,
        GPT2LMHeadModel,
        GPT2Tokenizer,
        ImageGPTForCausalImageModeling,
        PreTrainedModel,
        SpeechEncoderDecoderModel,
        top_k_top_p_filtering,
    )
    from transformers.generation import (
        BeamSampleDecoderOnlyOutput,
        BeamSampleEncoderDecoderOutput,
        BeamSearchDecoderOnlyOutput,
        BeamSearchEncoderDecoderOutput,
        BeamSearchScorer,
        ConstrainedBeamSearchScorer,
        DisjunctiveConstraint,
        ForcedBOSTokenLogitsProcessor,
        ForcedEOSTokenLogitsProcessor,
        GenerateEncoderDecoderOutput,
        GreedySearchDecoderOnlyOutput,
        GreedySearchEncoderDecoderOutput,
        HammingDiversityLogitsProcessor,
        InfNanRemoveLogitsProcessor,
        LogitsProcessorList,
        MaxLengthCriteria,
        MinLengthLogitsProcessor,
        NoBadWordsLogitsProcessor,
        NoRepeatNGramLogitsProcessor,
        PhrasalConstraint,
        RepetitionPenaltyLogitsProcessor,
        SampleDecoderOnlyOutput,
        SampleEncoderDecoderOutput,
        StoppingCriteria,
        StoppingCriteriaList,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )
    from transformers.generation.candidate_generator import AssistedCandidateGenerator, CandidateGenerator
    from transformers.generation.streamers import BaseStreamer

torch_device = "hpu"
adapt_transformers_to_gaudi()


class GenerationTesterMixin:
    model_tester = None
    all_generative_model_classes = ()
    input_name = "input_ids"

    def _update_default_model_kwargs(self, model_kwargs):
        model_kwargs["limit_hpu_graphs"] = False
        model_kwargs["reuse_cache"] = False
        model_kwargs["bucket_size"] = -1

    def _get_input_ids_and_config(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict[self.input_name]

        # cut to half length & take max batch_size 3
        sequence_length = input_ids.shape[-1] // 2
        input_ids = input_ids[:batch_size, :sequence_length]

        # generate max 3 tokens
        max_length = input_ids.shape[-1] + 3
        if config.eos_token_id is not None and config.pad_token_id is None:
            # hack to allow generate for models such as GPT2 as is done in `generate()`
            if isinstance(config.eos_token_id, int):
                config.eos_token_id = [config.eos_token_id]
            config.pad_token_id = config.eos_token_id[0]
        # TransfoXL has no attention mask
        if "transfoxl" in config.__class__.__name__.lower():
            attention_mask = None
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)[:batch_size, :sequence_length]

        return config, input_ids, attention_mask, max_length

    @staticmethod
    def _get_logits_processor_and_kwargs(
        input_length,
        eos_token_id,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        max_length=None,
        diversity_penalty=None,
    ):
        process_kwargs = {
            "min_length": input_length + 1 if max_length is None else max_length - 1,
            "bad_words_ids": [[1, 0]],
            "no_repeat_ngram_size": 2,
            "repetition_penalty": 1.2,
        }
        logits_processor = LogitsProcessorList(
            (
                [
                    HammingDiversityLogitsProcessor(diversity_penalty, num_beams=2, num_beam_groups=2),
                ]
                if diversity_penalty is not None
                else []
            )
            + (
                [
                    MinLengthLogitsProcessor(process_kwargs["min_length"], eos_token_id),
                ]
                if eos_token_id is not None
                else []
            )
            + (
                [
                    ForcedBOSTokenLogitsProcessor(forced_bos_token_id),
                ]
                if forced_bos_token_id is not None
                else []
            )
            + (
                [ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id)]
                if forced_eos_token_id is not None
                else []
            )
            + [
                NoBadWordsLogitsProcessor(process_kwargs["bad_words_ids"], eos_token_id),
                NoRepeatNGramLogitsProcessor(process_kwargs["no_repeat_ngram_size"]),
                RepetitionPenaltyLogitsProcessor(process_kwargs["repetition_penalty"]),
            ]
        )
        return process_kwargs, logits_processor

    @staticmethod
    def _get_warper_and_kwargs(num_beams):
        warp_kwargs = {"top_k": 10, "top_p": 0.7, "temperature": 0.7}
        logits_warper = LogitsProcessorList(
            [
                TemperatureLogitsWarper(warp_kwargs["temperature"]),
                TopKLogitsWarper(top_k=warp_kwargs["top_k"], min_tokens_to_keep=(2 if num_beams > 1 else 1)),
                TopPLogitsWarper(top_p=warp_kwargs["top_p"], min_tokens_to_keep=(2 if num_beams > 1 else 1)),
            ]
        )
        return warp_kwargs, logits_warper

    @staticmethod
    def _get_beam_scorer_and_kwargs(batch_size, max_length, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
        }
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=beam_kwargs["num_beams"],
            device=torch_device,
            length_penalty=beam_kwargs["length_penalty"],
            do_early_stopping=beam_kwargs["early_stopping"],
            num_beam_hyps_to_keep=num_return_sequences,
        )
        return beam_kwargs, beam_scorer

    @staticmethod
    def _get_diverse_beam_scorer_and_kwargs(batch_size, max_length, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
            "num_beam_groups": 2,  # one beam per group
            "diversity_penalty": 2.0,
        }
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=beam_kwargs["num_beams"],
            device=torch_device,
            length_penalty=beam_kwargs["length_penalty"],
            do_early_stopping=beam_kwargs["early_stopping"],
            num_beam_hyps_to_keep=num_return_sequences,
            num_beam_groups=beam_kwargs["num_beam_groups"],
        )
        return beam_kwargs, beam_scorer

    @staticmethod
    def _get_constrained_beam_scorer_and_kwargs(batch_size, max_length, constraints, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": num_return_sequences * 4,
            "num_return_sequences": num_return_sequences,
        }
        beam_scorer = ConstrainedBeamSearchScorer(
            batch_size=batch_size,
            constraints=constraints,
            num_beams=beam_kwargs["num_beams"],
            device=torch_device,
            length_penalty=beam_kwargs["length_penalty"],
            do_early_stopping=beam_kwargs["early_stopping"],
            num_beam_hyps_to_keep=num_return_sequences,
        )
        return beam_kwargs, beam_scorer

    @staticmethod
    def _get_encoder_outputs(
        model, input_ids, attention_mask, output_attentions=None, output_hidden_states=None, num_interleave=1
    ):
        encoder = model.get_encoder()
        encoder_outputs = encoder(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.repeat_interleave(
            num_interleave, dim=0
        )
        input_ids = torch.zeros_like(input_ids[:, :1]) + model._get_decoder_start_token_id()
        attention_mask = None
        return encoder_outputs, input_ids, attention_mask

    @staticmethod
    def _get_static_shapes():
        return False

    def _greedy_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        if model.config.is_encoder_decoder:
            max_length = 4
        logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
            input_ids.shape[-1],
            eos_token_id=model.config.eos_token_id,
            forced_bos_token_id=model.config.forced_bos_token_id,
            forced_eos_token_id=model.config.forced_eos_token_id,
            max_length=max_length,
        )

        kwargs = {}
        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        model.generation_config.static_shapes = self._get_static_shapes()
        output_generate = model.generate(
            input_ids,
            do_sample=False,
            num_beams=1,
            max_length=max_length,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            **logits_process_kwargs,
            **model_kwargs,
        )

        if model.config.is_encoder_decoder:
            encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            kwargs["encoder_outputs"] = encoder_outputs

        with torch.no_grad():
            model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
            self._update_default_model_kwargs(model_kwargs)
            output_greedy = model.greedy_search(
                input_ids,
                max_length=max_length,
                logits_processor=logits_processor,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
                **model_kwargs,
            )
        return output_greedy, output_generate

    def _sample_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
        num_return_sequences,
        logits_processor,
        logits_warper,
        logits_warper_kwargs,
        process_kwargs,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        torch.manual_seed(0)
        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        self._update_default_model_kwargs(model_kwargs)
        model.generation_config.static_shapes = self._get_static_shapes()
        output_generate = model.generate(
            input_ids,
            do_sample=True,
            num_beams=1,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            **logits_warper_kwargs,
            **process_kwargs,
            **model_kwargs,
        )

        torch.manual_seed(0)
        kwargs = {}
        if model.config.is_encoder_decoder:
            encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                num_interleave=num_return_sequences,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            kwargs["encoder_outputs"] = encoder_outputs
        elif attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        # prevent flaky generation test failures
        logits_processor.append(InfNanRemoveLogitsProcessor())

        with torch.no_grad():
            model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
            self._update_default_model_kwargs(model_kwargs)
            output_sample = model.sample(
                input_ids.repeat_interleave(num_return_sequences, dim=0),
                max_length=max_length,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                output_scores=output_scores,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
                **model_kwargs,
            )

        return output_sample, output_generate

    def _beam_search_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
        beam_scorer,
        beam_kwargs,
        logits_processor,
        logits_process_kwargs,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        self._update_default_model_kwargs(model_kwargs)
        model.generation_config.static_shapes = self._get_static_shapes()
        output_generate = model.generate(
            input_ids,
            do_sample=False,
            max_length=max_length,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            **beam_kwargs,
            **logits_process_kwargs,
            **model_kwargs,
        )

        # beam_search does not automatically interleave `batch_size` dim for `num_beams`
        kwargs = {}
        if model.config.is_encoder_decoder:
            encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                num_interleave=beam_scorer.num_beams,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            kwargs["encoder_outputs"] = encoder_outputs
        elif attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(beam_scorer.num_beams, dim=0)

        with torch.no_grad():
            model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
            self._update_default_model_kwargs(model_kwargs)
            output_beam_search = model.beam_search(
                input_ids.repeat_interleave(beam_scorer.num_beams, dim=0),
                beam_scorer,
                max_length=max_length,
                logits_processor=logits_processor,
                output_scores=output_scores,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
                **model_kwargs,
            )
        return output_generate, output_beam_search

    def _beam_sample_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
        beam_scorer,
        beam_kwargs,
        logits_warper,
        logits_warper_kwargs,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        torch.manual_seed(0)
        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        self._update_default_model_kwargs(model_kwargs)
        output_generate = model.generate(
            input_ids,
            do_sample=True,
            max_length=max_length,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            **beam_kwargs,
            **logits_warper_kwargs,
            **model_kwargs,
        )
        # beam_search does not automatically interleave `batch_size` dim for `num_beams`
        torch.manual_seed(0)
        kwargs = {}
        if model.config.is_encoder_decoder:
            encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                num_interleave=beam_scorer.num_beams,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            kwargs["encoder_outputs"] = encoder_outputs
        elif attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(beam_scorer.num_beams, dim=0)

        # prevent flaky generation test failures
        logits_processor = LogitsProcessorList()
        logits_processor.append(InfNanRemoveLogitsProcessor())

        with torch.no_grad():
            model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
            self._update_default_model_kwargs(model_kwargs)
            output_beam_sample = model.beam_sample(
                input_ids.repeat_interleave(beam_scorer.num_beams, dim=0),
                beam_scorer,
                max_length=max_length,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                output_scores=output_scores,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
                **model_kwargs,
            )

        return output_generate, output_beam_sample

    def _group_beam_search_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
        beam_scorer,
        beam_kwargs,
        logits_processor,
        logits_process_kwargs,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        self._update_default_model_kwargs(model_kwargs)
        output_generate = model.generate(
            input_ids,
            do_sample=False,
            max_length=max_length,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            **beam_kwargs,
            **logits_process_kwargs,
            **model_kwargs,
        )

        # group_beam_search does not automatically interleave `batch_size` dim for `num_beams`
        kwargs = {}
        if model.config.is_encoder_decoder:
            encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                num_interleave=beam_scorer.num_beams,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            kwargs["encoder_outputs"] = encoder_outputs
        elif attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(beam_scorer.num_beams, dim=0)

        with torch.no_grad():
            model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
            self._update_default_model_kwargs(model_kwargs)
            output_group_beam_search = model.group_beam_search(
                input_ids.repeat_interleave(beam_scorer.num_beams, dim=0),
                beam_scorer,
                max_length=max_length,
                logits_processor=logits_processor,
                output_scores=output_scores,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
                **model_kwargs,
            )
        return output_generate, output_group_beam_search

    def _constrained_beam_search_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
        constrained_beam_scorer,
        constraints,
        beam_kwargs,
        logits_processor,
        logits_process_kwargs,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        self._update_default_model_kwargs(model_kwargs)
        model.generation_config.static_shapes = self._get_static_shapes()
        output_generate = model.generate(
            input_ids,
            do_sample=False,
            max_length=max_length,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            constraints=constraints,
            **beam_kwargs,
            **logits_process_kwargs,
            **model_kwargs,
        )

        # group_beam_search does not automatically interleave `batch_size` dim for `num_beams`
        kwargs = {}
        if model.config.is_encoder_decoder:
            encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                num_interleave=constrained_beam_scorer.num_beams,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            kwargs["encoder_outputs"] = encoder_outputs
        elif attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(constrained_beam_scorer.num_beams, dim=0)

        with torch.no_grad():
            model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
            self._update_default_model_kwargs(model_kwargs)
            output_group_beam_search = model.constrained_beam_search(
                input_ids.repeat_interleave(constrained_beam_scorer.num_beams, dim=0),
                constrained_beam_scorer,
                max_length=max_length,
                logits_processor=logits_processor,
                output_scores=output_scores,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
                **model_kwargs,
            )
        return output_generate, output_group_beam_search

    def _contrastive_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        contrastive_search_kwargs = {
            "penalty_alpha": 0.6,
            "top_k": 5,
        }

        if model.config.is_encoder_decoder:
            max_length = 4
        logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
            input_ids.shape[-1],
            eos_token_id=model.config.eos_token_id,
            forced_bos_token_id=model.config.forced_bos_token_id,
            forced_eos_token_id=model.config.forced_eos_token_id,
            max_length=max_length,
        )

        kwargs = {}
        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        self._update_default_model_kwargs(model_kwargs)
        model.generation_config.static_shapes = self._get_static_shapes()
        output_generate = model.generate(
            input_ids,
            do_sample=False,
            num_beams=1,
            max_length=max_length,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            **logits_process_kwargs,
            **model_kwargs,
            **contrastive_search_kwargs,
        )

        if model.config.is_encoder_decoder:
            encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            kwargs["encoder_outputs"] = encoder_outputs

        with torch.no_grad():
            model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
            self._update_default_model_kwargs(model_kwargs)
            stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])
            output_contrastive = model.contrastive_search(
                input_ids,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
                **model_kwargs,
                **contrastive_search_kwargs,
            )
        return output_contrastive, output_generate

    def test_greedy_generate(self):
        # check `generate()` and `greedy_search()` are equal
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            # test old generation output for backwards compatibility
            model = model_class(config).to(torch_device).eval()
            output_greedy, output_generate = self._greedy_generate(
                model=model, input_ids=input_ids, attention_mask=attention_mask, max_length=max_length
            )
            self.assertListEqual(output_greedy.tolist(), output_generate.tolist())

    def test_greedy_generate_dict_outputs(self):
        for model_class in self.all_generative_model_classes:
            # disable cache
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            config.use_cache = False
            model = model_class(config).to(torch_device).eval()
            output_greedy, output_generate = self._greedy_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_greedy, GreedySearchEncoderDecoderOutput)
                self.assertIsInstance(output_generate, GreedySearchEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_greedy, GreedySearchDecoderOnlyOutput)
                self.assertIsInstance(output_generate, GreedySearchDecoderOnlyOutput)

            self.assertListEqual(output_generate.sequences.tolist(), output_greedy.sequences.tolist())

            for output in (output_greedy, output_generate):
                self._check_outputs(output, input_ids, model.config)

    def test_greedy_generate_dict_outputs_use_cache(self):
        for model_class in self.all_generative_model_classes:
            # enable cache
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            if not hasattr(config, "use_cache"):
                # only relevant if model has "use_cache"
                return

            config.use_cache = True
            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            output_greedy, output_generate = self._greedy_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertListEqual(output_generate.sequences.tolist(), output_greedy.sequences.tolist())

            for output in (output_greedy, output_generate):
                self._check_outputs(output, input_ids, model.config, use_cache=True)

    def test_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            model = model_class(config).to(torch_device).eval()

            if model.config.is_encoder_decoder:
                max_length = 4

            process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                model.config.eos_token_id,
                forced_bos_token_id=model.config.forced_bos_token_id,
                forced_eos_token_id=model.config.forced_eos_token_id,
                max_length=max_length,
            )
            logits_warper_kwargs, logits_warper = self._get_warper_and_kwargs(num_beams=2)

            # check `generate()` and `sample()` are equal
            output_sample, output_generate = self._sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                logits_warper_kwargs=logits_warper_kwargs,
                process_kwargs=process_kwargs,
            )
            self.assertListEqual(output_sample.tolist(), output_generate.tolist())

            # check `generate()` and `sample()` yield equal results for `num_return_sequences`
            output_sample, output_generate = self._sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=3,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                logits_warper_kwargs=logits_warper_kwargs,
                process_kwargs=process_kwargs,
            )
            self.assertListEqual(output_sample.tolist(), output_generate.tolist())

    def test_sample_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            # disable cache
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            config.use_cache = False
            model = model_class(config).to(torch_device).eval()
            if model.config.is_encoder_decoder:
                max_length = 4

            process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                model.config.eos_token_id,
                forced_bos_token_id=model.config.forced_bos_token_id,
                forced_eos_token_id=model.config.forced_eos_token_id,
                max_length=max_length,
            )
            logits_warper_kwargs, logits_warper = self._get_warper_and_kwargs(num_beams=1)

            output_sample, output_generate = self._sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=2,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                logits_warper_kwargs=logits_warper_kwargs,
                process_kwargs=process_kwargs,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_sample, SampleEncoderDecoderOutput)
                self.assertIsInstance(output_generate, SampleEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_sample, SampleDecoderOnlyOutput)
                self.assertIsInstance(output_generate, SampleDecoderOnlyOutput)

            self.assertListEqual(output_generate.sequences.tolist(), output_sample.sequences.tolist())

            for output in (output_sample, output_generate):
                self._check_outputs(output, input_ids, model.config, num_return_sequences=2)

    def test_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            model = model_class(config).to(torch_device).eval()
            if model.config.is_encoder_decoder:
                max_length = 4

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                config.eos_token_id,
                config.forced_bos_token_id,
                config.forced_eos_token_id,
                max_length,
            )
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(input_ids.shape[0], max_length)

            # check `generate()` and `beam_search()` are equal
            output_generate, output_beam_search = self._beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
                logits_processor=logits_processor,
            )

            self.assertListEqual(output_generate.tolist(), output_beam_search.tolist())

            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(input_ids.shape[0], max_length)

            output_generate, output_beam_search = self._beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
                logits_processor=logits_processor,
            )
            self.assertListEqual(output_generate.tolist(), output_beam_search.tolist())

    def test_beam_search_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # disable cache
            config.use_cache = False

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            model = model_class(config).to(torch_device).eval()
            if model.config.is_encoder_decoder:
                max_length = 4

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                config.eos_token_id,
                config.forced_bos_token_id,
                config.forced_eos_token_id,
                max_length,
            )
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(input_ids.shape[0], max_length)
            output_generate, output_beam_search = self._beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
                logits_processor=logits_processor,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )
            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_beam_search, BeamSearchEncoderDecoderOutput)
                self.assertIsInstance(output_generate, BeamSearchEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_beam_search, BeamSearchDecoderOnlyOutput)
                self.assertIsInstance(output_generate, BeamSearchDecoderOnlyOutput)

            self.assertListEqual(output_generate.sequences.tolist(), output_beam_search.sequences.tolist())
            self.assertTrue(
                torch.allclose(output_generate["sequences_scores"], output_beam_search["sequences_scores"], atol=1e-3)
            )
            self.assertTrue(output_generate["sequences_scores"].shape == (output_generate["sequences"].shape[0],))
            self.assertTrue((output_generate["sequences_scores"] < 0).all().item())

            for output in (output_beam_search, output_generate):
                self._check_outputs(output, input_ids, model.config, num_return_sequences=beam_scorer.num_beams)

    def test_beam_search_generate_dict_outputs_use_cache(self):
        for model_class in self.all_generative_model_classes:
            # enable cache
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            if not hasattr(config, "use_cache"):
                # only relevant if model has "use_cache"
                return

            model = model_class(config).to(torch_device).eval()
            if model.config.is_encoder_decoder:
                max_length = 4

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                config.eos_token_id,
                config.forced_bos_token_id,
                config.forced_eos_token_id,
                max_length,
            )

            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(input_ids.shape[0], max_length)

            config.use_cache = True
            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            output_beam, output_generate = self._beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
                logits_processor=logits_processor,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertListEqual(output_generate.sequences.tolist(), output_beam.sequences.tolist())

            for output in (output_beam, output_generate):
                self._check_outputs(
                    output, input_ids, model.config, use_cache=True, num_return_sequences=beam_scorer.num_beams
                )

    @pytest.mark.skip("Beam search sampling is not supported by optimum-habana yet")
    def test_beam_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            logits_warper_kwargs, logits_warper = self._get_warper_and_kwargs(num_beams=1)

            model = model_class(config).to(torch_device).eval()

            # check `generate()` and `beam_search()` are equal
            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(input_ids.shape[0], max_length)

            output_generate, output_beam_sample = self._beam_sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_warper=logits_warper,
                logits_warper_kwargs=logits_warper_kwargs,
            )
            self.assertListEqual(output_generate.tolist(), output_beam_sample.tolist())

    @pytest.mark.skip("Beam search sampling is not supported by optimum-habana yet")
    def test_beam_sample_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # disable cache
            config.use_cache = False

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            model = model_class(config).to(torch_device).eval()
            logits_warper_kwargs, logits_warper = self._get_warper_and_kwargs(num_beams=1)

            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(input_ids.shape[0], max_length)

            output_beam_sample, output_generate = self._beam_sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_warper=logits_warper,
                logits_warper_kwargs=logits_warper_kwargs,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_beam_sample, BeamSampleEncoderDecoderOutput)
                self.assertIsInstance(output_generate, BeamSampleEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_beam_sample, BeamSampleDecoderOnlyOutput)
                self.assertIsInstance(output_generate, BeamSampleDecoderOnlyOutput)

            self.assertListEqual(output_generate.sequences.tolist(), output_beam_sample.sequences.tolist())
            self.assertTrue(
                torch.allclose(output_generate["sequences_scores"], output_beam_sample["sequences_scores"], atol=1e-3)
            )
            self.assertTrue(output_generate["sequences_scores"].shape == (output_generate["sequences"].shape[0],))
            self.assertTrue((output_generate["sequences_scores"] < 0).all().item())

            for output in (output_beam_sample, output_generate):
                self._check_outputs(output, input_ids, model.config, num_return_sequences=beam_scorer.num_beams)

    def test_generate_without_input_ids(self):
        config, _, _, max_length = self._get_input_ids_and_config()

        # if no bos token id => cannot generate from None
        if config.bos_token_id is None:
            return

        for model_class in self.all_generative_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()

            output_ids_generate = model.generate(do_sample=False, max_length=max_length, remove_invalid_values=True)
            self.assertIsNotNone(output_ids_generate)

    @pytest.mark.skip("Group beam search is not supported by optimum-habana")
    def test_group_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            model = model_class(config).to(torch_device).eval()
            if model.config.is_encoder_decoder:
                max_length = 4

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                config.eos_token_id,
                config.forced_bos_token_id,
                config.forced_eos_token_id,
                max_length,
                diversity_penalty=2.0,
            )

            # check `generate()` and `group_beam_search()` are equal
            beam_kwargs, beam_scorer = self._get_diverse_beam_scorer_and_kwargs(input_ids.shape[0], max_length)
            output_generate, output_group_beam_search = self._group_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_processor=logits_processor,
                logits_process_kwargs=logits_process_kwargs,
            )
            self.assertListEqual(output_generate.tolist(), output_group_beam_search.tolist())

            # check `generate()` and `group_beam_search()` are equal for `num_return_sequences`
            num_return_sequences = 2
            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_diverse_beam_scorer_and_kwargs(
                input_ids.shape[0], max_length, num_return_sequences=num_return_sequences
            )
            output_generate, output_group_beam_search = self._group_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_processor=logits_processor,
                logits_process_kwargs=logits_process_kwargs,
            )
            self.assertListEqual(output_generate.tolist(), output_group_beam_search.tolist())

    @pytest.mark.skip("Group beam search is not supported by optimum-habana")
    def test_group_beam_search_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            config.use_cache = False

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            model = model_class(config).to(torch_device).eval()
            if model.config.is_encoder_decoder:
                max_length = 4

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                config.eos_token_id,
                config.forced_bos_token_id,
                config.forced_eos_token_id,
                max_length,
                diversity_penalty=2.0,
            )

            num_return_sequences = 1
            beam_kwargs, beam_scorer = self._get_diverse_beam_scorer_and_kwargs(
                input_ids.shape[0], max_length, num_return_sequences=num_return_sequences
            )
            output_generate, output_group_beam_search = self._group_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_processor=logits_processor,
                logits_process_kwargs=logits_process_kwargs,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )
            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_group_beam_search, BeamSearchEncoderDecoderOutput)
                self.assertIsInstance(output_generate, BeamSearchEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_group_beam_search, BeamSearchDecoderOnlyOutput)
                self.assertIsInstance(output_generate, BeamSearchDecoderOnlyOutput)

            self.assertListEqual(output_generate.sequences.tolist(), output_group_beam_search.sequences.tolist())
            self.assertTrue(
                torch.allclose(
                    output_generate["sequences_scores"], output_group_beam_search["sequences_scores"], atol=1e-3
                )
            )
            self.assertTrue(output_generate["sequences_scores"].shape == (output_generate["sequences"].shape[0],))
            self.assertTrue((output_generate["sequences_scores"] < 0).all().item())

            for output in (output_group_beam_search, output_generate):
                self._check_outputs(
                    output, input_ids, model.config, num_return_sequences=num_return_sequences * beam_scorer.num_beams
                )

    def test_constrained_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            model = model_class(config).to(torch_device).eval()
            max_length = 20

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                config.eos_token_id,
                config.forced_bos_token_id,
                config.forced_eos_token_id,
                max_length,
            )

            # check `generate()` and `constrained_beam_search()` are equal
            # Sample constraints
            if not input_ids.dtype == torch.float32:
                min_id = torch.min(input_ids) + 3
                max_id = torch.max(input_ids)
            else:
                # otherwise this throws an error for Speech2TextModel since its inputs are floating points
                min_id = 3
                max_id = 100

            force_tokens = torch.randint(min_id, max_id, (1, 2)).tolist()[0]
            constraints = [
                PhrasalConstraint(force_tokens),
            ]

            beam_kwargs, beam_scorer = self._get_constrained_beam_scorer_and_kwargs(
                input_ids.shape[0], max_length, constraints, num_return_sequences=1
            )
            output_generate, output_beam_search = self._constrained_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                constrained_beam_scorer=beam_scorer,
                constraints=constraints,
                beam_kwargs=beam_kwargs,
                logits_processor=logits_processor,
                logits_process_kwargs=logits_process_kwargs,
            )
            self.assertListEqual(output_generate.tolist(), output_beam_search.tolist())
            for generation_output in output_generate:
                self._check_sequence_inside_sequence(force_tokens, generation_output)

            # check `generate()` and `constrained_beam_search()` are equal for `num_return_sequences`
            # Sample constraints
            force_tokens = torch.randint(min_id, max_id, (1, 2)).tolist()[0]
            constraints = [
                PhrasalConstraint(force_tokens),
            ]

            num_return_sequences = 2
            max_length = 20

            beam_kwargs, beam_scorer = self._get_constrained_beam_scorer_and_kwargs(
                input_ids.shape[0], max_length, constraints, num_return_sequences=num_return_sequences
            )

            output_generate, output_beam_search = self._constrained_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                constrained_beam_scorer=beam_scorer,
                constraints=constraints,
                beam_kwargs=beam_kwargs,
                logits_processor=logits_processor,
                logits_process_kwargs=logits_process_kwargs,
            )
            self.assertListEqual(output_generate.tolist(), output_beam_search.tolist())

            for generation_output in output_generate:
                self._check_sequence_inside_sequence(force_tokens, generation_output)

    def test_constrained_beam_search_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # disable cache
            config.use_cache = False

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            model = model_class(config).to(torch_device).eval()
            if model.config.is_encoder_decoder:
                max_length = 20

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                config.eos_token_id,
                config.forced_bos_token_id,
                config.forced_eos_token_id,
                max_length,
            )

            # Sample constraints
            min_id = 3
            max_id = model.config.vocab_size
            force_tokens = torch.randint(min_id, max_id, (1, 2)).tolist()[0]
            constraints = [
                PhrasalConstraint(force_tokens),
            ]

            beam_kwargs, beam_scorer = self._get_constrained_beam_scorer_and_kwargs(
                input_ids.shape[0], max_length, constraints, num_return_sequences=1
            )
            output_generate, output_beam_search = self._constrained_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                constrained_beam_scorer=beam_scorer,
                constraints=constraints,
                beam_kwargs=beam_kwargs,
                logits_processor=logits_processor,
                logits_process_kwargs=logits_process_kwargs,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_beam_search, BeamSearchEncoderDecoderOutput)
                self.assertIsInstance(output_generate, BeamSearchEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_beam_search, BeamSearchDecoderOnlyOutput)
                self.assertIsInstance(output_generate, BeamSearchDecoderOnlyOutput)

            self.assertListEqual(output_generate.sequences.tolist(), output_beam_search.sequences.tolist())
            self.assertTrue(
                torch.allclose(output_generate["sequences_scores"], output_beam_search["sequences_scores"], atol=1e-3)
            )
            self.assertTrue(output_generate["sequences_scores"].shape == (output_generate["sequences"].shape[0],))
            self.assertTrue((output_generate["sequences_scores"] < 0).all().item())

            for output in (output_beam_search, output_generate):
                self._check_outputs(output, input_ids, model.config, num_return_sequences=beam_scorer.num_beams)

    def test_contrastive_generate(self):
        # check `generate()` and `contrastive_search()` are equal
        for model_class in self.all_generative_model_classes:
            # won't fix: FSMT and Reformer have a different cache variable type (and format).
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                return

            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # NOTE: contrastive search only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                return
            config.use_cache = True
            config.is_decoder = True

            # test old generation output for backwards compatibility
            model = model_class(config).to(torch_device).eval()
            output_contrastive, output_generate = self._contrastive_generate(
                model=model, input_ids=input_ids, attention_mask=attention_mask, max_length=max_length
            )
            self.assertListEqual(output_contrastive.tolist(), output_generate.tolist())

    def test_contrastive_generate_dict_outputs_use_cache(self):
        for model_class in self.all_generative_model_classes:
            # won't fix: FSMT and Reformer have a different cache variable type (and format).
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                return

            # enable cache
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # NOTE: contrastive search only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                return
            config.use_cache = True
            config.is_decoder = True

            model = model_class(config).to(torch_device).eval()
            output_contrastive, output_generate = self._contrastive_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertListEqual(output_generate.sequences.tolist(), output_contrastive.sequences.tolist())

            for output in (output_contrastive, output_generate):
                self._check_outputs(output, input_ids, model.config, use_cache=True)

    def test_contrastive_generate_low_memory(self):
        # Check that choosing 'low_memory' does not change the model output
        for model_class in self.all_generative_model_classes:
            # won't fix: FSMT, Reformer, gptbigcode, and speech2text have a different cache variable type (and format).
            if any(
                model_name in model_class.__name__.lower()
                for model_name in ["fsmt", "reformer", "gptbigcode", "speech2text"]
            ):
                return

            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config(batch_size=1)

            # NOTE: contrastive search only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                return

            config.use_cache = True
            config.is_decoder = True

            # test output equality of low versus high memory
            model = model_class(config).to(torch_device).eval()

            low_output = model.generate(
                input_ids,
                top_k=4,
                penalty_alpha=0.6,
                low_memory=True,
                max_length=max_length,
                attention_mask=attention_mask,
            )

            high_output = model.generate(
                input_ids,
                top_k=4,
                penalty_alpha=0.6,
                low_memory=False,
                max_length=max_length,
                attention_mask=attention_mask,
            )
            self.assertListEqual(low_output.tolist(), high_output.tolist())

        return

    @pytest.mark.skip(reason="Assisted decoding not yet supported by optimum-habana")
    @slow  # TODO(Joao): remove this. Some models (e.g. data2vec, xcom, roberta) have an error rate between 1 and 10%.
    def test_assisted_decoding_matches_greedy_search(self):
        # This test ensures that the assisted generation does not introduce output changes over greedy search.
        # It breaks the pattern in the tests above, for multiple reasons:
        # - assisted_decoding, contrarily to the other methods, can't be called on its own (e.g. needs to
        # prepare the assistant encoder outputs in the main generate body);
        # - assisted_decoding does not support `use_cache = False`
        # - assisted_decoding does not support `batch_size > 1`

        for model_class in self.all_generative_model_classes:
            # won't fix: FSMT and Reformer have a different cache variable type (and format).
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                return
            # may fix in the future: the following models fail with assisted decoding, and need model-specific fixes
            if any(
                model_name in model_class.__name__.lower()
                for model_name in ["bigbirdpegasus", "led", "mega", "speech2text", "git", "prophetnet"]
            ):
                return

            # This for loop is a naive and temporary effort to make the test less flaky.
            failed = 0
            for i in range(10):
                # enable cache
                config, input_ids, attention_mask, max_length = self._get_input_ids_and_config(batch_size=1)

                # NOTE: assisted generation only works with cache on at the moment.
                if not hasattr(config, "use_cache"):
                    return

                config.use_cache = True
                config.is_decoder = True
                model = model_class(config).to(torch_device).eval()
                output_greedy = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=1,
                    do_sample=False,
                    output_scores=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )
                # Note: with assisted generate, if the same model is used as assistant, then all assistant tokens will
                # be correct
                output_assisted = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=1,
                    do_sample=False,
                    assistant_model=model,
                    output_scores=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )

                try:
                    self.assertListEqual(output_greedy.sequences.tolist(), output_assisted.sequences.tolist())

                    for output in (output_greedy, output_assisted):
                        self._check_outputs(output, input_ids, model.config, use_cache=True)
                except AssertionError:
                    failed += 1
                    if failed > 1:
                        self.assertListEqual(output_greedy.sequences.tolist(), output_assisted.sequences.tolist())

                        for output in (output_greedy, output_assisted):
                            self._check_outputs(output, input_ids, model.config, use_cache=True)

    @pytest.mark.skip(reason="Assisted decoding not yet supported by optimum-habana")
    def test_assisted_decoding_sample(self):
        # In this test we don't check assisted vs non-assisted output -- seeded assisted decoding with sample will not
        # match sample for the same seed, as the forward pass does not return the exact same logits (due to matmul with
        # different shapes, see https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535).
        for model_class in self.all_generative_model_classes:
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest("Won't fix: old model with different cache format")
            if any(
                model_name in model_class.__name__.lower()
                for model_name in [
                    "bigbirdpegasus",
                    "led",
                    "mega",
                    "speech2text",
                    "git",
                    "prophetnet",
                    "seamlessm4t",
                    "clvp",
                ]
            ):
                self.skipTest("May fix in the future: need model-specific fixes")

            # enable cache
            config, input_ids, attention_mask, _ = self._get_input_ids_and_config(batch_size=1)

            # NOTE: assisted generation only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                self.skipTest("This model doesn't support caching")

            config.use_cache = True
            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            # Sets assisted generation arguments such that:
            # a) no EOS is generated, to ensure generation doesn't break early
            # b) the assistant model always generates two tokens when it is called, to ensure the input preparation of
            #    the assistant model is correct
            # c) there are at least two forward passes in the main model, to ensure the input preparation of
            #    the main model is correct
            assistant_model = model
            assistant_model.generation_config.num_assistant_tokens = 2  # see b)
            assistant_model.generation_config.num_assistant_tokens_schedule = "constant"  # see b)
            generation_kwargs = {
                "eos_token_id": -1,  # see a)
                "max_new_tokens": 4,  # see c)
                "num_beams": 1,
                "do_sample": True,
                "assistant_model": assistant_model,
                "output_scores": True,
                "output_hidden_states": True,
                "output_attentions": True,
                "return_dict_in_generate": True,
            }

            #######################################################################
            # Monkey patch assisted decoding function till SW issue is resolved
            import copy
            from types import MethodType
            from typing import List, Optional, Union

            from transformers.generation.utils import (
                GenerateDecoderOnlyOutput,
                _crop_past_key_values,
                _prepare_attention_mask,
                _prepare_token_type_ids,
                _split_model_outputs,
            )

            def _speculative_sampling(
                candidate_input_ids,
                candidate_logits,
                candidate_length,
                new_logits,
                last_assistant_token_is_eos,
                max_matches,
            ):
                """
                Applies sampling as in the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf, algorithm 1). Returns
                the selected tokens, as well as the number of candidate matches.

                NOTE: Unless otherwise stated, the variable names match those in the paper.
                """
                new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
                # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
                # selected by the assistant, respectively.
                q = candidate_logits.softmax(dim=-1)
                q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids.squeeze()].squeeze(0, 1)
                p = new_logits.softmax(dim=-1)
                p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids.squeeze()].squeeze(0, 1)
                probability_ratio = p_i / q_i

                # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
                # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
                # (= keep with p = probability_ratio). Keep all the tokens until the first rejection
                r_i = torch.rand_like(probability_ratio)
                is_accepted = r_i <= probability_ratio
                n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1

                # Ensure we don't generate beyond max_len or an EOS token (not in algorithm 1, but needed for correct behavior)
                if last_assistant_token_is_eos and n_matches == candidate_length:
                    # Output length is assumed to be `n_matches + 1`. Since we won't generate another token with the target model
                    # due to acceptance on EOS we fix `n_matches`
                    n_matches -= 1
                    valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
                else:
                    n_matches = min(n_matches, max_matches)

                    # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
                    gamma = min(candidate_logits.shape[1], max_matches)
                    p_n_plus_1 = p[:, n_matches, :]
                    if n_matches < gamma:
                        q_n_plus_1 = q[:, n_matches, :]
                        p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
                        p_prime.div_(p_prime.sum())
                    else:
                        p_prime = p_n_plus_1
                    t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

                    # The selected tokens include the matches (if any) plus the next sampled tokens
                    if n_matches > 0:
                        valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
                    else:
                        valid_tokens = t

                return valid_tokens, n_matches

            def assisted_decoding(
                self,
                input_ids: torch.LongTensor,
                assistant_model: Optional["PreTrainedModel"] = None,
                candidate_generator: Optional["CandidateGenerator"] = None,
                do_sample: bool = False,
                logits_processor: Optional[LogitsProcessorList] = None,
                logits_warper: Optional[LogitsProcessorList] = None,
                stopping_criteria: Optional[StoppingCriteriaList] = None,
                pad_token_id: Optional[int] = None,
                eos_token_id: Optional[Union[int, List[int]]] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                output_scores: Optional[bool] = None,
                return_dict_in_generate: Optional[bool] = None,
                synced_gpus: bool = False,
                streamer: Optional["BaseStreamer"] = None,
                **model_kwargs,
            ):
                r"""
                Generates sequences of token ids for models with a language modeling head using **greedy decoding** or
                **sample** (depending on `do_sample`), assisted by candidate sequences. Assisted generation is an example of a
                candidate decoding strategy. Can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text
                models.

                <Tip warning={true}>

                In most cases, you do not need to call [`~generation.GenerationMixin.candidate_decoding`] directly. Use
                generate() instead. For an overview of generation strategies and code examples, check the [following
                guide](../generation_strategies).

                </Tip>

                Parameters:
                    input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                        The sequence used as a prompt for the generation.
                    candidate_generator (`CandidateGenerator`, *optional*):
                        A derived instance of [`CandidateGenerator`] that defines how candidate sequences are generated. For
                        more information, the documentation of [`CandidateGenerator`] should be read. Only one of `assistant_model` or `candidate_generator` should be passed as input to this function.
                    assistant_model (`PreTrainedModel`, *optional*):
                        An assistant model that can be used to accelerate generation. The assistant model must have the exact
                        same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
                        is much faster than running generation with the model you're calling generate from. As such, the
                        assistant model should be much smaller.
                    do_sample (`bool`, *optional*, defaults to `False`):
                        Whether or not to use sampling ; use greedy decoding otherwise.
                    logits_processor (`LogitsProcessorList`, *optional*):
                        An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                        used to modify the prediction scores of the language modeling head applied at each generation step.
                    logits_warper (`LogitsProcessorList`, *optional*):
                        An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                        to warp the prediction score distribution of the language modeling head applied before multinomial
                        sampling at each generation step.
                    stopping_criteria (`StoppingCriteriaList`, *optional*):
                        An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                        used to tell if the generation loop should stop.
                    pad_token_id (`int`, *optional*):
                        The id of the *padding* token.
                    eos_token_id (`Union[int, List[int]]`, *optional*):
                        The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
                    output_attentions (`bool`, *optional*, defaults to `False`):
                        Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                        returned tensors for more details.
                    output_hidden_states (`bool`, *optional*, defaults to `False`):
                        Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                        for more details.
                    output_scores (`bool`, *optional*, defaults to `False`):
                        Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
                    return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                        Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
                    synced_gpus (`bool`, *optional*, defaults to `False`):
                        Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
                    streamer (`BaseStreamer`, *optional*):
                        Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                        through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
                    model_kwargs:
                        Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                        If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

                Return:
                    [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
                    `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
                    [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
                    `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
                    `model.config.is_encoder_decoder=True`.

                Examples:

                ```python
                >>> from transformers import (
                ...     AutoTokenizer,
                ...     AutoModelForCausalLM,
                ...     LogitsProcessorList,
                ...     MinLengthLogitsProcessor,
                ...     StoppingCriteriaList,
                ...     MaxLengthCriteria,
                ... )

                >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
                >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
                >>> assistant_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
                >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
                >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id
                >>> input_prompt = "It might be possible to"
                >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
                >>> # instantiate logits processors
                >>> logits_processor = LogitsProcessorList(
                ...     [
                ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
                ...     ]
                ... )
                >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
                >>> outputs = model.assisted_decoding(
                ...     input_ids,
                ...     assistant_model=assistant_model,
                ...     logits_processor=logits_processor,
                ...     stopping_criteria=stopping_criteria,
                ... )
                >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
                ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
                ```"""
                # handling deprecated arguments
                if (assistant_model is None) == (candidate_generator is None):
                    raise ValueError(
                        "One (and only one) of `assistant_model` and `candidate_generator` should be defined."
                    )

                if assistant_model is not None:
                    candidate_generator = AssistedCandidateGenerator(
                        input_ids=input_ids,
                        assistant_model=assistant_model,
                        logits_processor=logits_processor,
                        model_kwargs=model_kwargs,
                        eos_token_id=eos_token_id,
                    )
                    warnings.warn(
                        "Passing `assistant_model` to `assisted_decoding` is deprecated and will be removed in v4.38. "
                        "Pass the `candidate_generator` argument instead.",
                        FutureWarning,
                    )

                # init values
                logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
                logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
                stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
                pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
                eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
                if eos_token_id is not None and pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                if isinstance(eos_token_id, int):
                    eos_token_id = [eos_token_id]
                eos_token_id_tensor = (
                    torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
                )
                output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
                output_attentions = (
                    output_attentions if output_attentions is not None else self.generation_config.output_attentions
                )
                output_hidden_states = (
                    output_hidden_states
                    if output_hidden_states is not None
                    else self.generation_config.output_hidden_states
                )
                return_dict_in_generate = (
                    return_dict_in_generate
                    if return_dict_in_generate is not None
                    else self.generation_config.return_dict_in_generate
                )

                # init attention / hidden states / scores tuples
                scores = () if (return_dict_in_generate and output_scores) else None
                decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
                cross_attentions = () if (return_dict_in_generate and output_attentions) else None
                decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

                # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
                if return_dict_in_generate and self.config.is_encoder_decoder:
                    encoder_attentions = (
                        model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                    )
                    encoder_hidden_states = (
                        model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                    )

                # keep track of which sequences are already finished
                unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

                # other auxiliary variables
                max_len = stopping_criteria[0].max_length

                this_peer_finished = False  # used by synced_gpus only
                while True:
                    if synced_gpus:
                        # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                        # The following logic allows an early break if all peers finished generating their sequence
                        this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                        # send 0.0 if we finished, 1.0 otherwise
                        torch.dist.all_reduce(this_peer_finished_flag, op=torch.dist.ReduceOp.SUM)
                        # did all peers finish? the reduced sum will be 0.0 then
                        if this_peer_finished_flag.item() == 0.0:
                            break

                    cur_len = input_ids.shape[-1]

                    #  1. Fetch candidate sequences from a `CandidateGenerator`
                    candidate_input_ids, candidate_logits = candidate_generator.get_candidates(input_ids)
                    candidate_input_ids = candidate_input_ids.to(self.device)
                    if candidate_logits is not None:
                        candidate_logits = candidate_logits.to(self.device)

                    candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
                    last_assistant_token_is_eos = (
                        ~candidate_input_ids[:, -1]
                        .tile(eos_token_id_tensor.shape[0], 1)
                        .ne(eos_token_id_tensor.unsqueeze(1))
                        .prod(dim=0)
                        .bool()
                    )

                    # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
                    # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
                    # we use this forward pass to also pick the subsequent logits in the original model.

                    # 2.1. Prepare the model inputs
                    candidate_kwargs = copy.copy(model_kwargs)
                    candidate_kwargs = _prepare_attention_mask(
                        candidate_kwargs, candidate_input_ids.shape[1], self.config.is_encoder_decoder
                    )
                    candidate_kwargs = _prepare_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])

                    model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)

                    # 2.2. Run a forward pass on the candidate sequence
                    outputs = self(
                        **model_inputs,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )

                    # 2.3. Process the new logits
                    new_logits = outputs.logits[:, -candidate_length - 1 :]  # excludes the input prompt if present
                    if len(logits_processor) > 0:
                        for i in range(candidate_length + 1):
                            new_logits[:, i, :] = logits_processor(
                                candidate_input_ids[:, : cur_len + i], new_logits[:, i, :]
                            )
                    if len(logits_warper) > 0:
                        for i in range(candidate_length + 1):
                            new_logits[:, i, :] = logits_warper(
                                candidate_input_ids[:, : cur_len + i], new_logits[:, i, :]
                            )

                    # 3. Select the accepted tokens. There are two possible cases:
                    # Case 1: `do_sample=True` and we have logits for the candidates (originally from speculative decoding)
                    # 👉 Apply algorithm 1 from the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf).
                    max_matches = max_len - cur_len - 1
                    if do_sample and candidate_logits is not None:
                        valid_tokens, n_matches = _speculative_sampling(
                            candidate_input_ids,
                            candidate_logits,
                            candidate_length,
                            new_logits,
                            last_assistant_token_is_eos,
                            max_matches,
                        )

                    # Case 2: all other cases (originally from assisted generation) 👉 Compare the tokens selected from the
                    # original model logits with the candidate tokens. We can keep the candidate tokens until the first
                    # mismatch, or until the max length is reached.
                    else:
                        if do_sample:
                            probs = new_logits.softmax(dim=-1)
                            selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
                        else:
                            selected_tokens = new_logits.argmax(dim=-1)

                        candidate_new_tokens = candidate_input_ids[:, cur_len:]
                        n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()

                        # Ensure we don't generate beyond max_len or an EOS token
                        if last_assistant_token_is_eos and n_matches == candidate_length:
                            n_matches -= 1
                        n_matches = min(n_matches, max_matches)
                        valid_tokens = selected_tokens[:, : n_matches + 1]

                    # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
                    # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
                    # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
                    # is no match.

                    # 4.1. Get the valid continuation, after the matching tokens
                    input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
                    if streamer is not None:
                        streamer.put(valid_tokens.cpu())
                    new_cur_len = input_ids.shape[-1]

                    # 4.2. Discard past key values relative to unused assistant tokens
                    new_cache_size = new_cur_len - 1
                    outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)

                    # 5. Update the candidate generation strategy if needed
                    candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)

                    if synced_gpus and this_peer_finished:
                        continue  # don't waste resources running the code we don't need

                    # Store scores, attentions and hidden_states when required
                    # Assistant: modified to append one tuple element per token, as in the other generation methods.
                    if return_dict_in_generate:
                        if output_scores:
                            scores += tuple(new_logits[:, i, :] for i in range(n_matches + 1))

                        if "past_key_values" not in model_kwargs:
                            added_len = new_cur_len
                        else:
                            added_len = n_matches + 1

                        if output_attentions:
                            if self.config.is_encoder_decoder:
                                cross_attentions = _split_model_outputs(
                                    cross_attentions, outputs.cross_attentions, cur_len, added_len
                                )
                                decoder_attentions = _split_model_outputs(
                                    decoder_attentions,
                                    outputs.decoder_attentions,
                                    cur_len,
                                    added_len,
                                    is_decoder_attention=True,
                                )
                            else:
                                decoder_attentions = _split_model_outputs(
                                    decoder_attentions,
                                    outputs.attentions,
                                    cur_len,
                                    added_len,
                                    is_decoder_attention=True,
                                )
                        if output_hidden_states:
                            if self.config.is_encoder_decoder:
                                decoder_hidden_states = _split_model_outputs(
                                    decoder_hidden_states, outputs.decoder_hidden_states, cur_len, added_len
                                )
                            else:
                                decoder_hidden_states = _split_model_outputs(
                                    decoder_hidden_states, outputs.hidden_states, cur_len, added_len
                                )

                    model_kwargs = self._update_model_kwargs_for_generation(
                        outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                    )

                    # if eos_token was found in one sentence, set sentence to finished
                    if eos_token_id_tensor is not None:
                        unfinished_sequences = unfinished_sequences.mul(
                            input_ids[:, -1]
                            .tile(eos_token_id_tensor.shape[0], 1)
                            .ne(eos_token_id_tensor.unsqueeze(1))
                            .prod(dim=0)
                        )

                        # stop when each sentence is finished
                        if unfinished_sequences.max() == 0:
                            this_peer_finished = True

                    # stop if we exceed the maximum length
                    if stopping_criteria(input_ids, scores):
                        this_peer_finished = True

                    if this_peer_finished and not synced_gpus:
                        break

                if streamer is not None:
                    streamer.end()

                if return_dict_in_generate:
                    if self.config.is_encoder_decoder:
                        return GenerateEncoderDecoderOutput(
                            sequences=input_ids,
                            scores=scores,
                            encoder_attentions=encoder_attentions,
                            encoder_hidden_states=encoder_hidden_states,
                            decoder_attentions=decoder_attentions,
                            cross_attentions=cross_attentions,
                            decoder_hidden_states=decoder_hidden_states,
                            past_key_values=model_kwargs.get("past_key_values"),
                        )
                    else:
                        return GenerateDecoderOnlyOutput(
                            sequences=input_ids,
                            scores=scores,
                            attentions=decoder_attentions,
                            hidden_states=decoder_hidden_states,
                            past_key_values=model_kwargs.get("past_key_values"),
                        )
                else:
                    return input_ids

            model.assisted_decoding = MethodType(assisted_decoding, model)

            #######################################################################

            output_assisted = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)

            self._check_outputs(output_assisted, input_ids, model.config, use_cache=True)

    def test_generate_with_head_masking(self):
        """Test designed for encoder-decoder models to ensure the attention head masking is used."""
        attention_names = ["encoder_attentions", "decoder_attentions", "cross_attentions"]
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            # We want to test only encoder-decoder models
            if not config.is_encoder_decoder:
                continue
            model = model_class(config).to(torch_device)

            head_masking = {
                "head_mask": torch.zeros(config.encoder_layers, config.encoder_attention_heads, device=torch_device),
                "decoder_head_mask": torch.zeros(
                    config.decoder_layers, config.decoder_attention_heads, device=torch_device
                ),
                "cross_attn_head_mask": torch.zeros(
                    config.decoder_layers, config.decoder_attention_heads, device=torch_device
                ),
            }

            signature = inspect.signature(model.forward)
            # We want to test only models where encoder/decoder head masking is implemented
            if not set(head_masking.keys()) < {*signature.parameters.keys()}:
                continue

            for attn_name, (name, mask) in zip(attention_names, head_masking.items()):
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    num_beams=1,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    remove_invalid_values=True,
                    **{name: mask},
                )
                # We check the state of decoder_attentions and cross_attentions just from the last step
                attn_weights = out[attn_name] if attn_name == attention_names[0] else out[attn_name][-1]
                self.assertEqual(sum([w.sum().item() for w in attn_weights]), 0.0)

    def test_left_padding_compatibility(self):
        # The check done in this test is fairly difficult -- depending on the model architecture, passing the right
        # position index for the position embeddings can still result in a different output, due to numerical masking.
        # On the other hand, for some types of position embeddings, an incorrect position index can have a minimal
        # impact on the output.
        # There are two tricks employed to check whether left-padding compatibility is in place:
        # 1 - To reduce the negative impact of the numerical attention mask on a correct position index, we set the
        # padding size to 1.
        # 2 - To reduce the chance of false positives (i.e. passing when it should be failing), we run the check
        # multiple times with random inputs, and it has to pass with all of them.
        # NOTE: because of 2), there is some chance of false positives in this test.

        for model_class in self.all_generative_model_classes:
            config, _, _, _ = self._get_input_ids_and_config()
            if config.is_encoder_decoder:
                continue  # skip for encoder-decoder models -- they don't need left-padding compatibility
            model = model_class(config).to(torch_device).eval()
            signature = inspect.signature(model.forward).parameters.keys()

            no_failures = True
            for _ in range(10):  # there may be false positives with 10 runs, we rely on the CI to catch the flakiness
                _, input_ids, attention_mask, _ = self._get_input_ids_and_config()
                model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if "position_ids" in signature:
                    position_ids = torch.cumsum(attention_mask, dim=-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    model_kwargs["position_ids"] = position_ids
                next_logits_wo_padding = model(**model_kwargs).logits[:, -1, :]

                pad_size = (input_ids.shape[0], 1)
                padding = torch.ones(pad_size, dtype=input_ids.dtype, device=torch_device) * config.pad_token_id
                padded_input_ids = torch.cat((padding, input_ids), dim=1)
                padded_attention_mask = torch.cat((torch.zeros_like(padding), attention_mask), dim=1)
                model_kwargs = {"input_ids": padded_input_ids, "attention_mask": padded_attention_mask}
                if "position_ids" in signature:
                    position_ids = torch.cumsum(padded_attention_mask, dim=-1) - 1
                    position_ids.masked_fill_(padded_attention_mask == 0, 1)
                    model_kwargs["position_ids"] = position_ids
                next_logits_with_padding = model(**model_kwargs).logits[:, -1, :]
                if not torch.allclose(next_logits_wo_padding, next_logits_with_padding, atol=1e-7):
                    no_failures = False
                    break

            self.assertTrue(no_failures)

    def test_past_key_values_format(self):
        # Test that the KV cache is formatted correctly. Exceptions need to explicitly overwrite this test. Having a
        # standard KV cache format is important for a consistent API (and for advanced generation methods).
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            # If it doesn't support cache, pass the test
            if not hasattr(config, "use_cache"):
                return

            model = model_class(config).to(torch_device)
            if "use_cache" not in inputs:
                inputs["use_cache"] = True
            outputs = model(**inputs)

            # If "past_key_values" is not returned, pass the test (e.g. RWKV uses a different cache name and format)
            if "past_key_values" not in outputs:
                return

            num_hidden_layers = (
                getattr(config, "decoder_layers", None)
                or getattr(config, "num_decoder_layers", None)
                or config.num_hidden_layers
            )
            num_attention_heads = getattr(config, "decoder_attention_heads", config.num_attention_heads)
            embed_dim = getattr(config, "d_model", config.hidden_size)
            per_head_embed_dim = embed_dim // num_attention_heads

            past_kv = outputs["past_key_values"]
            self.assertEqual(len(past_kv), num_hidden_layers)

            # Encoder-Decoder checks
            if config.is_encoder_decoder:
                encoder_num_attention_heads = config.encoder_attention_heads
                encoder_per_head_embed_dim = embed_dim // encoder_num_attention_heads
                batch_size, seq_length = inputs["decoder_input_ids"].shape
                for i in range(num_hidden_layers):
                    self.assertEqual(len(past_kv[i]), 4)  # K V for the decoder + K V for the encoder = 4
                    self.assertEqual(
                        past_kv[i][0].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )
                    self.assertEqual(
                        past_kv[i][1].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )
                    # The sequence length for the encoder K V depends on the model. Since it is not manipulated in
                    # autoregressive generation, I'm keeping the test general and not checking the 3rd dim
                    self.assertEqual(
                        (past_kv[i][2].shape[0], past_kv[i][2].shape[1], past_kv[i][2].shape[3]),
                        (batch_size, encoder_num_attention_heads, encoder_per_head_embed_dim),
                    )
                    self.assertEqual(
                        (past_kv[i][3].shape[0], past_kv[i][3].shape[1], past_kv[i][3].shape[3]),
                        (batch_size, encoder_num_attention_heads, encoder_per_head_embed_dim),
                    )

            # Decoder-only checks
            else:
                # TODO: this line is only needed because of imagegpt, where "pixel_values" = "input_ids". Fix the
                # tests in imagegpt such that `prepare_config_and_inputs_for_common` returns the later (and the other
                # tests use it)
                key = "input_ids" if "input_ids" in inputs else "pixel_values"
                batch_size, seq_length = inputs[key].shape
                for i in range(num_hidden_layers):
                    self.assertEqual(len(past_kv[0]), 2)  # K V for the decoder = 2
                    self.assertEqual(
                        past_kv[i][0].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )
                    self.assertEqual(
                        past_kv[i][1].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )

    def test_generate_from_inputs_embeds_decoder_only(self):
        # When supported, tests that the decoder model can generate from `inputs_embeds` instead of `input_ids`
        # if fails, you should probably update the `prepare_inputs_for_generation` function
        for model_class in self.all_generative_model_classes:
            config, input_ids, _, _ = self._get_input_ids_and_config()

            # Ignore:
            # a) eos (to always output 20 tokens) and pad (so we don't try to infer the attn mask from the input_ids,
            #   which would cause a mismatch),
            config.pad_token_id = config.eos_token_id = -1
            # b) embedding scaling, the scaling factor applied after embeding from input_ids (requires knowledge of the
            #   variable that holds the scaling factor, which is model-dependent)
            if hasattr(config, "scale_embedding"):
                config.scale_embedding = False

            # This test is for decoder-only models (encoder-decoder models have native input embeddings support in the
            # decoder)
            if config.is_encoder_decoder:
                continue

            # Skip models without explicit support
            model = model_class(config).to(torch_device).eval()
            if "inputs_embeds" not in inspect.signature(model.prepare_inputs_for_generation).parameters.keys():
                continue

            # Traditional way of generating text
            outputs_from_ids = model.generate(input_ids)
            self.assertEqual(outputs_from_ids.shape, (2, 20))

            # Same thing, but from input embeddings (`input_ids` is passed so the prompt is present in the output)
            inputs_embeds = model.get_input_embeddings()(input_ids)
            outputs_from_embeds = model.generate(input_ids, inputs_embeds=inputs_embeds)
            self.assertListEqual(outputs_from_ids.tolist(), outputs_from_embeds.tolist())

            # But if we pass different inputs_embeds, we should get different outputs
            torch.manual_seed(0)
            random_embeds = torch.rand_like(inputs_embeds)
            outputs_from_rand_embeds = model.generate(input_ids, inputs_embeds=random_embeds)
            with self.assertRaises(AssertionError):
                self.assertListEqual(outputs_from_rand_embeds.tolist(), outputs_from_embeds.tolist())

            # input_ids is not a required input -- if we don't pass it, the newly generated tokens will be the same
            outputs_from_embeds_wo_ids = model.generate(
                inputs_embeds=inputs_embeds, max_new_tokens=20 - inputs_embeds.shape[1]
            )
            self.assertListEqual(
                outputs_from_embeds[:, inputs_embeds.shape[1] :].tolist(),
                outputs_from_embeds_wo_ids[:, 1:].tolist(),
            )

    def _check_outputs(self, output, input_ids, config, use_cache=False, num_return_sequences=1):
        batch_size, seq_length = input_ids.shape
        num_sequences_in_output = batch_size * num_return_sequences
        gen_len = (
            output.sequences.shape[-1] - 1 if config.is_encoder_decoder else output.sequences.shape[-1] - seq_length
        )

        # scores
        self._check_scores(num_sequences_in_output, output.scores, length=gen_len, config=config)

        # Attentions
        if config.is_encoder_decoder:
            # encoder
            self._check_encoder_attention_for_generate(output.encoder_attentions, batch_size, config, seq_length)
            # decoder
            self._check_attentions_for_generate(
                num_sequences_in_output,
                output.decoder_attentions,
                min_length=1,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )
        else:
            # if use_cache first input is equal to no use_cache, so skip here
            attentions = output.attentions if not use_cache else output.attentions[1:]
            min_length = seq_length if not use_cache else seq_length + 1
            self._check_attentions_for_generate(
                num_sequences_in_output,
                attentions=attentions,
                min_length=min_length,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )

        # Hidden States
        if config.is_encoder_decoder:
            # encoder
            self._check_encoder_hidden_states_for_generate(
                output.encoder_hidden_states, batch_size, config, seq_length
            )

            # decoder
            self._check_hidden_states_for_generate(
                num_sequences_in_output,
                output.decoder_hidden_states,
                min_length=1,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )
        else:
            # if use_cache first input is equal to no use_cache, so skip here
            hidden_states = output.hidden_states if not use_cache else output.hidden_states[1:]
            min_length = seq_length if not use_cache else seq_length + 1
            self._check_hidden_states_for_generate(
                num_sequences_in_output,
                hidden_states,
                min_length=min_length,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )

    def _check_scores(self, batch_size, scores, length, config):
        expected_shape = (batch_size, config.vocab_size)
        self.assertIsInstance(scores, tuple)
        self.assertEqual(len(scores), length)
        self.assertListEqual([iter_scores.shape for iter_scores in scores], [expected_shape] * len(scores))

    def _check_attentions_for_generate(
        self, batch_size, attentions, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [isinstance(iter_attentions, tuple) for iter_attentions in attentions], [True] * len(attentions)
        )
        self.assertEqual(len(attentions), (max_length - min_length) * num_beam_groups)

        for idx, iter_attentions in enumerate(attentions):
            tgt_len = min_length + idx if not use_cache else 1
            src_len = min_length + idx

            expected_shape = (
                batch_size * num_beam_groups,
                config.num_attention_heads,
                tgt_len,
                src_len,
            )
            # check attn size
            self.assertListEqual(
                [layer_attention.shape for layer_attention in iter_attentions], [expected_shape] * len(iter_attentions)
            )

    def _check_encoder_attention_for_generate(self, attentions, batch_size, config, seq_length):
        encoder_expected_shape = (batch_size, config.num_attention_heads, seq_length, seq_length)
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [layer_attentions.shape for layer_attentions in attentions],
            [encoder_expected_shape] * len(attentions),
        )

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, tuple) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (max_length - min_length) * num_beam_groups)

        for idx, iter_hidden_states in enumerate(hidden_states):
            seq_len = min_length + idx if not use_cache else 1
            expected_shape = (batch_size * num_beam_groups, seq_len, config.hidden_size)
            # check hidden size
            self.assertListEqual(
                [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states],
                [expected_shape] * len(iter_hidden_states),
            )

    def _check_encoder_hidden_states_for_generate(self, hidden_states, batch_size, config, seq_length):
        encoder_expected_shape = (batch_size, seq_length, config.hidden_size)
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [layer_hidden_states.shape for layer_hidden_states in hidden_states],
            [encoder_expected_shape] * len(hidden_states),
        )

    def _check_sequence_inside_sequence(self, tensor_1, tensor_2):
        # check if tensor_1 inside tensor_2 or tensor_2 inside tensor_1.
        # set to same device. we don't care what device.

        if not isinstance(tensor_1, list):
            tensor_1 = tensor_1.cpu().tolist()
        if not isinstance(tensor_2, list):
            tensor_2 = tensor_2.cpu().tolist()

        in_order = len(tensor_1) <= len(tensor_2)
        longer = tensor_2 if in_order else tensor_1
        shorter = tensor_1 if in_order else tensor_2

        flag = False
        chunk_size = len(shorter)
        for chunk_idx in range(len(longer) - chunk_size + 1):
            subseq = longer[chunk_idx : chunk_idx + chunk_size]
            if subseq == shorter:
                flag = True
                break

        self.assertTrue(flag)


@require_torch
class UtilsFunctionsTest(unittest.TestCase):
    # tests whether the top_k_top_p function behaves as expected
    def test_top_k_top_p_filtering(self):
        logits = torch.tensor(
            [
                [
                    8.2220991,  # 3rd highest value; idx. 0
                    -0.5620044,
                    5.23229752,
                    4.0386393,
                    -6.8798378,
                    -0.54785802,
                    -3.2012153,
                    2.92777176,
                    1.88171953,
                    7.35341276,
                    8.43207833,  # 2nd highest value; idx. 10
                    -9.85711836,
                    -5.96209236,
                    -1.13039161,
                    -7.1115294,
                    -0.8369633,
                    -5.3186408,
                    7.06427407,
                    0.81369344,
                    -0.82023817,
                    -5.9179796,
                    0.58813443,
                    -6.99778438,
                    4.71551189,
                    -0.18771637,
                    7.44020759,  # 4th highest value; idx. 25
                    9.38450987,  # 1st highest value; idx. 26
                    2.12662941,
                    -9.32562038,
                    2.35652522,
                ],  # cummulative prob of 4 highest values <= 0.6
                [
                    0.58425518,
                    4.53139238,
                    -5.57510464,
                    -6.28030699,
                    -7.19529503,
                    -4.02122551,
                    1.39337037,
                    -6.06707057,
                    1.59480517,
                    -9.643119,
                    0.03907799,
                    0.67231762,
                    -8.88206726,
                    6.27115922,  # 4th highest value; idx. 13
                    2.28520723,
                    4.82767506,
                    4.30421368,
                    8.8275313,  # 2nd highest value; idx. 17
                    5.44029958,
                    -4.4735794,
                    7.38579536,  # 3rd highest value; idx. 20
                    -2.91051663,
                    2.61946077,
                    -2.5674762,
                    -9.48959302,
                    -4.02922645,
                    -1.35416918,
                    9.67702323,  # 1st highest value; idx. 27
                    -5.89478553,
                    1.85370467,
                ],  # cummulative prob of 4 highest values <= 0.6
            ],
            dtype=torch.float,
            device=torch_device,
        )

        non_inf_expected_idx = torch.tensor(
            [[0, 0], [0, 10], [0, 25], [0, 26], [1, 13], [1, 17], [1, 20], [1, 27]],
            dtype=torch.long,
            device=torch_device,
        )  # expected non filtered idx as noted above

        non_inf_expected_output = torch.tensor(
            [
                8.2221,
                8.4321,
                7.4402,
                9.3845,
                6.2712,
                8.8275,
                7.3858,
                9.6770,
            ],  # expected non filtered values as noted above
            dtype=torch.float,
            device=torch_device,
        )

        output = top_k_top_p_filtering(logits, top_k=10, top_p=0.6, min_tokens_to_keep=4)
        non_inf_output = output[output != -float("inf")].to(device=torch_device)
        non_inf_idx = (output != -float("inf")).nonzero().to(device=torch_device)

        self.assertTrue(torch.allclose(non_inf_expected_output, non_inf_output, atol=1e-12))
        self.assertTrue(torch.all(torch.eq(non_inf_expected_idx, non_inf_idx)))

    # tests whether the function uses filter_value instead of default -inf
    def test_top_k_top_p_filtering_with_filter_value(self):
        logits = torch.tensor(
            [
                [
                    1,
                    1,
                    1,
                    0.99,  # get filtered by top-p filtering
                    0.98,  # get filtered by top-k filtering
                ]
            ],
            dtype=torch.float,
            device=torch_device,
        )

        expected_output = torch.tensor(
            [[1, 1, 1, 0, 0]],
            dtype=torch.float,
            device=torch_device,
        )

        output = top_k_top_p_filtering(logits, top_k=4, top_p=0.5, filter_value=0.0)

        self.assertTrue(torch.allclose(expected_output, output, atol=1e-12))


@require_torch
class GenerationIntegrationTests(unittest.TestCase, GenerationIntegrationTestsMixin):
    # setting framework_dependent_parameters needs to be gated, just like its contents' imports
    if is_torch_available():
        framework_dependent_parameters = {
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoModelForSpeechSeq2Seq": AutoModelForSpeechSeq2Seq,
            "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
            "AutoModelForVision2Seq": AutoModelForVision2Seq,
            "LogitsProcessorList": LogitsProcessorList,
            "MinLengthLogitsProcessor": MinLengthLogitsProcessor,
            "create_tensor_fn": torch.tensor,
            "floats_tensor": floats_tensor,
            "return_tensors": "pt",
        }

    @slow
    def test_diverse_beam_search(self):
        # PT-only test: TF doesn't have a diverse beam search implementation
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood.
        The celebrity couple announced the arrival of their son, Silas Randall Timberlake, in statements to People.
        "Silas was the middle name of Timberlake's maternal grandfather Bill Bomar, who died in 2012, while Randall is the musician's own middle name, as well as his father's first," People reports.
        The couple announced the pregnancy in January, with an Instagram post. It is the first baby for both."""

        bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(torch_device)
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        outputs = bart_model.generate(
            input_ids,
            num_beams=4,
            num_return_sequences=2,
            num_beam_groups=4,
            diversity_penalty=2.0,
            remove_invalid_values=True,
        )

        generated_text = bart_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The couple announced the birth of their son, Silas Randall Timberlake, in a statement. Silas was the"
                " middle name of Timberlake's maternal grandfather Bill Bomar. Randall is the musician's own middle"
                " name, as well as his father's first. It is the first baby for both of them.",
                "Justin Timberlake and Jessica Biel have a son. The baby is named Silas Randall Timberlake. It is the"
                " first child for both. The couple announced the pregnancy in January. The name Silas is the middle"
                " name of Timberlake's maternal grandfather. It's also his own middle name.",
            ],
        )

    def test_max_length_backward_compat_greedy(self):
        # PT-only test: TF doesn't have StoppingCriteria
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        bart_model = BartForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-bart").to(
            torch_device
        )
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        max_length = 20
        input_ids = input_ids.expand(2, -1)
        model_kwargs = bart_model._prepare_encoder_decoder_kwargs_for_generation(input_ids, {})
        input_ids, model_kwargs = bart_model._prepare_decoder_input_ids_for_generation(
            batch_size=input_ids.shape[0],
            model_input_name=bart_model.main_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=bart_model.config.decoder_start_token_id,
            bos_token_id=bart_model.config.bos_token_id,
        )

        with self.assertWarns(UserWarning):
            bart_model.greedy_search(
                input_ids,
                max_length=max_length,
                pad_token_id=bart_model.config.pad_token_id,
                eos_token_id=bart_model.config.eos_token_id,
                **model_kwargs,
            )

    def test_max_length_backward_compat_sample(self):
        # PT-only test: TF doesn't have StoppingCriteria
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        bart_model = BartForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-bart").to(
            torch_device
        )
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        max_length = 20
        input_ids = input_ids.expand(2, -1)
        model_kwargs = bart_model._prepare_encoder_decoder_kwargs_for_generation(input_ids, {})
        input_ids, model_kwargs = bart_model._prepare_decoder_input_ids_for_generation(
            batch_size=input_ids.shape[0],
            model_input_name=bart_model.main_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=bart_model.config.decoder_start_token_id,
            bos_token_id=bart_model.config.bos_token_id,
        )
        with torch.no_grad():
            with self.assertWarns(UserWarning):
                bart_model.sample(
                    input_ids,
                    max_length=max_length,
                    pad_token_id=bart_model.config.pad_token_id,
                    eos_token_id=bart_model.config.eos_token_id,
                    **model_kwargs,
                )

    def test_max_length_backward_compat_beam_search(self):
        # PT-only test: TF doesn't have StoppingCriteria
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        bart_model = BartForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-bart").to(
            torch_device
        )
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        batch_size = 1
        max_length = 20
        num_beams = 2

        input_ids = input_ids.expand(2, -1)
        model_kwargs = bart_model._prepare_encoder_decoder_kwargs_for_generation(input_ids, {})
        input_ids, model_kwargs = bart_model._prepare_decoder_input_ids_for_generation(
            batch_size=input_ids.shape[0],
            model_input_name=bart_model.main_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=bart_model.config.decoder_start_token_id,
            bos_token_id=bart_model.config.bos_token_id,
        )

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=torch_device,
        )
        with self.assertWarns(UserWarning):
            _ = bart_model.beam_search(
                input_ids, num_beams=num_beams, max_length=max_length, beam_scorer=beam_scorer, **model_kwargs
            )

    def test_max_length_backward_compat_group_beam_search(self):
        # PT-only test: TF doesn't have StoppingCriteria & group beam search
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        bart_model = BartForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-bart").to(
            torch_device
        )
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        batch_size = 1
        max_length = 20
        num_beams = 6
        num_beam_groups = 3
        num_return_sequences = num_beams * batch_size

        input_ids = input_ids.expand(6, -1)
        model_kwargs = bart_model._prepare_encoder_decoder_kwargs_for_generation(input_ids, {})
        input_ids, model_kwargs = bart_model._prepare_decoder_input_ids_for_generation(
            batch_size=input_ids.shape[0],
            model_input_name=bart_model.main_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=bart_model.config.decoder_start_token_id,
            bos_token_id=bart_model.config.bos_token_id,
        )

        diverse_beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=torch_device,
            num_beam_hyps_to_keep=num_return_sequences,
            num_beam_groups=num_beam_groups,
        )
        with self.assertWarns(UserWarning):
            bart_model.group_beam_search(
                input_ids, diverse_beam_scorer, num_beams=num_beams, max_length=max_length, **model_kwargs
            )

    def test_max_length_warning_if_different(self):
        # PT-only test: TF doesn't have StoppingCriteria
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        bart_model = BartForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-bart").to(
            torch_device
        )
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        batch_size = 1

        max_length = 20
        num_beams = 6
        num_beam_groups = 3
        num_return_sequences = num_beams * batch_size
        stopping_criteria_max_length = 18
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=stopping_criteria_max_length)])

        # Greedy
        input_ids = input_ids.expand(6, -1)
        model_kwargs = bart_model._prepare_encoder_decoder_kwargs_for_generation(input_ids, {})
        input_ids, model_kwargs = bart_model._prepare_decoder_input_ids_for_generation(
            batch_size=input_ids.shape[0],
            model_input_name=bart_model.main_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=bart_model.config.decoder_start_token_id,
            bos_token_id=bart_model.config.bos_token_id,
        )

        with self.assertWarns(UserWarning):
            bart_model.greedy_search(
                input_ids,
                max_length=max_length,
                pad_token_id=bart_model.config.pad_token_id,
                stopping_criteria=stopping_criteria,
                eos_token_id=bart_model.config.eos_token_id,
                **model_kwargs,
            )

        # Sample
        with self.assertWarns(UserWarning):
            with torch.no_grad():
                bart_model.sample(
                    input_ids,
                    max_length=max_length,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=bart_model.config.pad_token_id,
                    eos_token_id=bart_model.config.eos_token_id,
                    **model_kwargs,
                )

        # Beam
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=torch_device,
        )
        with self.assertWarns(UserWarning):
            with torch.no_grad():
                bart_model.beam_search(
                    input_ids,
                    num_beams=num_beams,
                    stopping_criteria=stopping_criteria,
                    max_length=max_length,
                    beam_scorer=beam_scorer,
                    **model_kwargs,
                )

        # Grouped beam search
        diverse_beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=torch_device,
            num_beam_hyps_to_keep=num_return_sequences,
            num_beam_groups=num_beam_groups,
        )
        with self.assertWarns(UserWarning):
            bart_model.group_beam_search(
                input_ids,
                diverse_beam_scorer,
                stopping_criteria=stopping_criteria,
                num_beams=num_beams,
                max_length=max_length,
                **model_kwargs,
            )

    def test_custom_stopping_criteria_overload_error(self):
        # PT-only test: TF doesn't have StoppingCriteria
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/bart-tiny-random")
        bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/bart-tiny-random").to(torch_device)

        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)
        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(MaxLengthCriteria(max_length=42))
        with self.assertRaises(ValueError):
            bart_model.generate(input_ids, stopping_criteria=stopping_criteria)
        with self.assertRaises(ValueError):
            bart_model.generate(input_ids, stopping_criteria=stopping_criteria, max_length=32)

    def test_custom_stopping_criteria(self):
        # PT-only test: TF doesn't have StoppingCriteria
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/bart-tiny-random")
        bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/bart-tiny-random").to(torch_device)
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        class DummyCriteria(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                return input_ids.shape[-1] >= 20

        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(DummyCriteria())

        self.assertEqual(
            list(bart_model.generate(input_ids, stopping_criteria=stopping_criteria, max_length=22).shape),
            [1, 20],
        )
        self.assertEqual(
            list(bart_model.generate(input_ids, stopping_criteria=stopping_criteria, max_length=18).shape),
            [1, 18],
        )

    def test_stop_sequence_stopping_criteria(self):
        # PT-only test: TF doesn't have StoppingCriteria
        prompt = """Hello I believe in"""
        generator = pipeline("text-generation", model="hf-internal-testing/tiny-random-bart")
        output = generator(prompt)
        self.assertEqual(
            output,
            [
                {
                    "generated_text": (
                        "Hello I believe in in in number number number number number number number number number"
                    )
                }
            ],
        )

        output = generator(prompt, stop_sequence=" number")
        self.assertEqual(output, [{"generated_text": "Hello I believe in in in number"}])

    def test_generate_non_nlp_input_ids_as_kwarg(self):
        # PT-only test: AFAIK there's no non-NLP model architecture in TF that supports `input_ids` as its only input
        model = ImageGPTForCausalImageModeling.from_pretrained(
            "hf-internal-testing/tiny-random-imagegpt", max_length=10
        ).to(torch_device)
        input_ids = ids_tensor((3, 5), vocab_size=10)

        output_sequences_kwargs = model.generate(input_ids=input_ids).cpu()
        output_sequences = model.generate(input_ids).cpu()

        self.assertListEqual(output_sequences.tolist(), output_sequences_kwargs.tolist())
        self.assertEqual(output_sequences.shape, (3, 10))

    def test_generate_input_values_as_encoder_kwarg(self):
        # PT-only test: AFAIK there's no generate-capable architecture in TF that supports `input_values` as its input
        input_values = floats_tensor((2, 250))
        model = SpeechEncoderDecoderModel.from_pretrained("hf-internal-testing/tiny-random-speech-encoder-decoder")
        model = model.to(torch_device)
        output_sequences_kwargs = model.generate(input_values=input_values, max_length=5).cpu()
        output_sequences = model.generate(input_values, max_length=5).cpu()

        self.assertListEqual(output_sequences.tolist(), output_sequences_kwargs.tolist())
        self.assertEqual(output_sequences.shape, (2, 5))

    def test_transition_scores_group_beam_search_encoder_decoder(self):
        # PT-only test: TF doesn't have group beam search
        articles = [
            "Justin Timberlake and Jessica Biel, welcome to parenthood.",
            "Michael Phelps is arguably the most decorated Olympian of all time.",
        ]
        tokenizer = BartTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        model = BartForConditionalGeneration.from_pretrained(
            "hf-internal-testing/tiny-random-bart",
            max_length=10,
            num_beams=2,
            num_beam_groups=2,
            num_return_sequences=2,
            diversity_penalty=1.0,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_scores=True,
            length_penalty=0.0,
        )
        model = model.to(torch_device)

        input_ids = tokenizer(articles, return_tensors="pt", padding=True).input_ids.to(torch_device)
        outputs = model.generate(input_ids=input_ids)

        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices)
        transition_scores_sum = transition_scores.sum(-1)

        self.assertTrue(torch.allclose(transition_scores_sum, outputs.sequences_scores, atol=1e-3))

    @slow
    def test_beam_search_example_integration(self):
        # PT-only test: TF doesn't have a BeamSearchScorer
        # exactly the example provided in the docstrings of beam search, which previously
        # failed after directly copying from it. Refer to PR #15555
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        encoder_input_str = "translate English to German: How old are you?"
        encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        # lets run beam search using 3 beams
        num_beams = 3
        # define decoder start token ids
        input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        input_ids = input_ids * model.config.decoder_start_token_id

        # add encoder_outputs to model keyword arguments
        model_kwargs = {
            "encoder_outputs": model.get_encoder()(
                encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
            )
        }

        # instantiate beam scorer
        beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=num_beams,
            device=model.device,
        )

        # instantiate logits processors
        logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
            ]
        )

        outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(outputs, ["Wie alt bist du?"])

    @slow
    def test_constrained_beam_search(self):
        # PT-only test: TF doesn't have constrained beam search
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        force_tokens = tokenizer("scared", add_prefix_space=True, add_special_tokens=False).input_ids
        force_tokens_2 = tokenizer("big weapons", add_prefix_space=True, add_special_tokens=False).input_ids

        constraints = [
            PhrasalConstraint(force_tokens),
            PhrasalConstraint(force_tokens_2),
        ]

        starting_text = ["The soldiers were not prepared and"]

        input_ids = tokenizer(starting_text, return_tensors="pt").input_ids.to(torch_device)

        outputs = model.generate(
            input_ids,
            constraints=constraints,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            max_length=30,
            remove_invalid_values=True,
        )

        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The soldiers were not prepared and didn't know what to do. They had no idea how they would react if"
                " the enemy attacked them, big weapons scared"
            ],
        )

    @slow
    def test_constrained_beam_search_mixed(self):
        # PT-only test: TF doesn't have constrained beam search
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        force_phrase = tokenizer("scared", add_prefix_space=True, add_special_tokens=False).input_ids
        flexible_phrases = tokenizer(
            ["scream", "screams", "screaming", "screamed"], add_prefix_space=True, add_special_tokens=False
        ).input_ids

        constraints = [
            PhrasalConstraint(force_phrase),
            DisjunctiveConstraint(flexible_phrases),
        ]

        starting_text = ["The soldiers", "The child"]

        input_ids = tokenizer(starting_text, return_tensors="pt").input_ids.to(torch_device)

        outputs = model.generate(
            input_ids,
            constraints=constraints,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            # max_length=20,
            remove_invalid_values=True,
        )

        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The soldiers, who had been stationed at the base for more than a year before being evacuated"
                " screaming scared",
                "The child was taken to a local hospital where he died.\n 'I don't think screaming scared",
            ],
        )

    @slow
    def test_constrained_beam_search_mixed_mixin(self):
        # PT-only test: TF doesn't have constrained beam search
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        force_word = "scared"
        force_flexible = ["scream", "screams", "screaming", "screamed"]

        force_words_ids = [
            tokenizer([force_word], add_prefix_space=True, add_special_tokens=False).input_ids,
            tokenizer(force_flexible, add_prefix_space=True, add_special_tokens=False).input_ids,
        ]

        starting_text = ["The soldiers", "The child"]

        input_ids = tokenizer(starting_text, return_tensors="pt").input_ids.to(torch_device)

        outputs = model.generate(
            input_ids,
            force_words_ids=force_words_ids,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
        )

        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The soldiers, who had been stationed at the base for more than a year before being evacuated"
                " screaming scared",
                "The child was taken to a local hospital where he died.\n 'I don't think screaming scared",
            ],
        )

    @slow
    def test_cfg_mixin(self):
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        input = tokenizer(["The dragon flew over Paris,"], return_tensors="pt", return_attention_mask=True)
        input["input_ids"] = input["input_ids"].to(torch_device)
        input["attention_mask"] = input["attention_mask"].to(torch_device)

        outputs = model.generate(**input, max_new_tokens=32, guidance_scale=1.5)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The dragon flew over Paris, landing in the Rue de la Bastille. The crowd was so excited "
                'that they had to leave the city.\n\n"We\'re going to Paris!"\n'
            ],
        )

        neg = tokenizer(["France,"], return_tensors="pt", return_attention_mask=True)
        neg["input_ids"] = neg["input_ids"].to(torch_device)
        neg["attention_mask"] = neg["attention_mask"].to(torch_device)
        outputs = model.generate(
            **input,
            max_new_tokens=32,
            guidance_scale=1.5,
            negative_prompt_ids=neg["input_ids"],
            negative_prompt_attention_mask=neg["attention_mask"],
        )
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                'The dragon flew over Paris, landing on the pavement.\n\n"Paris!"\n\n"Paris!"\n\n"'
                'Paris!"\n\n"Paris!"\n\n"Paris!"\n\n'
            ],
        )

    @slow
    def test_constrained_beam_search_example_translation_mixin(self):
        # PT-only test: TF doesn't have constrained beam search
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        encoder_input_str = "translate English to German: How old are you?"
        force_words = ["sind"]

        input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
        force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids

        outputs = model.generate(
            input_ids,
            force_words_ids=force_words_ids,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
        )

        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(outputs, ["Wie alt sind Sie?"])

    @slow
    def test_constrained_beam_search_example_integration(self):
        # PT-only test: TF doesn't have constrained beam search
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        encoder_input_str = "translate English to German: How old are you?"
        encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        # lets run beam search using 5 beams
        num_beams = 5
        # define decoder start token ids
        input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        input_ids = input_ids * model.config.decoder_start_token_id

        # add encoder_outputs to model keyword arguments
        model_kwargs = {
            "encoder_outputs": model.get_encoder()(
                encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
            )
        }

        constraint_str = "sind"
        constraint_token_ids = tokenizer.encode(constraint_str)[:-1]  # remove eos token
        constraints = [PhrasalConstraint(token_ids=constraint_token_ids)]

        # instantiate beam scorer
        beam_scorer = ConstrainedBeamSearchScorer(
            batch_size=1, num_beams=num_beams, device=model.device, constraints=constraints
        )

        # instantiate logits processors
        logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
            ]
        )

        outputs = model.constrained_beam_search(
            input_ids, beam_scorer, constraints=constraints, logits_processor=logits_processor, **model_kwargs
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(outputs, ["Wie alt sind Sie?"])

    def test_constrained_beam_search_mixin_type_checks(self):
        # PT-only test: TF doesn't have constrained beam search
        tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/t5-tiny-random")
        model = AutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/t5-tiny-random")

        encoder_input_str = "translate English to German: How old are you?"
        input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        with self.assertRaises(ValueError):
            force_words = ["sind"]
            force_words_ids = tokenizer(force_words, return_tensors="pt").input_ids
            model.generate(
                input_ids,
                force_words_ids=force_words_ids,
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=1,
                remove_invalid_values=True,
            )

        with self.assertRaises(ValueError):
            force_words = ["sind"]
            force_words_ids = [tokenizer(force_words, return_tensors="pt").input_ids]
            model.generate(
                input_ids,
                force_words_ids=force_words_ids,
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=1,
                remove_invalid_values=True,
            )

        with self.assertRaises(ValueError):
            model.generate(input_ids, force_words_ids=[])

        with self.assertRaises(ValueError):
            model.generate(input_ids, force_words_ids=[[-1]])

        with self.assertRaises(ValueError):
            model.generate(input_ids, force_words_ids=[[[-1]]])

    def test_contrastive_search_batched(self):
        # PT-only test: TF doesn't have constrained beam search
        # Tests that contrastive search works with batched inputs (i.e. has the same output as for non-batched inputs)
        articles = ["Foo", "Bar Baz"]
        tokenizer = BartTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        model = BartForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-bart").to(torch_device)

        model.config.eos_token_id = None
        input_ids_batched = tokenizer(articles, padding=True, return_tensors="pt").input_ids.to(torch_device)
        input_ids = tokenizer(articles[1], return_tensors="pt").input_ids.to(torch_device)

        output_sequences_batched = model.generate(
            input_ids=input_ids_batched, penalty_alpha=0.6, top_k=4, return_dict_in_generate=True, output_scores=True
        )
        output_sequences = model.generate(
            input_ids=input_ids, penalty_alpha=0.6, top_k=4, return_dict_in_generate=True, output_scores=True
        )

        batched_out = tokenizer.decode(output_sequences_batched.sequences[1], skip_special_tokens=True)
        out = tokenizer.decode(output_sequences.sequences[0], skip_special_tokens=True)
        self.assertEqual(batched_out, out)

        # output_sequences_batched.scores[0][1] -> 1st set of logits, 2nd sequence
        max_score_diff = (output_sequences_batched.scores[0][1] - output_sequences.scores[0][0]).abs().max()
        self.assertTrue(max_score_diff < 1e-5)

    def test_eos_token_id_int_and_list_top_k_top_sampling(self):
        # Has TF equivalent: this test relies on random sampling
        generation_kwargs = {
            "do_sample": True,
            "num_beams": 1,
            "top_p": 0.7,
            "top_k": 10,
            "temperature": 0.7,
        }
        expectation = 20

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        text = """Hello, my dog is cute and"""
        tokens = tokenizer(text, return_tensors="pt").to(torch_device)
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)

        # Only some seeds will work both on CPU/GPU for a fixed `expectation` value.
        # The selected seed is not guaranteed to work on all torch versions.
        torch.manual_seed(1)
        eos_token_id = 846
        generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        self.assertTrue(expectation == len(generated_tokens[0]))

        torch.manual_seed(1)
        eos_token_id = [846, 198]
        generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        self.assertTrue(expectation == len(generated_tokens[0]))

    def test_model_kwarg_encoder_signature_filtering(self):
        # Has TF equivalent: ample use of framework-specific code
        bart_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        article = """Hugging Face is a technology company based in New York and Paris."""
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)
        bart_model = BartForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-bart").to(
            torch_device
        )
        output = bart_model.generate(input_ids).cpu().numpy()

        # Let's create a fake model that has a different signature. In particular, this fake model accepts "foo" as an
        # argument. Because "foo" is not in the encoder signature and doesn't start with "decoder_", it will be part of
        # the encoder kwargs prior to signature filtering, which would lead to an exception. But filtering kicks in and
        # saves the day.
        class FakeBart(BartForConditionalGeneration):
            def forward(self, input_ids, foo=None, **kwargs):
                return super().forward(input_ids, **kwargs)

        bart_model = FakeBart.from_pretrained("hf-internal-testing/tiny-random-bart").to(torch_device)
        fake_output = bart_model.generate(input_ids, foo="bar").cpu().numpy()
        self.assertTrue(np.array_equal(output, fake_output))

        # Encoder signature filtering only kicks in if it doesn't accept wildcard kwargs. The following test will fail
        # because it doesn't do signature filtering.
        class FakeEncoder(bart_model.model.encoder.__class__):
            def forward(self, input_ids, **kwargs):
                return super().forward(input_ids, **kwargs)

        fake_encoder = FakeEncoder(bart_model.config, bart_model.model.shared).to(torch_device)
        bart_model.model.encoder = fake_encoder

        # Normal generation still works (the output will be different because the encoder weights are different)
        fake_output = bart_model.generate(input_ids).cpu().numpy()
        with self.assertRaises(TypeError):
            # FakeEncoder.forward() accepts **kwargs -> no filtering -> type error due to unexpected input "foo"
            bart_model.generate(input_ids, foo="bar")

    def test_default_max_length_warning(self):
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model.config.pad_token_id = tokenizer.eos_token_id

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)

        # Default generation config value of 20 -> emits warning
        with self.assertWarns(UserWarning):
            model.generate(input_ids)

        # Explicitly setting max_length to 20 -> no warning
        with warnings.catch_warnings(record=True) as warning_list:
            model.generate(input_ids, max_length=20)
            self.assertEqual(len(warning_list), 0)

        # Generation config max_length != 20 -> no warning
        with warnings.catch_warnings(record=True) as warning_list:
            model.generation_config.max_length = 10
            model.generation_config._from_model_config = False  # otherwise model.config.max_length=20 takes precedence
            model.generate(input_ids)
            self.assertEqual(len(warning_list), 0)
