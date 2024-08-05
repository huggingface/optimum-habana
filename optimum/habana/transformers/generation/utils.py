# coding=utf-8
# Copyright 2022 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import inspect
import math
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache, QuantizedCacheConfig
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.candidate_generator import (
    CandidateGenerator,
    PromptLookupCandidateGenerator,
    _crop_past_key_values,
    _prepare_attention_mask,
    _prepare_token_type_ids,
)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)
from transformers.generation.utils import (
    NEED_SETUP_CACHE_CLASSES_MAPPING,
    QUANT_BACKEND_CLASSES_MAPPING,
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
    GenerateBeamOutput,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
    GenerateOutput,
    GenerationMixin,
    GenerationMode,
    _ranking_fast,
    _split_model_inputs,
    _split_model_outputs,
    stack_model_outputs,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.utils import ModelOutput, is_hqq_available, is_quanto_available, is_torchdynamo_compiling

from optimum.utils import logging

from ...utils import HabanaGenerationtime, HabanaProfile
from ..integrations.deepspeed import unwrap_deepspeed_model
from .candidate_generator import GaudiAssistedCandidateGenerator
from .configuration_utils import GaudiGenerationConfig


if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.streamers import BaseStreamer
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from .candidate_generator import GaudiCandidateGenerator


MODELS_OPTIMIZED_WITH_STATIC_SHAPES = [
    "bloom",
    "gpt2",
    "opt",
    "gptj",
    "gpt_neox",
    "llama",
    "falcon",
    "codegen",
    "gpt_bigcode",
    "bart",
    "mpt",
    "t5",
    "mistral",
    "phi",
    "mixtral",
    "gemma",
    "blip_text_model",
    "seamless_m4t",
    "starcoder2",
    "persimmon",
    "qwen2",
    "llava",
    "llava_next",
    "stablelm",
    "mamba",
    "deci",
]


logger = logging.get_logger(__name__)


def incrementor(bucket_size, prompt_len):
    assert bucket_size > 0
    passnum = -1
    while True:
        passnum += 1
        if passnum == 0:
            token_idx = prompt_len
            allocated_space = int(math.ceil(prompt_len / bucket_size) * bucket_size)
            if prompt_len % bucket_size == 0:
                allocated_space += bucket_size
            need_expansion = True
        else:
            token_idx += 1
            need_expansion = token_idx >= allocated_space
            if need_expansion:
                assert (allocated_space - token_idx) <= bucket_size
                allocated_space += bucket_size
        yield {
            "allocated_space": allocated_space,
            "passnum": passnum,
            "token_idx": token_idx,
            "need_expansion": need_expansion,
        }


def get_final_stopping_criteria(x):
    if isinstance(x, bool):
        return x
    elif torch.is_tensor(x):
        return all(x)
    else:
        raise TypeError(f"The stopping criteria should be either a boolean or a torch.tensor but got {type(x)}.")


class GaudiGenerationMixin(GenerationMixin):
    """
    This class enables to perform fast generation in lazy mode and with HPU graphs.
    The only difference with GenerationMixin is that the various generation
    methods will generate sequences whose size is max_length. Having constant
    sizes allows to make the most of lazy mode and HPU graphs.
    """

    def _get_hpu_graphs_kwargs(self, model_kwargs):
        hpu_graphs_kwargs = {}
        if model_kwargs["limit_hpu_graphs"]:
            hpu_graphs_kwargs.update({"bypass_hpu_graphs": False})
            if "first_token" not in model_kwargs.keys():
                model_kwargs["first_token"] = True
                hpu_graphs_kwargs.update({"bypass_hpu_graphs": True})
        return hpu_graphs_kwargs

    def _prepare_decoder_attention_mask(
        self,
        max_steps: int,  # current stopping criteria
        batch_size: int,
        pad_token_id: int,
        device: str,
        dtype: str = bool,
    ) -> torch.Tensor:
        x = torch.zeros((batch_size, max_steps), device=device, dtype=dtype)
        return x.index_fill(1, torch.tensor(0), 1)  # First the position with pad_token_id

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: torch.Tensor,
        device: torch.device = None,
        max_new_tokens: int = None,
        pad_token_id: int = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        token_idx = model_kwargs.get("token_idx", None)

        # 2. `decoder_start_token_id` must have shape (batch_size, 1)
        if device is None:
            device = self.device
        if token_idx is None:
            if decoder_start_token_id.ndim == 1:
                if decoder_start_token_id.shape[0] != batch_size:
                    raise ValueError(
                        f"`decoder_start_token_id` expected to have length {batch_size} but got {decoder_start_token_id.shape[0]}"
                    )
                decoder_start_token_id = decoder_start_token_id.view(-1, 1)
            else:
                decoder_start_token_id = (
                    torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id
                )
        else:
            # creating padded decoder_input_ids to achieve static shapes. Later new tokens once generated are copied in to decoder_input_ids based on token_idx
            max_length = max_new_tokens + 1 if max_new_tokens is not None else self.generation_config.max_length
            decoder_start_token_id = (
                torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id
            )
            decoder_start_token_id = torch.nn.functional.pad(
                decoder_start_token_id, (0, max_length - 1), value=pad_token_id
            )

        # 3. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_start_token_id
        # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token. Note that the
        # original checkpoints can't be detected through `self.__class__.__name__.lower()`, needing custom logic.
        # See: https://github.com/huggingface/transformers/pull/31470
        elif "donut" in self.__class__.__name__.lower() or (
            self.config.model_type == "vision-encoder-decoder" and "donut" in self.config.encoder.model_type.lower()
        ):
            pass
        elif self.config.model_type in ["whisper"]:
            pass
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]).all().item():
            if token_idx is None:
                decoder_input_ids = torch.cat([decoder_start_token_id, decoder_input_ids], dim=-1)
            else:
                max_length = max_new_tokens + 2 if max_new_tokens is not None else self.generation_config.max_length
                if max_length != decoder_start_token_id.shape[-1]:
                    decoder_start_token_id = torch.nn.functional.pad(
                        decoder_start_token_id,
                        (0, max_length - decoder_start_token_id.shape[-1]),
                        value=pad_token_id,
                    )
                decoder_input_ids = decoder_start_token_id.index_copy(1, token_idx, decoder_input_ids)
                token_idx.add_(1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask
        else:
            if token_idx is not None:
                decoder_input_ids_len = decoder_input_ids.shape[-1]
                max_length = (
                    max_new_tokens + decoder_input_ids_len
                    if max_new_tokens is not None
                    else self.generation_config.max_length
                )
                decoder_input_ids = torch.nn.functional.pad(
                    decoder_input_ids, (0, max_length - decoder_input_ids_len), value=pad_token_id
                )
                token_idx.copy_(decoder_input_ids_len)
                if "decoder_attention_mask" in model_kwargs:
                    decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                    pad_len = max_length - decoder_attention_mask.shape[-1]
                    decoder_attention_mask = torch.cat(
                        (torch.ones_like(decoder_attention_mask)[:, :pad_len], decoder_attention_mask),
                        dim=-1,
                    )
                    model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """
        Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...].

        Copied from Transformers: https://github.com/huggingface/transformers/blob/527ab894e59b6582578008e3b47648a65063f73d/src/transformers/generation/utils.py#L704
        The tensor `token_idx` is not expanded.
        """

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "token_idx"
                    and key != "decoder_input_ids"
                    and key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def _pad_past_key_values(self, model_kwargs):
        pad_amount = model_kwargs.get("kv_cache_pad_len", 0)
        if model_kwargs["past_key_values"]:
            for i in range(len(model_kwargs["past_key_values"])):
                for j in range(len(model_kwargs["past_key_values"][i])):
                    if torch.is_tensor(model_kwargs["past_key_values"][i][j]):
                        model_kwargs["past_key_values"][i][j] = torch.nn.functional.pad(
                            model_kwargs["past_key_values"][i][j], (0, 0, 0, pad_amount)
                        )
                        if model_kwargs.get("lazy_mode", False):
                            self.htcore_generation.mark_step()

    def _remove_past_key_values(self, model_kwargs):
        if model_kwargs["past_key_values"]:
            for i in range(len(model_kwargs["past_key_values"])):
                for j in range(len(model_kwargs["past_key_values"][i])):
                    if torch.is_tensor(model_kwargs["past_key_values"][i][j]):
                        t = model_kwargs["past_key_values"][i][j]
                        del t
                        model_kwargs["past_key_values"][i][j] = None
        del model_kwargs["past_key_values"]
        model_kwargs["past_key_values"] = None

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        """
        Copied from Transformers: https://github.com/huggingface/transformers/blob/527ab894e59b6582578008e3b47648a65063f73d/src/transformers/generation/utils.py#L745

        Adds support for `token_idx`, which is necessary for using static shapes.
        """
        # mark to identify starting from second token
        model_kwargs["first_token"] = False
        if not model_kwargs.get("pad_done", False):
            # update past_key_values keeping its naming used in model code
            cache_name, cache = self._extract_past_from_model_output(
                outputs, standardize_cache_format=standardize_cache_format
            )
            model_kwargs[cache_name] = cache
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        token_idx = model_kwargs.get("token_idx", None)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                if token_idx is not None:
                    attention_mask.index_fill_(1, token_idx, 1)
                else:
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                    )
                model_kwargs["attention_mask"] = attention_mask
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                if token_idx is not None:
                    decoder_attention_mask.index_fill_(1, token_idx, 1)
                else:
                    decoder_attention_mask = torch.cat(
                        [
                            decoder_attention_mask,
                            decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1)),
                        ],
                        dim=-1,
                    )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        if token_idx is not None:
            token_idx.add_(1)
            if "token_idx_cpu" in model_kwargs:
                model_kwargs["token_idx_cpu"] += 1

        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            if model_kwargs.get("use_cache", True):
                model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
            else:
                past_positions = model_kwargs.pop("cache_position")
                new_positions = torch.arange(
                    past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
                ).to(past_positions.device)
                model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))

        return model_kwargs

    @torch.no_grad()
    def update_model_kwargs_for_bucketing(
        self, params, input_ids, model_kwargs, pad_token_id, bucket_size, reduce_recompile=False
    ):
        if params["need_expansion"]:
            # Pad inputs to have static shapes during generation, this gives better performance than dynamic shapes on HPUs
            pad_amount = params["allocated_space"] - input_ids.shape[-1]
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_amount), value=pad_token_id)
            if model_kwargs["attention_mask"] is not None:
                model_kwargs["attention_mask"] = torch.nn.functional.pad(
                    model_kwargs["attention_mask"], (0, pad_amount), value=0
                )
            else:
                assert False, "Not tested for cases where attn_mask isnt passed"
            if reduce_recompile and params["passnum"] == 0:
                position_ids_cpu = model_kwargs["attention_mask"].long().cumsum(-1) - 1
                position_ids_cpu.masked_fill_(model_kwargs["attention_mask"] == 0, 1)
                input_ids = input_ids.to(self.device)
                model_kwargs["attention_mask"] = model_kwargs["attention_mask"].to(self.device)

            if "past_key_values" in model_kwargs:

                def create_pad_arg(pad_amount, i, j):
                    if model_kwargs["past_key_values"][0][0].dim() == 3:
                        assert self.config.model_type == "bloom"
                        if j == 0:
                            return (0, pad_amount)
                        elif j == 1:
                            return (0, 0, 0, pad_amount)
                        else:
                            assert False
                    elif model_kwargs["past_key_values"][0][0].dim() == 4:
                        return (0, 0, 0, pad_amount)  # llama, falcon, qwen2
                    else:
                        assert False, "Unknown case, please handle, or dont use bucketing"

                new_kv = [None for i in range(len(model_kwargs["past_key_values"]))]
                if self.config.model_type == "gpt_bigcode" and model_kwargs["past_key_values"][0][0].dim() == 2:
                    # GPT_BIGCODE's kv cache is list of tensors.
                    new_kv = [None for i in range(len(model_kwargs["past_key_values"]))]
                    for i in range(len(model_kwargs["past_key_values"])):
                        pad = (0, 0, 0, pad_amount)
                        new_kv[i] = torch.nn.functional.pad(
                            model_kwargs["past_key_values"][i], pad, value=pad_token_id
                        )
                    model_kwargs["past_key_values"] = list(new_kv)
                else:
                    for i in range(len(model_kwargs["past_key_values"])):
                        tmp_lst = [None for j in range(len(model_kwargs["past_key_values"][i]))]
                        for j in range(len(model_kwargs["past_key_values"][i])):
                            pad_tuple = create_pad_arg(pad_amount, i, j)
                            # Different models might have different shapes of kv-cache
                            # create_pad_arg handles them on a per-model basis
                            # This is a necessary (but not sufficient) condition: what ever dimension we are padding, should be a multiple of bucket_size
                            # This check is added in case we get a new model with a new kv-cache structure, and we attempt to pad some wrong dimension
                            # in peft case, if there's virtual token. the model_kwargs["past_key_values"][i][j].shape[-(len(pad_tuple) // 2)] % bucket_size == num_virtual_token, no need of assert, the pad length of past_key_value should be aligned with input id and attention_mask
                            num_virtual_tokens = model_kwargs.get("num_virtual_tokens", 0)
                            assert (
                                model_kwargs["past_key_values"][i][j].shape[-(len(pad_tuple) // 2)] % bucket_size
                                == num_virtual_tokens
                            )
                            tmp_lst[j] = torch.nn.functional.pad(
                                model_kwargs["past_key_values"][i][j], pad_tuple, value=pad_token_id
                            )
                        new_kv[i] = tuple(tmp_lst)
                    model_kwargs["past_key_values"] = tuple(new_kv)

        if "token_idx" not in model_kwargs:
            model_kwargs["token_idx"] = torch.tensor(params["token_idx"], device=self.device)
        return input_ids, model_kwargs

    def _get_candidate_generator(
        self,
        generation_config: GaudiGenerationConfig,
        input_ids: torch.LongTensor,
        inputs_tensor: torch.Tensor,
        assistant_model: "PreTrainedModel",
        logits_processor: LogitsProcessorList,
        model_kwargs: Dict,
    ) -> CandidateGenerator:
        if generation_config.prompt_lookup_num_tokens is not None:
            candidate_generator = PromptLookupCandidateGenerator(
                num_output_tokens=generation_config.prompt_lookup_num_tokens,
                max_matching_ngram_size=generation_config.max_matching_ngram_size,
                max_length=generation_config.max_length,
            )
        else:
            candidate_generator = GaudiAssistedCandidateGenerator(
                input_ids=input_ids,
                assistant_model=assistant_model,
                generation_config=generation_config,
                model_kwargs=model_kwargs,
                inputs_tensor=inputs_tensor,
                logits_processor=logits_processor,
            )
        return candidate_generator

    def _get_stopping_criteria(
        self,
        generation_config: GaudiGenerationConfig,
        stopping_criteria: Optional[StoppingCriteriaList],
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        **kwargs,
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if generation_config.max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
        if generation_config.stop_strings is not None:
            if tokenizer is None:
                raise ValueError(
                    "There are one or more stop strings, either in the arguments to `generate` or in the "
                    "model's generation config, but we could not locate a tokenizer. When generating with "
                    "stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`."
                )
            criteria.append(StopStringCriteria(stop_strings=generation_config.stop_strings, tokenizer=tokenizer))
        if not generation_config.ignore_eos and generation_config._eos_token_tensor is not None:
            criteria.append(EosTokenCriteria(eos_token_id=generation_config._eos_token_tensor))
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        has_default_min_length,
        model_input_name,
        input_ids_length,
        inputs_tensor,
        has_token_idx,
    ):
        """Prepared max and min length in generaion configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            if has_token_idx:
                generation_config.max_length = input_ids_length
            else:
                generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        # if both `inputs_embeds` and `input_ids` are passed, we do not correct the length
        # otherwise we need total length [inputs-embeds-len + new-tokens-len] to not go beyond indicated `max_length``
        elif (
            model_input_name == "inputs_embeds"
            and input_ids_length != inputs_tensor.shape[1]
            and not self.config.is_encoder_decoder
        ):
            generation_config.max_length -= inputs_tensor.shape[1]

        # same for min length
        if generation_config.min_new_tokens is not None:
            if not has_default_min_length:
                logger.warning(
                    f"Both `min_new_tokens` (={generation_config.min_new_tokens}) and `min_length`(="
                    f"{generation_config.min_length}) seem to have been set. `min_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            if has_token_idx:
                generation_config.min_length = input_ids_length
            else:
                generation_config.min_length = generation_config.min_new_tokens + input_ids_length

        elif (
            model_input_name == "inputs_embeds"
            and input_ids_length != inputs_tensor.shape[1]
            and not self.config.is_encoder_decoder
        ):
            generation_config.min_length = max(generation_config.min_length - inputs_tensor.shape[1], 0)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[GaudiGenerationConfig], **kwargs: Dict
    ) -> Tuple[GaudiGenerationConfig, Dict]:
        """
        Copied from https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/generation/utils.py#L1230
        Differences:
        - add management of `static_shapes` and `ignore_eos` in the generation config
        - workaround for `token_type_ids` for Falcon
        """
        # TODO joao: when we can detect `fullgraph=True` in `torch.compile` (https://github.com/pytorch/pytorch/pull/120400)
        # replace `is_torchdynamo_compiling` by the corresponding check. As it is, we are being too restrictive with
        # the parameterization in `fullgraph=False` so as to enable `fullgraph=True`.

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # three conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same);
            # 3) the user must have set generation parameters in the model config.
            # NOTE: `torch.compile` can't compile `hash`, this legacy support is disabled with compilation.
            if (
                not is_torchdynamo_compiling()
                and self.generation_config._from_model_config
                and self.generation_config._original_object_hash == hash(self.generation_config)
                and self.config._has_non_default_generation_parameters()
            ):
                new_generation_config = GaudiGenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                    )
                    self.generation_config = new_generation_config
            using_model_generation_config = True
            generation_config = self.generation_config

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled.
        if is_torchdynamo_compiling():
            model_kwargs = kwargs
            generate_attributes_in_kwargs = [
                key for key, value in kwargs.items() if getattr(generation_config, key, None) != value
            ]
            if len(generate_attributes_in_kwargs) > 0:
                raise ValueError(
                    "`torch.compile` exception: all generation configuration attributes must be passed within a "
                    f"`generation_config` instance passed to `generate` (found: {generate_attributes_in_kwargs})."
                )
        else:
            generation_config = copy.deepcopy(generation_config)
            if generation_config.static_shapes is None:
                generation_config.static_shapes = self.config.model_type in MODELS_OPTIMIZED_WITH_STATIC_SHAPES
                if self.config.model_type == "vision-encoder-decoder":
                    generation_config.static_shapes = (
                        self.config.decoder.model_type in MODELS_OPTIMIZED_WITH_STATIC_SHAPES
                    )
            self.generation_config.static_shapes = generation_config.static_shapes
            if generation_config.ignore_eos is None:
                generation_config.ignore_eos = kwargs.get("ignore_eos", kwargs.get("lazy_mode", None))
                self.generation_config.ignore_eos = generation_config.ignore_eos
            model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
            if self.config.model_type == "falcon" and "token_type_ids" in kwargs.keys():
                for key in ["token_type_ids"]:
                    model_kwargs.pop(key, None)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.decoder_start_token_id is None:
                    generation_config.decoder_start_token_id = self.generation_config.decoder_start_token_id

        return generation_config, model_kwargs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GaudiGenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        lazy_mode: Optional[bool] = False,
        hpu_graphs: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        iteration_times: Optional[List[float]] = None,
        profiling_record_shapes: Optional[bool] = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in [`transformers.generation.generation_config`] which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate, e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Most of these parameters are explained in more detail in [this blog
        post](https://huggingface.co/blog/how-to-generate).
        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`transformers.generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
                `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
                generating before other GPUs. Otherwise it'll be set to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            hpu_graphs (`bool`, *optional*, defaults to `False`):
                Whether to use HPU graphs for inference.
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            profiling_record_shapes (`bool`, *optional*, defaults to False):
                Record shapes when enabling profiling.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`transformers.utils.ModelOutput`] or `torch.LongTensor`: A [`transformers.generationutils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.
                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`transformers.generationutils.ModelOutput`] types are:
                    - [`transformers.generation.GenerateDecoderOnlyOutput`],
                    - [`transformers.generation.GenerateBeamDecoderOnlyOutput`]
                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`transformers.generationutils.ModelOutput`] types are:
                    - [`transformers.generation.GenerateEncoderDecoderOutput`],
                    - [`transformers.generation.GenerateBeamEncoderDecoderOutput`]
        """
        if iteration_times is not None:
            hb_gen_time = HabanaGenerationtime(iteration_times=iteration_times)
            hb_gen_time.start()
        else:
            hb_gen_time = None
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        if hpu_graphs and not lazy_mode:
            raise ValueError(
                "`hpu_graphs` is True but `lazy_mode` is False. HPU graphs require `lazy_mode` to be set to True."
            )
        num_virtual_tokens = kwargs.pop("num_virtual_tokens", 0)
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        self.generation_config.max_length = generation_config.max_length

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )

        is_greedy_or_beam_and_bucket = (
            not generation_config.bucket_internal
            and generation_config.bucket_size > 0
            and generation_config.get_generation_mode(assistant_model)
            in [
                GenerationMode.GREEDY_SEARCH,
                GenerationMode.SAMPLE,
                GenerationMode.BEAM_SEARCH,
                GenerationMode.BEAM_SAMPLE,
                GenerationMode.CONTRASTIVE_SEARCH,
            ]
        )
        model_kwargs["bucket_size"] = generation_config.bucket_size if generation_config.static_shapes else -1
        model_kwargs["bucket_internal"] = generation_config.bucket_internal
        model_kwargs["reduce_recompile"] = (
            generation_config.reduce_recompile if generation_config.reduce_recompile is not None else False
        )
        if model_kwargs["reduce_recompile"]:
            assert generation_config.bucket_size
        # Below condition checked explicitly since llama supports bucket_internal even without reuse_cache
        if generation_config.bucket_internal:
            assert generation_config.bucket_size >= 0, "please set bucket_size to use bucket_internal"
        if generation_config.reuse_cache:
            assert self.config.model_type in [
                "llama",
                "mistral",
                "falcon",
                "mixtral",
                "phi",
                "qwen2",
                "gptj",
            ], "reuse_cache only supported by llama, mistral, falcon, mixtral, phi, qwen2 and gptj at the moment"
            if not generation_config.bucket_internal:
                assert (
                    generation_config.bucket_size <= 0
                ), "please set bucket_internal along with reuse_cache and bucket_size"
            else:
                assert generation_config.bucket_size >= 0, "please set valid bucket_size to use bucket_internal"

        if generation_config.static_shapes:
            # Pad inputs to have static shapes during generation, this gives better performance than dynamic shapes on HPUs
            # In encoder_decoder models, Inputs are already padded

            if not self.config.is_encoder_decoder:
                # only pad if bucket_size < -1. If we are bucketing (bucket_size > 0), then that is taken care in greedy_search()
                if not is_greedy_or_beam_and_bucket:
                    # token_idx is the current index in the generation process, it is incremented each time a new token is generated
                    token_idx = inputs_tensor.shape[-1]
                    model_kwargs["token_idx"] = torch.tensor(token_idx, device=inputs_tensor.device)
                    model_kwargs["token_idx_cpu"] = token_idx
                    if generation_config.max_new_tokens is None:
                        generation_config.max_new_tokens = generation_config.max_length - token_idx
                    inputs_tensor = torch.nn.functional.pad(
                        inputs_tensor, (0, generation_config.max_new_tokens), value=generation_config.pad_token_id
                    )
                    for other_inputs in ["attention_mask", "token_type_ids"]:
                        if model_kwargs.get(other_inputs) is not None:
                            model_kwargs[other_inputs] = torch.nn.functional.pad(
                                model_kwargs[other_inputs], (0, generation_config.max_new_tokens), value=0
                            )
            else:
                assert generation_config.bucket_size <= 0, "Untested path for bucket>0"
                token_idx = 1
                model_kwargs["token_idx"] = torch.tensor(token_idx, device=inputs_tensor.device)
                if model_kwargs.get("decoder_attention_mask", None) is None and generation_config.use_cache:
                    max_length = (
                        generation_config.max_new_tokens + 1
                        if generation_config.max_new_tokens is not None
                        else generation_config.max_length
                    )
                    model_kwargs["decoder_attention_mask"] = self._prepare_decoder_attention_mask(
                        max_length,
                        inputs_tensor.shape[0],
                        generation_config.pad_token_id,
                        inputs_tensor.device,
                    )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
                max_new_tokens=generation_config.max_new_tokens,
                pad_token_id=generation_config.pad_token_id,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
            has_token_idx="token_idx" in model_kwargs,
        )

        use_dynamic_cache_by_default = False
        if "mamba" in self.__class__.__name__.lower():
            cache_name = "cache_params"
        else:
            cache_name = "past_key_values"
        if generation_config.cache_implementation is not None and (model_kwargs.get(cache_name) is not None):
            raise ValueError(
                f"Passing both `cache_implementation` (used to initialize certain caches) and `{cache_name}` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                model_kwargs[cache_name] = self._get_cache(
                    generation_config.cache_implementation,
                    getattr(generation_config, "num_beams", 1) * batch_size,
                    generation_config.max_length,
                    model_kwargs,
                )
            elif generation_config.cache_implementation == "quantized":
                if not self._supports_quantized_cache:
                    raise ValueError(
                        "This model does not support the quantized cache. If you want your model to support quantized "
                        "cache, please open an issue."
                    )

                cache_config = (
                    generation_config.cache_config
                    if generation_config.cache_config is not None
                    else QuantizedCacheConfig()
                )
                cache_class = QUANT_BACKEND_CLASSES_MAPPING[cache_config.backend]

                if cache_config.backend == "quanto" and not is_quanto_available():
                    raise ImportError(
                        "You need to install `quanto` in order to use KV cache quantization with quanto backend. "
                        "Please install it via  with `pip install quanto`"
                    )
                elif cache_config.backend == "HQQ" and not is_hqq_available():
                    raise ImportError(
                        "You need to install `HQQ` in order to use KV cache quantization with HQQ backend. "
                        "Please install it via  with `pip install hqq`"
                    )

                model_kwargs[cache_name] = cache_class(cache_config)
        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        # elif generation_config.cache_implementation is None and self._supports_default_dynamic_cache():
        #     past = model_kwargs.get(cache_name, None)
        #     requires_cross_attention_cache = (
        #         self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
        #     )
        #     if past is None:
        #         model_kwargs[cache_name] = (
        #             DynamicCache()
        #             if not requires_cross_attention_cache
        #             else EncoderDecoderCache(DynamicCache(), DynamicCache())
        #         )
        #         use_dynamic_cache_by_default = True
        #     elif isinstance(past, tuple):
        #         model_kwargs[cache_name] = (
        #             DynamicCache.from_legacy_cache(past)
        #             if not requires_cross_attention_cache
        #             else EncoderDecoderCache.from_legacy_cache(past)
        #         )
        #         use_dynamic_cache_by_default = True

        self._validate_generated_length(
            generation_config,
            model_kwargs["token_idx"].item() if "token_idx" in model_kwargs else input_ids_length,
            has_default_max_length,
        )

        # determine whether introduce trim_logits feature
        model_kwargs["trim_logits"] = generation_config.trim_logits

        # determine whether attention softmax needs to execute in lower precision
        model_kwargs["attn_softmax_bf16"] = generation_config.attn_softmax_bf16

        # determine whether limit_hpu_graphs needs to be used
        model_kwargs["use_hpu_graphs"] = hpu_graphs
        model_kwargs["limit_hpu_graphs"] = generation_config.limit_hpu_graphs

        # prepare for allocate kv cache
        model_kwargs["reuse_cache"] = generation_config.reuse_cache

        # determine whether flash attention needs to be used
        model_kwargs["use_flash_attention"] = generation_config.use_flash_attention
        model_kwargs["flash_attention_recompute"] = True if generation_config.flash_attention_recompute else False
        model_kwargs["flash_attention_causal_mask"] = True if generation_config.flash_attention_causal_mask else False
        model_kwargs["flash_attention_fast_softmax"] = (
            True if generation_config.flash_attention_fast_softmax else False
        )
        model_kwargs["num_virtual_tokens"] = num_virtual_tokens

        if not self.config.is_encoder_decoder:
            calculated_max_length = input_ids.shape[-1] + num_virtual_tokens
            if not generation_config.static_shapes and generation_config.max_new_tokens is not None:
                calculated_max_length = input_ids.shape[-1] + generation_config.max_new_tokens + num_virtual_tokens
            if generation_config.use_cache and generation_config.reuse_cache:
                bs, _ = input_ids.shape
                if not is_greedy_or_beam_and_bucket:
                    unwrap_deepspeed_model(self).allocate_kv_cache(
                        bs * generation_config.num_beams, calculated_max_length, token_idx + num_virtual_tokens
                    )
            if generation_config.use_cache:
                model_kwargs["kv_cache_len"] = calculated_max_length
                model_kwargs["kv_cache_pad_len"] = generation_config.max_new_tokens

            if self.config.model_type in ["llama", "falcon", "mistral", "qwen2", "gptj"]:
                if self.config.max_position_embeddings < calculated_max_length:
                    unwrap_deepspeed_model(self).update_sincos_cache(seq_len=calculated_max_length)

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if generation_config.bucket_size > 0:
            assert generation_config.static_shapes, "bucket_size > 0 can be set only when static_shapes is set"
        # if generation_config.bucket_size <= 0, padding is handled by the generating fn (like greedy_search)
        if generation_config.static_shapes and generation_config.bucket_size > 0:
            assert generation_mode in [
                GenerationMode.GREEDY_SEARCH,
                GenerationMode.SAMPLE,
                GenerationMode.BEAM_SEARCH,
                GenerationMode.BEAM_SAMPLE,
                GenerationMode.CONTRASTIVE_SEARCH,
            ], "generation_config.bucket_size > 0 supported only for greedy mode"

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                (
                    "You are calling .generate() with the `input_ids` being on a device type different"
                    f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                    f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                    " Please make sure that you have put `input_ids` to the"
                    f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                    " running `.generate()`."
                ),
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        self.generation_config.generation_mode = generation_mode
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=tokenizer,
            **kwargs,
        )

        # In lazy mode, import Habana torch to be able to add mark_step()
        if lazy_mode:
            import habana_frameworks.torch.core as htcore

            self.htcore_generation = htcore

        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")
            if generation_config.cache_implementation == "static":
                raise ValueError("assisted generate is not supported with `static_cache`")
            if self._is_stateful:
                # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
                # which is not possible with stateful models (they can't reset to a previous subset of generated text)
                raise ValueError(
                    f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
                )

            # 11. Get the candidate generator, given the parameterization
            candidate_generator = self._get_candidate_generator(
                generation_config=generation_config,
                input_ids=input_ids,
                inputs_tensor=inputs_tensor,
                assistant_model=assistant_model,
                logits_processor=logits_processor,
                model_kwargs=model_kwargs,
            )

            # 12. prepare logits warper (if `do_sample` is `True`)
            prepared_logits_warper = (
                self._get_logits_warper(
                    generation_config,
                    device=input_ids.device,
                )
                if generation_config.do_sample
                else None
            )

            # 13. run assisted generate
            result = self._assisted_decoding(
                input_ids,
                candidate_generator=candidate_generator,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                lazy_mode=lazy_mode,
                ignore_eos=generation_config.ignore_eos,
                profiling_warmup_steps=profiling_warmup_steps,
                profiling_steps=profiling_steps,
                hb_gen_time=hb_gen_time,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.DOLA_GENERATION:
            if self._is_stateful:
                # DoLa decoding was not designed for stateful models, and would require some changes
                raise ValueError(
                    f"dola decoding is not supported with stateful models, such as {self.__class__.__name__}"
                )
            prepared_logits_warper = (
                self._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )
            result = self._dola_decoding(
                input_ids,
                dola_layers=generation_config.dola_layers,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")
            if self._is_stateful:
                # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
                raise ValueError(
                    f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}"
                )

            result = self._contrastive_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                lazy_mode=lazy_mode,
                ignore_eos=generation_config.ignore_eos,
                profiling_warmup_steps=profiling_warmup_steps,
                profiling_steps=profiling_steps,
                hb_gen_time=hb_gen_time,
                profiling_record_shapes=profiling_record_shapes,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. prepare logits warper
            prepared_logits_warper = (
                self._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                lazy_mode=lazy_mode,
                ignore_eos=generation_config.ignore_eos,
                profiling_warmup_steps=profiling_warmup_steps,
                profiling_steps=profiling_steps,
                hb_gen_time=hb_gen_time,
                profiling_record_shapes=profiling_record_shapes,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
            # 11. prepare logits warper
            prepared_logits_warper = (
                self._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )

            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            result = self._beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                lazy_mode=lazy_mode,
                profiling_warmup_steps=profiling_warmup_steps,
                profiling_steps=profiling_steps,
                hb_gen_time=hb_gen_time,
                profiling_record_shapes=profiling_record_shapes,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                lazy_mode=lazy_mode,
                profiling_warmup_steps=profiling_warmup_steps,
                profiling_steps=profiling_steps,
                hb_gen_time=hb_gen_time,
                profiling_record_shapes=profiling_record_shapes,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                lazy_mode=lazy_mode,
                profiling_warmup_steps=profiling_warmup_steps,
                profiling_steps=profiling_steps,
                hb_gen_time=hb_gen_time,
                profiling_record_shapes=profiling_record_shapes,
                **model_kwargs,
            )

        # Convert to legacy cache if needed
        if use_dynamic_cache_by_default and generation_config.return_legacy_cache:
            if isinstance(result, ModelOutput) and hasattr(result, "past_key_values"):
                if isinstance(result.past_key_values, (DynamicCache, EncoderDecoderCache)):
                    result.past_key_values = result.past_key_values.to_legacy_cache()

        return result

    def _dola_decoding(
        self,
        input_ids: torch.LongTensor,
        dola_layers: Union[str, List[int]],
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GaudiGenerationConfig,
        synced_gpus: bool,
        streamer: "BaseStreamer",
        logits_warper: Optional[LogitsProcessorList],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **dola decoding** and can be
        used for decoder-only text models.
        The method is based on the paper "DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language
        Models" (https://arxiv.org/abs/2309.03883) in ICLR 2024.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            dola_layers (`Union[str, List[int]]`):
                The candidate layers used in contrasting layers of DoLa. It can be either 1) 'low' or 'high', which
                means the lower part or higher part of the model layers, respectively, or 2) a list of layer indices
                to be used for candidate layers. The 0-th layer is the word embedding layer of the model.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """

        raise NotImplementedError("Dola decoding is not supported by optimum-habana yet.")

    @torch.no_grad()
    def _contrastive_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GaudiGenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        lazy_mode: Optional[bool] = False,
        ignore_eos: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        hb_gen_time: Optional[HabanaGenerationtime] = None,
        profiling_record_shapes: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **contrastive search** and can
        be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Adapted from: https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/generation/utils.py#L2453

        The changes are:
        - support lazy mode and HPU graphs on Gaudi
        - support static shapes and bucketing

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            ignore_eos (`bool`, *optional*, defaults to `False`):
                Whether to ignore finished sequences (faster in lazy mode and with HPU graphs) or not (eager mode).
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            profiling_record_shapes (`bool`, *optional*, defaults to False):
                Record shapes when enabling profiling.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`transformers.generation.GenerateDecoderOnlyOutput`],
            [`transformers.generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`: A `torch.LongTensor`
            containing the generated tokens (default behaviour) or a
            [`transformers.generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`transformers.generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        top_k = generation_config.top_k
        penalty_alpha = generation_config.penalty_alpha
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]

        if not ignore_eos:
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
            model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        this_peer_finished = False

        hb_profer = HabanaProfile(
            warmup=profiling_warmup_steps, active=profiling_steps, record_shapes=profiling_record_shapes
        )
        hb_profer.start()
        bucket_size = model_kwargs.get("bucket_size", -1)
        prev_idx = -1  # avoiding calculate cache_idx when its value is not changing
        bucket_internal = model_kwargs.get("bucket_internal", None)
        reduce_recompile = model_kwargs.get("reduce_recompile", False)

        if not bucket_internal:
            if bucket_size >= 0:
                inc = iter(incrementor(bucket_size, cur_len))
            if bucket_size > 0:
                assert "position_ids" not in model_kwargs, "Untested path"

        token_idx = model_kwargs.get("token_idx", None)
        top_k_ids = None
        if token_idx is not None:
            # Update cur_len in case of static shapes
            cur_len = token_idx.item()

        time_to_first_token_done = False
        model_kwargs["pad_done"] = False
        model_kwargs["lazy_mode"] = lazy_mode

        batch_indices = torch.arange(batch_size, device=input_ids.device)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if lazy_mode:
                self.htcore_generation.mark_step()

            if bucket_size > 0 and not bucket_internal:
                # it will not have been padded if bucket_size > 0
                params = next(inc)
                input_ids, model_kwargs = self.update_model_kwargs_for_bucketing(
                    params, input_ids, model_kwargs, pad_token_id, bucket_size, reduce_recompile
                )

            # if the first step in the loop, encode all the prefix and obtain: (1) past_key_values;
            # (2) last_hidden_states; (3) logit_for_next_step; (4) update model kwargs for the next step
            if model_kwargs.get("past_key_values") is None or (
                isinstance(model_kwargs["past_key_values"], (Cache, EncoderDecoderCache))
                and model_kwargs["past_key_values"].get_seq_length() == 0
            ):
                # prepare inputs
                model_kwargs["use_cache"] = True
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                hpu_graphs_kwargs = self._get_hpu_graphs_kwargs(model_kwargs)

                # encode the given prefix and prepare model inputs; encoder-decoder model process the prefix and save
                # the `encoder_outputs`
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                    **hpu_graphs_kwargs,
                )

                # last decoder hidden states will be used to compute the degeneration penalty (cosine similarity with
                # previous tokens)
                if self.config.is_encoder_decoder:
                    last_hidden_states = outputs.decoder_hidden_states[-1]
                else:
                    last_hidden_states = outputs.hidden_states[-1]

                # next logit for contrastive search to select top-k candidate tokens
                token_idx = model_kwargs.get("token_idx", None)
                if token_idx is not None and outputs.logits.shape[-2] > 1:
                    last_hidden_states = last_hidden_states[:, :token_idx, :]
                    # case1 (w/o KV caching): outputs.logits.shape: [batch_size, max_length, vocab_size]
                    if self.config.is_encoder_decoder:
                        logit_for_next_step = outputs.logits[:, token_idx - 1, :]
                    else:
                        logit_for_next_step = torch.index_select(outputs.logits, -2, token_idx - 1).squeeze(-2)
                else:
                    # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for this first iteration
                    # (the clone itself is always small)
                    logit_for_next_step = outputs.logits[:, -1, :].clone()

                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    standardize_cache_format=True,
                )

                if not sequential:
                    # Expands model inputs top_k times, for batched forward passes (akin to beam search).
                    _, model_kwargs = self._expand_inputs_for_generation(
                        expand_size=top_k, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
                    )

                past_key_values = model_kwargs.get("past_key_values")
                if past_key_values is None:
                    raise ValueError(
                        f"{self.__class__.__name__} does not support caching and therefore **can't** be used "
                        "for contrastive search."
                    )
                elif (
                    (
                        not isinstance(past_key_values[0], (tuple, torch.Tensor))
                        and not isinstance(past_key_values[0], (list, torch.Tensor))
                    )  # Added list type to support GaudiLlamaForCausalLM
                    or past_key_values[0][0].shape[0] != batch_size
                ):
                    raise ValueError(
                        f"{self.__class__.__name__} does not have a standard cache format and therefore **can't** be "
                        "used for contrastive search without further modifications."
                    )

                if lazy_mode:
                    self.htcore_generation.mark_step()

            # contrastive_search main logic start:
            # contrastive search decoding consists of two steps: (1) candidate tokens recall; (2) candidate re-rank by
            # degeneration penalty
            if token_idx is not None and self.config.is_encoder_decoder:
                processed_logit_for_next_step = logits_processor(input_ids[:, :token_idx], logit_for_next_step)
            else:
                processed_logit_for_next_step = logits_processor(input_ids, logit_for_next_step)

            next_probs = torch.nn.functional.softmax(processed_logit_for_next_step, dim=-1)

            if token_idx is not None:
                if top_k_ids is None:
                    top_k_ids = torch.full(
                        (batch_size, top_k, input_ids.shape[-1]), pad_token_id, dtype=torch.int64
                    ).to(input_ids.device)
                elif bucket_size > 0 and not bucket_internal:
                    if input_ids.shape[-1] > top_k_ids.shape[-1]:  # needs expansion
                        pad_amount = input_ids.shape[-1] - top_k_ids.shape[-1]
                        top_k_ids = torch.nn.functional.pad(top_k_ids, (0, pad_amount), value=pad_token_id)

                top_k_probs, top_k_prob_ids = torch.topk(next_probs, dim=-1, k=top_k)
                top_k_ids[:, :, token_idx - 1] = top_k_prob_ids
            else:
                top_k_probs, top_k_ids = torch.topk(next_probs, dim=-1, k=top_k)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_logits:
                    raw_logits += (logit_for_next_step,)
                if output_scores:
                    scores += (processed_logit_for_next_step,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # This is needed to properly delete outputs.logits which may be very large for this first iteration
            # Otherwise a reference to outputs.logits is kept all along until after the next call to self.forward()
            del outputs

            if not sequential:
                # Replicates the new past_key_values to match the `top_k` candidates
                past = model_kwargs["past_key_values"]
                # If it is a static cache, modify it in-place layer after layer to save memory
                if isinstance(past, DynamicCache) or (
                    isinstance(past, EncoderDecoderCache) and isinstance(past.self_attention_cache, DynamicCache)
                ):
                    past.batch_repeat_interleave(top_k)
                else:
                    new_key_values = []
                    for layer in past:
                        items = []
                        # item is either the key or the value matrix
                        for item in layer:
                            items.append(item.repeat_interleave(top_k, dim=0))
                        new_key_values.append(tuple(items))

                    past = tuple(new_key_values)

                model_kwargs["past_key_values"] = past

            if sequential:
                all_outputs = []
                for i in range(top_k):
                    # compute the candidate tokens by the language model and collect their hidden_states
                    if token_idx is not None:
                        next_model_inputs = self.prepare_inputs_for_generation(
                            top_k_ids[:, i, :].view(-1, input_ids.shape[-1]), **model_kwargs
                        )
                    else:
                        next_model_inputs = self.prepare_inputs_for_generation(
                            top_k_ids[:, i].view(-1, 1), **model_kwargs
                        )

                    outputs = self(
                        **next_model_inputs,
                        return_dict=True,
                        output_hidden_states=True,
                        output_attentions=output_attentions,
                    )
                    if isinstance(outputs["past_key_values"], DynamicCache) or (
                        isinstance(outputs["past_key_values"], EncoderDecoderCache)
                        and isinstance(outputs["past_key_values"].self_attention_cache, DynamicCache)
                    ):
                        # Remove past K-V from output since we don't need to stack later
                        outputs["past_key_values"] = None
                        # Remove last token from past K-V since we don't want to append it at this point
                        model_kwargs["past_key_values"].crop(-1)

                    all_outputs.append(outputs)
                outputs = stack_model_outputs(all_outputs)

            else:
                # compute the candidate tokens by the language model and collect their hidden_states
                # assembles top_k_ids into batch of size k
                if token_idx is not None:
                    next_model_inputs = self.prepare_inputs_for_generation(
                        top_k_ids.view(-1, input_ids.shape[-1]), **model_kwargs
                    )
                else:
                    next_model_inputs = self.prepare_inputs_for_generation(top_k_ids.view(-1, 1), **model_kwargs)

                outputs = self(
                    **next_model_inputs,
                    return_dict=True,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                )

            # This is essential to avoid having a last reference to the big past K-V and double the necesary memory
            # in the next loop
            del next_model_inputs

            # name is different for encoder-decoder and decoder-only models
            if self.config.is_encoder_decoder:
                next_hidden = outputs.decoder_hidden_states[-1]
                full_hidden_states = outputs.decoder_hidden_states
            else:
                next_hidden = outputs.hidden_states[-1]
                full_hidden_states = outputs.hidden_states

            logits = outputs.logits[:, -1, :]
            context_hidden = last_hidden_states.repeat_interleave(top_k, dim=0)

            # compute the degeneration penalty and re-rank the candidates based on the degeneration penalty and the
            # model confidence. Keeping `selected_idx` on CPU enables multi-device contrastive search and doesn't
            # introduce (noticeable) slowdowns on single-device runs.
            selected_idx = _ranking_fast(context_hidden, next_hidden, top_k_probs, penalty_alpha, top_k)

            # This will be used instead of the previous inneficient torch.stack(torch.split())
            augmented_idx = torch.tensor(
                [x + i * top_k for i, x in enumerate(selected_idx)], device=selected_idx.device
            )

            # prepare for the next step: (1) next token_id; (2) past_key_values; (3) last_hidden_states for computing
            # the degeneration penalty; (4) logits for selecting next top-k candidates; (5) selected tokens scores
            # (model confidence minus degeneration penalty); (6) decoder hidden_states
            top_k_indices = torch.arange(len(top_k_ids), device=input_ids.device)
            if token_idx is not None:
                next_tokens = top_k_ids[top_k_indices, selected_idx, token_idx - 1]
            else:
                next_tokens = top_k_ids[top_k_indices, selected_idx]
            next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), top_k))
            next_hidden = next_hidden[batch_indices, selected_idx, :]
            last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)

            next_decoder_hidden_states = ()
            for layer in full_hidden_states:
                layer = torch.stack(torch.split(layer, top_k))[batch_indices, selected_idx, :]
                next_decoder_hidden_states += (layer,)

            # generate past_key_values cache of only the selected token
            if sequential:
                if token_idx is not None:
                    next_model_input = self.prepare_inputs_for_generation(
                        top_k_ids[:, selected_idx, :].view(-1, input_ids.shape[-1]), **model_kwargs
                    )
                else:
                    next_model_input = self.prepare_inputs_for_generation(
                        top_k_ids[:, selected_idx].view(-1, 1), **model_kwargs
                    )

                selected_outputs = self(
                    **next_model_input,
                    return_dict=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                next_past_key_values = selected_outputs["past_key_values"]

            else:
                _, next_past_key_values = self._extract_past_from_model_output(outputs, standardize_cache_format=True)
                # Do it in-place layer per layer to save memory
                if isinstance(next_past_key_values, DynamicCache) or (
                    isinstance(next_past_key_values, EncoderDecoderCache)
                    and isinstance(next_past_key_values.self_attention_cache, DynamicCache)
                ):
                    next_past_key_values.batch_select_indices(augmented_idx)
                else:
                    new_key_values = []
                    for layer in next_past_key_values:
                        items = []
                        # item is either the key or the value matrix
                        for item in layer:
                            items.append(item[augmented_idx, ...])
                        new_key_values.append(tuple(items))

                next_past_key_values = tuple(new_key_values)

            logit_for_next_step = torch.stack(torch.split(logits, top_k))[batch_indices, selected_idx, :]

            # Rebuilds the relevant parts of the model output for the selected token, for use in the next iteration
            if self.config.is_encoder_decoder:
                next_step_cross_attentions = ()
                next_step_decoder_attentions = ()
                if output_attentions:
                    for layer in outputs.cross_attentions:
                        layer = torch.stack(torch.split(layer, top_k, dim=0))[batch_indices, selected_idx, ...]
                        next_step_cross_attentions += (layer,)
                    for layer in outputs.decoder_attentions:
                        layer = torch.stack(torch.split(layer, top_k, dim=0))[batch_indices, selected_idx, ...]
                        next_step_decoder_attentions += (layer,)
                outputs = Seq2SeqLMOutput(
                    past_key_values=next_past_key_values,
                    decoder_hidden_states=next_decoder_hidden_states,
                    decoder_attentions=next_step_decoder_attentions or None,
                    cross_attentions=next_step_cross_attentions or None,
                )
            else:
                next_step_attentions = ()
                if output_attentions:
                    for layer in outputs.attentions:
                        layer = torch.stack(torch.split(layer, top_k, dim=0))[batch_indices, selected_idx, ...]
                        next_step_attentions += (layer,)
                outputs = CausalLMOutputWithPast(
                    past_key_values=next_past_key_values,
                    hidden_states=next_decoder_hidden_states,
                    attentions=next_step_attentions or None,
                )
            # contrastive_search main logic end

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # finished sentences should have their next token be a padding token
            if not ignore_eos and has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            if token_idx is not None:
                # Use token_idx-1 since token index is incremented twice in first iteration
                input_ids.index_copy_(
                    1, token_idx - 1, next_tokens.unsqueeze(-1) if next_tokens.dim() == 1 else next_tokens
                )
            else:
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # increase cur_len
            cur_len = cur_len + 1
            if bucket_size > 0 and bucket_internal:
                # Calculate slice idx for kv cache during the decode phase.
                # Breaking down the kv cache in the attention block helps to reduce computation time.
                if model_kwargs.get("token_idx_cpu") <= (model_kwargs["kv_cache_len"] // bucket_size) * bucket_size:
                    idx = (model_kwargs.get("token_idx_cpu") - 1) // bucket_size
                    if prev_idx != idx:
                        model_kwargs["cache_idx"] = (idx + 1) * bucket_size
                        prev_idx = idx
                else:
                    model_kwargs["cache_idx"] = model_kwargs["kv_cache_len"]

            # stop when each sentence is finished
            if ignore_eos:
                this_peer_finished = stopping_criteria(
                    input_ids,
                    scores,
                    token_idx=cur_len,
                    ignore_eos=ignore_eos,
                    eos_token_id=generation_config.eos_token_id,
                )
            else:
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                    input_ids,
                    scores,
                    token_idx=cur_len,
                    ignore_eos=ignore_eos,
                    eos_token_id=generation_config.eos_token_id,
                )
                this_peer_finished = unfinished_sequences.max() == 0

            if (
                not model_kwargs.get("pad_done", False)
                and not model_kwargs.get("reuse_cache", False)
                and bucket_internal
            ):
                # Pad the returned pask key values tensors from prefill phase forward run to maximum length
                # before starting the decode phase.
                self._pad_past_key_values(model_kwargs)
                model_kwargs["pad_done"] = True

            hb_profer.step()

            if hb_gen_time is not None:
                if not time_to_first_token_done:
                    time_to_first_token_done = True
                    import habana_frameworks.torch.hpu as torch_hpu

                    torch_hpu.synchronize()
                hb_gen_time.step()

        if (
            model_kwargs.get("use_hpu_graphs", False)
            and model_kwargs.get("limit_hpu_graphs", False)
            and not model_kwargs.get("reuse_cache", False)
            and bucket_internal
        ):
            # Clear HPU graphs input tensors of the decode phase after the full generation while loop
            self.clear_inputs()
            # Delete past key value tensors
            self._remove_past_key_values(model_kwargs)

        hb_profer.stop()
        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            # Contrastive search works by forward looking at the next token, so we need to exclude it from
            # `past_key_values` to be consistent with the other decoding methods
            if model_kwargs.get("past_key_values") is not None:
                if isinstance(model_kwargs["past_key_values"], DynamicCache) or (
                    isinstance(model_kwargs["past_key_values"], EncoderDecoderCache)
                    and isinstance(model_kwargs["past_key_values"].self_attention_cache, DynamicCache)
                ):
                    model_kwargs["past_key_values"].crop(-1)
                else:
                    past_key_values = []
                    for layer in model_kwargs["past_key_values"]:
                        layer_past_key_values = []
                        for item in layer:
                            layer_past_key_values.append(item[..., :-1, :])
                        past_key_values.append(tuple(layer_past_key_values))
                    model_kwargs["past_key_values"] = tuple(past_key_values)

            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
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
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GaudiGenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        logits_warper: Optional[LogitsProcessorList],
        lazy_mode: Optional[bool] = False,
        ignore_eos: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        hb_gen_time: Optional[HabanaGenerationtime] = None,
        profiling_record_shapes: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`GaudiGenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            ignore_eos (`bool`, *optional*, defaults to `False`):
                Whether to ignore finished sequences (faster in lazy mode and with HPU graphs) or not (eager mode).
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            profiling_record_shapes (`bool`, *optional*, defaults to False):
                Record shapes when enabling profiling.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`transformers.generation.GenerateDecoderOnlyOutput`], [`transformers.generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`transformers.generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`transformers.generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        bucket_size = model_kwargs.get("bucket_size", -1)
        prev_idx = -1  # avoiding calculate cache_idx when its value is not changing
        bucket_internal = model_kwargs.get("bucket_internal", None)
        reduce_recompile = model_kwargs.get("reduce_recompile", False)

        hb_profer = HabanaProfile(
            warmup=profiling_warmup_steps, active=profiling_steps, record_shapes=profiling_record_shapes
        )
        hb_profer.start()

        if not bucket_internal:
            if bucket_size >= 0:
                inc = iter(incrementor(bucket_size, cur_len))
            if bucket_size > 0:
                assert "position_ids" not in model_kwargs, "Untested path"

        token_idx = model_kwargs.get("token_idx", None)
        if token_idx is not None:
            # Update cur_len in case of static shapes
            cur_len = token_idx.item()

        time_to_first_token_done = False
        model_kwargs["pad_done"] = False
        model_kwargs["lazy_mode"] = lazy_mode
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if lazy_mode:
                self.htcore_generation.mark_step()

            if bucket_size > 0 and not bucket_internal:
                # it will not have been padded if bucket_size > 0
                params = next(inc)
                input_ids, model_kwargs = self.update_model_kwargs_for_bucketing(
                    params, input_ids, model_kwargs, pad_token_id, bucket_size, reduce_recompile
                )

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            hpu_graphs_kwargs = self._get_hpu_graphs_kwargs(model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                **hpu_graphs_kwargs,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            token_idx = model_kwargs.get("token_idx", None)
            if token_idx is not None and outputs.logits.shape[-2] > 1:
                # case1 (w/o KV caching): outputs.logits.shape: [batch_size, max_length, vocab_size]
                if self.config.is_encoder_decoder:
                    next_token_logits = outputs.logits[:, token_idx - 1, :]
                    next_token_scores = logits_processor(input_ids[:, :token_idx], next_token_logits)
                else:
                    if model_kwargs.get("num_virtual_tokens", 0) > 0:
                        # for prompt tuning, the output logit shape > model_inputs["input_ids"].shape[-1]
                        if model_kwargs.get("reuse_cache", False):
                            output_idx = torch.tensor(outputs.logits.shape[-2], device=input_ids.device)
                        else:
                            output_idx = token_idx + outputs.logits.shape[-2] - input_ids.shape[-1]
                        next_token_logits = torch.index_select(outputs.logits, -2, output_idx - 1).squeeze(-2)
                    else:
                        next_token_logits = torch.index_select(outputs.logits, -2, token_idx - 1).squeeze(-2)
                    next_token_scores = logits_processor(input_ids, next_token_logits)
            else:
                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].clone()
                if token_idx is not None and self.config.is_encoder_decoder:
                    # case2 (with KV caching): outputs.logits.shape: [batch_size, 1, vocab_size]
                    next_token_scores = logits_processor(input_ids[:, :token_idx], next_token_logits)
                else:
                    # case3 (default case): token_idx is None
                    next_token_scores = logits_processor(input_ids, next_token_logits)

            # pre-process distribution
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            # TODO: no ignore_eos check here since there is a compilation error, will add ignore_eos here if fixed
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            if not lazy_mode:
                next_tokens = next_tokens.to(input_ids.dtype)

            if token_idx is not None:
                input_ids.index_copy_(
                    1, token_idx, next_tokens.unsqueeze(-1) if next_tokens.dim() == 1 else next_tokens
                )
            else:
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            cur_len = cur_len + 1
            if bucket_size > 0 and bucket_internal:
                # Calculate slice idx for kv cache during the decode phase.
                # Breaking down the kv cache in the attention block helps to reduce computation time.
                if model_kwargs.get("token_idx_cpu") <= (model_kwargs["kv_cache_len"] // bucket_size) * bucket_size:
                    idx = (model_kwargs.get("token_idx_cpu") - 1) // bucket_size
                    if prev_idx != idx:
                        model_kwargs["cache_idx"] = (idx + 1) * bucket_size
                        prev_idx = idx
                else:
                    model_kwargs["cache_idx"] = model_kwargs["kv_cache_len"]

            if ignore_eos:
                this_peer_finished = stopping_criteria(
                    input_ids,
                    scores,
                    token_idx=cur_len,
                    ignore_eos=ignore_eos,
                    eos_token_id=generation_config.eos_token_id,
                )
            else:
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                    input_ids,
                    scores,
                    token_idx=cur_len,
                    ignore_eos=ignore_eos,
                    eos_token_id=generation_config.eos_token_id,
                )
                this_peer_finished = unfinished_sequences.max() == 0

            hb_profer.step()
            if hb_gen_time is not None:
                if not time_to_first_token_done:
                    time_to_first_token_done = True
                    import habana_frameworks.torch.hpu as torch_hpu

                    torch_hpu.synchronize()
                hb_gen_time.step()

            if (
                not model_kwargs.get("pad_done", False)
                and not model_kwargs.get("reuse_cache", False)
                and bucket_internal
            ):
                # Pad the returned past key values tensors from prefill phase forward run to maximum length
                # before starting the decode phase.
                if outputs.past_key_values[0][0].shape[2] == model_inputs["input_ids"].shape[1]:
                    self._pad_past_key_values(model_kwargs)
                model_kwargs["pad_done"] = True

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if (
            model_kwargs.get("use_hpu_graphs", False)
            and model_kwargs.get("limit_hpu_graphs", False)
            and not model_kwargs.get("reuse_cache", False)
            and bucket_internal
        ):
            # Clear HPU graphs input tensors of the decode phase after the full generation while loop
            self.clear_inputs()
            # Delete past key value tensors
            self._remove_past_key_values(model_kwargs)

        hb_profer.stop()

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
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
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GaudiGenerationConfig,
        synced_gpus: bool,
        logits_warper: Optional[LogitsProcessorList],
        lazy_mode: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        hb_gen_time: Optional[HabanaGenerationtime] = None,
        profiling_record_shapes: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`:
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`GaudiGenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            profiling_record_shapes (`bool`, *optional*, defaults to False):
                Record shapes when enabling profiling.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`transformers.generation.utils.GenerateBeamDecoderOnlyOutput`], [`transformers.generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`transformers.generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`transformers.generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        eos_token_id = generation_config._eos_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        token_idx = model_kwargs.get("token_idx", None)
        if token_idx is not None:
            # Update cur_len in case of static shapes
            cur_len = token_idx.item()
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        if self.generation_config.static_shapes:
            beam_trace_scores = torch.zeros(
                (input_ids.shape[1], 2 * batch_size * num_beams), device=input_ids.device, dtype=torch.float32
            )
            beam_trace_indices = torch.zeros(
                (input_ids.shape[1], 2 * batch_size * num_beams), device=input_ids.device, dtype=torch.int64
            )
            beam_trace_tokens = torch.zeros(
                (input_ids.shape[1], 2 * batch_size * num_beams), device=input_ids.device, dtype=torch.int64
            )
            beam_trace_idx = torch.tensor(0, device=input_ids.device)
            num_eos_tokens = torch.zeros((1), device=input_ids.device, dtype=torch.int64)
            num_beams_tensor = torch.tensor(num_beams, device=input_ids.device, dtype=torch.int64)

        def finalize_beams(initial_ids, beam_trace, model_config, length_penalty):
            beam_trace_idx, beam_trace_scores, beam_trace_indices, beam_trace_tokens = beam_trace
            bs = initial_ids.shape[0]
            num_beams = beam_trace_scores.shape[1] // (2 * bs)

            beam_trace_idx = beam_trace_idx.item()
            beam_trace_scores = beam_trace_scores[:beam_trace_idx, :]
            beam_trace_indices = beam_trace_indices[:beam_trace_idx, :]
            beam_trace_tokens = beam_trace_tokens[:beam_trace_idx, :]

            # (score, parent_beam, token_id, is_finished)
            root = (float("-inf"), None, None, False)

            def resolve_beam(beam):
                if beam == root:
                    return []
                score, prev, tok, is_finished = beam
                rest = resolve_beam(prev)
                rest.append(tok)
                return rest

            prev_beams = [[root] * num_beams] * bs
            best = [root] * bs

            def beam_score(beam):
                return (beam[3], beam[0])

            for step, (scores, indices, tokens) in enumerate(
                zip(beam_trace_scores, beam_trace_indices, beam_trace_tokens)
            ):
                cur_beams = [[] for _ in range(bs)]
                for idx, (s, i, t) in enumerate(zip(scores, indices, tokens)):
                    batch = idx // (num_beams * 2)
                    idx = idx % (num_beams * 2)
                    b_len = 1 + step
                    b_score = s.item() / (b_len**length_penalty)
                    b_tok = t.item()
                    is_finished = b_tok == model_config.eos_token_id
                    if len(cur_beams[batch]) >= num_beams:
                        continue
                    beam = (b_score, prev_beams[batch][i], b_tok, is_finished)
                    if not is_finished:
                        cur_beams[batch].append(beam)
                    if is_finished or (step + 1 == beam_trace_idx):
                        if beam_score(best[batch]) < beam_score(beam):
                            best[batch] = beam
                prev_beams = cur_beams

            def expand_if_needed(tensor, new_size, value, dim=-1):
                orig_len = tensor.shape[dim]
                padding_len = new_size - orig_len
                import torch.nn.functional as F

                if padding_len > 0:
                    if dim == -1:
                        return F.pad(tensor, (0, padding_len), value=value)
                    elif dim == -2:
                        return F.pad(tensor, (0, 0, 0, padding_len), value=value)
                    else:
                        assert False, f"Unsupported dim value: {dim}"
                return tensor

            result = [
                torch.cat(
                    [initial_ids[i], torch.tensor(resolve_beam(b), dtype=initial_ids.dtype, device=initial_ids.device)]
                )
                for i, b in enumerate(best)
            ]
            max_length = max([t.shape[-1] for t in result])
            result = [expand_if_needed(res, max_length, model_config.pad_token_id) for res in result]
            input_ids = torch.stack(result)
            return input_ids

        hb_profer = HabanaProfile(
            warmup=profiling_warmup_steps, active=profiling_steps, record_shapes=profiling_record_shapes
        )
        hb_profer.start()
        this_peer_finished = False

        bucket_size = model_kwargs.get("bucket_size", -1)
        reduce_recompile = model_kwargs.get("reduce_recompile", False)
        prompt_len = input_ids.shape[-1]
        if bucket_size >= 0:
            inc = iter(incrementor(bucket_size, prompt_len))
        if bucket_size > 0:
            assert "position_ids" not in model_kwargs, "Untested path"
        if self.generation_config.static_shapes:
            initial_ids = input_ids[::num_beams, 0:cur_len]

        time_to_first_token_done = False
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if lazy_mode:
                self.htcore_generation.mark_step()

            if bucket_size > 0:
                # it will not have been padded if bucket_size > 0
                params = next(inc)
                input_ids, model_kwargs = self.update_model_kwargs_for_bucketing(
                    params, input_ids, model_kwargs, pad_token_id, bucket_size, reduce_recompile
                )

            model_kwargs["lazy_mode"] = lazy_mode
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # if sequential is True, split the input to batches of batch_size and run sequentially
            if sequential:
                if any(
                    model_name in self.__class__.__name__.lower()
                    for model_name in [
                        "fsmt",
                        "reformer",
                        "bloom",
                        "ctrl",
                        "gpt_bigcode",
                        "transo_xl",
                        "xlnet",
                        "cpm",
                        "jamba",
                    ]
                ):
                    raise RuntimeError(
                        f"Currently generation for {self.__class__.__name__} is not supported "
                        f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                    )

                inputs_per_sub_batches = _split_model_inputs(
                    model_inputs, split_size=batch_size, full_batch_size=batch_beam_size
                )
                outputs_per_sub_batch = [
                    self(**inputs_per_sub_batch, return_dict=True) for inputs_per_sub_batch in inputs_per_sub_batches
                ]

                outputs = stack_model_outputs(outputs_per_sub_batch)
            else:
                hpu_graphs_kwargs = self._get_hpu_graphs_kwargs(model_kwargs)
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    **hpu_graphs_kwargs,
                )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            token_idx = model_kwargs.get("token_idx", None)
            if token_idx is not None and outputs.logits.shape[-2] > 1:
                if model_kwargs.get("num_virtual_tokens", 0) > 0:
                    # for prompt tuning, the output logit shape may > model_inputs["input_ids"].shape[-1]
                    if model_kwargs.get("reuse_cache", False):
                        output_idx = torch.tensor(outputs.logits.shape[-2], device=input_ids.device)
                    else:
                        output_idx = token_idx + outputs.logits.shape[-2] - input_ids.shape[-1]
                    next_token_logits = torch.index_select(outputs.logits, -2, output_idx - 1).squeeze(-2)
                else:
                    next_token_logits = torch.index_select(outputs.logits, -2, token_idx - 1).squeeze(-2)
            else:
                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].clone()

            next_token_scores = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            if token_idx is not None:
                next_token_scores_processed = logits_processor(input_ids[:, :token_idx], next_token_scores)
            else:
                next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            if do_sample:
                next_token_scores_processed = logits_warper(input_ids, next_token_scores_processed)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
            # non eos token per beam.
            n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
            n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
            if do_sample:
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)
            else:
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
                )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            if self.generation_config.static_shapes:
                beam_scores = next_token_scores.flatten()
                next_indices_flattened = next_indices.flatten()
                static_beam_indices = (
                    next_indices_flattened
                    + torch.tensor(
                        [[batch_idx * num_beams] * next_indices.shape[1] for batch_idx in range(batch_size)],
                        device=next_indices.device,
                    ).flatten()
                )

                beam_tokens = next_tokens.remainder(vocab_size).flatten()

                beam_trace_scores.index_copy_(0, beam_trace_idx, beam_scores.unsqueeze(0))
                beam_trace_indices.index_copy_(0, beam_trace_idx, next_indices_flattened.unsqueeze(0))
                beam_trace_tokens.index_copy_(0, beam_trace_idx, beam_tokens.unsqueeze(0))
                beam_trace_idx.add_(1)

                if self.generation_config.early_stopping:
                    num_eos_tokens.add_(beam_tokens[0:num_beams].eq(self.config.eos_token_id).sum())

                beam_scores.add_(torch.where(beam_tokens.eq(self.config.eos_token_id), float("-inf"), 0.0))
                beam_scores = beam_scores.view(batch_size, -1).unsqueeze(0)
                _, selected = torch.topk(beam_scores, k=num_beams, dim=-1, largest=True, sorted=True)
                offset = torch.arange(0, torch.numel(beam_scores), beam_scores.shape[-1]).unsqueeze(-1)
                selected = (selected + offset).flatten()
                beam_scores = beam_scores.flatten().index_select(0, selected)
                beam_tokens = beam_tokens.index_select(0, selected)
                static_beam_indices = static_beam_indices.index_select(0, selected)

                prev_beams = outputs.logits.shape[0] // batch_size

                beam_offsets = torch.arange(0, 1, prev_beams, dtype=torch.int32)
                beam_offsets = beam_offsets.to(device=outputs.logits.device)
                static_beam_indices = (static_beam_indices.view(batch_size, -1) + beam_offsets.unsqueeze(-1)).flatten()

                next_tokens = beam_tokens.unsqueeze(-1)
                beam_next_tokens = next_tokens
                beam_idx = static_beam_indices
            else:
                next_tokens = next_tokens % vocab_size

                # stateless
                beam_outputs = beam_scorer.process(
                    input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=beam_indices,
                    decoder_prompt_len=prompt_len,
                )
                beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

            if token_idx is not None:
                input_ids = torch.index_select(input_ids, 0, beam_idx)
                input_ids.index_copy_(
                    1, token_idx, beam_next_tokens.unsqueeze(-1) if beam_next_tokens.dim() == 1 else beam_next_tokens
                )
            else:
                input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if model_kwargs.get("past_key_values", None) is not None:
                if model_kwargs["reuse_cache"]:
                    model_kwargs["past_key_values"] = unwrap_deepspeed_model(self).reorder_kv_cache(beam_idx)
                else:
                    model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                        model_kwargs["past_key_values"], beam_idx
                    )

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
            # (that way the memory peak does not include outputs.logits)
            del outputs

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            hb_profer.step()
            if self.generation_config.static_shapes:
                is_min_length_reached = (
                    self.generation_config.min_length and cur_len >= self.generation_config.min_length
                )
                if (
                    self.generation_config.early_stopping
                    and is_min_length_reached
                    and num_eos_tokens >= num_beams_tensor
                ):
                    break
                elif get_final_stopping_criteria(stopping_criteria(input_ids, scores, token_idx=cur_len)):
                    break
            elif get_final_stopping_criteria(stopping_criteria(input_ids, scores)) or (
                beam_scorer.is_done and not lazy_mode
            ):
                this_peer_finished = True

            if hb_gen_time is not None:
                if not time_to_first_token_done:
                    time_to_first_token_done = True
                    import habana_frameworks.torch.hpu as torch_hpu

                    torch_hpu.synchronize()
                hb_gen_time.step()
        hb_profer.stop()

        if self.generation_config.static_shapes:
            beam_trace = (beam_trace_idx, beam_trace_scores, beam_trace_indices, beam_trace_tokens)
            from collections import UserDict

            def map_tensors(obj, fn):
                constructor = type(obj)
                if isinstance(obj, tuple):
                    return constructor(map_tensors(v, fn) for v in obj)
                if isinstance(obj, list):
                    return constructor([map_tensors(v, fn) for v in obj])
                if isinstance(obj, dict) or isinstance(obj, UserDict):
                    return constructor({k: map_tensors(v, fn) for k, v in obj.items()})
                if isinstance(obj, torch.Tensor):
                    return fn(obj)
                return obj

            def move(obj, device):
                return map_tensors(obj, lambda t: t.to(device))

            sequence_outputs = {}
            sequence_outputs["sequences"] = finalize_beams(
                initial_ids.cpu(), move(beam_trace, "cpu"), self.config, self.generation_config.length_penalty
            )
        else:
            sequence_outputs = beam_scorer.finalize(
                input_ids,
                beam_scores,
                next_tokens,
                beam_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                max_length=stopping_criteria.max_length,
                beam_indices=beam_indices,
                decoder_prompt_len=prompt_len,
            )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]

    def _group_beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GaudiGenerationConfig,
        synced_gpus: bool,
        lazy_mode: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        hb_gen_time: Optional[HabanaGenerationtime] = None,
        profiling_record_shapes: Optional[bool] = False,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **diverse beam search
        decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`GaudiGenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            profiling_record_shapes (`bool`, *optional*, defaults to False):
                Record shapes when enabling profiling.
            model_kwargs:
                Additional model specific kwargs that will be forwarded to the `forward` function of the model. If
                model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`transformers.generation.GenerateBeamDecoderOnlyOutput`], [`transformers.generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`transformers.generation.GenerateBeamDecoderOnlyOutput`] if [`transformers.generation.BeamSearchDecoderOnlyOutput`] if
            `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or a
            [`transformers.generation.GenerateBeamEncoderDecoderOutput`] if `model.config.is_encoder_decoder=True`.
        """

        raise NotImplementedError("Group beam search is not supported by optimum-habana yet.")

    def _constrained_beam_search(
        self,
        input_ids: torch.LongTensor,
        constrained_beam_scorer: ConstrainedBeamSearchScorer,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GaudiGenerationConfig,
        synced_gpus: bool,
        lazy_mode: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        hb_gen_time: Optional[HabanaGenerationtime] = None,
        profiling_record_shapes: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **constrained beam search
        decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            constrained_beam_scorer (`ConstrainedBeamSearchScorer`):
                A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation, while satisfying a list of positive constraints. For more information, the
                documentation of [`ConstrainedBeamSearchScorer`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            generation_config ([`GaudiGenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            profiling_record_shapes (`bool`, *optional*, defaults to False):
                Record shapes when enabling profiling.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`transformers.generation.utils.GenerateBeamDecoderOnlyOutput`], [`transformers.generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`transformers.generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`transformers.generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        eos_token_id = generation_config._eos_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        batch_size = len(constrained_beam_scorer._beam_hyps)
        num_beams = constrained_beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        token_idx = model_kwargs.get("token_idx", None)
        if token_idx is not None:
            # Update cur_len in case of static shapes
            cur_len = token_idx.item()
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

        hb_profer = HabanaProfile(
            warmup=profiling_warmup_steps, active=profiling_steps, record_shapes=profiling_record_shapes
        )
        hb_profer.start()

        time_to_first_token_done = False
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_kwargs["lazy_mode"] = lazy_mode
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            hpu_graphs_kwargs = self._get_hpu_graphs_kwargs(model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                **hpu_graphs_kwargs,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            if token_idx is not None and outputs.logits.shape[-2] > 1:
                if model_kwargs.get("num_virtual_tokens", 0) > 0:
                    # for prompt tuning, the output logit shape > model_inputs["input_ids"].shape[-1]
                    if model_kwargs.get("reuse_cache", False):
                        output_idx = torch.tensor(outputs.logits.shape[-2], device=input_ids.device)
                    else:
                        output_idx = token_idx + outputs.logits.shape[-2] - input_ids.shape[-1]
                    next_token_logits = torch.index_select(outputs.logits, -2, output_idx - 1).squeeze(-2)
                else:
                    next_token_logits = torch.index_select(outputs.logits, -2, token_idx - 1).squeeze(-2)
            else:
                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].clone()

            next_token_scores = torch.nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)

            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            scores_for_all_vocab = next_token_scores.clone()

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = (next_tokens / vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = constrained_beam_scorer.process(
                input_ids[:, :cur_len],
                next_token_scores,
                next_tokens,
                next_indices,
                scores_for_all_vocab,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            if token_idx is not None:
                input_ids = input_ids[beam_idx, :]
                input_ids.index_copy_(
                    1, token_idx, beam_next_tokens.unsqueeze(-1) if beam_next_tokens.dim() == 1 else beam_next_tokens
                )
            else:
                input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
            # (that way the memory peak does not include outputs.logits)
            del outputs

            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            hb_profer.step()

            if constrained_beam_scorer.is_done or get_final_stopping_criteria(
                stopping_criteria(input_ids, scores, token_idx=cur_len)
            ):
                this_peer_finished = True

            if hb_gen_time is not None:
                if not time_to_first_token_done:
                    time_to_first_token_done = True
                    import habana_frameworks.torch.hpu as torch_hpu

                    torch_hpu.synchronize()
                hb_gen_time.step()

        hb_profer.stop()
        sequence_outputs = constrained_beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]

    def _assisted_decoding(
        self,
        input_ids: torch.LongTensor,
        candidate_generator: "GaudiCandidateGenerator",
        logits_processor: LogitsProcessorList,
        logits_warper: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GaudiGenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        lazy_mode: Optional[bool] = False,
        ignore_eos: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        hb_gen_time: Optional[HabanaGenerationtime] = None,
        profiling_record_shapes: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** or
        **sample** (depending on `do_sample`), assisted by candidate sequences. Assisted generation is an example of a
        candidate decoding strategy. Can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text
        models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            candidate_generator (`CandidateGenerator`):
                A derived instance of [`CandidateGenerator`] that defines how candidate sequences are generated. For
                more information, the documentation of [`CandidateGenerator`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only used if sampling is active.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            lazy_mode (`bool`, *optional*, defaults to `False`):
                Whether the run is executed in lazy mode or not (i.e. eager mode).
            profiling_warmup_steps (`int`, *optional*, defaults to 0):
                Number of steps to ignore for profling.
            profiling_steps (`int`, *optional*, defaults to 0):
                Number of steps to be captured when enabling profiling.
            profiling_record_shapes (`bool`, *optional*, defaults to False):
                Record shapes when enabling profiling.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`transformers.generation.GenerateDecoderOnlyOutput`], [`transformers.generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`transformers.generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`transformers.generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        do_sample = logits_warper is not None
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if not ignore_eos:
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # This is needed if return_dict_in_generate is True
        start_from_empty_dynamic_cache = False
        past_key_values = model_kwargs.get("past_key_values", None)
        if isinstance(past_key_values, DynamicCache) or (
            isinstance(past_key_values, EncoderDecoderCache)
            and isinstance(past_key_values.self_attention_cache, DynamicCache)
        ):
            if len(past_key_values) == 0:
                start_from_empty_dynamic_cache = True

        hb_profer = HabanaProfile(warmup=profiling_warmup_steps, active=profiling_steps)
        hb_profer.start()
        this_peer_finished = False

        token_idx = model_kwargs.get("token_idx", None)
        time_to_first_token_done = False
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if lazy_mode:
                self.htcore_generation.mark_step()

            if token_idx is not None:
                # Update cur_len in case of static shapes
                cur_len = token_idx.item()
            else:
                cur_len = input_ids.shape[-1]

            # prepare model inputs
            model_kwargs["lazy_mode"] = lazy_mode
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            #  1. Fetch candidate sequences from a `CandidateGenerator`

            candidate_input_ids, candidate_logits = candidate_generator.get_candidates(input_ids[:, :cur_len])
            candidate_input_ids = candidate_input_ids.to(self.device)
            if candidate_logits is not None:
                candidate_logits = candidate_logits.to(self.device)

            if self.generation_config.static_shapes:
                candidate_length = candidate_input_ids.shape[1] - cur_len
            else:
                candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
            is_done_candidate = stopping_criteria(candidate_input_ids, None)

            # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
            # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
            # we use this forward pass to also pick the subsequent logits in the original model.

            # 2.1. Prepare the model inputs
            candidate_kwargs = copy.copy(model_kwargs)
            candidate_kwargs = _prepare_attention_mask(
                candidate_kwargs, candidate_input_ids.shape[1], self.config.is_encoder_decoder
            )
            candidate_kwargs = _prepare_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])
            if "cache_position" in candidate_kwargs:
                candidate_kwargs["cache_position"] = torch.cat(
                    (
                        candidate_kwargs["cache_position"],
                        torch.arange(cur_len, cur_len + candidate_length, device=input_ids.device, dtype=torch.long),
                    ),
                    dim=0,
                )

            model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)
            if "num_logits_to_keep" in model_inputs:
                model_inputs["num_logits_to_keep"] = candidate_length + 1

            hpu_graphs_kwargs = self._get_hpu_graphs_kwargs(model_kwargs)

            # 2.2. Run a forward pass on the candidate sequence
            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            outputs = self(
                **model_inputs,
                **hpu_graphs_kwargs,
            )

            # 2.3. Process the new logits
            new_logits = outputs.logits[:, -candidate_length - 1 :]  # excludes the input prompt if present
            next_token_logits = new_logits.clone()
            if len(logits_processor) > 0:
                for i in range(candidate_length + 1):
                    new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])
            if do_sample and len(logits_warper) > 0:
                for i in range(candidate_length + 1):
                    new_logits[:, i, :] = logits_warper(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])

            # 3. Select the accepted tokens. There are two possible cases:
            # Case 1: `do_sample=True` and we have logits for the candidates (originally from speculative decoding)
            # 👉 Apply algorithm 1 from the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf).
            if do_sample and candidate_logits is not None:
                from transformers.generation.utils import _speculative_sampling

                valid_tokens, n_matches = _speculative_sampling(
                    candidate_input_ids,
                    candidate_logits,
                    candidate_length,
                    new_logits,
                    is_done_candidate,
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
                if is_done_candidate and n_matches == candidate_length:
                    n_matches -= 1
                valid_tokens = selected_tokens[:, : n_matches + 1]

            # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
            # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
            # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
            # is no match.

            # 4.1. Get the valid continuation, after the matching tokens
            if self.generation_config.static_shapes:
                input_ids[:, cur_len : cur_len + n_matches + 1] = valid_tokens
            else:
                input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
            if streamer is not None:
                streamer.put(valid_tokens.cpu())
            new_cur_len = input_ids.shape[-1]

            # 4.2. Discard past key values relative to unused assistant tokens
            new_cache_size = new_cur_len - 1
            outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)

            # 5. Update the candidate generation strategy if needed
            candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)

            # Store scores, attentions and hidden_states when required
            # Assistant: modified to append one tuple element per token, as in the other generation methods.
            if return_dict_in_generate:
                if output_scores:
                    scores += tuple(new_logits[:, i, :] for i in range(n_matches + 1))
                if output_logits:
                    raw_logits += (next_token_logits,)

                if "past_key_values" not in model_kwargs or start_from_empty_dynamic_cache:
                    added_len = new_cur_len
                    # set it to false for other iterations
                    start_from_empty_dynamic_cache = False
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
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                num_new_tokens=n_matches + 1,
            )

            if ignore_eos:
                this_peer_finished = stopping_criteria(
                    input_ids,
                    scores,
                    token_idx=None,
                    ignore_eos=ignore_eos,
                    eos_token_id=generation_config.eos_token_id,
                )
            else:
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                    input_ids,
                    scores,
                    token_idx=None,
                    ignore_eos=ignore_eos,
                    eos_token_id=generation_config.eos_token_id,
                )
                this_peer_finished = unfinished_sequences.max() == 0

            hb_profer.step()
            if hb_gen_time is not None:
                if not time_to_first_token_done:
                    time_to_first_token_done = True
                    import habana_frameworks.torch.hpu as torch_hpu

                    torch_hpu.synchronize()
                hb_gen_time.step()

            if this_peer_finished and not synced_gpus:
                break

        hb_profer.stop()
        if streamer is not None:
            streamer.end()

        if (
            hasattr(candidate_generator, "assistant_model")
            and candidate_generator.assistant_model.generation_config.num_assistant_tokens_schedule == "heuristic"
        ):
            candidate_generator.assistant_model.generation_config.num_assistant_tokens = (
                candidate_generator.num_assistant_tokens
            )
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
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
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
