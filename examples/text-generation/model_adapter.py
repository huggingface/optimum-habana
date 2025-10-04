# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

###############################################################################
# Copyright (C) 2020-2025 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import logging
from typing import Any, Literal, Optional, Union

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM, TemplateLM
from lm_eval.models.utils import get_dtype, stop_sequences_criteria

# Local imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


logger = logging.getLogger(__name__)


class HabanaModelAdapter(HFLM):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        args: argparse.Namespace,
        options: GenerationConfig,
        backend: Literal["default", "causal", "seq2seq"] = "default",
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        softmax_dtype: Union[str, torch.dtype, None] = None,
        mixed_precision_dtype: Union[str, torch.dtype, None] = None,
        add_bos_token: Optional[bool] = True,
        prefix_token_id: Optional[int] = None,
        delta: Optional[str] = None,
        # end token for thinking, either the string or int token id.
        # splits to get response after this token (if provided).
        think_end_token: Optional[Union[str, int]] = None,
        enable_thinking: Optional[bool] = None,
        chat_template_args: Optional[dict] = None,
        **kwargs,
    ) -> None:
        # To skip cuda code of the HFLM init
        TemplateLM.__init__(self)
        self.tokenizer = tokenizer
        self._model = model
        self._config = self._model.config
        self._batch_size = args.batch_size
        self.buckets: list[int] = sorted(args.buckets)
        self.options = options
        self.device_ = args.device
        self.pretrained = model
        self.peft = args.peft_model
        self.delta = delta
        self.custom_prefix_token_id = prefix_token_id
        if isinstance(think_end_token, str) and think_end_token.isdigit():
            self.think_end_token = int(think_end_token)
        else:
            self.think_end_token = think_end_token

        self.chat_template_args = chat_template_args or {}
        if enable_thinking is not None:
            self.chat_template_args.update({"enable_thinking": enable_thinking})

        # determine which of 'causal' and 'seq2seq' backends to use for HF models
        self._get_backend(config=self._config, backend=backend, trust_remote_code=args.trust_remote_code)
        self.truncation = truncation
        self.logits_cache = logits_cache
        self.add_bos_token = add_bos_token
        self._max_length = max_length
        self.softmax_dtype = get_dtype(softmax_dtype) if softmax_dtype is not None else None
        self.mixed_precision_dtype = get_dtype(mixed_precision_dtype) if mixed_precision_dtype is not None else None
        self.hpu_graphs = args.use_hpu_graphs
        self.use_lazy_mode = True
        self.ignore_eos = args.ignore_eos
        if args.torch_compile:
            self.use_lazy_mode = False
        _vision_models = ["gemma-3"]
        _multimodal = True if any(model in args.model_name_or_path for model in _vision_models) else False
        self.vocab_size = (
            self._model.language_model.config.vocab_size if _multimodal else self._model.config.vocab_size
        )
        if "gemma" in getattr(self._config, "model_type", ""):
            self.add_bos_token = True
            logger.info(
                f"Model type is '{self._config.model_type}', part of the Gemma family--a BOS token will be used as Gemma underperforms without it."
            )
        self.batch_size_per_gpu = int(args.batch_size)
        self.revision = args.model_revision
        self.model_inputs = {"use_cache": self.options.use_cache}
        if self._model.config.model_type in [
            "llama",
            "mistral",
            "falcon",
            "phi",
            "mixtral",
            "qwen2",
            "gptj",
            "starcoder2",
            "gemma",
            "baichuan",
        ]:
            self.model_inputs.update(
                {
                    "reuse_cache": self.options.reuse_cache,
                }
            )

        if self.model.config.model_type in [
            "llama",
            "mistral",
            "qwen2",
            "falcon",
            "starcoder2",
            "gemma",
            "baichuan",
            "gpt_bigcode",
        ]:
            if self.model.config.model_type not in ["falcon", "gpt_bigcode"]:
                self.model_inputs.update(
                    {
                        "attn_softmax_bf16": self.options.attn_softmax_bf16,
                    }
                )
            self.model_inputs.update(
                {
                    "use_flash_attention": self.options.use_flash_attention,
                    "flash_attention_recompute": self.options.flash_attention_recompute,
                    "flash_attention_causal_mask": self.options.flash_attention_causal_mask,
                }
            )
            if self.model.config.model_type in ["llama", "qwen2", "baichuan", "gpt_bigcode"]:
                self.model_inputs.update({"flash_attention_fast_softmax": self.options.flash_attention_fast_softmax})
            if self.model.config.model_type in ["llama"]:
                self.model_inputs.update({"use_flex_attention": self.options.use_flex_attention})
        if args.warmup:
            self.warm_up()

    def warm_up(self) -> None:
        for bucket_size in reversed(self.buckets):
            inps = torch.ones((self._batch_size, bucket_size), dtype=torch.int64)
            self._model_call(inps)

    @property
    def eot_token_id(self) -> int:
        return self._model.config.eos_token_id

    @property
    def max_length(self) -> int:
        # Legacy
        return self._max_length if self._max_length else self.buckets[-1]

    @property
    def device(self):
        # We need to do padding ourselves, otherwise we'll end up with recompilations
        # Returning 'cpu' to keep tensors on CPU in lm_eval code
        return "cpu"

    @max_length.setter
    def max_length(self, value: int) -> None:
        self._max_length = value

    def find_bucket(self, length: int, key=lambda b, length: b >= length) -> int:
        for b in self.buckets:
            if key(b, length):
                return b
        new_bucket = length
        self.buckets.append(new_bucket)
        self.buckets.sort()
        return new_bucket

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        bs, seq_length = inps.shape
        padding_length = 0
        if self.options.static_shapes:
            bucket_length = self.find_bucket(seq_length)
            if self.options.use_cache and self.options.reuse_cache:
                self._model.allocate_kv_cache(bs, bucket_length + 1, bucket_length)
            padding_length = bucket_length - seq_length
            inps = F.pad(inps, (0, padding_length), value=self._model.config.pad_token_id)
        logits = self._model(inps.to(self.device_), **self.model_inputs)["logits"].cpu()

        if self.options.static_shapes and padding_length > 0:
            logits = logits[:, :-padding_length, :]
        logits = logits.to(torch.float32)

        return logits

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:
        """
        Override to change only max_length property
        """
        legacy_max_length = self.max_length
        self.max_length = super().max_length
        # Call the parent class's implementation for the unchanged parts
        res = super().generate_until(requests, disable_tqdm)
        self.max_length = legacy_max_length
        return res

    def _model_generate(
        self,
        context,
        max_length: int,
        stop: list[str],
        **generation_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """
        Patched method
        source: https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.1/lm_eval/models/huggingface.py#L951
        """
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample")
        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(self.tokenizer, stop, context.shape[1], context.shape[0])
        # to avoid graph recompilation
        if self.options.static_shapes:
            self.options.bucket_internal = True
            bucket_length = self.find_bucket(context.shape[1])
            padding_length = bucket_length - context.shape[1]
            max_gen_toks = max_length - context.shape[1]
            if padding_length > 0 and self.hpu_graphs:
                # Static shapes require right-padding (left-padding due to batch encoding is performed at tok_batch_encode level)
                # See https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.1/lm_eval/models/huggingface.py#L869
                context = F.pad(context, (0, padding_length), value=self.tokenizer.pad_token_id)
                generation_kwargs["attention_mask"] = F.pad(
                    generation_kwargs["attention_mask"], (0, padding_length), value=0
                )
        # move context & attention_mask to hpu
        context = context.to("hpu")
        generation_kwargs["attention_mask"] = generation_kwargs["attention_mask"].to("hpu")
        with torch.autocast(
            device_type="hpu",
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            return self.model.generate(
                input_ids=context,
                max_new_tokens=max_gen_toks,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                hpu_graphs=self.hpu_graphs,
                lazy_mode=self.use_lazy_mode,
                **generation_kwargs,
            )
