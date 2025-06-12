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
import json
import multiprocessing as mp
import os
from typing import Literal, Optional

import psutil
import torch
import torch.nn.functional as F
from lm_eval import evaluator, utils
from lm_eval.models.huggingface import HFLM, TemplateLM
from lm_eval.models.utils import stop_sequences_criteria

# Local imports
from run_generation import setup_parser
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from utils import finalize_quantization, initialize_model, save_model

from optimum.habana.utils import HabanaGenerationTime, get_hpu_memory_stats


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = utils.eval_logger

# This hack is a workaround to limitations of lm_eval which always allocates
# mp.Pool with max cpu count which explodes on multinode scenarios and for hpu
# create multiprocess with spawn context
OrigPool = mp.Pool


def LimitedSpawnPool(_):
    spawn_context = mp.get_context("spawn")
    physical_cpu_count = psutil.cpu_count(logical=False)
    pool_size = physical_cpu_count
    world_size = int(os.getenv("WORLD_SIZE", 1))
    pool_size //= max(world_size, 1)
    if (pool_size * world_size) != physical_cpu_count:
        pool_size -= 1
    return spawn_context.Pool(pool_size)


mp.Pool = LimitedSpawnPool


def setup_lm_eval_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Evaluation script for HPU"
    )
    parser.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        help="Input length buckets to use with static_shapes",
        default=[16, 32, 64, 128, 189, 284, 384, 985],
    )

    parser.add_argument(
        "--output_file", "-o", type=str, help="Output file with end results and runtime parameters", required=True
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Tasks to run",
        default=["hellaswag", "lambada_openai", "piqa", "winogrande"],
    )
    parser.add_argument("--limit_iters", type=int, help="limit examples to run that many iterations", default=None)
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="If True, prints extra-logs for all tasks",
    )
    parser.add_argument("--max_graphs", type=int, help="Maximum number of HPU graphs", default=None)
    args = setup_parser(parser)

    return args


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
        add_bos_token: Optional[bool] = True,
        prefix_token_id: Optional[int] = None,
        delta: Optional[str] = None,
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
        # determine which of 'causal' and 'seq2seq' backends to use for HF models
        self._get_backend(config=self._config, backend=backend, trust_remote_code=args.trust_remote_code)
        self.truncation = truncation
        self.logits_cache = logits_cache
        self.add_bos_token = add_bos_token
        self._max_length = max_length
        self.hpu_graphs = args.use_hpu_graphs
        self.use_lazy_mode = True
        if args.torch_compile:
            self.use_lazy_mode = False
        self.vocab_size = self._model.config.vocab_size
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
    def device(self):
        # We need to do padding ourselves, otherwise we'll end up with recompilations
        # Returning 'cpu' to keep tensors on CPU in lm_eval code
        return "cpu"

    def find_bucket(self, length: int) -> list[int]:
        return [b for b in self.buckets if b >= length][0]

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

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """
        Patched method
        source: https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/models/huggingface.py/#L858
        """

        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(self.tokenizer, stop, context.shape[1], context.shape[0])
        # to avoid graph recompilation
        if self.options.static_shapes:
            # Filter buckets greater than or equal to the given number
            greater_or_equal = [x for x in self.buckets if x >= context.shape[1]]
            # Return the smallest value from the filtered list, or the context shape, if no such value exists
            bucket = min(greater_or_equal, default=context.shape[1])
            max_gen_toks = max_length - context.shape[1]
            max_length = max(max_length, max_gen_toks + bucket)
        # move context & attention_mask to hpu
        context = context.to("hpu")
        generation_kwargs["attention_mask"] = generation_kwargs["attention_mask"].to("hpu")
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            hpu_graphs=self.hpu_graphs,
            lazy_mode=self.use_lazy_mode,
            **generation_kwargs,
        )

    def get_model_info(self) -> dict:
        """
        Patched method to get Hugging Face model information for experiment reproducibility.
        source: https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/models/huggingface.py/#L1375
        Remove from SynapseAI 1.21
        """

        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            else:
                return ""

        def get_model_sha(pretrained: str, revision: str) -> str:
            return ""

        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
            "model_revision": self.revision,
            "model_sha": get_model_sha(self.pretrained, self.revision),
        }
        if self.peft:
            model_info["peft_sha"] = get_model_sha(self.peft, self.revision)
        if self.delta:
            model_info["delta_sha"] = get_model_sha(self.delta, self.revision)
        return model_info


def main() -> None:
    # Modified based on cli_evaluate function in https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/__main__.py/#L268
    args = setup_lm_eval_parser()
    model, _, tokenizer, generation_config = initialize_model(args, logger)

    with torch.no_grad():
        lm = HabanaModelAdapter(tokenizer, model, args, generation_config)

    with HabanaGenerationTime() as timer:
        with torch.no_grad():
            log_samples = args.log_samples
            results = evaluator.simple_evaluate(lm, tasks=args.tasks, limit=args.limit_iters, log_samples=log_samples)
        if args.device == "hpu":
            import habana_frameworks.torch.hpu as torch_hpu

            torch_hpu.synchronize()

    results["args"] = vars(args)
    results["duration"] = timer.last_duration

    if args.local_rank == 0:
        if args.device == "hpu":
            mem = get_hpu_memory_stats()
            for k, v in mem.items():
                print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))

        json_str = json.dumps(results, indent=2, default=utils.handle_non_serializable, ensure_ascii=False)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(json_str)
        if args.show_config:
            print(json_str)

    if args.quant_config:
        finalize_quantization(model)
    if args.save_quantized_model_with_inc:
        save_model(model, tokenizer, args.saved_model_path)
    if args.pt2e_save and args.pt2e_path:
        from quantization_tools.pt2e import pt2e_save

        pt2e_save(model)

    if args.const_serialization_path and os.path.isdir(args.const_serialization_path):
        import shutil

        shutil.rmtree(args.const_serialization_path)


if __name__ == "__main__":
    main()
