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
# Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import json
import logging
import multiprocessing as mp
import os
import time

import lm_eval.evaluator
import lm_eval.tasks
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import get_dtype
import psutil
import torch
import torch.nn.functional as F

# Local imports
from run_generation import setup_parser
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from utils import finalize_quantization, initialize_model

from optimum.habana.utils import get_hpu_memory_stats

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = logging.getLogger(__name__)


# This hack is a workaround to limitations of lm_eval which always allocates
# mp.Pool with max cpu count which explodes on multinode scenarios and for hpu
# create multiprocess with spawn context
OrigPool = mp.Pool


def LimitedSpawnPool(_):
    spawn_context = mp.get_context("spawn")
    physical_cpu_count = psutil.cpu_count(logical=False)
    pool_size = physical_cpu_count
    world_size = int(os.getenv("WORLD_SIZE", 1))
    if world_size == 0:
        world_size = 1
    pool_size //= world_size
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
        default=[16, 32, 64, 128, 189, 284, 384],
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
    args = setup_parser(parser)

    return args


class HabanaModelAdapter(HFLM):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        args: argparse.Namespace,
        options: GenerationConfig,
    ) -> None:
        super().__init__(device=args.device, pretrained=args.model_name_or_path)
        self.tokenizer = tokenizer
        self._model = model
        self._batch_size = args.batch_size
        self.buckets: list[int] = sorted(args.buckets)
        self.options = options
        self.device_ = args.device
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
        if self._model.config.model_type in ["llama", "mistral", "qwen2", "falcon", "starcoder2", "gemma", "baichuan"]:
            if self._model.config.model_type != "falcon":
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
        return self.buckets[-1]

    #@property
    #def max_gen_toks(self):
    #    raise NotImplementedError()

    #@property
    #def batch_size(self):
    #    return self._batch_size

    @property
    def device(self):
        # We need to do padding ourselves, otherwise we'll end up with recompilations
        # Returning 'cpu' to keep tensors on CPU in lm_eval code
        return "cpu"

    #def tok_encode(self, string):
    #    return self.tokenizer.encode(string)

    #def tok_decode(self, tokens):
    #    return self.tokenizer.decode(tokens)

    #def _model_generate(self, context, max_length, eos_token_id):
    #    raise NotImplementedError()

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

    def _create_model(
        self,
        pretrained: str,
        revision="main",
        dtype="auto",
        trust_remote_code=False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize=False,
        gpus=None,
        max_memory_per_gpu=None,
        max_cpu_memory=None,
        offload_folder="./offload",
        # PEFT, delta weights and quantization options
        peft=None,
        delta=None,
        autogptq=False,
        gptqmodel=False,
        **kwargs,
    ) -> None:
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
        adapt_transformers_to_gaudi()

        self._model = self.AUTO_MODEL_CLASS.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=get_dtype(dtype),
            trust_remote_code=trust_remote_code,
        )

def main() -> None:
    args = setup_lm_eval_parser()
    model, _, tokenizer, generation_config = initialize_model(args, logger)
    if args.trust_remote_code:
        # trust_remote_code fix was introduced in lm_eval 0.4.3
        # https://github.com/EleutherAI/lm-evaluation-harness/pull/1998/files
        # We need to cherry-pick the fix manually untill we upgrade (SW-190418)
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    #lm_tasks = lm_eval.tasks.get_task_dict(args.tasks)
    with torch.no_grad():
        lm = HabanaModelAdapter(tokenizer, model, args, generation_config)

    eval_start = time.perf_counter()
    with torch.no_grad():
        #results = lm_eval.evaluator.evaluate(lm, lm_tasks, limit=args.limit_iters)
        results = lm_eval.evaluator.simple_evaluate(lm, tasks=args.tasks, limit=args.limit_iters)
    if args.device == "hpu":
        import habana_frameworks.torch.hpu as torch_hpu

        torch_hpu.synchronize()
    eval_end = time.perf_counter()

    results["args"] = vars(args)
    results["duration"] = eval_end - eval_start

    if args.local_rank == 0:
        if args.device == "hpu":
            mem = get_hpu_memory_stats()
            for k, v in mem.items():
                print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))
        json.dump(results, open(args.output_file, "w"), indent=2)
        print(json.dumps(results, indent=2))
    if args.quant_config:
        finalize_quantization(model)

    if args.const_serialization_path and os.path.isdir(args.const_serialization_path):
        import shutil

        shutil.rmtree(args.const_serialization_path)


if __name__ == "__main__":
    main()
