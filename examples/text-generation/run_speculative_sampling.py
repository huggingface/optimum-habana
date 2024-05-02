#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
Conditional text generation on Habana Gaudi/Gaudi2.
"""

import argparse
import json
import logging
import math
import os
import time
from itertools import cycle
from pathlib import Path
import torch
from utils import adjust_batch, count_hpu_graphs, initialize_model
from optimum.habana.utils import get_hpu_memory_stats
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def setup_parser(parser):
    # Arguments management
    parser.add_argument("--device", "-d", type=str, choices=["hpu"], help="Device to run", default="hpu")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model (on the HF Hub or locally).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=0,
        help="If > 0 then pad and truncate the input sequences to this specified length of tokens. \
            if == 0, then truncate to 16 (original default) \
            if < 0, then do not truncate, use full input prompt",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size must be 1 for assisted decoding.")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of inference iterations for benchmarking.")
    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="Local process rank.")
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling for generation.",
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="Number of beams used for beam search generation. 1 means greedy search will be performed.",
    )
    parser.add_argument(
        "--trim_logits",
        action="store_true",
        help="Calculate logits only for the last token to save memory in the first step.",
    )
    parser.add_argument(
        "--seed",
        default=27,
        type=int,
        help="Seed to use for random generation. Useful to reproduce your runs with `--do_sample`.",
    )
    parser.add_argument(
        "--profiling_warmup_steps",
        default=0,
        type=int,
        help="Number of steps to ignore for profiling.",
    )
    parser.add_argument(
        "--profiling_steps",
        default=0,
        type=int,
        help="Number of steps to capture for profiling.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        type=str,
        nargs="*",
        help='Optional argument to give a prompt of your choice as input. Can be a single string (eg: --prompt "Hello world"), or a list of space-separated strings (eg: --prompt "Hello world" "How are you?")',
    )
    parser.add_argument(
        "--bad_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that are not allowed to be generated.",
    )
    parser.add_argument(
        "--force_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that must be generated.",
    )
    parser.add_argument(
        "--peft_model",
        default=None,
        type=str,
        help="Optional argument to give a path to a PEFT model.",
    )
    
    parser.add_argument("--num_return_sequences", type=int, default=1) #Must be kept 1 for assisted decoding
    parser.add_argument(
        "--token",
        default=None,
        type=str,
        help="The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
        "generated when running `huggingface-cli login` (stored in `~/.huggingface`).",
    )
    parser.add_argument(
        "--model_revision",
        default="main",
        type=str,
        help="The specific model version to use (can be a branch name, tag name or commit id).",
    )
    parser.add_argument(
        "--attn_softmax_bf16",
        action="store_true",
        help="Whether to run attention softmax layer in lower precision provided that the model supports it and "
        "is also running in lower precision.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output directory to store results in.",
    )
    parser.add_argument(
        "--bucket_size",
        default=-1,
        type=int,
        help="Bucket size to maintain static shapes. If this number is negative (default is -1) \
            then we use `shape = prompt_length + max_new_tokens`. If a positive number is passed \
            we increase the bucket in steps of `bucket_size` instead of allocating to max (`prompt_length + max_new_tokens`).",
    )
    parser.add_argument(
        "--bucket_internal",
        action="store_true",
        help="Split kv sequence into buckets in decode phase. It improves throughput when max_new_tokens is large.",
    )
    parser.add_argument(
        "--dataset_max_samples",
        default=-1,
        type=int,
        help="If a negative number is passed (default = -1) perform inference on the whole dataset, else use only `dataset_max_samples` samples.",
    )
    parser.add_argument(
        "--limit_hpu_graphs",
        action="store_true",
        help="Skip HPU Graph usage for first token to save memory",
    )
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        help="Whether to reuse key/value cache for decoding. It should save memory.",
    )
    parser.add_argument(
        "--use_cache",
        default=True,
        help="Assisted generation requires use_cache=True",
    )
    parser.add_argument("--verbose_workers", action="store_true", help="Enable output from non-master workers")
    parser.add_argument(
        "--simulate_dyn_prompt",
        default=None,
        type=int,
        nargs="*",
        help="If empty, static prompt is used. If a comma separated list of integers is passed, we warmup and use those shapes for prompt length.",
    )
    parser.add_argument(
        "--reduce_recompile",
        action="store_true",
        help="Preprocess on cpu, and some other optimizations. Useful to prevent recompilations when using dynamic prompts (simulate_dyn_prompt)",
    )
    parser.add_argument(
        "--assistant_model",
        default=None,
        help="Path to or name of the assistant model to use for assisted decoding.",
    )
    parser.add_argument("--fp8", action="store_true", help="Enable Quantization to fp8")
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Whether to enable Habana Flash Attention, provided that the model supports it.",
    )
    parser.add_argument(
        "--flash_attention_recompute",
        action="store_true",
        help="Whether to enable Habana Flash Attention in recompute mode on first token generation. This gives an opportunity of splitting graph internally which helps reduce memory consumption.",
    )
    parser.add_argument(
        "--flash_attention_causal_mask",
        action="store_true",
        help="Whether to enable Habana Flash Attention in causal mode on first token generation.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Whether to use torch compiled model or not.",
    )
    parser.add_argument("--temperature", default=1.0, type=float, help="Temperature value for text generation")
    parser.add_argument("--top_p", default=1.0, type=float, help="Top_p value for generating text via sampling")
    parser.add_argument(
        "--const_serialization_path",
        "--csp",
        type=str,
        help="Path to serialize const params. Const params will be held on disk memory instead of being allocated on host memory.",
    )
    parser.add_argument(
        "--disk_offload",
        action="store_true",
        help="Whether to enable device map auto. In case no space left on cpu, weights will be offloaded to disk.",
    )
    args = parser.parse_args()

    if args.torch_compile:
        args.use_hpu_graphs = False

    if not args.use_hpu_graphs:
        args.limit_hpu_graphs = False

    args.quant_config = os.getenv("QUANT_CONFIG", "")
    return args

def main():
    parser = argparse.ArgumentParser()
    args = setup_parser(parser)
    model, tokenizer, generation_config = initialize_model(args, logger)
    use_lazy_mode = True
    if args.torch_compile and model.config.model_type == "llama":
        use_lazy_mode = False
    import habana_frameworks.torch.hpu as torch_hpu
    #Get stats for custom input prompts
    if args.prompt:
        input_sentences = args.prompt
    if args.batch_size < len(input_sentences):
        input_sentences = input_sentences[: args.batch_size]
    elif args.batch_size > len(input_sentences):
        # Dynamically extends to support larger batch sizes
        num_sentences_to_add = args.batch_size - len(input_sentences)
        for i in range(num_sentences_to_add):
            input_sentences.append(input_sentences[i % len(input_sentences)])
    
    def generate(size=None, reduce_recompile=False):
        """Generates sequences from the input sentences and returns them."""
        # Tokenization
        if args.max_input_tokens > 0:
            input_tokens = tokenizer.batch_encode_plus(
                input_sentences,
                return_tensors="pt",
                padding="max_length",
                max_length=args.max_input_tokens,
                truncation=True,
            )
        else:
            input_tokens = tokenizer.batch_encode_plus(input_sentences, return_tensors="pt", padding=True)

        if size is not None:
            input_tokens = adjust_batch(input_tokens, size)
        if not reduce_recompile:
            # Move inputs to target device(s)
            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to(args.device)

        outputs = model.generate(
            **input_tokens,
            generation_config=generation_config,
            lazy_mode=use_lazy_mode,
            hpu_graphs=args.use_hpu_graphs,
            profiling_steps=args.profiling_steps,
            profiling_warmup_steps=args.profiling_warmup_steps,
        ).cpu()
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    from optimum.habana.utils import HabanaProfile

    # compilation stage disable profiling
    HabanaProfile.disable()
    # Compilation
    logger.info("Graph compilation...")
    dyn_prompt_lens = args.simulate_dyn_prompt
    t0 = time.perf_counter()
    # The first three iterations take longer because of graph compilation
    if dyn_prompt_lens is None or len(set(dyn_prompt_lens)) == 1:
        for _ in range(args.warmup):
            if dyn_prompt_lens is None:
                print("Warming up", flush=True)
                generate(None, args.reduce_recompile)
            else:
                print("Warming up for shape,", dyn_prompt_lens[0], flush=True)
                generate(dyn_prompt_lens[0], args.reduce_recompile)
    else:
        if args.bucket_size > 0:
            mn = min(dyn_prompt_lens)
            mx = max(dyn_prompt_lens)

            def rounder(x):
                return int(math.ceil(x / args.bucket_size) * args.bucket_size)

            min_prompt_len = rounder(mn)
            max_sentence_len = rounder(mx)
            for _ in range(args.warmup):
                lst = list(range(min_prompt_len, max_sentence_len + 1, args.bucket_size))
                for sz in lst:
                    print("Warming up for shape,", sz - 1, flush=True)
                    generate(sz - 1, args.reduce_recompile)
    torch_hpu.synchronize()
    compilation_duration = time.perf_counter() - t0
    HabanaProfile.enable()
    total_new_tokens_generated = 0
    logger.info("Running generate...")
    t0 = time.perf_counter()
    # Benchmark over n_iterations iterations
    if dyn_prompt_lens is None:
        for i in range(args.n_iterations):
            generated = generate(None, args.reduce_recompile)
    else:
        repeated_prompt_len = cycle(dyn_prompt_lens)
        for i in range(args.n_iterations):
            prompt_len = next(repeated_prompt_len)
            print("Generating for shape,", prompt_len)
            generated = generate(prompt_len, args.reduce_recompile)
    duration = time.perf_counter() - t0
    total_new_tokens_generated = args.n_iterations * args.batch_size * args.max_new_tokens
    throughput = total_new_tokens_generated / duration

    print()
    print("Input/outputs:")
    for i, input_sentence in enumerate(zip(input_sentences)):
        print(f"input {i+1}: {input_sentence}")
        for j, output in enumerate(
            zip(generated[args.num_return_sequences * i : args.num_return_sequences * (i + 1)])
        ):
            print(f"output {j+1}: {output}")
        print()

    # Store results if necessary
    if args.output_dir is not None and args.global_rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "throughput": throughput,
            "output": output,
        }
        with (output_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    stats = f"Throughput (including tokenization) = {throughput} tokens/second"
    stats = stats + f"\nNumber of HPU graphs                = {count_hpu_graphs()}"
    separator = "-" * len(stats)
    print()
    print("Stats:")
    print(separator)
    print(stats)
    mem = get_hpu_memory_stats()
    for k, v in mem.items():
        print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))
    print(f"Graph compilation duration          = {compilation_duration} seconds")
    print(separator)
    print()
            
if __name__ == "__main__":
    main()
        

    