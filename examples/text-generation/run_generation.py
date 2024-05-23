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
    class StoreTrueFalseAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if isinstance(values, bool) or values is None:
                # Flag passed without any value -> set to True
                setattr(namespace, self.dest, True)
            else:
                # Flag passed with value -> pattern match and set accordingly
                value_str = values.lower()
                if value_str in ('true', '1', 'yes'):
                    setattr(namespace, self.dest, True)
                elif value_str in ('false', '0', 'no'):
                    setattr(namespace, self.dest, False)
                else:
                    raise ValueError(f"Invalid value for {option_string}: {values}")


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
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
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
        "--dataset_name",
        default=None,
        type=str,
        help="Optional argument if you want to assess your model on a given dataset of the HF Hub.",
    )
    parser.add_argument(
        "--column_name",
        default=None,
        type=str,
        help="If `--dataset_name` was given, this will be the name of the column to use as prompts for generation.",
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
        "--profiling_record_shapes",
        default=False,
        type=bool,
        help="Record shapes when enabling profiling.",
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
        "--assistant_model",
        default=None,
        type=str,
        help="Optional argument to give a path to a draft/assistant model for assisted decoding.",
    )
    parser.add_argument(
        "--peft_model",
        default=None,
        type=str,
        help="Optional argument to give a path to a PEFT model.",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1)
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
        "--use_flash_attention",
        nargs='?',
        const=True,
        default=False,
        action=StoreTrueFalseAction,
        help="Whether to enable Habana Flash Attention, provided that the model supports it."
    )
    parser.add_argument(
        "--flash_attention_recompute",
        nargs='?',
        const=True,
        default=False,
        action=StoreTrueFalseAction,
        help="Whether to enable Habana Flash Attention in recompute mode on first token generation. This gives an opportunity of splitting graph internally which helps reduce memory consumption."
    )
    parser.add_argument(
        "--flash_attention_causal_mask",
        nargs='?',
        const=True,
        default=False,
        action=StoreTrueFalseAction,
        help="Whether to enable Habana Flash Attention in causal mode on first token generation."
    )
    parser.add_argument(
        "--flash_attention_fast_softmax",
        nargs='?',
        const=None,  # Default value handled post-parsing
        action=StoreTrueFalseAction,
        help="Whether to enable Habana Flash Attention in fast softmax mode."
    )
    parser.add_argument(
        "--book_source",
        action="store_true",
        help="Whether to use project Guttenberg books data as input. Usefull for testing large sequence lenghts.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Whether to use torch compiled model or not.",
    )
    parser.add_argument(
        "--ignore_eos",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to ignore eos, set False to disable it",
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
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust the execution of code from datasets/models defined on the Hub. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.",
    )
    parser.add_argument(
        "--parallel_strategy",
        type=str,
        choices=["tp", "none"],  # Add other strategies as needed
        default="none",
        help="Run multi card with the specified parallel strategy. Choices are 'tp' for Tensor Parallel Strategy or 'none'.",
    )

    args = parser.parse_args()

    if args.torch_compile:
        args.use_hpu_graphs = False

    if not args.use_hpu_graphs:
        args.limit_hpu_graphs = False

    if args.flash_attention_fast_softmax is None:
        args.flash_attention_fast_softmax = args.use_flash_attention

    args.quant_config = os.getenv("QUANT_CONFIG", "")
    if args.quant_config == "" and args.disk_offload:
        logger.warning(
            "`--disk_offload` was tested only with fp8, it may not work with full precision. If error raises try to remove the --disk_offload flag."
        )
    return args


def main():
    parser = argparse.ArgumentParser()
    args = setup_parser(parser)
    model, assistant_model, tokenizer, generation_config = initialize_model(args, logger)

    use_lazy_mode = True
    if args.torch_compile and model.config.model_type == "llama":
        use_lazy_mode = False

    import habana_frameworks.torch.hpu as torch_hpu

    if args.dataset_name is None:
        # Benchmark over the prompts below
        if args.prompt:
            input_sentences = args.prompt
        elif args.book_source:

            def download_book(book_id):
                import os

                import requests

                url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                response = requests.get(url)
                if response.status_code == 200:
                    pid = os.getpid()
                    save_path = f"/tmp/{book_id}_{pid}.txt"
                    with open(save_path, "wb") as file:
                        file.write(response.content)
                    print(f"Book downloaded and saved to: {save_path}")
                    return save_path
                else:
                    print("Failed to download book! Exiting...")
                    import sys

                    sys.exit()

            def assemble_prompt(prompt_size, book_path):
                prompt = ""
                counter = 0
                book_lines = open(book_path).readlines()
                for line in book_lines:
                    for word in line.split():
                        counter += 1
                        prompt += word + " "
                        if counter == prompt_size:
                            return [prompt] * args.batch_size

            book_ids = [
                2701,  # Moby Dick; Or, The Whale
                1513,  # Romeo and Juliet
                1342,  # Pride and Prejudice
            ]
            input_sentences = assemble_prompt(prompt_size=args.max_input_tokens, book_path=download_book(book_ids[0]))
        else:
            input_sentences = [
                "DeepSpeed is a machine learning framework",
                "He is working on",
                "He has a",
                "He got all",
                "Everyone is happy and I can",
                "The new movie that got Oscar this year",
                "In the far far distance from our galaxy,",
                "Peace is the only way",
            ]

        if args.batch_size > len(input_sentences):
            # Dynamically extends to support larger batch sizes
            num_sentences_to_add = args.batch_size - len(input_sentences)
            for i in range(num_sentences_to_add):
                input_sentences.append(input_sentences[i % len(input_sentences)])
        elif args.batch_size < len(input_sentences):
            input_sentences = input_sentences[: args.batch_size]

        def generate(size=None, reduce_recompile=False):
            """Generates sequences from the input sentences and returns them."""
            encode_t0 = time.perf_counter()
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
            encode_duration = time.perf_counter() - encode_t0

            if size is not None:
                input_tokens = adjust_batch(input_tokens, size)
            if not reduce_recompile:
                # Move inputs to target device(s)
                for t in input_tokens:
                    if torch.is_tensor(input_tokens[t]):
                        input_tokens[t] = input_tokens[t].to(args.device)
            iteration_times = []
            outputs = model.generate(
                **input_tokens,
                generation_config=generation_config,
                assistant_model=assistant_model,
                lazy_mode=use_lazy_mode,
                hpu_graphs=args.use_hpu_graphs,
                profiling_steps=args.profiling_steps,
                profiling_warmup_steps=args.profiling_warmup_steps,
                ignore_eos=args.ignore_eos,
                iteration_times=iteration_times,
                profiling_record_shapes=args.profiling_record_shapes,
            ).cpu()
            first_token_time = iteration_times[0] + encode_duration
            logger.info(f"Time to first token = {first_token_time*1000}ms")
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
            for i in range(args.warmup):
                if dyn_prompt_lens is None:
                    print(f"Warming up iteration {i+1}/{args.warmup}", flush=True)
                    generate(None, args.reduce_recompile)
                else:
                    print(f"Warming up for shape {dyn_prompt_lens[0]} iteration {i+1}/{args.warmup}", flush=True)
                    generate(dyn_prompt_lens[0], args.reduce_recompile)
        else:
            if args.bucket_size > 0:
                mn = min(dyn_prompt_lens)
                mx = max(dyn_prompt_lens)

                def rounder(x):
                    return int(math.ceil(x / args.bucket_size) * args.bucket_size)

                min_prompt_len = rounder(mn)
                max_sentence_len = rounder(mx)
                for i in range(args.warmup):
                    lst = list(range(min_prompt_len, max_sentence_len + 1, args.bucket_size))
                    for sz in lst:
                        print(f"Warming up for shape {sz - 1} iteration {i+1}/{args.warmup}", flush=True)
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
    else:
        # Downloading and loading a dataset from the hub.
        from datasets import load_dataset
        from torch.utils.data import DataLoader

        assert not args.simulate_dyn_prompt, "Both dataset_name and simulate_dyn_prompt are set"

        raw_dataset = load_dataset(args.dataset_name)
        if "test" in raw_dataset:
            split = "test"
        elif "validation" in raw_dataset:
            split = "validation"
        else:
            split = "train"
        raw_dataset = (
            raw_dataset[split]
            .shuffle()
            .select(range(args.dataset_max_samples if args.dataset_max_samples > 0 else (raw_dataset[split]).num_rows))
        )

        if args.column_name is None:
            # If no column name is given, take the first column that has strings
            column_name = [key for key in raw_dataset.features.keys() if raw_dataset.features[key].dtype == "string"][
                0
            ]
            logger.info(
                f"No column name was given so automatically choosing '{column_name}' for prompts. If you would like to use another column of the dataset, you can set the argument `--column_name`."
            )
        else:
            column_name = args.column_name

        # Remove unused columns
        raw_dataset = raw_dataset.remove_columns([name for name in raw_dataset.column_names if name != column_name])

        # Set the prompt length to args.max_input_tokens if > 0 else (if 0 truncate to 16, otherwise use full length)
        prompt_length = args.max_input_tokens if args.max_input_tokens > 0 else (-1, 16)[args.max_input_tokens == 0]

        def preprocess_function(examples):
            # Tokenize the texts
            return tokenizer(
                examples[column_name],
                padding="max_length",
                max_length=prompt_length if prompt_length > 0 else None,
                truncation=prompt_length > 0,
            )

        raw_dataset = raw_dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )
        # After tokenization, we can remove the column of interest
        raw_dataset = raw_dataset.remove_columns([column_name])
        raw_dataset.set_format(type="torch")

        if prompt_length <= 0:
            # Todo please check if this collate function is suitable for your model
            # This has been tested for OPT, llama, and Bloom
            assert model.config.model_type in ["opt", "bloom", "llama"]

            def collate_fn(data):
                collect = {k: [dt[k] for dt in data] for k in data[0]}
                result = {}
                for k in collect:
                    tensors = collect[k]
                    max_shape = max([item.shape[0] for item in tensors])
                    result[k] = torch.stack(
                        [torch.cat((torch.zeros(max_shape - t.shape[0], dtype=t.dtype), t)) for t in tensors], 0
                    )
                return result

        else:
            collate_fn = None

        dataloader = DataLoader(raw_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

        def generate_dataset(batch):
            prompt = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            # Move inputs to target device(s)
            for t in batch:
                if torch.is_tensor(batch[t]):
                    batch[t] = batch[t].to(args.device)
            # Generate new sequences
            outputs = model.generate(
                **batch,
                generation_config=generation_config,
                lazy_mode=use_lazy_mode,
                hpu_graphs=args.use_hpu_graphs,
                profiling_steps=args.profiling_steps,
                profiling_warmup_steps=args.profiling_warmup_steps,
                ignore_eos=args.ignore_eos,
                profiling_record_shapes=args.profiling_record_shapes,
            ).cpu()
            return prompt, outputs

        # warmup
        if prompt_length > 0:
            from optimum.habana.utils import HabanaProfile

            # compilation stage disable profiling
            HabanaProfile.disable()
            # Compilation
            logger.info("Graph compilation...")
            t0 = time.perf_counter()
            for i, batch in enumerate(dataloader):
                generate_dataset(batch)
                # The first three iterations take longer because of graph compilation
                if (i + 1) == 3:
                    break
            torch_hpu.synchronize()
            compilation_duration = time.perf_counter() - t0
            HabanaProfile.enable()

        total_new_tokens_generated = 0
        duration = 0
        separator = "-" * 50
        logger.info("Running generate dataset...")
        t_start = time.time()
        for i, batch in enumerate(dataloader):
            t0 = time.perf_counter()
            prompt, outputs = generate_dataset(batch)
            duration += time.perf_counter() - t0
            total_new_tokens_generated += args.batch_size * args.max_new_tokens
            print(separator)
            print(f"Batch n°{i+1}")
            print(f"Input: {prompt[:args.batch_size]}")
            print(
                f"Output: {tokenizer.batch_decode(outputs, skip_special_tokens=True)[:args.batch_size*args.num_return_sequences]}"
            )
            print(separator)
        t_end = time.time()

        throughput = total_new_tokens_generated / duration
        # Print Stats

        stats = f"Throughput (including tokenization) = {throughput} tokens/second"
        separator = "-" * len(stats)
        print()
        print("Stats:")
        print(separator)
        print(stats)
        print("Total runtime for dataset:", t_end - t_start)
        mem = get_hpu_memory_stats()
        for k, v in mem.items():
            print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))
        if prompt_length > 0:
            print(f"Graph compilation duration          = {compilation_duration} seconds")
        print(separator)
    if args.quant_config:
        import habana_quantization_toolkit

        habana_quantization_toolkit.finish_measurements(model)
    if args.const_serialization_path and os.path.isdir(args.const_serialization_path):
        import shutil

        shutil.rmtree(args.const_serialization_path)


if __name__ == "__main__":
    main()
