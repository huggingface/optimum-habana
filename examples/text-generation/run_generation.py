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
import copy
import json
import logging
import os
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import check_min_version

from optimum.habana.checkpoint_utils import (
    get_ds_injection_policy,
    get_repo_root,
    model_is_optimized,
    model_on_meta,
    write_checkpoints_json,
)


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.33.0")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    # Arguments management
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--local_rank", type=int, default=-1, metavar="N", help="Local process rank.")
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
        "--dataset_min_max",
        default="",
        type=str,
        help="For dataset run, specify 2 integers (comma separated) defining min and max sentence length, so we can perform warmup. To be used with --max_input_tokens=-1 and --bucket_size set to > 0 (bucketed uncropped dataset)",
    )

    args = parser.parse_args()

    # If the DeepSpeed launcher is used, the env variable _ will be equal to /usr/local/bin/deepspeed
    # For multi node, the value of the env variable WORLD_SIZE should be larger than 8
    use_deepspeed = "deepspeed" in os.environ["_"] or (
        "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 8
    )
    if use_deepspeed:
        # Set necessary env variables
        os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
        os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")

    if not args.use_hpu_graphs:
        args.limit_hpu_graphs = False
        args.reuse_cache = False

    # Device is HPU
    args.device = "hpu"
    import habana_frameworks.torch.hpu as torch_hpu

    # Get world size, rank and local rank
    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

    world_size, rank, args.local_rank = initialize_distributed_hpu()

    if use_deepspeed:
        # Check if DeepSpeed is installed
        from transformers.integrations.deepspeed import is_deepspeed_available

        if not is_deepspeed_available():
            raise ImportError(
                "This script requires deepspeed: `pip install"
                " git+https://github.com/HabanaAI/DeepSpeed.git@1.12.0`."
            )
        import deepspeed

        # Initialize process(es) for DeepSpeed
        deepspeed.init_distributed(dist_backend="hccl")
        logger.info("DeepSpeed is enabled.")
    else:
        logger.info("Single-device run.")

    # Tweak generation so that it runs faster on Gaudi
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

    adapt_transformers_to_gaudi()

    # Set seed before initializing model.
    from optimum.habana.utils import set_seed

    try:
        from optimum.habana.utils import check_optimum_habana_min_version
    except ImportError:

        def check_optimum_habana_min_version(*a, **b):
            return ()

    # Will error if the minimal version of Optimum Habana is not installed. Remove at your own risks.
    check_optimum_habana_min_version("1.8.0.dev0")

    set_seed(args.seed)

    tokenizer_kwargs = {
        "revision": args.model_revision,
        "token": args.token,
    }
    if args.bad_words is not None or args.force_words is not None:
        tokenizer_kwargs["add_prefix_space"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)

    if use_deepspeed or args.bf16:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float
        args.attn_softmax_bf16 = False

    model_kwargs = {
        "revision": args.model_revision,
        "token": args.token,
    }

    if use_deepspeed:
        config = AutoConfig.from_pretrained(args.model_name_or_path, **model_kwargs)
        is_optimized = model_is_optimized(config)
        load_to_meta = model_on_meta(config)

        if load_to_meta:
            # Construct model with fake meta tensors, later will be replaced on devices during ds-inference ckpt load
            with deepspeed.OnDevice(dtype=model_dtype, device="meta"):
                model = AutoModelForCausalLM.from_config(config, torch_dtype=model_dtype)
        else:
            get_repo_root(args.model_name_or_path, local_rank=args.local_rank, token=args.token)
            # TODO: revisit placement on CPU when auto-injection is possible
            with deepspeed.OnDevice(dtype=model_dtype, device="cpu"):
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path, torch_dtype=model_dtype, **model_kwargs
                )
        model = model.eval()

        # Initialize the model
        ds_inference_kwargs = {"dtype": model_dtype}
        ds_inference_kwargs["tensor_parallel"] = {"tp_size": world_size}
        ds_inference_kwargs["enable_cuda_graph"] = args.use_hpu_graphs

        if load_to_meta:
            # model loaded to meta is managed differently
            checkpoints_json = "checkpoints.json"
            write_checkpoints_json(args.model_name_or_path, args.local_rank, checkpoints_json)

        # Make sure all devices/nodes have access to the model checkpoints
        torch.distributed.barrier()

        ds_inference_kwargs["injection_policy"] = get_ds_injection_policy(config)
        if load_to_meta:
            ds_inference_kwargs["checkpoint"] = checkpoints_json

        model = deepspeed.init_inference(model, **ds_inference_kwargs)
        model = model.module
    else:
        get_repo_root(args.model_name_or_path, token=args.token)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype, **model_kwargs)
        model = model.eval().to(args.device)
        is_optimized = model_is_optimized(model.config)

        if args.use_hpu_graphs:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph

            model = wrap_in_hpu_graph(model)

    if not model.config.is_encoder_decoder:
        tokenizer.padding_side = "left"
    # Some models like GPT2 do not have a PAD token so we have to set it if necessary
    if model.config.model_type == "llama":
        # unwind broken decapoda-research config
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        tokenizer.bos_token_id = model.generation_config.bos_token_id
        tokenizer.eos_token_id = model.generation_config.eos_token_id
        tokenizer.pad_token_id = model.generation_config.pad_token_id
        tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    if rank in [-1, 0]:
        logger.info(f"Args: {args}")
        logger.info(f"device: {args.device}, n_hpu: {world_size}, bf16: {use_deepspeed or args.bf16}")

    bad_words_ids = None
    force_words_ids = None
    if args.bad_words is not None:
        bad_words_ids = [tokenizer.encode(bad_word, add_special_tokens=False) for bad_word in args.bad_words]
    if args.force_words is not None:
        force_words_ids = [tokenizer.encode(force_word, add_special_tokens=False) for force_word in args.force_words]

    if args.peft_model:
        import importlib.util

        if importlib.util.find_spec("peft") is None:
            raise ImportError("The `peft` package is not installed, please run: `pip install peft`.")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.peft_model)
        model = model.to(model_dtype)

    # Generation configuration
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.use_cache = args.use_kv_cache
    generation_config.static_shapes = is_optimized
    generation_config.bucket_size = args.bucket_size if is_optimized else -1
    generation_config.do_sample = args.do_sample
    generation_config.num_beams = args.num_beams
    generation_config.bad_words_ids = bad_words_ids
    generation_config.force_words_ids = force_words_ids
    generation_config.num_return_sequences = args.num_return_sequences
    generation_config.trim_logits = args.trim_logits
    generation_config.attn_softmax_bf16 = args.attn_softmax_bf16
    generation_config.limit_hpu_graphs = args.limit_hpu_graphs
    generation_config.reuse_cache = args.reuse_cache

    if args.dataset_name is None:
        # Benchmark over the prompts below
        if args.prompt:
            input_sentences = args.prompt
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

        def generate():
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

            # Move inputs to target device(s)
            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to(args.device)

            outputs = model.generate(
                **input_tokens,
                generation_config=generation_config,
                lazy_mode=True,
                hpu_graphs=args.use_hpu_graphs,
                profiling_steps=args.profiling_steps,
                profiling_warmup_steps=args.profiling_warmup_steps,
            ).cpu()
            return tokenizer.batch_decode(outputs, skip_special_tokens=True)

        from optimum.habana.utils import HabanaProfile

        # compilation stage disable profiling
        HabanaProfile.disable()
        # Compilation
        if rank in [-1, 0]:
            logger.info("Graph compilation...")
        t0 = time.perf_counter()
        # The first three iterations take longer because of graph compilation
        for _ in range(args.warmup):
            generate()
        torch_hpu.synchronize()
        compilation_duration = time.perf_counter() - t0
        HabanaProfile.enable()
        total_new_tokens_generated = 0
        if rank in [-1, 0]:
            logger.info("Running generate...")
        t0 = time.perf_counter()
        # Benchmark over n_iterations iterations
        for i in range(args.n_iterations):
            generated = generate()
        duration = time.perf_counter() - t0
        total_new_tokens_generated = args.n_iterations * args.batch_size * args.max_new_tokens
        throughput = total_new_tokens_generated / duration

        if rank in [-1, 0]:
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
            if args.output_dir is not None:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                results = {
                    "throughput": throughput,
                    "output": output,
                }
                with (output_dir / "results.json").open("w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
            from optimum.habana.utils import get_hpu_memory_stats

            stats = f"Throughput (including tokenization) = {throughput} tokens/second"
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
            if rank in [-1, 0]:
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
                lazy_mode=True,
                hpu_graphs=args.use_hpu_graphs,
                profiling_steps=args.profiling_steps,
                profiling_warmup_steps=args.profiling_warmup_steps,
            ).cpu()
            return prompt, outputs

        # warmup
        from optimum.habana.utils import HabanaProfile
        # compilation stage disable profiling
        HabanaProfile.disable()
        # Compilation
        if rank in [-1, 0]:
            logger.info("Graph compilation...")
        t0 = time.perf_counter()
        if prompt_length > 0:
            for i, batch in enumerate(dataloader):
                generate_dataset(batch)
                # The first three iterations take longer because of graph compilation
                if (i + 1) == args.warmup:
                    break
        elif len(args.dataset_min_max) > 0 and args.bucket_size > -1:
            assert model.config.model_type in ["llama", 'opt'], 'This path only tested for opt, llama.'
            # warm in the case we have dataset, min, max shape is 
            mn, mx = args.dataset_min_max.split(',') # if dataset is given we Could compute min/max from it
            mn = int(mn)
            mx = int(mx)
            import math
            rounder = lambda x : int(math.ceil(x/args.bucket_size) * args.bucket_size)
            min_prompt_len = rounder(mn)
            max_sentence_len = rounder(mx + args.max_new_tokens)
            # we need to compile models for length:
            # min_prompt_len, min_prompt_len+b, min_prompt_len+2b, ... max_sentence_len
            for i, batch in enumerate(dataloader):
                break
            def adjust_batch(batch, size):
                curr_size = batch['input_ids'].shape[1]
                if curr_size >= size:
                    adjusted_batch = {'input_ids': batch['input_ids'][:,:size], 'attention_mask': batch['attention_mask'][:,:size]}
                else:
                    adjusted_batch = {}
                    for k in batch.keys():
                        last_colm = batch[k][:,-1]
                        expanded = last_colm.tile((size-curr_size,1)).T
                        adjusted_batch[k] = torch.concat([batch[k], expanded],1)
                assert adjusted_batch['input_ids'].shape[1] == size
                assert adjusted_batch['attention_mask'].shape[1] == size
                return adjusted_batch
            for _ in range(args.warmup):
                for sz in range(min_prompt_len, max_sentence_len+1, args.bucket_size):
                    print('Compiling for size', sz)
                    adjusted_size_batch = adjust_batch(batch, sz)
                    generate_dataset(adjusted_size_batch)
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
            if rank in [-1, 0]:
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
        if rank in [-1, 0]:
            from optimum.habana.utils import get_hpu_memory_stats

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


if __name__ == "__main__":
    main()
