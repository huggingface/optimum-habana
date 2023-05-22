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
import logging
import os
import time

import torch
import torch.nn.functional as F
from checkpoint_utils import model_is_bloom, write_checkpoints_json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


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
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of inference iterations.")
    parser.add_argument("--local_rank", type=int, default=-1, metavar="N", help="Local process rank.")
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )
    parser.add_argument("--use_hpu_graphs", action="store_true", help="Whether to use HPU graphs or not.")
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
        help="Optional argument if you want to assess your model on a given dataset of the HF Hub, this will be the name of the column to use as prompts for generation.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling for generation.",
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

    # Device is HPU
    args.device = "hpu"
    import habana_frameworks.torch.hpu as torch_hpu

    # Get world size, rank and local rank
    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

    world_size, rank, args.local_rank = initialize_distributed_hpu()

    if use_deepspeed:
        # Check if DeepSpeed is installed
        from transformers.deepspeed import is_deepspeed_available

        if not is_deepspeed_available():
            raise ImportError(
                "This script requires deepspeed: `pip install" " git+https://github.com/HabanaAI/DeepSpeed.git@1.9.0`."
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if use_deepspeed or args.bf16:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float

    if use_deepspeed:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        is_bloom = model_is_bloom(config)

        if is_bloom:
            # Construct model with fake meta tensors, later will be replaced on devices during ds-inference ckpt load
            with deepspeed.OnDevice(dtype=model_dtype, device="meta"):
                model = AutoModelForCausalLM.from_config(config, torch_dtype=model_dtype)
        else:
            with deepspeed.OnDevice(dtype=model_dtype, device=args.device):
                model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype)
        model = model.eval()

        # Initialize the model
        ds_inference_kwargs = {"dtype": model_dtype}
        ds_inference_kwargs["tensor_parallel"] = {"tp_size": world_size}
        ds_inference_kwargs["enable_cuda_graph"] = args.use_hpu_graphs

        # BLOOM is managed differently
        if is_bloom:
            checkpoints_json = "checkpoints.json"
            write_checkpoints_json(args.model_name_or_path, args.local_rank, checkpoints_json)

            # Make sure all devices/nodes have access to the model checkpoints
            torch.distributed.barrier()

            from transformers.models.bloom.modeling_bloom import BloomBlock

            ds_inference_kwargs["injection_policy"] = {BloomBlock: ("self_attention.dense", "mlp.dense_4h_to_h")}
            ds_inference_kwargs["checkpoint"] = checkpoints_json

        model = deepspeed.init_inference(model, **ds_inference_kwargs)
        if is_bloom:
            model.module.split_lm_head()
        model = model.module
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype)
        model = model.eval().to(args.device)
        is_bloom = model_is_bloom(model.config)

        if args.use_hpu_graphs:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph

            model = wrap_in_hpu_graph(model)

    # Some models like GPT2 do not have a PAD token so we have to set it if necessary
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    if rank in [-1, 0]:
        logger.info(f"Args: {args}")
        logger.info(f"device: {args.device}, n_hpu: {world_size}, bf16: {use_deepspeed or args.bf16}")

    # Generation configuration
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        use_cache=args.use_kv_cache,
        do_sample=args.do_sample,
    )

    if args.dataset_name is None:
        # Benchmark over the prompts below
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
            input_tokens = tokenizer.batch_encode_plus(input_sentences, return_tensors="pt", padding=True)

            # Pad inputs to have static shapes during generation, this gives better performance than dynamic shapes on HPUs
            input_token_len = input_tokens.input_ids.shape[-1]
            input_tokens["input_ids"] = F.pad(
                input_tokens.input_ids, (0, args.max_new_tokens), value=model.config.pad_token_id
            )
            input_tokens["attention_mask"] = F.pad(input_tokens.attention_mask, (0, args.max_new_tokens), value=0)
            if is_bloom:
                # token_idx is the current index in the generation process, it is incremented each time a new token is generated
                kwargs = {"token_idx": torch.tensor(input_token_len, device=args.device)}
            else:
                kwargs = {}

            # Move inputs to target device(s)
            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to(args.device)

            outputs = model.generate(
                **input_tokens,
                **kwargs,
                generation_config=generation_config,
                lazy_mode=True,
                hpu_graphs=args.use_hpu_graphs,
            ).cpu()
            return tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Compilation
        if rank in [-1, 0]:
            logger.info("Graph compilation...")
        t0 = time.perf_counter()
        # The first three iterations take longer because of graph compilation
        for _ in range(3):
            generate()
        torch_hpu.synchronize()
        compilation_duration = time.perf_counter() - t0

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
            stats = f"Throughput (including tokenization) = {throughput} tokens/second"
            separator = "-" * len(stats)
            print()
            print("Stats:")
            print(separator)
            print(stats)
            if args.use_hpu_graphs:
                print(f"Graph compilation duration = {compilation_duration} seconds")
            print(separator)
            print()
            print("Input/outputs:")
            print(separator)
            for i, (input_sentence, output) in enumerate(zip(input_sentences, generated)):
                print(f"input {i+1}: {input_sentence}")
                print(f"output {i+1}: {output}")
                print(separator)
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
        raw_dataset = raw_dataset[split]

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

        # Set the prompt length to 16
        prompt_length = 16

        def preprocess_function(examples):
            # Tokenize the texts
            return tokenizer(examples[column_name], padding="max_length", max_length=prompt_length, truncation=True)

        raw_dataset = raw_dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )
        # After tokenization, we can remove the column of interest
        raw_dataset = raw_dataset.remove_columns([column_name])
        raw_dataset.set_format(type="torch")

        separator = None

        logger.info("Running generation...")

        dataloader = DataLoader(raw_dataset, batch_size=args.batch_size)
        for i, batch in enumerate(dataloader):
            prompt = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            # Pad inputs to have static shapes during generation, this gives better performance than dynamic shapes on HPUs
            batch["input_ids"] = F.pad(batch["input_ids"], (0, args.max_new_tokens), value=model.config.pad_token_id)
            batch["attention_mask"] = F.pad(batch["attention_mask"], (0, args.max_new_tokens), value=0)
            # prompt = batch.pop(column_name)
            # Move inputs to target device(s)
            for t in batch:
                if torch.is_tensor(batch[t]):
                    batch[t] = batch[t].to(args.device)
            if is_bloom:
                # token_idx is the current index in the generation process, it is incremented each time a new token is generated
                batch["token_idx"] = torch.tensor(prompt_length, device=args.device)

            # Generate new sequences
            outputs = model.generate(
                **batch,
                generation_config=generation_config,
                lazy_mode=args.use_hpu_graphs,
                hpu_graphs=args.use_hpu_graphs,
            ).cpu()

            # Print outputs
            if separator is None:
                separator = "-" * len(prompt[0])
            if rank in [-1, 0]:
                print(separator)
                print(f"Batch n°{i+1}")
                print(f"Input: {prompt[:args.batch_size]}")
                print(f"Output: {tokenizer.batch_decode(outputs, skip_special_tokens=True)[:args.batch_size]}")


if __name__ == "__main__":
    main()
