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
""" Conditional text generation with the auto-regressive models of the library (GPT2/BLOOM)
"""

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.utils import is_offline_mode


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def get_repo_root(model_name_or_path, local_rank):
    # checks if online or not
    if is_offline_mode():
        if local_rank == 0:
            print("Offline mode: forcing local_files_only=True")

    # download only on first process
    if local_rank == 0:
        snapshot_download(
            model_name_or_path,
            local_files_only=is_offline_mode(),
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            ignore_patterns=["*.safetensors"],
        )

    torch.distributed.barrier()

    return snapshot_download(
        model_name_or_path,
        local_files_only=is_offline_mode(),
        cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
        ignore_patterns=["*.safetensors"],
    )


def get_checkpoint_files(model_name_or_path, local_rank):
    cached_repo_dir = get_repo_root(model_name_or_path, local_rank)

    # extensions: .bin | .pt
    # creates a list of paths from all downloaded files in cache dir
    file_list = [str(entry) for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]") if entry.is_file()]
    return file_list


def write_checkpoints_json(model_name_or_path, local_rank, checkpoints_json):
    checkpoint_files = get_checkpoint_files(model_name_or_path, local_rank)
    if local_rank == 0:
        data = {"type": "BLOOM", "checkpoints": checkpoint_files, "version": 1.0}
        json.dump(data, open(checkpoints_json, "w"))


#
# Functions to prepare models' input
#


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    # Arguments management
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model",
    )
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of inference iterations")
    parser.add_argument("--local_rank", type=int, default=-1, metavar="N", help="Local process rank.")
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )
    parser.add_argument("--use_hpu_graphs", action="store_true", help="Whether to use HPU graphs or not.")
    parser.add_argument(
        "--gaudi_config_name_or_path",
        default=None,
        type=str,
        help="Path to the Gaudi configuration",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        help="Optional argument if you want to assess ypur model on a given dataset of the HF Hub.",
    )

    args = parser.parse_args()

    # Disable lazy mode if HPU graphs are not used
    if not args.use_hpu_graphs:
        os.environ["PT_HPU_LAZY_MODE"] = "2"

    # If the DeepSpeed launcher is used, the env variable _ will be equal to /usr/local/bin/deepspeed
    # For multi node, the value of the env variable WORLD_SIZE should be larger than 8
    use_deepspeed = "deepspeed" in os.environ["_"] or (
        "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 8
    )
    if use_deepspeed:
        # Set necessary env variables
        os.environ.setdefault("WA_BETA_ALIBI", "1")
        os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
        os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")

    # Device is HPU
    args.device = "hpu"
    import habana_frameworks.torch.hpu as torch_hpu

    # Get world size, rank and local rank
    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

    world_size, rank, args.local_rank = initialize_distributed_hpu()

    # Initialize the various processes if this is a multi-device run
    if args.local_rank != -1:
        # if not torch.distributed.is_initialized():
        #     torch.distributed.init_process_group(backend="hccl", rank=rank, world_size=world_size)
        logger.info("Enabled distributed run.")
    else:
        logger.info("Single node run.")

    # Tweak generation so that it runs faster on Gaudi
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

    adapt_transformers_to_gaudi()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # print("HEEEEEEEERE", tokenizer.pad_token, tokenizer.eos_token)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.padding_side = "left"

    # Single device
    if args.local_rank == -1:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        model = model.eval().to(args.device)
    # Multi device
    else:
        # DeepSpeed inference
        if use_deepspeed:
            if "bloom" not in args.model_name_or_path:
                raise ValueError(
                    f"DeepSpeed-inference on Gaudi is only available with BLOOM at the moment, got {args.model_name_or_path}."
                )

            import deepspeed

            deepspeed.init_distributed(dist_backend="hccl")

            config = AutoConfig.from_pretrained(args.model_name_or_path)

            # Construct model with fake meta tensors, later will be replaced on devices during ds-inference ckpt load
            with deepspeed.OnDevice(dtype=torch.bfloat16, device="meta"):
                model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
            model = model.eval()

            checkpoints_json = "checkpoints.json"
            write_checkpoints_json(args.model_name_or_path, args.local_rank, checkpoints_json)
            torch.distributed.barrier()

            model = deepspeed.init_inference(
                model,
                mp_size=world_size,
                dtype=torch.bfloat16,
                injection_policy={BloomBlock: ("self_attention.dense", "mlp.dense_4h_to_h")},
                checkpoint=checkpoints_json,
                args=args,
                enable_cuda_graph=args.use_hpu_graphs,
            )
            model.module.split_lm_head()
            model = model.module
        # Torch DDP
        else:
            torch.distributed.init_process_group(backend="hccl", rank=rank, world_size=world_size)
            model = torch.nn.parallel.DistributedDataParallel(model)

    if not use_deepspeed:
        # Wrapper for HPU graphs
        if args.use_hpu_graphs:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph

            model = wrap_in_hpu_graph(model)

        # Load Gaudi configuration
        from optimum.habana import GaudiConfig

        # gaudi_config = GaudiConfig.from_pretrained("args.gaudi_config_name_or_path")
        gaudi_config = GaudiConfig(use_habana_mixed_precision=True)

        # Tell Gaudi the bf16 ops to use
        if gaudi_config.use_habana_mixed_precision:
            from habana_frameworks.torch.hpex import hmp

            # Open temporary files to write mixed-precision ops
            with tempfile.NamedTemporaryFile() as hmp_bf16_file:
                with tempfile.NamedTemporaryFile() as hmp_fp32_file:
                    # hmp.convert needs ops to be written in text files
                    gaudi_config.write_bf16_fp32_ops_to_text_files(
                        hmp_bf16_file.name,
                        hmp_fp32_file.name,
                    )
                    hmp.convert(
                        bf16_file_path=hmp_bf16_file.name,
                        fp32_file_path=hmp_fp32_file.name,
                        isVerbose=gaudi_config.hmp_is_verbose,
                    )

    # if hasattr(model.config, "max_position_embeddings"):
    #     max_position_embeddings = model.config.max_position_embeddings
    # else:
    #     max_position_embeddings = 1024

    # args.length = adjust_length_to_model(args.length, max_sequence_length=max_position_embeddings)
    if rank in [-1, 0]:
        logger.info(f"Args: {args}")
        logger.info(
            f"device: {args.device}, n_hpu: {world_size}, bf16: {use_deepspeed or gaudi_config.use_habana_mixed_precision}"
        )

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        use_cache=args.use_kv_cache,
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
            # dynamically extend to support larger batch sizes
            num_sentences_to_add = args.batch_size - len(input_sentences)
            for i in range(num_sentences_to_add):
                input_sentences.append(input_sentences[i % len(input_sentences)])
        elif args.batch_size < len(input_sentences):
            input_sentences = input_sentences[: args.batch_size]

        def generate():
            """Returns a list of zipped outputs and number of new tokens."""

            # Tokenization
            input_tokens = tokenizer.batch_encode_plus(input_sentences, return_tensors="pt", padding=True)

            # Pad inputs to have static shapes during genration, this gives better performance than dynamic shapes on HPUs
            input_token_len = input_tokens.input_ids.shape[-1]
            input_tokens["input_ids"] = F.pad(
                input_tokens.input_ids, (0, args.max_new_tokens), value=model.config.pad_token_id
            )
            input_tokens["attention_mask"] = F.pad(input_tokens.attention_mask, (0, args.max_new_tokens), value=0)
            # token_idx is the current index in the generation process, it is incremented each time a new token is generated
            kwargs = {"token_idx": torch.tensor(input_token_len, device=args.device)}

            # Move inputs to target device(s)
            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to(args.device)

            outputs = model.generate(
                **input_tokens,
                **kwargs,
                generation_config=generation_config,
                lazy_mode=args.use_hpu_graphs,
                hpu_graphs=args.use_hpu_graphs,
            ).cpu()
            return tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Compilation
        if args.use_hpu_graphs:
            if rank in [-1, 0]:
                print("Graph compilation...")
            t0 = time.perf_counter()
            # The first two iterations take longer because of graph compilation
            for _ in range(2):
                generate()
            torch_hpu.synchronize()
            compilation_duration = time.perf_counter() - t0

        total_new_tokens_generated = 0
        if rank in [-1, 0]:
            print("Running generate...")
        t0 = time.perf_counter()
        # Benchmark over n_iterations iterations
        for i in range(args.n_iterations):
            generated = generate()
        duration = time.perf_counter() - t0
        total_new_tokens_generated = args.n_iterations * args.batch_size * args.max_new_tokens
        throughput = total_new_tokens_generated / duration

        if rank in [-1, 0]:
            print("*** Summary ***")
            print(f"Throughput (including tokenization) = {throughput} tokens/second")
            if args.use_hpu_graphs:
                print(f"Graph compilation duration = {compilation_duration} seconds")
            print()
            print("Input sentences:")
            for i, input_sentence in enumerate(input_sentences):
                print(f"{i+1}: {input_sentence}")
            print()
            print("Outputs:")
            for i, output in enumerate(generated):
                print(f"{i+1}: {output}")
    else:
        # Compute the perplexity over the given dataset
        # Downloading and loading a dataset from the hub.
        from datasets import load_dataset
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        raw_dataset = load_dataset(args.dataset_name)["test"]
        prompt_length = 16

        def preprocess_function(examples):
            # Tokenize the texts
            return tokenizer(examples["headline"], padding="max_length", max_length=prompt_length, truncation=True)

        raw_dataset = raw_dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )
        raw_dataset = raw_dataset.remove_columns(["content", "category"])
        raw_dataset.set_format(type="torch")

        dataloader = DataLoader(raw_dataset, batch_size=args.batch_size)
        for i, batch in enumerate(dataloader):
            # Move inputs to target device(s)
            batch["input_ids"] = F.pad(batch["input_ids"], (0, args.max_new_tokens), value=model.config.pad_token_id)
            batch["attention_mask"] = F.pad(batch["attention_mask"], (0, args.max_new_tokens), value=0)
            prompt = batch.pop("headline")
            for t in batch:
                if torch.is_tensor(batch[t]):
                    batch[t] = batch[t].to(args.device)
            batch["token_idx"] = torch.tensor(prompt_length, device=args.device)

            outputs = model.generate(
                **batch,
                generation_config=generation_config,
                lazy_mode=args.use_hpu_graphs,
                hpu_graphs=args.use_hpu_graphs,
            ).cpu()
            if rank in [-1, 0]:
                print(
                    f"Sample {i+1}; Input: {prompt[0]}; Output: ",
                    tokenizer.batch_decode(outputs, skip_special_tokens=True)[0],
                )


if __name__ == "__main__":
    main()
