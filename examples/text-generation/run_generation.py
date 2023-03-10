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
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.utils import is_offline_mode


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def generate_dataset(prompt, batch_size=4, sequence_length=32, n_iterations=80):
    my_dict = {"prompt": [prompt] * batch_size * n_iterations}
    return Dataset.from_dict(my_dict)


def get_repo_root(model_name_or_path, global_rank):
    # checks if online or not
    if is_offline_mode():
        if global_rank == 0:
            print("Offline mode: forcing local_files_only=True")

    # download only on first process
    if global_rank == 0:
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


def get_checkpoint_files(model_name_or_path, global_rank):
    cached_repo_dir = get_repo_root(model_name_or_path, global_rank)

    # extensions: .bin | .pt
    # creates a list of paths from all downloaded files in cache dir
    file_list = [str(entry) for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]") if entry.is_file()]
    return file_list


def write_checkpoints_json(model_name_or_path, global_rank, checkpoints_json):
    checkpoint_files = get_checkpoint_files(model_name_or_path, global_rank)
    if global_rank == 0:
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
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--seed", type=int, default=27, help="random seed for initialization")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size")
    parser.add_argument("--input_size", type=int, default=128, help="Length of the input sequence")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for text generation")
    parser.add_argument("--n_iterations", type=int, default=80, help="Number of inference iterations")
    parser.add_argument("--local_rank", type=int, default=-1, metavar="N", help="Local process rank.")
    parser.add_argument("--use_hpu_graphs", action="store_true", help="Whether to use HPU graphs or not.")
    parser.add_argument(
        "--gaudi_config_name_or_path",
        default=None,
        type=str,
        help="Path to the Gaudi configuration",
    )

    args = parser.parse_args()

    # Disable lazy mode if HPU graphs are not used
    if not args.use_hpu_graphs:
        os.environ["PT_HPU_LAZY_MODE"] = "2"

    # If the DeepSpeed launcher is used, the env variable _ will be equal to /usr/local/bin/deepspeed
    use_deepspeed = "deepspeed" in os.environ["_"]
    if use_deepspeed:
        # Set necessary env variables
        os.environ.setdefault("WA_BETA_ALIBI", "1")
        os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
        os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")

    # Device is HPU
    args.device = "hpu"
    # Get world size, rank and local rank
    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

    world_size, rank, args.local_rank = initialize_distributed_hpu()
    from optimum.habana.utils import set_seed

    set_seed(args.seed)

    # If the DeepSpeed launcher is used, the env variable _ will be equal to /usr/local/bin/deepspeed
    use_deepspeed = "deepspeed" in os.environ["_"]

    # Initialize the various processes if this is a multi-device run
    if args.local_rank != -1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="hccl", rank=rank, world_size=world_size)
            logger.info("Enabled distributed run.")
    else:
        logger.info("Single node run.")

    # Tweak generation so that it runs faster on Gaudi
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

    adapt_transformers_to_gaudi()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

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
            write_checkpoints_json(args.model_name_or_path, rank, checkpoints_json)
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

        gaudi_config = GaudiConfig.from_pretrained(args.gaudi_config_name_or_path)

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
    if args.local_rank in [-1, 0]:
        logger.info(f"Args: {args}")
        logger.info(
            f"device: {args.device}, n_hpu: {world_size}, bf16: {use_deepspeed or gaudi_config.use_habana_mixed_precision}"
        )

    # Generate the dataset to use
    prompt = "My name is Michael and I live in"
    dummy_dataset = generate_dataset(prompt, args.batch_size, args.input_size, args.n_iterations)

    def tokenize_function(examples):
        return tokenizer(examples["prompt"])

    tokenized_dataset = dummy_dataset.map(tokenize_function)
    tokenized_dataset = tokenized_dataset.with_format("torch", columns=["input_ids", "attention_mask"])

    if args.local_rank != -1:
        # In distributed mode, a distributed sampler is required
        sampler = DistributedSampler(
            tokenized_dataset,
            num_replicas=world_size,
            rank=args.local_rank,
            seed=args.seed,
        )
    else:
        sampler = None

    dataloader = DataLoader(dataset=tokenized_dataset, sampler=sampler, batch_size=args.batch_size)

    # TQDM
    if args.local_rank in [-1, 0]:
        steps_progress_bar = tqdm(total=len(dataloader.dataset) // (world_size * args.batch_size))
    else:
        steps_progress_bar = None

    if args.local_rank != -1:
        # Synchronize all processes before generation in distributed mode
        torch.distributed.barrier()

    for batch in dataloader:
        # Update TQDM bar
        if steps_progress_bar is not None:
            steps_progress_bar.update(1)
        # Generation
        output_sequences = model.generate(
            input_ids=batch["input_ids"].to(args.device),
            max_length=args.length + batch["input_ids"].shape[-1],
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            # do_sample=True,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            hpu_graphs=args.use_hpu_graphs,
        )

    if steps_progress_bar is not None:
        steps_progress_bar.close()

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    if args.local_rank != -1:
        torch.distributed.barrier()

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(args.stop_token) if args.stop_token else None]

        # # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        # total_sequence = (
        #     text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        # )

        generated_sequences.append(text)
        print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1}, process {args.local_rank} === {text}")

    return generated_sequences


if __name__ == "__main__":
    main()
