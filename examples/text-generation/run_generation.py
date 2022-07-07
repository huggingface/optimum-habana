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
import logging
import os
import tempfile

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from habana_frameworks.torch.utils.library_loader import load_habana_module
from optimum.habana import GaudiConfig
from optimum.habana.modeling_utils import to_gaudi_for_accelerated_generation
from transformers import BloomForCausalLM, BloomTokenizerFast, GPT2LMHeadModel, GPT2Tokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
}


def generate_dummy_dataset(batch_size=32, sequence_length=32, n_iterations=80):
    my_dict = {"prompt": ["a" * sequence_length] * batch_size * n_iterations}
    return Dataset.from_dict(my_dict)


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # TODO: multi HPUs seed setting should be possible with SynapseAI 1.5
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)


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
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
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
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size")
    parser.add_argument("--input_size", type=int, default=128, help="Length of the input sequence")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for text generation")
    parser.add_argument("--n_iterations", type=int, default=80, help="Number of inference iterations")
    parser.add_argument("--bf16", action="store_true", help="Whether to use bf16 mixed-precision")
    parser.add_argument("--local_rank", type=int, default=-1, metavar="N", help="Local process rank.")
    args = parser.parse_args()

    # The following line is needed for lazy mode
    load_habana_module()

    args.device = torch.device("hpu")

    # The following lines are needed for distributed runs on HPUs
    _n_gpu = 1
    world_size = 1
    rank = -1
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        if "LOCAL_RANK" in os.environ:
            args.local_rank = int(os.environ["LOCAL_RANK"])
        logger.info("Torch distributed launch used")
    elif (
        "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ
        and "OMPI_COMM_WORLD_SIZE" in os.environ
        and "OMPI_COMM_WORLD_RANK" in os.environ
    ):
        args.local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        logger.info("MPI environment variables set")
    else:
        try:
            global mpi_comm
            from mpi4py import MPI

            mpi_comm = MPI.COMM_WORLD
            world_size = mpi_comm.Get_size()
            if world_size > 1:
                rank = mpi_comm.Get_rank()
                args.local_rank = rank
            else:
                raise ("Single MPI process")
        except Exception as e:
            logger.info("Single node run")
    if args.local_rank != -1:
        try:
            import habana_frameworks.torch.core.hccl
        except ImportError as error:
            error.msg = f"Could not import habana_frameworks.torch.core.hccl. {error.msg}."
            raise error
        os.environ["ID"] = str(args.local_rank)
        torch.distributed.init_process_group(backend="hccl", rank=args.local_rank, world_size=world_size)
        logger.info("Enabled distributed run.")

    # Load Gaudi configuration from the hub to get the list of bf16 ops if necessary
    if args.bf16:
        # Default BERT Gaudi configuration uses some bf16 ops
        gaudi_config_name = "bert-base-uncased"
    else:
        # Default gpt2 Gaudi configuration does not use any bf16 ops
        gaudi_config_name = "gpt2"
    gaudi_config = GaudiConfig.from_pretrained(f"Habana/{gaudi_config_name}")

    logger.warning(f"device: {args.device}, n_hpu: {world_size} bf16: {gaudi_config.use_habana_mixed_precision}")

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    # Convert the model to have custom Gaudi generation
    model = to_gaudi_for_accelerated_generation(model).eval()
    model.to(args.device)

    if hasattr(model.config, "max_position_embeddings"):
        max_position_embeddings = model.config.max_position_embeddings
    else:
        max_position_embeddings = 1024

    # Torch DDP in distributed mode
    if args.local_rank != -1:
        kwargs = {}
        kwargs["bucket_cap_mb"] = 230
        kwargs["gradient_as_bucket_view"] = True
        kwargs["find_unused_parameters"] = False
        model_wrapped = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=None,
            output_device=None,
            **kwargs,
        )
    else:
        model_wrapped = model

    args.length = adjust_length_to_model(args.length, max_sequence_length=max_position_embeddings)
    logger.info(args)

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
                    opt_level=gaudi_config.hmp_opt_level,
                    bf16_file_path=hmp_bf16_file.name,
                    fp32_file_path=hmp_fp32_file.name,
                    isVerbose=gaudi_config.hmp_is_verbose,
                )

    # Generate the dataset to use
    dummy_dataset = generate_dummy_dataset(args.batch_size, args.input_size, args.n_iterations)

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

    if args.local_rank != -1:
        # With DDP the generate method of the model is accessed differently
        generate_method_to_call = model_wrapped.module.generate
    else:
        generate_method_to_call = model_wrapped.generate

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
        output_sequences = generate_method_to_call(
            input_ids=batch["input_ids"].to(args.device),
            max_length=args.length + batch["input_ids"].shape[-1],
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            # do_sample=True,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            lazy_mode=True,
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
