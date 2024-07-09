#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import logging
import time

import habana_frameworks.torch as ht
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


# Adapted from https://huggingface.co/Supabase/gte-small example

adapt_transformers_to_gaudi()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


SOURCE_SENTENCE = "what is the capital of China?"
COMPARE_TEXTS = [
    "how to implement quick sort in Python?",
    "Beijing",
    "sorting algorithms",
]


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="Supabase/gte-small",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--source_sentence",
        default=SOURCE_SENTENCE,
        type=str,
        help="Source sentence to compare with",
    )
    parser.add_argument(
        "--input_texts",
        default=COMPARE_TEXTS,
        type=str,
        nargs="+",
        help='Text input. Can be a single string (eg: --input_texts "text1"), or a list of space-separated strings (eg: --input_texts "text1" "text2")',
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to wrap model in HPU graph mode (recommended)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations for benchmarking.",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=5,
        help="Number of inference iterations for benchmarking.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path).to("hpu")
    if args.use_hpu_graphs:
        model = ht.hpu.wrap_in_hpu_graph(model)
    input_texts = [args.source_sentence] + args.input_texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt").to("hpu")

    if args.warmup:
        logger.info(f"Initializing warmup for {args.warmup} iterations")
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.bf16), torch.no_grad():
            for _ in tqdm(range(args.warmup), leave=False):
                model(**batch_dict)
        torch.hpu.synchronize()

    start_time = time.time()
    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.bf16), torch.no_grad():
        for _ in tqdm(range(args.n_iterations), leave=False):
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    torch.hpu.synchronize()
    end_time = time.time()
    logger.info(f"Total time: {end_time - start_time:.5f} s")
    logger.info(f"Average time per iteration: {(end_time - start_time) * 1000 / args.n_iterations:.5f} ms")
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:1] @ embeddings[1:].T) * 100
    logger.info(f"Scores for input texts relating to the source sentence: {scores.tolist()}")


if __name__ == "__main__":
    main()
