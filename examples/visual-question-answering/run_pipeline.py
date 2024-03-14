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

import PIL.Image
import requests
import torch
from transformers import pipeline

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--image_path",
        default=None,
        type=str,
        nargs="*",
        help='Path to image as input. Can be a single string (eg: --image_path "URL1"), or a list of space-separated strings (eg: --image_path "URL1" "URL2")',
    )
    parser.add_argument(
        "--topk",
        default=1,
        type=int,
        help="topk num",
    )
    parser.add_argument(
        "--question",
        default=None,
        type=str,
        nargs="*",
        help='question as input. Can be a single string (eg: --question "Q1"), or a list of space-separated strings (eg: --question "Q1" "Q2")',
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform in bf16 precision.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of inference iterations for benchmarking.")
    args = parser.parse_args()

    adapt_transformers_to_gaudi()
    image_paths = args.image_path
    image_paths_len = len(image_paths)

    if args.batch_size > image_paths_len:
        # Dynamically extends to support larger batch sizes
        num_path_to_add = args.batch_size - image_paths_len
        for i in range(num_path_to_add):
            image_paths.append(image_paths[i % image_paths_len])
    elif args.batch_size < image_paths_len:
        image_paths = image_paths[: args.batch_size]

    questions = args.question
    questions_len = len(questions)
    if args.batch_size > questions_len:
        # Dynamically extends to support larger batch sizes
        num_question_to_add = args.batch_size - questions_len
        for i in range(num_question_to_add):
            questions.append(questions[i % questions_len])
    elif args.batch_size < questions_len:
        questions = questions[: args.batch_size]

    images = []

    for image_path in image_paths:
        images.append(PIL.Image.open(requests.get(image_path, stream=True, timeout=3000).raw).convert("RGB"))

    if args.bf16:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    generator = pipeline(
        "visual-question-answering",
        model=args.model_name_or_path,
        torch_dtype=model_dtype,
        device="hpu",
    )
    if args.use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        generator.model = wrap_in_hpu_graph(generator.model)

    autocast_enable = model_dtype == torch.bfloat16
    model_input = []
    for i in range(args.batch_size):
        model_input.append({"image": images[i], "question": questions[i]})

    # warm up
    for i in range(args.warmup):
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=autocast_enable):
            generator(model_input, batch_size=args.batch_size, topk=args.topk)

    start = time.time()
    for i in range(args.n_iterations):
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=autocast_enable):
            result = generator(model_input, batch_size=args.batch_size, topk=args.topk)
    end = time.time()
    logger.info(f"result = {result}, time = {(end-start) * 1000/args.n_iterations}ms")


if __name__ == "__main__":
    main()
