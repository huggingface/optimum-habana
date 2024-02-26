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
        default="Salesforce/blip-image-captioning-large",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--image_path",
        default="https://ankur3107.github.io/assets/images/image-captioning-example.png",
        type=str,
        help="Path to image",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    args = parser.parse_args()

    adapt_transformers_to_gaudi()

    image = PIL.Image.open(requests.get(args.image_path, stream=True, timeout=3000).raw)

    generator = pipeline(
        "image-to-text",
        model=args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device="hpu",
    )
    generate_kwargs = {"lazy_mode": True, "hpu_graphs": args.use_hpu_graphs}
    if args.use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        generator.model = wrap_in_hpu_graph(generator.model)

    # warm up
    for i in range(5):
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
            generator(image, generate_kwargs=generate_kwargs)

    start = time.time()
    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
        result = generator(image, generate_kwargs=generate_kwargs)
    end = time.time()
    logger.info(f"result = {result}, time = {(end-start) * 1000}ms")


if __name__ == "__main__":
    main()
