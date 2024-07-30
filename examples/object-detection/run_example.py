#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# Copied from https://huggingface.co/docs/transformers/model_doc/owlvit

import argparse
import time

import habana_frameworks.torch as ht
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, DetrForObjectDetection

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="facebook/detr-resnet-101",
        type=str,
        help="Path of the pre-trained model",
    )
    parser.add_argument(
        "--image_path",
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        type=str,
        help='Path of the input image. Should be a single string (eg: --image_path "URL")',
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use bf16 precision for object detection.",
    )
    parser.add_argument(
        "--detect_threshold",
        type=float,
        default=0.9,
        help="Detection threshold score (otherwise dismissed)",
    )
    parser.add_argument(
        "--print_result",
        action="store_true",
        help="Whether to print the detection results.",
    )

    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument(
        "--n_iterations", type=int, default=10, help="Number of inference iterations for benchmarking."
    )

    args = parser.parse_args()

    adapt_transformers_to_gaudi()

    # you can specify the revision tag if you don't want the timm dependency
    processor = AutoProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")

    image = Image.open(requests.get(args.image_path, stream=True).raw)

    inputs = processor(images=image, return_tensors="pt").to("hpu")
    model.to("hpu")

    if args.use_hpu_graphs:
        model = ht.hpu.wrap_in_hpu_graph(model)

    autocast = torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.bf16)

    with torch.no_grad(), autocast:
        for i in range(args.warmup):
            inputs = processor(images=image, return_tensors="pt").to("hpu")
            outputs = model(**inputs)
            torch.hpu.synchronize()

        total_model_time = 0
        for i in range(args.n_iterations):
            inputs = processor(images=image, return_tensors="pt").to("hpu")
            model_start_time = time.time()
            outputs = model(**inputs)
            torch.hpu.synchronize()
            model_end_time = time.time()
            total_model_time = total_model_time + (model_end_time - model_start_time)

    if args.print_result:
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=args.detect_threshold
        )[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

tot_stat = f"Total latency (ms): {str(total_model_time * 1000)} (for n_iterations={str(args.n_iterations)}) "
avg_stat = f"Average latency (ms): {str(total_model_time * 1000 / args.n_iterations)} (per iteration) "
separator = "-" * max(len(tot_stat), len(avg_stat))
print()
print("Stats:")
print(separator)
print(tot_stat)
print(avg_stat)
print(separator)
