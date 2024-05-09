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

# Adapted from https://huggingface.co/docs/transformers/main/en/model_doc/table-transformer

import argparse
import os

import habana_frameworks.torch as ht
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


adapt_transformers_to_gaudi()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="microsoft/table-transformer-detection",
        type=str,
        help="Path of the pre-trained model.",
    )
    parser.add_argument(
        "--dataset_name",
        default="nielsr/example-pdf",
        type=str,
        help="HuggingFace dataset repository name.",
    )
    parser.add_argument(
        "--filename",
        default="example_pdf.png",
        type=str,
        help="Filename of the image within the dataset repository or locally.",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use bf16 precision.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Download models if needed
    if os.path.isfile(args.filename):
        file_path = args.filename
    else:
        file_path = hf_hub_download(repo_id=args.dataset_name, repo_type="dataset", filename=args.filename)
    image = Image.open(file_path).convert("RGB")

    image_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)
    model = TableTransformerForObjectDetection.from_pretrained(args.model_name_or_path).to("hpu")
    if args.use_hpu_graphs:
        model = ht.hpu.wrap_in_hpu_graph(model)

    inputs = image_processor(images=image, return_tensors="pt").to("hpu")
    target_sizes = torch.tensor([image.size[::-1]])

    # Forward
    with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.bf16):
        outputs = model(**inputs)
    torch.hpu.synchronize()

    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = box.tolist()
        print(f"Detected {model.config.id2label[label.item()]} with confidence {score.item():.5f} at location {box}")


if __name__ == "__main__":
    main()
