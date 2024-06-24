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

# Copied from https://huggingface.co/docs/transformers/main/en/model_doc/clipseg

import argparse
import time

import habana_frameworks.torch as ht
import requests
import torch
from PIL import Image
from torchvision.utils import save_image
from transformers import AutoProcessor, CLIPSegForImageSegmentation

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="CIDAS/clipseg-rd64-refined",
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
        "--prompt",
        default="a cat,a remote,a blanket",
        type=str,
        help='Prompt for classification. It should be a string seperated by comma. (eg: --prompt "a photo of a cat, a photo of a dog")',
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use bf16 precision for classification.",
    )
    parser.add_argument(
        "--print_result",
        action="store_true",
        help="Whether to print the classification results.",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of inference iterations for benchmarking.")

    args = parser.parse_args()

    adapt_transformers_to_gaudi()

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = CLIPSegForImageSegmentation.from_pretrained(
        args.model_name_or_path
    )  # Use CLIPSegForImageSegmentation instead of automodel.
    #  The output will contains the logits which are required to generated segmented images

    image = Image.open(requests.get(args.image_path, stream=True).raw)
    texts = []
    for text in args.prompt.split(","):
        texts.append(text)

    if args.use_hpu_graphs:
        model = ht.hpu.wrap_in_hpu_graph(model)

    autocast = torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.bf16)
    model.to("hpu")

    with torch.no_grad(), autocast:
        for i in range(args.warmup):
            inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt").to("hpu")
            outputs = model(**inputs)
            torch.hpu.synchronize()

        total_model_time = 0
        for i in range(args.n_iterations):
            inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt").to("hpu")
            model_start_time = time.time()
            outputs = model(**inputs)
            torch.hpu.synchronize()
            model_end_time = time.time()
            total_model_time = total_model_time + (model_end_time - model_start_time)

            if args.print_result:
                if i == 0:  # generate/output once only
                    logits = outputs.logits
                    for j in range(logits.shape[0]):
                        threshold = 0.5
                        segmented_image = ((torch.sigmoid(logits[j]) > threshold) * 255).unsqueeze(0)
                        segmented_image = segmented_image.to(torch.float32)
                        save_image(segmented_image, "segmented_" + texts[j].strip() + ".png")
                    print("Segmented images are generated.")

    print("n_iterations: " + str(args.n_iterations))
    print("Total latency (ms): " + str(total_model_time * 1000))
    print("Average latency (ms): " + str(total_model_time * 1000 / args.n_iterations))
