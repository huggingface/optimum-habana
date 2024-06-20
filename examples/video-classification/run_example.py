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
# limitations under the License.

# Loosely adapted from https://github.com/huggingface/optimum-habana/pull/783/files#diff-8361a5cbb8a1de8387eaff47125cce70f695f2a5994c66725c942c071835e82b

import argparse
import io
import logging
import os
import time

import decord
import habana_frameworks.torch as ht
import requests
import torch
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


adapt_transformers_to_gaudi()


def load_video(path):
    vr = decord.VideoReader(path)
    batch = vr.get_batch(list(range(16))).asnumpy()
    buf = [batch[i, :, :, :] for i in range(16)]
    logging.info(batch.shape)
    return buf


def download_file(link: str):
    resp = requests.get(link)
    return io.BytesIO(resp.content)


def get_image_buffers(video_paths: list[str]):
    for vp in video_paths:
        logging.info(f"Extracting images from {vp}")
        try:
            if vp.startswith("https://") or vp.startswith("http://"):
                file = download_file(vp)
                yield load_video(file)
            elif os.path.isfile(vp):
                yield load_video(vp)
            else:
                logging.error(f"Video path {vp} is not link or a file.")
        except Exception as e:
            logging.error(f"Error extracting video information from {vp}")
            logging.error(f"Trace: {e}")
            continue


def infer(model, inputs, cast_bf16: bool):
    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=cast_bf16), torch.no_grad():
        outputs = model(**inputs)
    torch.hpu.synchronize()
    predicted_class_idx = outputs.logits.argmax(-1).item()
    class_str = model.config.id2label[predicted_class_idx]
    return class_str


def run(
    model_name: str,
    video_paths: list[str],
    warm_up_epcohs: int,
    use_hpu_graphs: bool,
    cast_bf16: bool,
):
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    device = torch.device("hpu")
    model = VideoMAEForVideoClassification.from_pretrained(model_name)
    if use_hpu_graphs:
        model = ht.hpu.wrap_in_hpu_graph(model)
    model = model.to(device)
    model.eval()

    bufs = list(get_image_buffers(video_paths))

    start_time = time.time()
    if warm_up_epcohs:
        logging.info(f"Warming up model with {warm_up_epcohs} epochs")
    for i in tqdm(range(warm_up_epcohs), leave=False):
        for buf in bufs:
            inputs = processor(buf, return_tensors="pt")
            inputs.to(device)
            infer(model, inputs, cast_bf16)
    if warm_up_epcohs:
        end_time = time.time()
        logging.info(f"Completed warm up in {end_time - start_time:.3e} seconds")

    for i, buf in enumerate(bufs):
        start_time = time.time()
        inputs = processor(buf, return_tensors="pt")
        inputs.to(device)
        class_str = infer(model, inputs, cast_bf16)
        end_time = time.time()

        print(
            f"Predicted class for {video_paths[i].split('/')[-1]} is {class_str} and took {end_time - start_time:.3e} seconds"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="MCG-NJU/videomae-base-finetuned-kinetics",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--video_paths",
        default=[
            "https://ak.picdn.net/shutterstock/videos/21179416/preview/stock-footage-aerial-shot-winter-forest.mp4"
        ],
        type=str,
        nargs="*",
        help="Paths to video input. Can specify multiple in a space-separated list",
    )
    parser.add_argument(
        "--warm_up_epochs",
        "-w",
        default=0,
        type=int,
        help="Number of epochs to warm up the model",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        "-g",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--bf16",
        "-b",
        action="store_true",
        help="Whether to perform in bf16 precision.",
    )
    parser.add_argument(
        "--log_level",
        default=None,
        type=int,
        help="Log level for printout information",
    )

    args = parser.parse_args()

    logging_config = {"format": "[%(levelname)s]%(asctime)s : %(message)s"}
    if args.log_level:
        logging_config["level"] = args.log_level
    logging.basicConfig(**logging_config)
    logging.info(f"Config: {args}")

    if args.warm_up_epochs <= 0:
        logging.warning("No warm up sequence, inference time may be inaccurate.")

    run(
        args.model_name_or_path,
        args.video_paths,
        args.warm_up_epochs,
        args.use_hpu_graphs,
        args.bf16,
    )


if __name__ == "__main__":
    main()
