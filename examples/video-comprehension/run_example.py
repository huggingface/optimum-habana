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
import json
import logging
import os
import time
from pathlib import Path

import av
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import VideoLlavaProcessor

from optimum.habana.transformers.modeling_utils import (
    GaudiVideoLlavaForConditionalGeneration,
    adapt_transformers_to_gaudi,
)

from optimum.habana.transformers.gaudi_configuration import GaudiConfig
from diffusers.utils.export_utils import export_to_video



logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(

        "--video_path",
        default=None,
        type=str,
        nargs="*",
        help='Path to video as input. Can be a single string (eg: --image_path "URL1"), or a list of space-separated strings (eg: --video_path "URL1" "URL2")',
    )
    parser.add_argument(
        "--prompt",
        default=None,
        type=str,
        help='Optional argument to give a prompt of your choice as input. is a single string (eg: --prompt "Hello world")',
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output directory to store results in.",
    )
    parser.add_argument(
        "--token",
        default=None,
        type=str,
        help="The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
        "generated when running `huggingface-cli login` (stored in `~/.huggingface`).",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of inference iterations for benchmarking.")
    parser.add_argument(
        "--ignore_eos",
        action="store_true",
        help="Whether to disable stopping with eos token when calling `generate`.",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Whether to enable Habana Flash Attention, provided that the model supports it.",
    )
    parser.add_argument(
        "--flash_attention_recompute",
        action="store_true",
        help="Whether to enable Habana Flash Attention in recompute mode on first token generation. This gives an opportunity of splitting graph internally which helps reduce memory consumption.",
    )

    args = parser.parse_args()

    os.environ.setdefault("EXPERIMENTAL_WEIGHT_SHARING", "FALSE")

    if args.video_path is None:
        args.video_path = [
            hf_hub_download(
                repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset"
            )
        ]

    if args.prompt is None:
        args.prompt = ["USER: <video>Why is this video funny? ASSISTANT:"]
    video_paths = args.video_path
    video_paths_len = len(video_paths)

    prompts = args.prompt
    if args.batch_size > video_paths_len:
        # Dynamically extends to support larger batch sizes
        num_path_to_add = args.batch_size - video_paths_len
        for i in range(num_path_to_add):
            video_paths.append(video_paths[i % video_paths_len])
            prompts.append(prompts[i % video_paths_len])
    elif args.batch_size < video_paths_len:
        video_paths = video_paths[: args.batch_size]

    video_clips = []

    for video_path in video_paths:
        container = av.open(video_path)
        num_frames = container.streams.video[0].frames
        indices = np.arange(0, num_frames, num_frames / 8).astype(int)
        clip = read_video_pyav(container, indices)
        video_clips.append(clip)

    if args.bf16:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    adapt_transformers_to_gaudi()
    model = GaudiVideoLlavaForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model = model.to(model_dtype)
    device = torch.device("hpu")
    model = model.to(device)
    if args.use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        model = wrap_in_hpu_graph(model)

    processor = VideoLlavaProcessor.from_pretrained(args.model_name_or_path)
    processor.tokenizer.padding_side = "left"
    inputs = processor(text=prompts, videos=video_clips, return_tensors="pt")
    inputs = inputs.to(device)

    # warm up
    for i in range(args.warmup):
        generate_ids = model.generate(
            **inputs,
            lazy_mode=True,
            hpu_graphs=args.use_hpu_graphs,
            max_new_tokens=args.max_new_tokens,
            ignore_eos=args.ignore_eos,
            use_flash_attention=args.use_flash_attention,
            flash_attention_recompute=args.flash_attention_recompute,
        )
    torch.hpu.synchronize()

    start = time.perf_counter()
    for i in range(args.n_iterations):
        generate_ids = model.generate(
            **inputs,
            lazy_mode=True,
            hpu_graphs=args.use_hpu_graphs,
            max_new_tokens=args.max_new_tokens,
            ignore_eos=args.ignore_eos,
            use_flash_attention=args.use_flash_attention,
            flash_attention_recompute=args.flash_attention_recompute,
        )
        generate_texts = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    end = time.perf_counter()
    duration = end - start

    # Let's calculate the number of generated tokens
    n_input_tokens = inputs["input_ids"].shape[1]
    n_output_tokens = 0
    for i in range(generate_ids.shape[0]):
        n_input_tokens = torch.sum(inputs["attention_mask"][i, :]).item()
        # We have to subtract the number of input tokens as they are part of the returned sequence
        n_output_tokens += len(generate_ids[i]) - n_input_tokens

    total_new_tokens_generated = args.n_iterations * n_output_tokens
    throughput = total_new_tokens_generated / duration
    logger.info(f"result = {generate_texts}")
    logger.info(
        f"time = {(end - start) * 1000 / args.n_iterations}ms, Throughput (including tokenization) = {throughput} tokens/second"
    )

    # Store results if necessary
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "throughput": throughput,
            "output": generate_texts,
        }
        with (output_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
