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

import PIL.Image
import requests
import torch
from transformers import AutoConfig, LlavaNextProcessor, LlavaProcessor, pipeline

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_quantization(model, args):
    if os.getenv("USE_INC", "1") != "0":
        try:
            from neural_compressor.torch.quantization import FP8Config, convert, prepare
        except ImportError:
            raise ImportError(
                "Module neural_compressor is missing. Please use a newer Synapse version to use quantization, or set the environment variable to USE_INC=0"
            )

        config = FP8Config.from_json_file(args.quant_config)
        if config.measure:
            model = prepare(model, config)
        elif config.quantize:
            model = convert(model, config)
    else:
        import habana_frameworks.torch.core as htcore
        import habana_quantization_toolkit

        habana_quantization_toolkit.prep_model(model)
        htcore.hpu_initialize(model)

    return model


def finalize_quantization(model):
    if os.getenv("USE_INC", "1") != "0":
        try:
            from neural_compressor.torch.quantization import finalize_calibration
        except ImportError:
            raise ImportError(
                "Module neural_compressor is missing. Please use a newer Synapse version to use quantization, or set the environment variable to USE_INC=0"
            )

        finalize_calibration(model)
    else:
        import habana_quantization_toolkit

        habana_quantization_toolkit.finish_measurements(model)


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
    parser.add_argument(
        "--use_kv_cache",
        type=str2bool,
        default=True,
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )

    args = parser.parse_args()

    # set args.quant_config with env variable if it is set
    args.quant_config = os.getenv("QUANT_CONFIG", "")

    adapt_transformers_to_gaudi()

    model_type = AutoConfig.from_pretrained(args.model_name_or_path).model_type
    if args.image_path is None and model_type == "llava":
        args.image_path = ["https://llava-vl.github.io/static/images/view.jpg"]
    elif args.image_path is None and model_type == "llava_next":
        args.image_path = [
            "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
        ]
    if args.prompt is None:
        if model_type == "llava":
            processor = LlavaProcessor.from_pretrained(args.model_name_or_path)
        elif model_type == "llava_next":
            processor = LlavaNextProcessor.from_pretrained(args.model_name_or_path)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is shown in this image?"},
                    {"type": "image"},
                ],
            }
        ]
        args.prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    image_paths = args.image_path
    image_paths_len = len(image_paths)

    if args.batch_size > image_paths_len:
        # Dynamically extends to support larger batch sizes
        num_path_to_add = args.batch_size - image_paths_len
        for i in range(num_path_to_add):
            image_paths.append(image_paths[i % image_paths_len])
    elif args.batch_size < image_paths_len:
        image_paths = image_paths[: args.batch_size]

    images = []

    for image_path in image_paths:
        images.append(PIL.Image.open(requests.get(image_path, stream=True, timeout=3000).raw))

    if args.bf16:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    if args.quant_config:
        import habana_frameworks.torch.core as htcore

        htcore.hpu_set_env()

    generator = pipeline(
        "image-to-text",
        model=args.model_name_or_path,
        torch_dtype=model_dtype,
        device="hpu",
    )
    generate_kwargs = {
        "lazy_mode": True,
        "use_cache": args.use_kv_cache,
        "hpu_graphs": args.use_hpu_graphs,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
        "use_flash_attention": args.use_flash_attention,
        "flash_attention_recompute": args.flash_attention_recompute,
    }
    if args.use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        generator.model = wrap_in_hpu_graph(generator.model)

    if args.quant_config:
        generator.model = setup_quantization(generator.model, args)

    # warm up
    for i in range(args.warmup):
        generator(images, prompt=args.prompt, batch_size=args.batch_size, generate_kwargs=generate_kwargs)
    torch.hpu.synchronize()
    if args.quant_config:
        finalize_quantization(generator.model)

    start = time.perf_counter()
    for i in range(args.n_iterations):
        result = generator(images, prompt=args.prompt, batch_size=args.batch_size, generate_kwargs=generate_kwargs)
    end = time.perf_counter()
    duration = end - start

    # Let's calculate the number of generated tokens
    n_input_tokens = len(generator.tokenizer(args.prompt).input_ids) if args.prompt is not None else 0
    n_output_tokens = 0
    for sequence in result:
        # We have to subtract the number of input tokens as they are part of the returned sequence
        n_output_tokens += len(generator.tokenizer(sequence[0]["generated_text"]).input_ids) - n_input_tokens

    total_new_tokens_generated = args.n_iterations * n_output_tokens
    throughput = total_new_tokens_generated / duration
    logger.info(f"result = {result}")
    logger.info(
        f"time = {(end-start) * 1000 / args.n_iterations }ms, Throughput (including tokenization) = {throughput} tokens/second"
    )

    # Store results if necessary
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "throughput": throughput,
            "output": result,
        }
        with (output_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
