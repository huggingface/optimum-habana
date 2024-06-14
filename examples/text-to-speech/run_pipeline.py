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

import soundfile as sf
import torch
from datasets import load_dataset
from transformers import pipeline

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
from optimum.habana.utils import set_seed


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
        "--text",
        default=None,
        type=str,
        nargs="*",
        help='Text input. Can be a single string (eg: --text "text1"), or a list of space-separated strings (eg: --text "text1" "text2")',
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of inference iterations for benchmarking.")
    parser.add_argument("--seed", type=int, default=555, help="make speech generation deterministic")
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    args = parser.parse_args()

    adapt_transformers_to_gaudi()
    text = args.text
    text_bs = len(text)
    set_seed(args.seed)

    if args.batch_size > text_bs:
        # Dynamically extends to support larger batch sizes
        text_to_add = args.batch_size - text_bs
        for i in range(text_to_add):
            text.append(text[i % text_bs])
    elif args.batch_size < text_bs:
        text = text[: args.batch_size]

    if args.bf16:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    generator = pipeline(
        "text-to-speech",
        model=args.model_name_or_path,
        torch_dtype=model_dtype,
        device="hpu",
    )

    if args.use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        generator.model = wrap_in_hpu_graph(generator.model)

    forward_params = None
    if generator.model.config.model_type == "speecht5":
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to("hpu")
        forward_params = {"speaker_embeddings": speaker_embedding}
    if generator.model.config.model_type == "seamless_m4t":
        forward_params = {"tgt_lang": "eng"}

    generate_kwargs = None
    if generator.model.can_generate():
        generate_kwargs = {"lazy_mode": True, "ignore_eos": False, "hpu_graphs": args.use_hpu_graphs}

    with torch.autocast("hpu", torch.bfloat16, enabled=args.bf16), torch.inference_mode():
        # warm up
        for i in range(args.warmup):
            if generator.model.config.model_type == "speecht5":
                # SpeechT5 forces a dropout with training=True, which may zero out some elements randomly.
                # A random dropout may need different lengths of spectrograms to fit probability thresholds,
                # which violates the HPU static shape, so we have to fix the seed here.
                set_seed(args.seed)
            generator(text, batch_size=args.batch_size, forward_params=forward_params, generate_kwargs=generate_kwargs)

        start = time.time()
        for i in range(args.n_iterations):
            if generator.model.config.model_type == "speecht5":
                # SpeechT5 forces a dropout with training=True, which may zero out some elements randomly.
                # A random dropout may need different lengths of spectrograms to fit probability thresholds,
                # which violates the HPU static shape, so we have to fix the seed here.
                set_seed(args.seed)
            speech = generator(
                text, batch_size=args.batch_size, forward_params=forward_params, generate_kwargs=generate_kwargs
            )
        end = time.time()
        logger.info(f"speech = {speech} time = {(end-start) * 1000 / args.n_iterations }ms")
        sf.write("speech.wav", speech[0]["audio"].squeeze(), samplerate=speech[0]["sampling_rate"])


if __name__ == "__main__":
    main()
