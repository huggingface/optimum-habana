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

# Adapted from ../stable-diffusion/text_to_image_generation.py

import argparse
import logging
import sys
from pathlib import Path

import torch
from diffusers.utils.export_utils import export_to_video

from optimum.habana.diffusers import GaudiTextToVideoSDPipeline
from optimum.habana.transformers.gaudi_configuration import GaudiConfig
from optimum.habana.utils import set_seed


try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


# Will error if the minimal version of Optimum Habana is not installed. Remove at your own risks.
check_optimum_habana_min_version("1.14.0.dev0")


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--model_name_or_path",
        default="ali-vilab/text-to-video-ms-1.7b",
        type=str,
        help="Path to pre-trained model",
    )
    # Pipeline arguments
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default="Spiderman is surfing",
        help="The prompt or prompts to guide the video generation.",
    )
    parser.add_argument(
        "--num_videos_per_prompt", type=int, default=1, help="The number of videos to generate per prompt."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The number of videos in a batch.")
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="The height in pixels of the generated videos (0=default from model config).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="The width in pixels of the generated videos (0=default from model config).",
    )
    parser.add_argument("--num_frames", type=int, default=20, help="The number of frames in the generated videos.")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help=(
            "The number of denoising steps. More denoising steps usually lead to a higher quality videos at the expense"
            " of slower inference."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help=(
            "Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)."
            " Higher guidance scale encourages to generate videos that are closely linked to the text `prompt`,"
            " usually at the expense of lower video quality."
        ),
    )
    parser.add_argument(
        "--negative_prompts",
        type=str,
        nargs="*",
        default=None,
        help="The prompt or prompts not to guide the video generation.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502.",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        choices=["mp4", "np"],
        default="mp4",
        help="Whether to return mp4 or Numpy arrays.",
    )

    parser.add_argument(
        "--pipeline_save_dir",
        type=str,
        default=None,
        help="The directory where the generation pipeline will be saved.",
    )
    parser.add_argument(
        "--video_save_dir",
        type=str,
        default="./generated-videos",
        help="The directory where videos will be saved.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")

    # HPU-specific arguments
    parser.add_argument("--use_habana", action="store_true", help="Use HPU.")
    parser.add_argument(
        "--use_hpu_graphs", action="store_true", help="Use HPU graphs on HPU. This should lead to faster generations."
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp32", "autocast_bf16"],
        help="Which runtime dtype to perform generation in.",
    )
    args = parser.parse_args()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    logger.info(f"Arguments: {args}")

    # Set video resolution
    kwargs_call = {}
    if args.width > 0 and args.height > 0:
        kwargs_call["width"] = args.width
        kwargs_call["height"] = args.height
    kwargs_call["num_frames"] = args.num_frames

    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    if args.dtype == "autocast_bf16":
        gaudi_config_kwargs["use_torch_autocast"] = True

    gaudi_config = GaudiConfig(**gaudi_config_kwargs)
    logger.info(f"Gaudi Config: {gaudi_config}")

    kwargs = {
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": gaudi_config,
    }
    if args.dtype == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif args.dtype == "fp32":
        kwargs["torch_dtype"] = torch.float32

    # Generate images
    pipeline: GaudiTextToVideoSDPipeline = GaudiTextToVideoSDPipeline.from_pretrained(
        args.model_name_or_path, **kwargs
    )
    set_seed(args.seed)
    outputs = pipeline(
        prompt=args.prompts,
        num_videos_per_prompt=args.num_videos_per_prompt,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompts,
        eta=args.eta,
        output_type="pil" if args.output_type == "mp4" else args.output_type,  # Naming inconsistency in base class
        **kwargs_call,
    )

    # Save the pipeline in the specified directory if not None
    if args.pipeline_save_dir is not None:
        pipeline.save_pretrained(args.pipeline_save_dir)

    # Save images in the specified directory if not None and if they are in PIL format
    if args.video_save_dir is not None:
        if args.output_type == "mp4":
            video_save_dir = Path(args.video_save_dir)
            video_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving images in {video_save_dir.resolve()}...")

            for i, video in enumerate(outputs.videos):
                filename = video_save_dir / f"video_{i + 1}.mp4"
                export_to_video(video, str(filename.resolve()))
        else:
            logger.warning("--output_type should be equal to 'mp4' to save images in --video_save_dir.")


if __name__ == "__main__":
    main()
