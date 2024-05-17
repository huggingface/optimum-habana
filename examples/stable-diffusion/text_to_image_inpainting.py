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

"""
Adapted from: https://github.com/huggingface/optimum-habana/blob/main/examples/stable-diffusion/text_to_image_generation.py
- Use the AutoPipelineForInpainting to load the Gaudi inpaint pipeline.
- Add the inpaint examples from https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint
"""
import argparse
import logging
import sys
from io import BytesIO
from pathlib import Path

import PIL
import requests
import torch

from optimum.habana.diffusers import (
    AutoPipelineForInpainting,
    GaudiDDIMScheduler,
    GaudiEulerAncestralDiscreteScheduler,
    GaudiEulerDiscreteScheduler,
)
from optimum.habana.utils import set_seed


logger = logging.getLogger(__name__)

try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


check_optimum_habana_min_version("1.10.0")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="runwayml/stable-diffusion-inpainting",
        type=str,
        help="Path to pre-trained model",
    )

    # Pipeline arguments
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default="concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k",
        help="The prompt or prompts to guide the image generation. The delimiter is semicolon(;)",
    )
    parser.add_argument(
        "--base_image",
        type=str,
        default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
        help=("Path to inpaint base image"),
    )
    parser.add_argument(
        "--mask_image",
        type=str,
        default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png",
        help=("Path to inpaint mask image"),
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1, help="The number of images to generate per prompt."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="The batch size to generate per step."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="The height in pixels of the generated images (0=default from model config).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="The width in pixels of the generated images (0=default from model config).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help=(
            "The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense"
            " of slower inference."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help=(
            "Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)."
            " Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,"
            " usually at the expense of lower image quality."
        ),
    )
    parser.add_argument(
        "--scheduler",
        default="ddim",
        choices=["euler_discrete", "euler_ancestral_discrete", "ddim"],
        type=str,
        help="Name of scheduler",
    )
    parser.add_argument(
        "--timestep_spacing",
        default="linspace",
        choices=["linspace", "leading", "trailing"],
        type=str,
        help="The way the timesteps should be scaled.",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        choices=["pil", "np"],
        default="pil",
        help="Whether to return PIL images or Numpy arrays.",
    )
    parser.add_argument(
        "--pipeline_save_dir",
        type=str,
        default=None,
        help="The directory where the generation pipeline will be saved.",
    )
    parser.add_argument(
        "--image_save_dir",
        type=str,
        default="./stable-diffusion-inpainting-images",
        help="The directory where images will be saved.",
    )
    parser.add_argument("--seed", type=int, default=92, help="Random seed for initialization.")

    # HPU-specific arguments
    parser.add_argument("--use_habana", action="store_true", help="Use HPU.")
    parser.add_argument(
        "--use_hpu_graphs", action="store_true", help="Use HPU graphs on HPU. This should lead to faster generations."
    )
    parser.add_argument(
        "--gaudi_config_name",
        type=str,
        default="Habana/stable-diffusion",
        help=(
            "Name or path of the Gaudi configuration. In particular, it enables to specify how to apply Habana Mixed"
            " Precision."
        ),
    )
    parser.add_argument("--bf16", default=True, help="Whether to perform generation in bf16 precision.")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # Set seed before running the model
    if args.seed:
        logger.info("Set the random seed {}!".format(args.seed))
        set_seed(args.seed)

    def download_image(url):
        response = requests.get(url)
        return PIL.Image.open(BytesIO(response.content)).convert("RGB")

    img_url = args.base_image
    mask_url = args.mask_image

    init_image = download_image(img_url).resize((512, 512))
    mask_image = download_image(mask_url).resize((512, 512))

    # Initialize the scheduler and the generation pipeline
    kwargs = {"timestep_spacing": args.timestep_spacing}
    if args.scheduler == "euler_discrete":
        scheduler = GaudiEulerDiscreteScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **kwargs
        )
    elif args.scheduler == "euler_ancestral_discrete":
        scheduler = GaudiEulerAncestralDiscreteScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **kwargs
        )
    else:
        scheduler = GaudiDDIMScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler", **kwargs)


    init_kwargs = {
        "scheduler": scheduler,
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": args.gaudi_config_name,
        "torch_dtype": torch.bfloat16 if args.bf16 else None
    }

    pipe = AutoPipelineForInpainting.from_pretrained(
        args.model_name_or_path,
        **init_kwargs
    )

    kwargs={
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": args.num_images_per_prompt,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "output_type": args.output_type,
        "batch_size": args.batch_size
    }

    prompts = args.prompts
    logger.info(f"prompts={prompts}")
    outputs = pipe(prompt=prompts, image=init_image, mask_image=mask_image, **kwargs)

    # Save images in the specified directory if not None and if they are in PIL format
    if args.image_save_dir is not None:
        if args.output_type == "pil":
            image_save_dir = Path(args.image_save_dir)
            image_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving images in {image_save_dir.resolve()}...")
            init_image.save(image_save_dir / "init_image.png")
            mask_image.save(image_save_dir / "mask_image.png")
            for i, image in enumerate(outputs.images):
                image.save(image_save_dir / f"image_{i+1}.png")
        else:
            logger.warning("--output_type should be equal to 'pil' to save images in --image_save_dir.")

if __name__ == "__main__":
    main()
