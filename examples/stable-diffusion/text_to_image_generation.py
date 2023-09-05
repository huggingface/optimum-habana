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
import sys
from pathlib import Path

import torch

from optimum.habana.diffusers import GaudiDDIMScheduler
from optimum.habana.utils import set_seed


try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


# Will error if the minimal version of Optimum Habana is not installed. Remove at your own risks.
check_optimum_habana_min_version("1.7.2")


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="runwayml/stable-diffusion-v1-5",
        type=str,
        help="Path to pre-trained model",
    )

    # Pipeline arguments
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default="An image of a squirrel in Picasso style",
        help="The prompt or prompts to guide the image generation.",
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1, help="The number of images to generate per prompt."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The number of images in a batch.")
    parser.add_argument("--height", type=int, default=512, help="The height in pixels of the generated images.")
    parser.add_argument("--width", type=int, default=512, help="The width in pixels of the generated images.")
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
        "--negative_prompts",
        type=str,
        nargs="*",
        default=None,
        help="The prompt or prompts not to guide the image generation.",
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
        default="./stable-diffusion-generated-images",
        help="The directory where images will be saved.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")

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
    parser.add_argument("--bf16", action="store_true", help="Whether to perform generation in bf16 precision.")
    parser.add_argument(
        "--ldm3d", action="store_true", help="Use LDM3D to generate an image and a depth map from a given text prompt."
    )

    args = parser.parse_args()

    if args.ldm3d:
        from optimum.habana.diffusers import GaudiStableDiffusionLDM3DPipeline as GaudiStableDiffusionPipeline

        if args.model_name_or_path == "runwayml/stable-diffusion-v1-5":
            args.model_name_or_path = "Intel/ldm3d-4c"
    else:
        from optimum.habana.diffusers import GaudiStableDiffusionPipeline

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # Initialize the scheduler and the generation pipeline
    scheduler = GaudiDDIMScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler")
    kwargs = {
        "scheduler": scheduler,
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": args.gaudi_config_name,
    }
    if args.bf16:
        kwargs["torch_dtype"] = torch.bfloat16
    pipeline = GaudiStableDiffusionPipeline.from_pretrained(
        args.model_name_or_path,
        **kwargs,
    )

    # Set seed before running the model
    set_seed(args.seed)

    # Generate images
    outputs = pipeline(
        prompt=args.prompts,
        num_images_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompts,
        eta=args.eta,
        output_type=args.output_type,
    )

    # Save the pipeline in the specified directory if not None
    if args.pipeline_save_dir is not None:
        pipeline.save_pretrained(args.pipeline_save_dir)

    # Save images in the specified directory if not None and if they are in PIL format
    if args.image_save_dir is not None:
        if args.output_type == "pil":
            image_save_dir = Path(args.image_save_dir)
            image_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving images in {image_save_dir.resolve()}...")
            if args.ldm3d:
                for i, rgb in enumerate(outputs.rgb):
                    rgb.save(image_save_dir / f"rgb_{i+1}.png")
                for i, depth in enumerate(outputs.depth):
                    depth.save(image_save_dir / f"depth_{i+1}.png")
            else:
                for i, image in enumerate(outputs.images):
                    image.save(image_save_dir / f"image_{i+1}.png")
        else:
            logger.warning("--output_type should be equal to 'pil' to save images in --image_save_dir.")


if __name__ == "__main__":
    main()
