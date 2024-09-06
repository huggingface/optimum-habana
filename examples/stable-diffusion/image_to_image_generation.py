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

import PIL
import requests
import torch
from torchvision import transforms

from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiEulerAncestralDiscreteScheduler,
    GaudiEulerDiscreteScheduler,
)
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="CompVis/stable-diffusion-v1-4",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--src_image_path",
        type=str,
        required=True,
        help="Path to source image",
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
        "--prompts_2",
        type=str,
        nargs="*",
        default=None,
        help="The second prompt or prompts to guide the image generation (applicable to SDXL).",
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1, help="The number of images to generate per prompt."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The number of images in a batch.")
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="The height in pixels of the generated images (0=default from model config).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
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
        "--image_guidance_scale",
        type=float,
        default=1.5,
        help=(
            "Image guidance scale is to push the generated image towards the inital image `image`. Image guidance"
            "scale is enabled by setting `image_guidance_scale > 1`. Higher image guidance scale encourages to"
            "generate images that are closely linked to the source image `image`, usually at the expense of lower"
            "image quality. This pipeline requires a value of at least `1`.used in intruct_pix2pix"
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
        "--negative_prompts_2",
        type=str,
        nargs="*",
        default=None,
        help="The second prompt or prompts not to guide the image generation (applicable to SDXL).",
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
    parser.add_argument(
        "--profiling_warmup_steps",
        default=0,
        type=int,
        help="Number of steps to ignore for profiling.",
    )
    parser.add_argument(
        "--profiling_steps",
        default=0,
        type=int,
        help="Number of steps to capture for profiling.",
    )
    parser.add_argument(
        "--throughput_warmup_steps",
        type=int,
        default=None,
        help="Number of steps to ignore for throughput calculation.",
    )
    args = parser.parse_args()

    # Set image resolution
    res = {}
    if args.width > 0 and args.height > 0:
        res["width"] = args.width
        res["height"] = args.height
    sdxl_models = ["stable-diffusion-xl", "sdxl"]
    sdxl = False
    kwargs = {
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": args.gaudi_config_name,
    }

    # Import selected pipeline
    if any(model in args.model_name_or_path for model in sdxl_models):
        from optimum.habana.diffusers import GaudiStableDiffusionXLImg2ImgPipeline as Img2ImgPipeline

        sdxl = True
    elif "instruct-pix2pix" in args.model_name_or_path:
        from optimum.habana.diffusers import GaudiStableDiffusionInstructPix2PixPipeline as Img2ImgPipeline

        kwargs["safety_checker"] = None
        res["image_guidance_scale"] = args.image_guidance_scale
    elif "image-variations" in args.model_name_or_path:
        from optimum.habana.diffusers import GaudiStableDiffusionImageVariationPipeline as Img2ImgPipeline

        kwargs["revision"] = "v2.0"
    else:
        from optimum.habana.diffusers import GaudiStableDiffusionImg2ImgPipeline as Img2ImgPipeline

    if "image-variations" in args.model_name_or_path:
        im = PIL.Image.open(requests.get(args.src_image_path, stream=True).raw)
        tform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=False,
                ),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
            ]
        )
        image = tform(im).unsqueeze(0)
    else:
        image = PIL.Image.open(requests.get(args.src_image_path, stream=True).raw)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    if args.bf16:
        kwargs["torch_dtype"] = torch.bfloat16

    if args.throughput_warmup_steps is not None:
        kwargs["throughput_warmup_steps"] = args.throughput_warmup_steps

    pipeline = Img2ImgPipeline.from_pretrained(
        args.model_name_or_path,
        **kwargs,
    )
    if pipeline.scheduler.config._class_name == "EulerAncestralDiscreteScheduler":
        pipeline.scheduler = GaudiEulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif pipeline.scheduler.config._class_name == "EulerDiscreteScheduler":
        pipeline.scheduler = GaudiEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    else:
        pipeline.scheduler = GaudiDDIMScheduler.from_config(pipeline.scheduler.config)
    # Set seed before running the model
    set_seed(args.seed)
    # Generate images
    if sdxl:
        outputs = pipeline(
            image=image,
            prompt=args.prompts,
            prompt_2=args.prompts_2,
            num_images_per_prompt=args.num_images_per_prompt,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompts,
            negative_prompt_2=args.negative_prompts_2,
            eta=args.eta,
            output_type=args.output_type,
            profiling_warmup_steps=args.profiling_warmup_steps,
            profiling_steps=args.profiling_steps,
            **res,
        )
    else:
        outputs = pipeline(
            image=image,
            prompt=args.prompts,
            num_images_per_prompt=args.num_images_per_prompt,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompts,
            eta=args.eta,
            output_type=args.output_type,
            profiling_warmup_steps=args.profiling_warmup_steps,
            profiling_steps=args.profiling_steps,
            **res,
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
