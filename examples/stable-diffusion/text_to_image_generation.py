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

import numpy as np
import torch
from accelerate import PartialState

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
check_optimum_habana_min_version("1.11.0")


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="runwayml/stable-diffusion-v1-5",
        type=str,
        help="Path to pre-trained model",
    )

    parser.add_argument(
        "--controlnet_model_name_or_path",
        default="lllyasviel/sd-controlnet-canny",
        type=str,
        help="Path to pre-trained model",
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
        "--base_image",
        type=str,
        default=None,
        help=("Path to inpaint base image"),
    )
    parser.add_argument(
        "--mask_image",
        type=str,
        default=None,
        help=("Path to inpaint mask image"),
    )
    parser.add_argument(
        "--control_image",
        type=str,
        default=None,
        help=("Path to the controlnet conditioning image"),
    )
    parser.add_argument(
        "--control_preprocessing_type",
        type=str,
        default="canny",
        help=(
            "The type of preprocessing to apply on contol image. Only `canny` is supported."
            " Defaults to `canny`. Set to unsupported value to disable preprocessing."
        ),
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
        "--throughput_warmup_steps",
        type=int,
        default=None,
        help="Number of steps to ignore for throughput calculation.",
    )
    parser.add_argument(
        "--profiling_warmup_steps",
        type=int,
        default=0,
        help="Number of steps to ignore for profiling.",
    )
    parser.add_argument(
        "--profiling_steps",
        type=int,
        default=0,
        help="Number of steps to capture for profiling.",
    )
    parser.add_argument("--distributed", action="store_true", help="Use distributed inference on multi-cards")
    parser.add_argument(
        "--unet_adapter_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--text_encoder_adapter_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--lora_id",
        default=None,
        type=str,
        help="Path to lora id",
    )
    parser.add_argument(
        "--use_cpu_rng",
        action="store_true",
        help="Enable deterministic generation using CPU Generator",
    )
    parser.add_argument(
        "--ip_adapter_path",
        default=None,
        type=str,
        help="Path to ip adapter",
    )
    args = parser.parse_args()

    # Set image resolution
    kwargs_call = {}
    if args.width > 0 and args.height > 0:
        kwargs_call["width"] = args.width
        kwargs_call["height"] = args.height

    # ControlNet
    if args.control_image is not None:
        from diffusers.utils import load_image
        from PIL import Image
        import cv2

        # get control image
        control_image = load_image(args.control_image)
        if args.control_preprocessing_type == "canny":
            image = np.array(control_image)
            # get canny image
            image = cv2.Canny(image, 100, 200)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            control_image = Image.fromarray(image)

    # Import selected pipeline
    sdxl_models = ["stable-diffusion-xl", "sdxl"]
    sdxl = True if any(model in args.model_name_or_path for model in sdxl_models) else False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

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

    kwargs = {
        "scheduler": scheduler,
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": args.gaudi_config_name,
    }

    if args.bf16:
        kwargs["torch_dtype"] = torch.bfloat16

    negative_prompts = args.negative_prompts
    if args.distributed:
        distributed_state = PartialState()
        if args.negative_prompts is not None:
            with distributed_state.split_between_processes(args.negative_prompts) as negative_prompt:
                negative_prompts = negative_prompt

    kwargs_common = {
        "num_images_per_prompt": args.num_images_per_prompt,
        "batch_size": args.batch_size,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": negative_prompts,
        "eta": args.eta,
        "output_type": args.output_type,
        "profiling_warmup_steps": args.profiling_warmup_steps,
        "profiling_steps": args.profiling_steps,
    }

    kwargs_call.update(kwargs_common)
    if args.throughput_warmup_steps is not None:
        kwargs_call["throughput_warmup_steps"] = args.throughput_warmup_steps

    if args.use_cpu_rng:
        # Patch for the deterministic generation - Need to specify CPU as the torch generator
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
    else:
        generator = None
    kwargs_call["generator"] = generator

    # Generate images
    if args.control_image is not None:
        from diffusers import ControlNetModel

        model_dtype = torch.bfloat16 if args.bf16 else None
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, torch_dtype=model_dtype)

        if sdxl and args.ip_adapter_path:
            from optimum.habana.diffusers import GaudiStableDiffusionXLInstantIDPipeline
            from insightface.app import FaceAnalysis

            # prepare 'antelopev2' under ./models
            app = FaceAnalysis(name='antelopev2', root='./', providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))

            pipeline = GaudiStableDiffusionXLInstantIDPipeline.from_pretrained(args.model_name_or_path, controlnet=controlnet, **kwargs)
            pipeline.load_ip_adapter_instantid(args.ip_adapter_path)

            # prepare face emb
            face_info = app.get(cv2.cvtColor(np.array(control_image), cv2.COLOR_RGB2BGR))
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
            face_emb = face_info['embedding']
            face_kps = GaudiStableDiffusionXLInstantIDPipeline.draw_kps(control_image, face_info['kps'])
            
            if args.lora_id:
                pipeline.load_lora_weights(args.lora_id)

            # Set seed before running the model
            set_seed(args.seed)

            outputs = pipeline(
                prompt=args.prompts,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                negative_prompt=args.negative_prompts,
                eta=args.eta,
                output_type=args.output_type,
                image_embeds=face_emb,
                image=face_kps,
                num_images_per_prompt=args.num_images_per_prompt,
                batch_size=args.batch_size,
                controlnet_conditioning_scale=0.8,
                ip_adapter_scale=0.8,
                profiling_warmup_steps=args.profiling_warmup_steps,
                profiling_steps=args.profiling_steps,
                **kwargs_call,
            )
        else:
            from optimum.habana.diffusers import GaudiStableDiffusionControlNetPipeline

            pipeline = GaudiStableDiffusionControlNetPipeline.from_pretrained(
                args.model_name_or_path,
                controlnet=controlnet,
                **kwargs,
            )
            if args.lora_id:
                pipeline.load_lora_weights(args.lora_id)

        kwargs_call["image"] = control_image

    elif (args.base_image is not None) and (args.mask_image is not None):
        from diffusers.utils import load_image

        pipeline = AutoPipelineForInpainting.from_pretrained(args.model_name_or_path, **kwargs)
        init_image = load_image(args.base_image)
        mask_image = load_image(args.mask_image)
        kwargs_call["image"] = init_image
        kwargs_call["mask_image"] = mask_image

    elif sdxl:
        from optimum.habana.diffusers import GaudiStableDiffusionXLPipeline

        pipeline = GaudiStableDiffusionXLPipeline.from_pretrained(
            args.model_name_or_path,
            **kwargs,
        )
        if args.lora_id:
            pipeline.load_lora_weights(args.lora_id)

        prompts_2 = args.prompts_2
        negative_prompts_2 = args.negative_prompts_2
        if args.distributed and args.prompts_2 is not None:
            with distributed_state.split_between_processes(args.prompts_2) as prompt_2:
                prompts_2 = prompt_2
        if args.distributed and args.negative_prompts_2 is not None:
            with distributed_state.split_between_processes(args.negative_prompts_2) as negative_prompt_2:
                negative_prompts_2 = negative_prompt_2

        kwargs_call["prompt_2"] = prompts_2
        kwargs_call["negative_prompt_2"] = negative_prompts_2

    else:
        if args.ldm3d:
            from optimum.habana.diffusers import GaudiStableDiffusionLDM3DPipeline as GaudiStableDiffusionPipeline

            if args.model_name_or_path == "runwayml/stable-diffusion-v1-5":
                args.model_name_or_path = "Intel/ldm3d-4c"
        else:
            from optimum.habana.diffusers import GaudiStableDiffusionPipeline

        pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            args.model_name_or_path,
            **kwargs,
        )
        if args.unet_adapter_name_or_path is not None:
            from peft import PeftModel

            pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args.unet_adapter_name_or_path)
            pipeline.unet = pipeline.unet.merge_and_unload()
        if args.text_encoder_adapter_name_or_path is not None:
            from peft import PeftModel

            pipeline.text_encoder = PeftModel.from_pretrained(
                pipeline.text_encoder, args.text_encoder_adapter_name_or_path
            )
            pipeline.text_encoder = pipeline.text_encoder.merge_and_unload()

    set_seed(args.seed)

    if args.distributed:
        with distributed_state.split_between_processes(args.prompts) as prompt:
            outputs = pipeline(prompt=prompt, **kwargs_call)
    else:
        outputs = pipeline(prompt=args.prompts, **kwargs_call)

    # Save the pipeline in the specified directory if not None
    if args.pipeline_save_dir is not None:
        save_dir = args.pipeline_save_dir
        if args.distributed:
            save_dir = f"{args.pipeline_save_dir}_{distributed_state.process_index}"
        pipeline.save_pretrained(save_dir)

    # Save images in the specified directory if not None and if they are in PIL format
    if args.image_save_dir is not None:
        if args.output_type == "pil":
            image_save_dir = Path(args.image_save_dir)
            if args.distributed:
                image_save_dir = Path(f"{image_save_dir}_{distributed_state.process_index}")

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
