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
import os
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from compel import Compel, ReturnedEmbeddingsType

from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiEulerAncestralDiscreteScheduler,
    GaudiEulerDiscreteScheduler,
    GaudiFlowMatchEulerDiscreteScheduler,
)
from optimum.habana.utils import set_seed


try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


# Will error if the minimal version of Optimum Habana is not installed. Remove at your own risks.
check_optimum_habana_min_version("1.18.0.dev0")


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
        "--controlnet_model_name_or_path",
        default="lllyasviel/sd-controlnet-canny",
        type=str,
        help="Path to pre-trained model",
    )

    parser.add_argument(
        "--scheduler",
        default="ddim",
        choices=["default", "euler_discrete", "euler_ancestral_discrete", "ddim", "flow_match_euler_discrete"],
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
        help="The second prompt or prompts to guide the image generation (applicable to SDXL and SD3).",
    )
    parser.add_argument(
        "--prompts_3",
        type=str,
        nargs="*",
        default=None,
        help="The third prompt or prompts to guide the image generation (applicable to SD3).",
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
        help="The second prompt or prompts not to guide the image generation (applicable to SDXL and SD3).",
    )
    parser.add_argument(
        "--negative_prompts_3",
        type=str,
        nargs="*",
        default=None,
        help="The third prompt or prompts not to guide the image generation (applicable to SD3).",
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
        "--sdp_on_bf16", action="store_true", help="Allow pyTorch to use reduced precision in the SDPA math backend"
    )
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
        "--use_compel",
        action="store_true",
        help="Use compel for prompt weighting",
    )
    parser.add_argument(
        "--use_freeu",
        action="store_true",
        help="Use freeu for improving generation quality",
    )
    parser.add_argument(
        "--use_zero_snr",
        action="store_true",
        help="Use rescale_betas_zero_snr for controlling image brightness",
    )
    parser.add_argument("--optimize", action="store_true", help="Use optimized pipeline.")
    parser.add_argument(
        "--quant_mode",
        default="disable",
        choices=["measure", "quantize", "quantize-mixed", "disable"],
        type=str,
        help="Quantization mode 'measure', 'quantize', 'quantize-mixed' or 'disable'",
    )
    parser.add_argument(
        "--use_distributed_cfg",
        action="store_true",
        help="Use distributed CFG (classifier-free guidance) across 2 devices for SD3 pipeline. Requires even world size.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="The file with prompts (for large number of images generation).",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=None,
        help="A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.",
    )
    args = parser.parse_args()

    if args.optimize and not args.use_habana:
        raise ValueError("--optimize can only be used with --use-habana.")

    # Select stable diffuson pipeline based on input
    sdxl_models = ["stable-diffusion-xl", "sdxl"]
    sd3_models = ["stable-diffusion-3", "sd3"]
    flux_models = ["FLUX.1", "flux"]
    sdxl = True if any(model in args.model_name_or_path for model in sdxl_models) else False
    sd3 = True if any(model in args.model_name_or_path for model in sd3_models) else False
    flux = True if any(model in args.model_name_or_path for model in flux_models) else False
    controlnet = True if args.control_image is not None else False
    inpainting = True if (args.base_image is not None) and (args.mask_image is not None) else False

    # Set the scheduler
    kwargs = {"timestep_spacing": args.timestep_spacing, "rescale_betas_zero_snr": args.use_zero_snr}

    if flux or sd3 or args.scheduler == "flow_match_euler_discrete":
        scheduler = GaudiFlowMatchEulerDiscreteScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **kwargs
        )
    elif args.scheduler == "euler_discrete":
        scheduler = GaudiEulerDiscreteScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **kwargs
        )
        if args.optimize:
            scheduler.hpu_opt = True
    elif args.scheduler == "euler_ancestral_discrete":
        scheduler = GaudiEulerAncestralDiscreteScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **kwargs
        )
    elif args.scheduler == "ddim":
        scheduler = GaudiDDIMScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler", **kwargs)
    else:
        scheduler = None

    # Set pipeline class instantiation options
    kwargs = {
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": args.gaudi_config_name,
        "sdp_on_bf16": args.sdp_on_bf16,
    }

    if scheduler is not None:
        kwargs["scheduler"] = scheduler

    if args.bf16:
        kwargs["torch_dtype"] = torch.bfloat16

    # Set pipeline call options
    kwargs_call = {
        "num_images_per_prompt": args.num_images_per_prompt,
        "batch_size": args.batch_size,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "eta": args.eta,
        "output_type": args.output_type,
        "profiling_warmup_steps": args.profiling_warmup_steps,
        "profiling_steps": args.profiling_steps,
    }

    if args.width > 0 and args.height > 0:
        kwargs_call["width"] = args.width
        kwargs_call["height"] = args.height

    if args.use_cpu_rng:
        kwargs_call["generator"] = torch.Generator(device="cpu").manual_seed(args.seed)
    else:
        kwargs_call["generator"] = None

    if args.throughput_warmup_steps is not None:
        kwargs_call["throughput_warmup_steps"] = args.throughput_warmup_steps

    if args.lora_scale is not None:
        kwargs_call["lora_scale"] = args.lora_scale

    negative_prompts = args.negative_prompts
    if args.distributed:
        distributed_state = PartialState()
        if args.negative_prompts is not None:
            with distributed_state.split_between_processes(args.negative_prompts) as negative_prompt:
                negative_prompts = negative_prompt
    kwargs_call["negative_prompt"] = negative_prompts

    if sdxl or sd3 or flux:
        prompts_2 = args.prompts_2
        if args.distributed and args.prompts_2 is not None:
            with distributed_state.split_between_processes(args.prompts_2) as prompt_2:
                prompts_2 = prompt_2
        kwargs_call["prompt_2"] = prompts_2

    if sdxl or sd3:
        negative_prompts_2 = args.negative_prompts_2
        if args.distributed and args.negative_prompts_2 is not None:
            with distributed_state.split_between_processes(args.negative_prompts_2) as negative_prompt_2:
                negative_prompts_2 = negative_prompt_2
        kwargs_call["negative_prompt_2"] = negative_prompts_2

    if sd3:
        prompts_3 = args.prompts_3
        negative_prompts_3 = args.negative_prompts_3
        if args.distributed and args.prompts_3 is not None:
            with distributed_state.split_between_processes(args.prompts_3) as prompt_3:
                prompts_3 = prompt_3
        if args.distributed and args.negative_prompts_3 is not None:
            with distributed_state.split_between_processes(args.negative_prompts_3) as negative_prompt_3:
                negative_prompts_3 = negative_prompt_3
        kwargs_call["prompt_3"] = prompts_3
        kwargs_call["negative_prompt_3"] = negative_prompts_3
        if args.use_distributed_cfg:
            kwargs_call["use_distributed_cfg"] = True

    if inpainting:
        from diffusers.utils import load_image

        init_image = load_image(args.base_image)
        mask_image = load_image(args.mask_image)
        kwargs_call["image"] = init_image
        kwargs_call["mask_image"] = mask_image

    if controlnet:
        from diffusers.utils import load_image
        from PIL import Image

        control_image = load_image(args.control_image)
        if args.control_preprocessing_type == "canny":
            # Generate Canny image for ControlNet
            import cv2

            image = np.array(control_image)
            image = cv2.Canny(image, 100, 200)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            control_image = Image.fromarray(image)
        kwargs_call["image"] = control_image

    kwargs_call["quant_mode"] = args.quant_mode

    # Instantiate a Stable Diffusion pipeline class
    quant_config_path = os.getenv("QUANT_CONFIG")
    if sdxl:
        # SDXL pipelines
        if controlnet:
            # Import SDXL+ControlNet pipeline
            raise ValueError("SDXL+ControlNet pipeline is not currenly supported")

        elif inpainting:
            # Import SDXL Inpainting pipeline
            from optimum.habana.diffusers import AutoPipelineForInpainting

            pipeline = AutoPipelineForInpainting.from_pretrained(args.model_name_or_path, **kwargs)

        elif args.optimize:
            # Import SDXL pipeline
            # set PATCH_SDPA to enable fp8 varient of softmax in sdpa
            os.environ["PATCH_SDPA"] = "1"
            import habana_frameworks.torch.hpu as torch_hpu

            from optimum.habana.diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_mlperf import (
                StableDiffusionXLPipeline_HPU,
            )

            pipeline = StableDiffusionXLPipeline_HPU.from_pretrained(
                args.model_name_or_path,
                **kwargs,
            )

            pipeline.unet.set_default_attn_processor(pipeline.unet)
            pipeline.to(torch.device("hpu"))

            if quant_config_path:
                import habana_frameworks.torch.core as htcore
                from neural_compressor.torch.quantization import FP8Config, convert, prepare

                htcore.hpu_set_env()

                config = FP8Config.from_json_file(quant_config_path)

                if config.measure:
                    logger.info("Running measurements")
                    pipeline.unet = prepare(pipeline.unet, config)
                elif config.quantize:
                    logger.info("Running quantization")
                    pipeline.unet = convert(pipeline.unet, config)
                htcore.hpu_initialize(pipeline.unet, mark_only_scales_as_const=True)

            if args.use_hpu_graphs:
                pipeline.unet = torch_hpu.wrap_in_hpu_graph(pipeline.unet)

        else:
            from optimum.habana.diffusers import GaudiStableDiffusionXLPipeline

            pipeline = GaudiStableDiffusionXLPipeline.from_pretrained(
                args.model_name_or_path,
                **kwargs,
            )

    elif sd3:
        # SD3 pipelines
        if controlnet:
            # Import SD3+ControlNet pipeline
            raise ValueError("SD3+ControlNet pipeline is not currenly supported")
        elif inpainting:
            # Import SD3 Inpainting pipeline
            raise ValueError("SD3 Inpainting pipeline is not currenly supported")
        elif args.use_compel:
            raise ValueError("SD3 pipeline + Compel is not currenly supported")
        else:
            # Import SD3 pipeline
            from optimum.habana.diffusers import GaudiStableDiffusion3Pipeline

            pipeline = GaudiStableDiffusion3Pipeline.from_pretrained(
                args.model_name_or_path,
                **kwargs,
            )

    elif flux:
        # Flux pipelines
        if controlnet:
            raise ValueError("Flux+ControlNet pipeline is not currenly supported")
        elif inpainting:
            raise ValueError("Flux Inpainting pipeline is not currenly supported")
        else:
            # Import Flux pipeline
            from optimum.habana.diffusers import GaudiFluxPipeline

            pipeline = GaudiFluxPipeline.from_pretrained(
                args.model_name_or_path,
                **kwargs,
            )

    else:
        # SD pipelines (SD1.x, SD2.x)
        if controlnet:
            # SD+ControlNet pipeline
            from diffusers import ControlNetModel

            from optimum.habana.diffusers import GaudiStableDiffusionControlNetPipeline

            model_dtype = torch.bfloat16 if args.bf16 else None
            controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, torch_dtype=model_dtype)
            pipeline = GaudiStableDiffusionControlNetPipeline.from_pretrained(
                args.model_name_or_path,
                controlnet=controlnet,
                **kwargs,
            )

        elif inpainting:
            # SD Inpainting pipeline
            from optimum.habana.diffusers import AutoPipelineForInpainting

            pipeline = AutoPipelineForInpainting.from_pretrained(args.model_name_or_path, **kwargs)

        else:
            # SD pipeline
            if not args.ldm3d:
                from optimum.habana.diffusers import GaudiStableDiffusionPipeline

                pipeline = GaudiStableDiffusionPipeline.from_pretrained(
                    args.model_name_or_path,
                    **kwargs,
                )
                pipeline.unet.set_default_attn_processor(pipeline.unet)

                if args.unet_adapter_name_or_path is not None:
                    from peft import PeftModel, tuners

                    tuners.boft.layer._FBD_CUDA = False

                    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args.unet_adapter_name_or_path)
                    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.bf16):
                        pipeline.unet = pipeline.unet.merge_and_unload()

                if args.text_encoder_adapter_name_or_path is not None:
                    from peft import PeftModel, tuners

                    tuners.boft.layer._FBD_CUDA = False

                    pipeline.text_encoder = PeftModel.from_pretrained(
                        pipeline.text_encoder, args.text_encoder_adapter_name_or_path
                    )
                    with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.bf16):
                        pipeline.text_encoder = pipeline.text_encoder.merge_and_unload()
            else:
                # SD LDM3D use-case
                from optimum.habana.diffusers import GaudiStableDiffusionLDM3DPipeline as GaudiStableDiffusionPipeline

                if args.model_name_or_path == "CompVis/stable-diffusion-v1-4":
                    args.model_name_or_path = "Intel/ldm3d-4c"
                pipeline = GaudiStableDiffusionPipeline.from_pretrained(
                    args.model_name_or_path,
                    **kwargs,
                )

    # Load LoRA weights if provided
    if args.lora_id:
        pipeline.load_lora_weights(args.lora_id)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # Set RNG seed
    seed_dist_offset = int(os.getenv("RANK", "0"))
    if args.use_distributed_cfg:
        # Same seed needed for a pair of workers with distributed CFG for SD3
        seed_dist_offset = seed_dist_offset // 2
    set_seed(args.seed + seed_dist_offset)
    if args.use_compel:
        tokenizer = [pipeline.tokenizer]
        text_encoder = [pipeline.text_encoder]
        if sdxl:
            tokenizer.append(pipeline.tokenizer_2)
            text_encoder.append(pipeline.text_encoder_2)
            compel = Compel(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                device=torch.device("cpu"),
            )
        else:
            compel = Compel(tokenizer=tokenizer, text_encoder=text_encoder, device=torch.device("cpu"))

    if args.use_freeu:
        if args.use_hpu_graphs:
            raise ValueError("Freeu cannot support the HPU graph model, please disable it.")

        pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.5, b2=1.6)

    # If prompts file is specified override prompts from the file
    if args.prompts_file is not None:
        lines = []
        with open(args.prompts_file, "r") as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        args.prompts = lines

    # Generate images using auto-selected diffuser pipeline
    logger.info(
        f"Generating images using pipeline {pipeline.__class__.__name__} with scheduler {pipeline.scheduler.__class__.__name__}"
    )
    if args.distributed:
        with distributed_state.split_between_processes(args.prompts) as prompt:
            if args.use_compel:
                if sdxl:
                    conditioning, pooled = compel(prompt)
                    outputs = pipeline(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, **kwargs_call)
                else:
                    prompt_embeds = compel(prompt)
                    outputs = pipeline(prompt_embeds=prompt_embeds, **kwargs_call)
            else:
                outputs = pipeline(prompt=prompt, **kwargs_call)

    else:
        if args.use_compel:
            if sdxl:
                conditioning, pooled = compel(args.prompts)
                outputs = pipeline(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, **kwargs_call)
            else:
                prompt_embeds = compel(args.prompts)
                outputs = pipeline(prompt_embeds=prompt_embeds, **kwargs_call)
        else:
            outputs = pipeline(prompt=args.prompts, **kwargs_call)

    if args.optimize and quant_config_path and config.measure:
        from neural_compressor.torch.quantization import finalize_calibration

        logger.info("Finalizing calibration...")
        finalize_calibration(pipeline.unet)

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
            rank = int(os.getenv("RANK", "0"))
            world_size = int(os.getenv("WORLD_SIZE", "1"))
            rank_ext = f"_rank{rank}" if world_size > 1 else ""
            if args.ldm3d:
                for i, rgb in enumerate(outputs.rgb):
                    rgb.save(image_save_dir / f"rgb_{i + 1}{rank_ext}.png")
                for i, depth in enumerate(outputs.depth):
                    depth.save(image_save_dir / f"depth_{i + 1}{rank_ext}.png")
            else:
                skip_rank = False
                if args.use_distributed_cfg and world_size > 1:
                    rank_ext += f"and{rank + 1}"
                    skip_rank = rank % 2 == 1
                if not skip_rank:
                    for i, image in enumerate(outputs.images):
                        image.save(image_save_dir / f"image_{i + 1}{rank_ext}.png")
        else:
            logger.warning("--output_type should be equal to 'pil' to save images in --image_save_dir.")


if __name__ == "__main__":
    main()
