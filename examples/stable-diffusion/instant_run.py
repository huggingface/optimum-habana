import argparse
from diffusers.utils import load_image
from diffusers.models import ControlNetModel

import cv2
import torch
import numpy as np
import gdown
import os
from pathlib import Path

from insightface.app import FaceAnalysis
from gaudi_pipeline_stable_diffusion_xl_instantid import GaudiStableDiffusionXLControlNetPipeline, draw_kps

from optimum.habana.diffusers import GaudiDDIMScheduler

import debugpy

# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for client to attach...")
# debugpy.wait_for_client()


def download_instantID_model():
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/config.json",
        local_dir="./checkpoints",
    )

    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/diffusion_pytorch_model.safetensors",
        local_dir="./checkpoints",
    )

    hf_hub_download(
        repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./checkpoints"
    )

    hf_hub_download(
        repo_id="latent-consistency/lcm-lora-sdxl",
        filename="pytorch_lora_weights.safetensors",
        local_dir="./checkpoints",
    )


def download_antelopev2():
    antelopev2_path = "./models/antelopev2"
    if os.path.exists(antelopev2_path) and os.path.isdir(antelopev2_path) and os.listdir(antelopev2_path):
        return
    
    gdown.download(url="https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing", output="./models/", quiet=False, fuzzy=True)
    os.system("unzip ./models/antelopev2.zip -d ./models/")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="stabilityai/stable-diffusion-xl-base-1.0",
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
        "--control_image",
        type=str,
        default=None,
        help=("Path to the controlnet conditioning image"),
    )
    # parser.add_argument(
    #     "--num_images_per_prompt", type=int, default=1, help="The number of images to generate per prompt."
    # )
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
        "--negative_prompts",
        type=str,
        nargs="*",
        default=None,
        help="The prompt or prompts not to guide the image generation.",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        choices=["pil", "np"],
        default="pil",
        help="Whether to return PIL images or Numpy arrays.",
    )
    parser.add_argument(
        "--image_save_dir",
        type=str,
        default="./stable-diffusion-generated-images",
        help="The directory where images will be saved.",
    )

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
    
    return parser.parse_args()


if __name__ == "__main__":
    download_instantID_model()
    download_antelopev2()

    args = parse_args()

    app = FaceAnalysis(name='antelopev2', root='./', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    face_adapter = f'./checkpoints/ip-adapter.bin'

    # Initialize the scheduler and the generation pipeline
    kwargs = {"timestep_spacing": args.timestep_spacing}
    scheduler = GaudiDDIMScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler", **kwargs)

    kwargs = {
        "scheduler": scheduler,
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": args.gaudi_config_name,
    }

    if args.bf16:
        kwargs["torch_dtype"] = torch.bfloat16

    # load IdentityNet
    model_dtype = torch.bfloat16 if args.bf16 else None
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, torch_dtype=model_dtype)

    pipe = GaudiStableDiffusionXLControlNetPipeline.from_pretrained(args.model_name_or_path, controlnet=controlnet, **kwargs)

    # load adapter
    pipe.load_ip_adapter_instantid(face_adapter)

    # load an image
    face_image = load_image("./examples/yann-lecun_resize.jpg")

    # prepare face emb
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])

    # generate image
    image = pipe(
        args.prompts,
        num_inference_steps=args.num_inference_steps,
        negative_prompt=args.negative_prompts,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
        profiling_warmup_steps=args.profiling_warmup_steps,
        profiling_steps=args.profiling_steps,
    ).images[0]

    image_save_dir = Path(args.image_save_dir)
    image_save_dir.mkdir(parents=True, exist_ok=True)
    image.save(image_save_dir / f"image.png")

    print('SUCESS')
