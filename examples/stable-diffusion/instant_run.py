# !pip install opencv-python transformers accelerate insightface

# import habana_frameworks.torch.gpu_migration
import habana_frameworks.torch.core as htcore

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel

import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from insightface.app import FaceAnalysis
# from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from gaudi_pipeline_stable_diffusion_xl_instantid import GaudiStableDiffusionXLControlNetPipeline, draw_kps

from optimum.habana.diffusers import GaudiDDIMScheduler

import debugpy

debugpy.listen(("0.0.0.0", 5678))
print("Waiting for client to attach...")
debugpy.wait_for_client()


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

    # hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="./checkpoints")
    # hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="./checkpoints")
    # hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./checkpoints")


if __name__ == "__main__":
    download_instantID_model()

    # prepare 'antelopev2' under ./models
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # prepare models under ./checkpoints
    face_adapter = f'/home/optimum-habana/examples/stable-diffusion/checkpoints/ip-adapter.bin'
    controlnet_path = f'/home/optimum-habana/examples/stable-diffusion/checkpoints/ControlNetModel'
    sd_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    gaudi_config_name = 'Habana/stable-diffusion'

    scheduler = GaudiDDIMScheduler.from_pretrained(sd_model_name, subfolder="scheduler")

    kwargs = {
        "scheduler": scheduler,
        "use_habana": True,
        "use_hpu_graphs": True,
        "gaudi_config": gaudi_config_name,
    }

    # load IdentityNet
    # controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained(controlnet_path)

    # pipe = StableDiffusionXLInstantIDPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16)
    pipe = GaudiStableDiffusionXLControlNetPipeline.from_pretrained(sd_model_name, controlnet=controlnet, **kwargs)

    # load adapter
    pipe.load_ip_adapter_instantid(face_adapter)

    # load an image
    face_image = load_image("./examples/yann-lecun_resize.jpg")

    # prepare face emb
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])

    # prompt
    prompt = "film noir style, ink sketch|vector, male man, highly detailed, sharp focus, ultra sharpness, monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic"
    negative_prompt = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful"

    # generate image
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
    ).images[0]

    # image_save_dir = Path('/home/optimum-habana/examples/stable-diffusion/controlnet_images')
    # image_save_dir.mkdir(parents=True, exist_ok=True)
    # image.save(image_save_dir / f"image.png")

    print('SUCESS')
