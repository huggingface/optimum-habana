import torch
from diffusers.utils import export_to_gif, load_image

from optimum.habana.diffusers import GaudiI2VGenXLPipeline

def main() :
    torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)
    pipeline = GaudiI2VGenXLPipeline.from_pretrained(
        "ali-vilab/i2vgen-xl",
        torch_dtype=torch.bfloat16,
        use_hpu_graphs=True,
        use_habana=True,
        gaudi_config="Habana/stable-diffusion",
    )
    #torch_dtype=torch.float,
    # import pdb;pdb.set_trace()
    # breakpoint()
    pipeline.enable_model_cpu_offload(device="hpu")

    image_url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"
    image = load_image(image_url).convert("RGB")

    prompt = "Papers were floating in the air on a table in the library"
    negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
    generator = torch.manual_seed(8888)
    # import pdb;pdb.set_trace()
    # breakpoint()
    frames = pipeline(
        prompt=prompt,
        image=image,
        num_inference_steps=50,
        negative_prompt=negative_prompt,
        guidance_scale=9.0,
        generator=generator,
    ).frames[0]
    export_to_gif(frames, "i2v.gif")

if __name__ == "__main__":
    main()
