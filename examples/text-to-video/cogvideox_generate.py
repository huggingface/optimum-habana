import argparse
import logging

import torch
from diffusers.utils import export_to_video

from optimum.habana.diffusers.pipelines.cogvideox.cogvideoX_gaudi import adapt_cogvideo_to_gaudi
from optimum.habana.diffusers.pipelines.cogvideox.pipeline_cogvideox_gaudi import GaudiCogVideoXPipeline
from optimum.habana.transformers.gaudi_configuration import GaudiConfig


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument(
        "--model_name_or_path",
        default="THUDM/CogVideoX-2b",
        type=str,
        help="Path to pre-trained model",
    )
    # Pipeline arguments
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default="A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance.",
        help="The prompt or prompts to guide the video generation.",
    )
    parser.add_argument(
        "--output_name",
        default="panda_gaudi.mp4",
        type=str,
        help="Path to pre-trained model",
    )

    args = parser.parse_args()

    gaudi_config_kwargs = {"use_fused_adam": True, "use_fused_clip_norm": True}
    gaudi_config_kwargs["use_torch_autocast"] = True

    gaudi_config = GaudiConfig(**gaudi_config_kwargs)
    logger.info(f"Gaudi Config: {gaudi_config}")


    kwargs = {
        "use_habana": True,
        "use_hpu_graphs": True,
        "gaudi_config": gaudi_config,
    }
    kwargs["torch_dtype"] = torch.bfloat16


    print('now to load model.....')
    pipe = GaudiCogVideoXPipeline.from_pretrained(
        args.model_name_or_path,
        **kwargs
    )
    print('pipe line load done!')

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    print('now to generate video.')
    video = pipe(
        prompt=args.prompts,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).frames[0]

    print('generate video done!')

    export_to_video(video, args.output_name, fps=8)



if __name__ == "__main__":
    main()


