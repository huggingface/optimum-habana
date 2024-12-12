import argparse
import logging
import sys
from pathlib import Path

import torch
from pipeline_cogvideox_gaudi import GaudiCogVideoXPipeline
#from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from optimum.habana.transformers.gaudi_configuration import GaudiConfig
from optimum.habana.utils import set_seed
logger = logging.getLogger(__name__)

prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

#prompt = "A 360-degree panoramic view of a lush mountain valley with a flowing river, birds flying across the sky, and a soft orange-pink sunrise."
#prompt = "Spiderman is surfing, Darth Vader is also surfing and following Spiderman"
#prompt = "An astronaut riding a horse"
#prompt = "A drone shot flying above vibrant red and orange foliage with occasional sunlight beams piercing through the canopy."
#prompt = "Skyscrapers with glowing neon signs, flying cars zipping between buildings, and a massive digital billboard displaying a news broadcast."
#prompt = "Bright, surreal waves of color blending and transforming into abstract shapes in rhythm with gentle ambient music."
#prompt = "A first-person view of a runner jumping between rooftops, flipping over obstacles, and climbing walls."

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
model_path = "/mnt/disk2/libo/hf_models/CogVideoX-2b/"
#model_path = "/mnt/disk2/libo/hf_models/CogVideoX-5b/"
pipe = GaudiCogVideoXPipeline.from_pretrained(
    model_path,
    **kwargs
)
print('pipe line load done!')

pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

print('now to generate video.')
video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cpu").manual_seed(42),
).frames[0]

print('generate video done!')

export_to_video(video, "panda_gaudi.mp4", fps=8)
#export_to_video(video, "output_gaudi.mp4", fps=8)
#export_to_video(video, "Spiderman_gaudi.mp4", fps=8)
#export_to_video(video, "astronaut_gaudi.mp4", fps=8)
#export_to_video(video, "drone_gaudi.mp4", fps=8)
#export_to_video(video, "Skyscrapers_gaudi.mp4", fps=8)
#export_to_video(video, "waves_gaudi.mp4", fps=8)




 
