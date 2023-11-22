#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
# limitations under the License.

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionControlNetPipeline
from diffusers.utils import load_image
import numpy as np
import torch

import cv2
from PIL import Image

# download an image
image = load_image(
"https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)
image = np.array(image)

# get canny image
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

# load control net and stable diffusion v1-5
model_id = "runwayml/stable-diffusion-v1-5"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
scheduler = GaudiDDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = GaudiStableDiffusionControlNetPipeline.from_pretrained(
model_id, 
controlnet=controlnet,
scheduler=scheduler,
use_habana=True,
#use_hpu_graphs=False,
use_hpu_graphs=True,
gaudi_config="Habana/stable-diffusion",
)

# generate image
generator = torch.manual_seed(0)

outputs = pipe(
"futuristic-looking woman", num_inference_steps=50, generator=generator, input_image=canny_image, num_images_per_prompt=20, batch_size=4
)

for i, image in enumerate(outputs.images):
    image.save(f"image_{i+1}.png")
