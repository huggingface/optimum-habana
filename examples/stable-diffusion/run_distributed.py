# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

"""
Adapted from: https://huggingface.co/docs/diffusers/en/training/distributed_inference
 - Use the GaudiStableDiffusionPipeline
"""
import torch
import logging
import argparse
from accelerate import PartialState
from optimum.habana.diffusers import GaudiStableDiffusionPipeline
from optimum.habana.utils import set_seed

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
        default=["a dog", "a cat"],
        help="The prompt or prompts to guide the image generation.",
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1, help="The number of images to generate per prompt."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for initialization.")
    parser.add_argument("--bf16", action="store_true", help="Whether to perform generation in bf16 precision.")
    parser.add_argument(
        "--gaudi_config",
        type=str,
        default="Habana/stable-diffusion",
        help=(
            "Name or path of the Gaudi configuration. In particular, it enables to specify how to apply Habana Mixed"
            " Precision."
        ),
    )
    # HPU-specific arguments
    parser.add_argument("--use_habana", action="store_true", help="Use HPU.")
    parser.add_argument(
        "--use_hpu_graphs", action="store_true", help="Use HPU graphs on HPU. This should lead to faster generations."
    )
    args = parser.parse_args()
     # Set seed before running the model
    if args.seed:
        logger.info("Set the random seed {}!".format(args.seed))
        set_seed(args.seed)

    kwargs = {
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": args.gaudi_config,
        "torch_dtype": torch.bfloat16 if args.bf16 else None
    }
    print(f"kwargs={kwargs}")
    pipeline = GaudiStableDiffusionPipeline.from_pretrained(
        args.model_name_or_path, use_safetensors=True, **kwargs
    )
    distributed_state = PartialState()
    kwargs= {
        "num_images_per_prompt": args.num_images_per_prompt
    }
    with distributed_state.split_between_processes(args.prompts) as prompt:
        outputs = pipeline(prompt, **kwargs)
        for i, image in enumerate(outputs.images):
            image.save(f"result_{distributed_state.process_index}_{i}.png")

if __name__ == "__main__":
    main()
