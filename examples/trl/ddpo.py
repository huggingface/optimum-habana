# Copyright 2023 metric-space, The HuggingFace Team. All rights reserved.
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

# Adapted from: https://github.com/huggingface/trl/blob/v0.7.8/examples/scripts/ddpo.py
# The only differences are:
#   - add new args gaudi_config
#   - use GaudiDefaultDDPOStableDiffusionPipeline instead of DefaultDDPOStableDiffusionPipeline
#   - use GaudiDDPOTrainer instead of DDPOTrainer
#   - use hpu in aesthetic_scorer
#   - cast model to bf16.

"""
python ddpo.py \
    --num_epochs=200 \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=50 \
    --sample_batch_size=6 \
    --train_batch_size=3 \
    --sample_num_batches_per_epoch=4 \
    --train_learning_rate=1e-05 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="tensorboard" \
    --bf16
"""

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser
from trl import DDPOConfig

from optimum.habana import GaudiConfig
from optimum.habana.trl import GaudiDDPOTrainer, GaudiDefaultDDPOStableDiffusionPipeline


@dataclass
class ScriptArguments:
    hf_user_access_token: str = field(
        default=None, metadata={"help": "Hugging Face token. If None, token is retrieved from env or cache."}
    )
    pretrained_model: str = field(
        default="CompVis/stable-diffusion-v1-4", metadata={"help": "the pretrained model to use"}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
    hf_hub_model_id: str = field(
        default="ddpo-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to"}
    )
    hf_hub_aesthetic_model_id: str = field(
        default="trl-lib/ddpo-aesthetic-predictor",
        metadata={"help": "HuggingFace model ID for aesthetic scorer model weights"},
    )
    hf_hub_aesthetic_model_filename: str = field(
        default="aesthetic-model.pth",
        metadata={"help": "HuggingFace model filename for aesthetic scorer model weights"},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})
    bf16: bool = field(default=False, metadata={"help": "Whether to use bf16 mixed precision."})
    gaudi_config_name: str = field(
        default="Habana/stable-diffusion", metadata={"help": "Name or path of the Gaudi configuration"}
    )
    push_to_hub: bool = field(default=False, metadata={"help": "Whether or not to push the model to the Hub."})
    use_habana: bool = field(default=True, metadata={"help": "Whether or not to use HPU."})
    use_hpu_graphs: bool = field(default=True, metadata={"help": "Whether or not to use hpu graphs."})


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    """
    This model attempts to predict the aesthetic score of an image. The aesthetic score
    is a numerical approximation of how much a specific image is liked by humans on average.
    This is from https://github.com/christophschuhmann/improved-aesthetic-predictor
    """

    def __init__(self, *, dtype, model_id, model_filename):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        try:
            cached_path = hf_hub_download(model_id, model_filename)
        except EntryNotFoundError:
            cached_path = os.path.join(model_id, model_filename)
        state_dict = torch.load(cached_path, map_location=torch.device("cpu"))
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


def aesthetic_scorer(hub_model_id, model_filename, use_habana=True):
    scorer = AestheticScorer(
        model_id=hub_model_id,
        model_filename=model_filename,
        dtype=torch.float32,
    )
    if use_habana:
        scorer = scorer.to("hpu")
    else:
        scorer = scorer.to("cpu")

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


# list of example prompts to feed stable diffusion
animals = [
    "cat",
    "dog",
    "horse",
    "monkey",
    "rabbit",
    "zebra",
    "spider",
    "bird",
    "sheep",
    "deer",
    "cow",
    "goat",
    "lion",
    "frog",
    "chicken",
    "duck",
    "goose",
    "bee",
    "pig",
    "turkey",
    "fly",
    "llama",
    "camel",
    "bat",
    "gorilla",
    "hedgehog",
    "kangaroo",
]


def prompt_fn():
    return np.random.choice(animals), {}


def image_outputs_logger(image_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts, _, rewards, _ = image_data[-1]

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()

    accelerate_logger.log_images(
        result,
        step=global_step,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    args, ddpo_config = parser.parse_args_into_dataclasses()
    ddpo_config.mixed_precision = "bf16" if args.bf16 else "no"
    ddpo_config.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": "./save",
    }

    # 1. initialize Gaudi config:
    gaudi_config = GaudiConfig.from_pretrained(args.gaudi_config_name) if args.use_habana else None

    pipeline = GaudiDefaultDDPOStableDiffusionPipeline(
        args.pretrained_model,
        pretrained_model_revision=args.pretrained_revision,
        use_lora=args.use_lora,
        use_habana=args.use_habana,
        use_hpu_graphs=args.use_hpu_graphs,
        gaudi_config=gaudi_config,
    )

    trainer = GaudiDDPOTrainer(
        ddpo_config,
        aesthetic_scorer(
            args.hf_hub_aesthetic_model_id,
            args.hf_hub_aesthetic_model_filename,
            args.use_habana,
        ),
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
        gaudi_config=gaudi_config,
        use_habana=args.use_habana,
        use_hpu_graphs=args.use_hpu_graphs,
    )

    trainer.train()

    if args.push_to_hub:
        trainer.push_to_hub(args.hf_hub_model_id, token=args.hf_user_access_token)
    else:
        trainer.save_pretrained(args.hf_hub_model_id)
