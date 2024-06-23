# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import random
import re
import gc
import copy
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Union
from unittest import TestCase, skipUnless

import numpy as np
import requests
import safetensors
import torch
from diffusers import (
    AutoencoderKL,
    AutoencoderKLTemporalDecoder,
    ControlNetModel,
    UNet2DConditionModel,
    UNetSpatioTemporalConditionModel,
    UniPCMultistepScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
)
from diffusers.pipelines.controlnet.pipeline_controlnet import MultiControlNetModel
from diffusers.utils import load_image, numpy_to_pil
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    slow,
)
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from parameterized import parameterized
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)
from transformers.testing_utils import parse_flag_from_env, slow

from optimum.habana import GaudiConfig
from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiDiffusionPipeline,
    GaudiEulerAncestralDiscreteScheduler,
    GaudiEulerDiscreteScheduler,
    GaudiStableDiffusionControlNetPipeline,
    GaudiStableDiffusionLDM3DPipeline,
    GaudiStableDiffusionPipeline,
    GaudiStableDiffusionUpscalePipeline,
    GaudiStableDiffusionXLPipeline,
    GaudiStableVideoDiffusionPipeline,
    GaudiStableDiffusionInpaintPipeline,
    GaudiStableDiffusionXLInpaintPipeline,
)
from optimum.habana.utils import set_seed

from .clip_coco_utils import calculate_clip_score, download_files
from .test_pipelines_common import PipelineKarrasSchedulerTesterMixin, PipelineLatentTesterMixin, PipelineTesterMixin


IS_GAUDI2 = os.environ.get("GAUDI2_CI", "0") == "0"


if IS_GAUDI2:
    THROUGHPUT_BASELINE_BF16 = 1.086
    THROUGHPUT_BASELINE_AUTOCAST = 0.394
    TEXTUAL_INVERSION_THROUGHPUT = 104.29806
    TEXTUAL_INVERSION_RUNTIME = 114.1344320399221
    CONTROLNET_THROUGHPUT = 92.886919836857
    CONTROLNET_RUNTIME = 537.4276602957398
    INPAINT_THROUGHPUT_BASELINE_BF16 = 0.84
    INPAINT_XL_THROUGHPUT_BASELINE_BF16 = 0.84
else:
    THROUGHPUT_BASELINE_BF16 = 0.309
    THROUGHPUT_BASELINE_AUTOCAST = 0.114
    TEXTUAL_INVERSION_THROUGHPUT = 60.5991479573174
    TEXTUAL_INVERSION_RUNTIME = 196.43840550999994
    CONTROLNET_THROUGHPUT = 44.7278034963213
    CONTROLNET_RUNTIME = 1116.084316640001
    INPAINT_THROUGHPUT_BASELINE_BF16 = 1.422
    INPAINT_XL_THROUGHPUT_BASELINE_BF16 = 0.265


_run_custom_bf16_ops_test_ = parse_flag_from_env("CUSTOM_BF16_OPS", default=False)


def custom_bf16_ops(test_case):
    """
    Decorator marking a test as needing custom bf16 ops.
    Custom bf16 ops must be declared before `habana_frameworks.torch.core` is imported, which is not possible if some other tests are executed before.

    Such tests are skipped by default. Set the CUSTOM_BF16_OPS environment variable to a truthy value to run them.

    """
    return skipUnless(_run_custom_bf16_ops_test_, "test requires custom bf16 ops")(test_case)


class GaudiPipelineUtilsTester(TestCase):
    """
    Tests the features added on top of diffusers/pipeline_utils.py.
    """

    def test_use_hpu_graphs_raise_error_without_habana(self):
        with self.assertRaises(ValueError):
            _ = GaudiDiffusionPipeline(
                use_habana=False,
                use_hpu_graphs=True,
            )

    def test_gaudi_config_raise_error_without_habana(self):
        with self.assertRaises(ValueError):
            _ = GaudiDiffusionPipeline(
                use_habana=False,
                gaudi_config=GaudiConfig(),
            )

    def test_device(self):
        pipeline_1 = GaudiDiffusionPipeline(
            use_habana=True,
            gaudi_config=GaudiConfig(),
        )
        self.assertEqual(pipeline_1._device.type, "hpu")

        pipeline_2 = GaudiDiffusionPipeline(
            use_habana=False,
        )
        self.assertEqual(pipeline_2._device.type, "cpu")

    def test_gaudi_config_types(self):
        # gaudi_config is a string
        _ = GaudiDiffusionPipeline(
            use_habana=True,
            gaudi_config="Habana/stable-diffusion",
        )

        # gaudi_config is instantiated beforehand
        gaudi_config = GaudiConfig.from_pretrained("Habana/stable-diffusion")
        _ = GaudiDiffusionPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
        )

    def test_default(self):
        pipeline = GaudiDiffusionPipeline(
            use_habana=True,
            gaudi_config=GaudiConfig(),
        )

        self.assertTrue(hasattr(pipeline, "htcore"))

    def test_use_hpu_graphs(self):
        pipeline = GaudiDiffusionPipeline(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(),
        )

        self.assertTrue(hasattr(pipeline, "ht"))
        self.assertTrue(hasattr(pipeline, "hpu_stream"))
        self.assertTrue(hasattr(pipeline, "cache"))

    def test_save_pretrained(self):
        model_name = "hf-internal-testing/tiny-stable-diffusion-torch"
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            gaudi_config=GaudiConfig(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)
            self.assertTrue(Path(tmp_dir, "gaudi_config.json").is_file())


class GaudiStableDiffusionPipelineTester(TestCase):
    """
    Tests the StableDiffusionPipeline for Gaudi.
    """

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            sample_size=32,
            time_cond_proj_dim=time_cond_proj_dim,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            norm_num_groups=2,
        )
        scheduler = GaudiDDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[4, 8],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=64,
            layer_norm_eps=1e-05,
            num_attention_heads=8,
            num_hidden_layers=3,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_ddim(self):
        device = "cpu"

        components = self.get_dummy_components()
        gaudi_config = GaudiConfig(use_torch_autocast=False)

        sd_pipe = GaudiStableDiffusionPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images[0]

        image_slice = image[-3:, -3:, -1]

        self.assertEqual(image.shape, (64, 64, 3))
        expected_slice = np.array([0.3203, 0.4555, 0.4711, 0.3505, 0.3973, 0.4650, 0.5137, 0.3392, 0.4045])

        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-2)

    def test_stable_diffusion_no_safety_checker(self):
        gaudi_config = GaudiConfig()
        scheduler = GaudiDDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        pipe = GaudiStableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            scheduler=scheduler,
            safety_checker=None,
            use_habana=True,
            gaudi_config=gaudi_config,
        )
        self.assertIsInstance(pipe, GaudiStableDiffusionPipeline)
        self.assertIsInstance(pipe.scheduler, GaudiDDIMScheduler)
        self.assertIsNone(pipe.safety_checker)

        image = pipe("example prompt", num_inference_steps=2).images[0]
        self.assertIsNotNone(image)

        # Check that there's no error when saving a pipeline with one of the models being None
        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe = GaudiStableDiffusionPipeline.from_pretrained(
                tmpdirname,
                use_habana=True,
                gaudi_config=tmpdirname,
            )

        # Sanity check that the pipeline still works
        self.assertIsNone(pipe.safety_checker)
        image = pipe("example prompt", num_inference_steps=2).images[0]
        self.assertIsNotNone(image)

    @parameterized.expand(["pil", "np", "latent"])
    def test_stable_diffusion_output_types(self, output_type):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        num_prompts = 2
        num_images_per_prompt = 3

        outputs = sd_pipe(
            num_prompts * [prompt],
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=2,
            output_type=output_type,
        )

        self.assertEqual(len(outputs.images), 2 * 3)
        # TODO: enable safety checker
        # if output_type == "latent":
        #     self.assertIsNone(outputs.nsfw_content_detected)
        # else:
        #     self.assertEqual(len(outputs.nsfw_content_detected), 2 * 3)

    # TODO: enable this test when PNDMScheduler is adapted to Gaudi
    # def test_stable_diffusion_negative_prompt(self):
    #     device = "cpu"  # ensure determinism for the device-dependent torch.Generator
    #     unet = self.dummy_cond_unet
    #     scheduler = PNDMScheduler(skip_prk_steps=True)
    #     vae = self.dummy_vae
    #     bert = self.dummy_text_encoder
    #     tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

    #     # make sure here that pndm scheduler skips prk
    #     sd_pipe = StableDiffusionPipeline(
    #         unet=unet,
    #         scheduler=scheduler,
    #         vae=vae,
    #         text_encoder=bert,
    #         tokenizer=tokenizer,
    #         safety_checker=None,
    #         feature_extractor=self.dummy_extractor,
    #     )
    #     sd_pipe = sd_pipe.to(device)
    #     sd_pipe.set_progress_bar_config(disable=None)

    #     prompt = "A painting of a squirrel eating a burger"
    #     negative_prompt = "french fries"
    #     generator = torch.Generator(device=device).manual_seed(0)
    #     output = sd_pipe(
    #         prompt,
    #         negative_prompt=negative_prompt,
    #         generator=generator,
    #         guidance_scale=6.0,
    #         num_inference_steps=2,
    #         output_type="np",
    #     )

    #     image = output.images
    #     image_slice = image[0, -3:, -3:, -1]

    #     assert image.shape == (1, 128, 128, 3)
    #     expected_slice = np.array([0.4851, 0.4617, 0.4765, 0.5127, 0.4845, 0.5153, 0.5141, 0.4886, 0.4719])
    #     assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_num_images_per_prompt(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # Test num_images_per_prompt=1 (default)
        images = sd_pipe(prompt, num_inference_steps=2, output_type="np").images

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (64, 64, 3))

        # Test num_images_per_prompt=1 (default) for several prompts
        num_prompts = 3
        images = sd_pipe([prompt] * num_prompts, num_inference_steps=2, output_type="np").images

        self.assertEqual(len(images), num_prompts)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        images = sd_pipe(
            prompt, num_inference_steps=2, output_type="np", num_images_per_prompt=num_images_per_prompt
        ).images

        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Test num_images_per_prompt for several prompts
        num_prompts = 2
        images = sd_pipe(
            [prompt] * num_prompts,
            num_inference_steps=2,
            output_type="np",
            num_images_per_prompt=num_images_per_prompt,
        ).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    def test_stable_diffusion_batch_sizes(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # Test num_images > 1 where num_images is a divider of the total number of generated images
        batch_size = 3
        num_images_per_prompt = batch_size**2
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Same test for several prompts
        num_prompts = 3
        images = sd_pipe(
            [prompt] * num_prompts,
            num_inference_steps=2,
            output_type="np",
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Test num_images when it is not a divider of the total number of generated images for a single prompt
        num_images_per_prompt = 7
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Same test for several prompts
        num_prompts = 2
        images = sd_pipe(
            [prompt] * num_prompts,
            num_inference_steps=2,
            output_type="np",
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    def test_stable_diffusion_bf16(self):
        """Test that stable diffusion works with bf16"""
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device="cpu").manual_seed(0)
        image = sd_pipe([prompt], generator=generator, num_inference_steps=2, output_type="np").images[0]

        self.assertEqual(image.shape, (64, 64, 3))

    def test_stable_diffusion_default(self):
        components = self.get_dummy_components()

        sd_pipe = GaudiStableDiffusionPipeline(
            use_habana=True,
            gaudi_config="Habana/stable-diffusion",
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device="cpu").manual_seed(0)
        images = sd_pipe(
            [prompt] * 2,
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            batch_size=3,
            num_images_per_prompt=5,
        ).images

        self.assertEqual(len(images), 10)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    def test_stable_diffusion_hpu_graphs(self):
        components = self.get_dummy_components()

        sd_pipe = GaudiStableDiffusionPipeline(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device="cpu").manual_seed(0)
        images = sd_pipe(
            [prompt] * 2,
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            batch_size=3,
            num_images_per_prompt=5,
        ).images

        self.assertEqual(len(images), 10)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    @slow
    def test_no_throughput_regression_bf16(self):
        prompts = [
            "An image of a squirrel in Picasso style",
            "High quality photo of an astronaut riding a horse in space",
        ]
        num_images_per_prompt = 11
        batch_size = 4
        model_name = "runwayml/stable-diffusion-v1-5"
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

        pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig.from_pretrained("Habana/stable-diffusion"),
            torch_dtype=torch.bfloat16,
        )
        set_seed(27)
        outputs = pipeline(
            prompt=prompts,
            num_images_per_prompt=num_images_per_prompt,
            batch_size=batch_size,
        )
        self.assertEqual(len(outputs.images), num_images_per_prompt * len(prompts))
        self.assertGreaterEqual(outputs.throughput, 0.95 * THROUGHPUT_BASELINE_BF16)

    @custom_bf16_ops
    @slow
    def test_no_throughput_regression_autocast(self):
        prompts = [
            "An image of a squirrel in Picasso style",
            "High quality photo of an astronaut riding a horse in space",
        ]
        num_images_per_prompt = 11
        batch_size = 4
        model_name = "stabilityai/stable-diffusion-2-1"
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

        pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig.from_pretrained("Habana/stable-diffusion-2"),
        )
        set_seed(27)
        outputs = pipeline(
            prompt=prompts,
            num_images_per_prompt=num_images_per_prompt,
            batch_size=batch_size,
            height=768,
            width=768,
        )
        self.assertEqual(len(outputs.images), num_images_per_prompt * len(prompts))
        self.assertGreaterEqual(outputs.throughput, 0.95 * THROUGHPUT_BASELINE_AUTOCAST)

    @slow
    def test_no_generation_regression(self):
        seed = 27
        set_seed(seed)
        model_name = "CompVis/stable-diffusion-v1-4"
        # fp32
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            safety_checker=None,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(use_torch_autocast=False),
        )

        prompt = "An image of a squirrel in Picasso style"
        generator = torch.manual_seed(seed)
        outputs = pipeline(
            prompt=prompt,
            generator=generator,
            output_type="np",
        )

        if IS_GAUDI2:
            target_score = 29.8925
        else:
            target_score = 36.774

        image = outputs.images[0]
        pil_image = numpy_to_pil(image)[0]
        pil_image.save("test_no_generation_regression_output.png")

        clip_score = calculate_clip_score(np.expand_dims(image, axis=0), [prompt])

        self.assertEqual(image.shape, (512, 512, 3))
        self.assertGreaterEqual(clip_score, target_score)

    @slow
    def test_no_generation_regression_ldm3d(self):
        seed = 27
        set_seed(seed)
        model_name = "Intel/ldm3d-4c"
        # fp32
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipeline = GaudiStableDiffusionLDM3DPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            safety_checker=None,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(),
        )

        prompt = "An image of a squirrel in Picasso style"
        generator = torch.manual_seed(seed)
        outputs = pipeline(
            prompt=prompt,
            generator=generator,
            output_type="np",
        )

        if IS_GAUDI2:
            target_score = 28.0894
        else:
            target_score = 35.81

        rgb = outputs.rgb[0]
        depth = outputs.depth[0]

        rgb_clip_score = calculate_clip_score(np.expand_dims(rgb, axis=0), [prompt])

        self.assertEqual(rgb.shape, (512, 512, 3))
        self.assertEqual(depth.shape, (512, 512, 1))
        self.assertGreaterEqual(rgb_clip_score, target_score)

    @slow
    def test_no_generation_regression_upscale(self):
        model_name = "stabilityai/stable-diffusion-x4-upscaler"
        # fp32
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipeline = GaudiStableDiffusionUpscalePipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(use_torch_autocast=False),
        )
        set_seed(27)

        url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
        response = requests.get(url)
        low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
        low_res_img = low_res_img.resize((128, 128))
        prompt = "a white cat"
        upscaled_image = pipeline(prompt=prompt, image=low_res_img, output_type="np").images[0]
        if IS_GAUDI2:
            expected_slice = np.array(
                [
                    0.16527882,
                    0.161616,
                    0.15665859,
                    0.1660901,
                    0.1594379,
                    0.14936888,
                    0.1578255,
                    0.15342498,
                    0.14590919,
                ]
            )
        else:
            expected_slice = np.array(
                [
                    0.1652787,
                    0.16161594,
                    0.15665877,
                    0.16608998,
                    0.1594378,
                    0.14936894,
                    0.15782538,
                    0.15342498,
                    0.14590913,
                ]
            )
        self.assertEqual(upscaled_image.shape, (512, 512, 3))
        self.assertLess(np.abs(expected_slice - upscaled_image[-3:, -3:, -1].flatten()).max(), 5e-3)

    @slow
    def test_textual_inversion(self):
        path_to_script = (
            Path(os.path.dirname(__file__)).parent
            / "examples"
            / "stable-diffusion"
            / "training"
            / "textual_inversion.py"
        )

        with tempfile.TemporaryDirectory() as data_dir:
            snapshot_download(
                "diffusers/cat_toy_example", local_dir=data_dir, repo_type="dataset", ignore_patterns=".gitattributes"
            )
            with tempfile.TemporaryDirectory() as run_dir:
                cmd_line = [
                    "python3",
                    f"{path_to_script.parent.parent.parent / 'gaudi_spawn.py'}",
                    "--use_mpi",
                    "--world_size",
                    "8",
                    f"{path_to_script}",
                    "--pretrained_model_name_or_path runwayml/stable-diffusion-v1-5",
                    f"--train_data_dir {data_dir}",
                    '--learnable_property "object"',
                    '--placeholder_token "<cat-toy>"',
                    '--initializer_token "toy"',
                    "--resolution 512",
                    "--train_batch_size 4",
                    "--max_train_steps 375",
                    "--learning_rate 5.0e-04",
                    "--scale_lr",
                    '--lr_scheduler "constant"',
                    "--lr_warmup_steps 0",
                    f"--output_dir {run_dir}",
                    "--save_as_full_pipeline",
                    "--gaudi_config_name Habana/stable-diffusion",
                    "--throughput_warmup_steps 3",
                    "--seed 27",
                ]
                pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
                cmd_line = [x for y in cmd_line for x in re.split(pattern, y) if x]

                # Run textual inversion
                p = subprocess.Popen(cmd_line)
                return_code = p.wait()

                # Ensure the run finished without any issue
                self.assertEqual(return_code, 0)

                # Assess throughput
                with open(Path(run_dir) / "speed_metrics.json") as fp:
                    results = json.load(fp)
                self.assertGreaterEqual(results["train_samples_per_second"], 0.95 * TEXTUAL_INVERSION_THROUGHPUT)
                self.assertLessEqual(results["train_runtime"], 1.05 * TEXTUAL_INVERSION_RUNTIME)

                # Assess generated image
                pipe = GaudiStableDiffusionPipeline.from_pretrained(
                    run_dir,
                    torch_dtype=torch.bfloat16,
                    use_habana=True,
                    use_hpu_graphs=True,
                    gaudi_config=GaudiConfig(use_habana_mixed_precision=False),
                )
                prompt = "A <cat-toy> backpack"
                set_seed(27)
                image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, output_type="np").images[0]

                # TODO: see how to generate images in a reproducible way
                # expected_slice = np.array(
                #     [0.57421875, 0.5703125, 0.58203125, 0.58203125, 0.578125, 0.5859375, 0.578125, 0.57421875, 0.56640625]
                # )
                self.assertEqual(image.shape, (512, 512, 3))
                # self.assertLess(np.abs(expected_slice - image[-3:, -3:, -1].flatten()).max(), 5e-3)


class GaudiStableDiffusionXLPipelineTester(TestCase):
    """
    Tests the StableDiffusionXLPipeline for Gaudi.
    """

    def get_dummy_components(self, time_cond_proj_dim=None, timestep_spacing="leading"):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(2, 4),
            layers_per_block=2,
            time_cond_proj_dim=time_cond_proj_dim,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
            norm_num_groups=1,
        )
        scheduler = GaudiEulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing=timestep_spacing,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "image_encoder": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_xl_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig(use_torch_autocast=False)
        sd_pipe = GaudiStableDiffusionXLPipeline(use_habana=True, gaudi_config=gaudi_config, **components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images[0]

        image_slice = image[-3:, -3:, -1]

        self.assertEqual(image.shape, (64, 64, 3))
        expected_slice = np.array([0.5552, 0.5569, 0.4725, 0.4348, 0.4994, 0.4632, 0.5142, 0.5012, 0.47])

        # The threshold should be 1e-2 below but it started failing
        # from Diffusers v0.24. However, generated images still look similar.
        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-1)

    def test_stable_diffusion_xl_euler_ancestral(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig(use_torch_autocast=False)
        sd_pipe = GaudiStableDiffusionXLPipeline(use_habana=True, gaudi_config=gaudi_config, **components)
        sd_pipe.scheduler = GaudiEulerAncestralDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images[0]

        image_slice = image[-3:, -3:, -1]

        self.assertEqual(image.shape, (64, 64, 3))
        expected_slice = np.array([0.4675, 0.5173, 0.4611, 0.4067, 0.5250, 0.4674, 0.5446, 0.5094, 0.4791])
        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-2)

    def test_stable_diffusion_xl_turbo_euler_ancestral(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(timestep_spacing="trailing")
        gaudi_config = GaudiConfig(use_torch_autocast=False)

        sd_pipe = GaudiStableDiffusionXLPipeline(use_habana=True, gaudi_config=gaudi_config, **components)
        sd_pipe.scheduler = GaudiEulerAncestralDiscreteScheduler.from_config(sd_pipe.scheduler.config)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images[0]

        image_slice = image[-3:, -3:, -1]

        self.assertEqual(image.shape, (64, 64, 3))
        expected_slice = np.array([0.4675, 0.5173, 0.4611, 0.4067, 0.5250, 0.4674, 0.5446, 0.5094, 0.4791])
        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-2)

    @parameterized.expand(["pil", "np", "latent"])
    def test_stable_diffusion_xl_output_types(self, output_type):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionXLPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        num_prompts = 2
        num_images_per_prompt = 3

        outputs = sd_pipe(
            num_prompts * [prompt],
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=2,
            output_type=output_type,
        )

        self.assertEqual(len(outputs.images), 2 * 3)

    def test_stable_diffusion_xl_num_images_per_prompt(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionXLPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # Test num_images_per_prompt=1 (default)
        images = sd_pipe(prompt, num_inference_steps=2, output_type="np").images

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (64, 64, 3))

        # Test num_images_per_prompt=1 (default) for several prompts
        num_prompts = 3
        images = sd_pipe([prompt] * num_prompts, num_inference_steps=2, output_type="np").images

        self.assertEqual(len(images), num_prompts)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        images = sd_pipe(
            prompt, num_inference_steps=2, output_type="np", num_images_per_prompt=num_images_per_prompt
        ).images

        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Test num_images_per_prompt for several prompts
        num_prompts = 2
        images = sd_pipe(
            [prompt] * num_prompts,
            num_inference_steps=2,
            output_type="np",
            num_images_per_prompt=num_images_per_prompt,
        ).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    def test_stable_diffusion_xl_batch_sizes(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionXLPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # Test batch_size > 1 where batch_size is a divider of the total number of generated images
        batch_size = 3
        num_images_per_prompt = batch_size**2
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
        ).images
        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Same test for several prompts
        num_prompts = 3
        images = sd_pipe(
            [prompt] * num_prompts,
            num_inference_steps=2,
            output_type="np",
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Test batch_size when it is not a divider of the total number of generated images for a single prompt
        num_images_per_prompt = 7
        images = sd_pipe(
            prompt,
            num_inference_steps=2,
            output_type="np",
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Same test for several prompts
        num_prompts = 2
        images = sd_pipe(
            [prompt] * num_prompts,
            num_inference_steps=2,
            output_type="np",
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    def test_stable_diffusion_xl_bf16(self):
        """Test that stable diffusion works with bf16"""
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionXLPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device="cpu").manual_seed(0)
        image = sd_pipe([prompt], generator=generator, num_inference_steps=2, output_type="np").images[0]

        self.assertEqual(image.shape, (64, 64, 3))

    def test_stable_diffusion_xl_default(self):
        components = self.get_dummy_components()

        sd_pipe = GaudiStableDiffusionXLPipeline(
            use_habana=True,
            gaudi_config="Habana/stable-diffusion",
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device="cpu").manual_seed(0)
        images = sd_pipe(
            [prompt] * 2,
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            batch_size=3,
            num_images_per_prompt=5,
        ).images

        self.assertEqual(len(images), 10)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    def test_stable_diffusion_xl_hpu_graphs(self):
        components = self.get_dummy_components()

        sd_pipe = GaudiStableDiffusionXLPipeline(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device="cpu").manual_seed(0)
        images = sd_pipe(
            [prompt] * 2,
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            batch_size=3,
            num_images_per_prompt=5,
        ).images

        self.assertEqual(len(images), 10)
        self.assertEqual(images[-1].shape, (64, 64, 3))


class GaudiStableDiffusionControlNetPipelineTester(TestCase):
    """
    Tests the StableDiffusionControlNetPipeline for Gaudi.
    """

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            sample_size=32,
            time_cond_proj_dim=time_cond_proj_dim,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            norm_num_groups=1,
        )

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal(m.weight)
                m.bias.data.fill_(1.0)

        torch.manual_seed(0)
        controlnet = ControlNetModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
            norm_num_groups=1,
        )
        controlnet.controlnet_down_blocks.apply(init_weights)

        scheduler = GaudiDDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[4, 8],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        generator = torch.Generator(device=device).manual_seed(seed)
        controlnet_embedder_scale_factor = 2
        images = [
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
                device=torch.device(device),
            ),
        ]
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "image": images,
        }
        return inputs

    def test_stable_diffusion_controlnet_num_images_per_prompt(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionControlNetPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device="cpu")
        prompt = inputs["prompt"]
        # Test num_images_per_prompt=1 (default)
        images = sd_pipe(**inputs).images

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (64, 64, 3))

        # Test num_images_per_prompt=1 (default) for several prompts
        num_prompts = 3
        inputs["prompt"] = [prompt] * num_prompts
        images = sd_pipe(**inputs).images

        self.assertEqual(len(images), num_prompts)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        inputs["prompt"] = prompt
        images = sd_pipe(num_images_per_prompt=num_images_per_prompt, **inputs).images

        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        ## Test num_images_per_prompt for several prompts
        num_prompts = 2
        inputs["prompt"] = [prompt] * num_prompts
        images = sd_pipe(num_images_per_prompt=num_images_per_prompt, **inputs).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    def test_stable_diffusion_controlnet_batch_sizes(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionControlNetPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device="cpu")
        prompt = inputs["prompt"]
        # Test batch_size > 1 where batch_size is a divider of the total number of generated images
        batch_size = 3
        num_images_per_prompt = batch_size**2
        images = sd_pipe(
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            **inputs,
        ).images
        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Same test for several prompts
        num_prompts = 3
        inputs["prompt"] = [prompt] * num_prompts

        images = sd_pipe(
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            **inputs,
        ).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        inputs["prompt"] = prompt
        # Test batch_size when it is not a divider of the total number of generated images for a single prompt
        num_images_per_prompt = 7
        images = sd_pipe(
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            **inputs,
        ).images

        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Same test for several prompts
        num_prompts = 2
        inputs["prompt"] = [prompt] * num_prompts
        images = sd_pipe(batch_size=batch_size, num_images_per_prompt=num_images_per_prompt, **inputs).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    def test_stable_diffusion_controlnet_bf16(self):
        """Test that stable diffusion works with bf16"""
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionControlNetPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device="cpu")
        image = sd_pipe(**inputs).images[0]

        self.assertEqual(image.shape, (64, 64, 3))

    def test_stable_diffusion_controlnet_default(self):
        components = self.get_dummy_components()

        sd_pipe = GaudiStableDiffusionControlNetPipeline(
            use_habana=True,
            gaudi_config="Habana/stable-diffusion",
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device="cpu")
        inputs["prompt"] = [inputs["prompt"]] * 2
        images = sd_pipe(
            batch_size=3,
            num_images_per_prompt=5,
            **inputs,
        ).images

        self.assertEqual(len(images), 10)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    def test_stable_diffusion_controlnet_hpu_graphs(self):
        components = self.get_dummy_components()

        sd_pipe = GaudiStableDiffusionControlNetPipeline(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device="cpu")
        inputs["prompt"] = [inputs["prompt"]] * 2

        images = sd_pipe(
            batch_size=3,
            num_images_per_prompt=5,
            **inputs,
        ).images

        self.assertEqual(len(images), 10)
        self.assertEqual(images[-1].shape, (64, 64, 3))


class GaudiStableDiffusionMultiControlNetPipelineTester(TestCase):
    """
    Tests the StableDiffusionControlNetPipeline for Gaudi.
    """

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            sample_size=32,
            time_cond_proj_dim=time_cond_proj_dim,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            norm_num_groups=1,
        )

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal(m.weight)
                m.bias.data.fill_(1.0)

        torch.manual_seed(0)
        controlnet1 = ControlNetModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
            norm_num_groups=1,
        )
        controlnet1.controlnet_down_blocks.apply(init_weights)

        torch.manual_seed(0)
        controlnet2 = ControlNetModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
            norm_num_groups=1,
        )
        controlnet2.controlnet_down_blocks.apply(init_weights)

        scheduler = GaudiDDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[4, 8],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        controlnet = MultiControlNetModel([controlnet1, controlnet2])

        components = {
            "unet": unet,
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        generator = torch.Generator(device=device).manual_seed(seed)
        controlnet_embedder_scale_factor = 2
        images = [
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
                device=torch.device(device),
            ),
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
                device=torch.device(device),
            ),
        ]
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "image": images,
        }
        return inputs

    def test_stable_diffusion_multicontrolnet_num_images_per_prompt(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionControlNetPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device="cpu")
        prompt = inputs["prompt"]
        # Test num_images_per_prompt=1 (default)
        images = sd_pipe(**inputs).images

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (64, 64, 3))

        # Test num_images_per_prompt=1 (default) for several prompts
        num_prompts = 3
        inputs["prompt"] = [prompt] * num_prompts
        images = sd_pipe(**inputs).images

        self.assertEqual(len(images), num_prompts)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        inputs["prompt"] = prompt
        images = sd_pipe(num_images_per_prompt=num_images_per_prompt, **inputs).images

        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        ## Test num_images_per_prompt for several prompts
        num_prompts = 2
        inputs["prompt"] = [prompt] * num_prompts
        images = sd_pipe(num_images_per_prompt=num_images_per_prompt, **inputs).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    def test_stable_diffusion_multicontrolnet_batch_sizes(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionControlNetPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device="cpu")
        prompt = inputs["prompt"]
        # Test batch_size > 1 where batch_size is a divider of the total number of generated images
        batch_size = 3
        num_images_per_prompt = batch_size**2
        images = sd_pipe(
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            **inputs,
        ).images
        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Same test for several prompts
        num_prompts = 3
        inputs["prompt"] = [prompt] * num_prompts

        images = sd_pipe(
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            **inputs,
        ).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        inputs["prompt"] = prompt
        # Test batch_size when it is not a divider of the total number of generated images for a single prompt
        num_images_per_prompt = 7
        images = sd_pipe(
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            **inputs,
        ).images

        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

        # Same test for several prompts
        num_prompts = 2
        inputs["prompt"] = [prompt] * num_prompts
        images = sd_pipe(batch_size=batch_size, num_images_per_prompt=num_images_per_prompt, **inputs).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    def test_stable_diffusion_multicontrolnet_bf16(self):
        """Test that stable diffusion works with bf16"""
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionControlNetPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device="cpu")
        image = sd_pipe(**inputs).images[0]

        self.assertEqual(image.shape, (64, 64, 3))

    def test_stable_diffusion_multicontrolnet_default(self):
        components = self.get_dummy_components()

        sd_pipe = GaudiStableDiffusionControlNetPipeline(
            use_habana=True,
            gaudi_config="Habana/stable-diffusion",
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device="cpu")
        inputs["prompt"] = [inputs["prompt"]] * 2
        images = sd_pipe(
            batch_size=3,
            num_images_per_prompt=5,
            **inputs,
        ).images

        self.assertEqual(len(images), 10)
        self.assertEqual(images[-1].shape, (64, 64, 3))

    def test_stable_diffusion_multicontrolnet_hpu_graphs(self):
        components = self.get_dummy_components()

        sd_pipe = GaudiStableDiffusionControlNetPipeline(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            **components,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device="cpu")
        inputs["prompt"] = [inputs["prompt"]] * 2

        images = sd_pipe(
            batch_size=3,
            num_images_per_prompt=5,
            **inputs,
        ).images

        self.assertEqual(len(images), 10)
        self.assertEqual(images[-1].shape, (64, 64, 3))


class TrainTextToImage(TestCase):
    """
    Tests the Stable Diffusion text_to_image Training for Gaudi.
    """

    def test_train_text_to_image_script(self):
        path_to_script = (
            Path(os.path.dirname(__file__)).parent
            / "examples"
            / "stable-diffusion"
            / "training"
            / "train_text_to_image_sdxl.py"
        )

        cmd_line = f"""ls {path_to_script}""".split()

        # check find existence
        p = subprocess.Popen(cmd_line)
        return_code = p.wait()

        # Ensure the run finished without any issue
        self.assertEqual(return_code, 0)

    @slow
    def test_train_text_to_image_sdxl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_to_script = (
                Path(os.path.dirname(__file__)).parent
                / "examples"
                / "stable-diffusion"
                / "training"
                / "train_text_to_image_sdxl.py"
            )

            cmd_line = f"""
                 python3
                 {path_to_script}
                 --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0
                 --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix
                 --dataset_name lambdalabs/naruto-blip-captions
                 --resolution 64
                 --crop_resolution 64
                 --random_flip
                 --proportion_empty_prompts=0.2
                 --train_batch_size 16
                 --learning_rate 1e-05
                 --max_grad_norm 1
                 --lr_scheduler constant
                 --lr_warmup_steps 0
                 --gaudi_config_name Habana/stable-diffusion
                 --throughput_warmup_steps 3
                 --dataloader_num_workers 8
                 --use_hpu_graphs_for_training
                 --use_hpu_graphs_for_inference
                 --bf16
                 --adjust_throughput
                 --center_crop
                 --max_train_steps 2
                 --checkpointing_steps 2
                 --output_dir {tmpdir}
                """.split()

            # Run train_text_to_image_sdxl.y
            p = subprocess.Popen(cmd_line)
            return_code = p.wait()

            # Ensure the run finished without any issue
            self.assertEqual(return_code, 0)

            # save_pretrained smoke test
            self.assertTrue(
                os.path.isfile(os.path.join(tmpdir, "checkpoint-2", "unet", "diffusion_pytorch_model.safetensors"))
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "checkpoint-2", "unet", "config.json")))


class TrainControlNet(TestCase):
    """
    Tests the train_controlnet.py script for Gaudi.
    """

    def test_train_controlnet_script(self):
        path_to_script = (
            Path(os.path.dirname(__file__)).parent
            / "examples"
            / "stable-diffusion"
            / "training"
            / "train_controlnet.py"
        )

        cmd_line = f"""ls {path_to_script}""".split()

        # check find existence
        p = subprocess.Popen(cmd_line)
        return_code = p.wait()

        # Ensure the run finished without any issue
        self.assertEqual(return_code, 0)

    @slow
    def test_train_controlnet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_to_script = (
                Path(os.path.dirname(__file__)).parent
                / "examples"
                / "stable-diffusion"
                / "training"
                / "train_controlnet.py"
            )

            download_files(
                [
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png",
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png",
                ],
                path=tmpdir,
            )

            cmd_line = f"""
                    python3
                    {path_to_script.parent.parent.parent / 'gaudi_spawn.py'}
                    --use_mpi
                    --world_size 8
                    {path_to_script}
                    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5
                    --dataset_name fusing/fill50k
                    --resolution 512
                    --train_batch_size 4
                    --learning_rate 1e-05
                    --validation_steps 1000
                    --validation_image "{tmpdir}/conditioning_image_1.png" "{tmpdir}/conditioning_image_2.png"
                    --validation_prompt "red circle with blue background" "cyan circle with brown floral background"
                    --checkpointing_steps 1000
                    --throughput_warmup_steps 3
                    --use_hpu_graphs
                    --bf16
                    --num_train_epochs 1
                    --output_dir {tmpdir}
                """.split()

            # Run train_controlnet.y
            p = subprocess.Popen(cmd_line)
            return_code = p.wait()

            # Ensure the run finished without any issue
            self.assertEqual(return_code, 0)

            # Assess throughput
            with open(Path(tmpdir) / "speed_metrics.json") as fp:
                results = json.load(fp)
            self.assertGreaterEqual(results["train_samples_per_second"], 0.95 * CONTROLNET_THROUGHPUT)
            self.assertLessEqual(results["train_runtime"], 1.05 * CONTROLNET_RUNTIME)

            # Assess generated image
            controlnet = ControlNetModel.from_pretrained(tmpdir, torch_dtype=torch.bfloat16)
            pipe = GaudiStableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.bfloat16,
                use_habana=True,
                use_hpu_graphs=True,
                gaudi_config=GaudiConfig(use_habana_mixed_precision=False),
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

            control_image = load_image(f"{tmpdir}/conditioning_image_1.png")
            prompt = "pale golden rod circle with old lace background"

            generator = set_seed(27)
            image = pipe(
                prompt, num_inference_steps=20, generator=generator, image=control_image, output_type="np"
            ).images[0]

            self.assertEqual(image.shape, (512, 512, 3))


def install_requirements(requirements_filename: Union[str, os.PathLike]):
    """
    Installs the necessary requirements to run the example if the provided file exists, otherwise does nothing.
    """

    if not Path(requirements_filename).exists():
        return

    cmd_line = f"pip install -r {requirements_filename}".split()
    p = subprocess.Popen(cmd_line)
    return_code = p.wait()
    assert return_code == 0


class DreamBooth(TestCase):
    def _test_dreambooth(self, extra_config, train_text_encoder=False):
        path_to_script = (
            Path(os.path.dirname(__file__)).parent
            / "examples"
            / "stable-diffusion"
            / "training"
            / "train_dreambooth.py"
        )
        install_requirements(path_to_script.parent / "requirements.txt")
        instance_prompt = "soccer player kicking a ball"
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                python3
                {path_to_script}
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --instance_data_dir {Path(os.path.dirname(__file__))/'resource/img'}
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --train_text_encoder
                --max_train_steps 1
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --gaudi_config_name Habana/stable-diffusion
                --output_dir {tmpdir}
                """.split()

            test_args.append("--instance_prompt")
            test_args.append(instance_prompt)
            if "oft" not in extra_config:
                test_args.append("--use_hpu_graphs_for_training")
                test_args.append("--use_hpu_graphs_for_inference")
            if train_text_encoder:
                test_args.append("--train_text_encoder")
            test_args.append(extra_config)
            p = subprocess.Popen(test_args)
            return_code = p.wait()

            # Ensure the run finished without any issue
            self.assertEqual(return_code, 0)
            # save_pretrained smoke test
            if "full" in extra_config:
                self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.safetensors")))
                if train_text_encoder:
                    self.assertTrue(os.path.isfile(os.path.join(tmpdir, "text_encoder", "model.safetensors")))
                self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))
            else:
                self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "adapter_model.safetensors")))
                if train_text_encoder:
                    self.assertTrue(os.path.isfile(os.path.join(tmpdir, "text_encoder", "adapter_model.safetensors")))

    def test_dreambooth_full(self):
        self._test_dreambooth("full")

    def test_dreambooth_full_with_text_encoder(self):
        self._test_dreambooth("full", train_text_encoder=True)

    def test_dreambooth_lora(self):
        self._test_dreambooth("lora")

    def test_dreambooth_lora_with_text_encoder(self):
        self._test_dreambooth("lora", train_text_encoder=True)

    def test_dreambooth_lokr(self):
        self._test_dreambooth("lokr")

    def test_dreambooth_lokr_with_text_encoder(self):
        self._test_dreambooth("lokr", train_text_encoder=True)

    def test_dreambooth_loha(self):
        self._test_dreambooth("loha")

    def test_dreambooth_loha_with_text_encoder(self):
        self._test_dreambooth("loha", train_text_encoder=True)

    def test_dreambooth_oft(self):
        self._test_dreambooth("oft")

    def test_dreambooth_oft_with_text_encoder(self):
        self._test_dreambooth("oft", train_text_encoder=True)


class DreamBoothLoRASDXL(TestCase):
    def _test_dreambooth_lora_sdxl(self, train_text_encoder=False):
        path_to_script = (
            Path(os.path.dirname(__file__)).parent
            / "examples"
            / "stable-diffusion"
            / "training"
            / "train_dreambooth_lora_sdxl.py"
        )
        install_requirements(path_to_script.parent / "requirements.txt")

        instance_prompt = "soccer player kicking a ball"
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                python3
                {path_to_script}
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-xl-pipe
                --instance_data_dir {Path(os.path.dirname(__file__))/'resource/img'}
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 1
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --gaudi_config_name Habana/stable-diffusion
                --use_hpu_graphs_for_training
                --use_hpu_graphs_for_inference
                --output_dir {tmpdir}
                """.split()
            if train_text_encoder:
                test_args.append("--train_text_encoder")
            test_args.append("--instance_prompt")
            test_args.append(instance_prompt)
            p = subprocess.Popen(test_args)
            return_code = p.wait()

            # Ensure the run finished without any issue
            self.assertEqual(return_code, 0)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            # make sure the state_dict has the correct naming in the parameters.
            lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            is_lora = all("lora" in k for k in lora_state_dict.keys())
            self.assertTrue(is_lora)

            # when not training the text encoder, all the parameters in the state dict should start
            # with `"unet"` in their names.
            if train_text_encoder:
                starts_with_unet = all(
                    k.startswith("unet") or k.startswith("text_encoder") or k.startswith("text_encoder_2")
                    for k in lora_state_dict.keys()
                )
            else:
                starts_with_unet = all(key.startswith("unet") for key in lora_state_dict.keys())
            self.assertTrue(starts_with_unet)

    def test_dreambooth_lora_sdxl_with_text_encoder(self):
        self._test_dreambooth_lora_sdxl(train_text_encoder=True)

    def test_dreambooth_lora_sdxl(self):
        self._test_dreambooth_lora_sdxl(train_text_encoder=False)


class GaudiStableVideoDiffusionPipelineTester(TestCase):
    """
    Tests the StableVideoDiffusionPipeline for Gaudi.
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.24.0-release/tests/pipelines/stable_video_diffusion/test_stable_video_diffusion.py
    """

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNetSpatioTemporalConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=8,
            out_channels=4,
            down_block_types=(
                "CrossAttnDownBlockSpatioTemporal",
                "DownBlockSpatioTemporal",
            ),
            up_block_types=("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal"),
            cross_attention_dim=32,
            num_attention_heads=8,
            projection_class_embeddings_input_dim=96,
            addition_time_embed_dim=32,
        )
        scheduler = GaudiEulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            interpolation_type="linear",
            num_train_timesteps=1000,
            prediction_type="v_prediction",
            sigma_max=700.0,
            sigma_min=0.002,
            steps_offset=1,
            timestep_spacing="leading",
            timestep_type="continuous",
            trained_betas=None,
            use_karras_sigmas=True,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLTemporalDecoder(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            latent_channels=4,
        )

        torch.manual_seed(0)
        config = CLIPVisionConfig(
            hidden_size=32,
            projection_dim=32,
            num_hidden_layers=5,
            num_attention_heads=4,
            image_size=32,
            intermediate_size=37,
            patch_size=1,
        )
        image_encoder = CLIPVisionModelWithProjection(config)

        torch.manual_seed(0)
        feature_extractor = CLIPImageProcessor(crop_size=32, size=32)
        components = {
            "unet": unet,
            "image_encoder": image_encoder,
            "scheduler": scheduler,
            "vae": vae,
            "feature_extractor": feature_extractor,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        image = floats_tensor((1, 3, 32, 32), rng=random.Random(0)).to(device)
        inputs = {
            "generator": generator,
            "image": image,
            "num_inference_steps": 2,
            "output_type": "pt",
            "min_guidance_scale": 1.0,
            "max_guidance_scale": 2.5,
            "num_frames": 2,
            "height": 32,
            "width": 32,
        }
        return inputs

    def test_stable_video_diffusion_single_video(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig(use_torch_autocast=False)
        sd_pipe = GaudiStableVideoDiffusionPipeline(use_habana=True, gaudi_config=gaudi_config, **components)
        for component in sd_pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        outputs = sd_pipe(
            **self.get_dummy_inputs(device),
        ).frames
        image = outputs[0]
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(len(outputs), 1)
        self.assertEqual(image.shape, (2, 3, 32, 32))

        expected_slice = np.array([0.5910, 0.5797, 0.5521, 0.6628, 0.6212, 0.6422, 0.5681, 0.5232, 0.5343])

        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-2)

    @slow
    def test_stable_video_diffusion_no_throughput_regression_bf16(self):
        image_url = (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"
        )
        model_name = "stabilityai/stable-video-diffusion-img2vid-xt"
        scheduler = GaudiEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")

        pipeline = GaudiStableVideoDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig.from_pretrained("Habana/stable-diffusion"),
            torch_dtype=torch.bfloat16,
        )
        set_seed(42)
        prompt_image = load_image(image_url)
        outputs = pipeline(
            image=prompt_image,
            num_videos_per_prompt=1,
            batch_size=1,
            height=256,
            width=256,
        )

        self.assertEqual(len(outputs.frames[0]), 25)
        if IS_GAUDI2:
            self.assertGreaterEqual(outputs.throughput, 0.95 * 0.012)


"""
Copied from: https://github.com/huggingface/diffusers/blob/v0.26.3/tests/pipelines/pipeline_params.py
"""

TEXT_TO_IMAGE_PARAMS = frozenset(
    [
        "prompt",
        "height",
        "width",
        "guidance_scale",
        "negative_prompt",
        "prompt_embeds",
        "negative_prompt_embeds",
        "cross_attention_kwargs",
    ]
)

TEXT_TO_IMAGE_BATCH_PARAMS = frozenset(["prompt", "negative_prompt"])

TEXT_TO_IMAGE_IMAGE_PARAMS = frozenset([])

IMAGE_TO_IMAGE_IMAGE_PARAMS = frozenset(["image"])

IMAGE_VARIATION_PARAMS = frozenset(
    [
        "image",
        "height",
        "width",
        "guidance_scale",
    ]
)

IMAGE_VARIATION_BATCH_PARAMS = frozenset(["image"])

TEXT_GUIDED_IMAGE_VARIATION_PARAMS = frozenset(
    [
        "prompt",
        "image",
        "height",
        "width",
        "guidance_scale",
        "negative_prompt",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]
)

TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS = frozenset(["prompt", "image", "negative_prompt"])

TEXT_GUIDED_IMAGE_INPAINTING_PARAMS = frozenset(
    [
        # Text guided image variation with an image mask
        "prompt",
        "image",
        "mask_image",
        "height",
        "width",
        "guidance_scale",
        "negative_prompt",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]
)

TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS = frozenset(["prompt", "image", "mask_image", "negative_prompt"])

IMAGE_INPAINTING_PARAMS = frozenset(
    [
        # image variation with an image mask
        "image",
        "mask_image",
        "height",
        "width",
        "guidance_scale",
    ]
)

IMAGE_INPAINTING_BATCH_PARAMS = frozenset(["image", "mask_image"])

IMAGE_GUIDED_IMAGE_INPAINTING_PARAMS = frozenset(
    [
        "example_image",
        "image",
        "mask_image",
        "height",
        "width",
        "guidance_scale",
    ]
)

IMAGE_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS = frozenset(["example_image", "image", "mask_image"])

CLASS_CONDITIONED_IMAGE_GENERATION_PARAMS = frozenset(["class_labels"])

CLASS_CONDITIONED_IMAGE_GENERATION_BATCH_PARAMS = frozenset(["class_labels"])

UNCONDITIONAL_IMAGE_GENERATION_PARAMS = frozenset(["batch_size"])

UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS = frozenset([])

UNCONDITIONAL_AUDIO_GENERATION_PARAMS = frozenset(["batch_size"])

UNCONDITIONAL_AUDIO_GENERATION_BATCH_PARAMS = frozenset([])

TEXT_TO_AUDIO_PARAMS = frozenset(
    [
        "prompt",
        "audio_length_in_s",
        "guidance_scale",
        "negative_prompt",
        "prompt_embeds",
        "negative_prompt_embeds",
        "cross_attention_kwargs",
    ]
)

TEXT_TO_AUDIO_BATCH_PARAMS = frozenset(["prompt", "negative_prompt"])
TOKENS_TO_AUDIO_GENERATION_PARAMS = frozenset(["input_tokens"])

TOKENS_TO_AUDIO_GENERATION_BATCH_PARAMS = frozenset(["input_tokens"])

TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS = frozenset(["prompt_embeds"])

VIDEO_TO_VIDEO_BATCH_PARAMS = frozenset(["prompt", "negative_prompt", "video"])


"""
Copied from: https://github.com/huggingface/diffusers/blob/v0.26.3/tests/pipelines/stable_diffusion_2/test_stable_diffusion_inpaint.py
- Modified pipeline to Gaudi pipeline.
- Modified the get_dummy_components to add the Gaudi pipeline parameters: use_habana, use_hpu_graphs, gaudi_config, bf16_full_eval
- Added testcases:
    test_stable_diffusion_inpaint_no_safety_checker
    test_stable_diffusion_inpaint_enable_safety_checker
    test_stable_diffusion_inpaint_no_throughput_regression
"""

enable_full_determinism()

class StableDiffusionInpaintPipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, TestCase
):
    pipeline_class = GaudiStableDiffusionInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset(
        []
    )  # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess
    image_latents_params = frozenset([])
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"mask", "masked_image_latents"})

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=9,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
        )
        scheduler = PNDMScheduler(skip_prk_steps=True)
        set_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        set_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=512,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
            "use_habana": True,
            "use_hpu_graphs": True,
            "gaudi_config": "Habana/stable-diffusion-2",
            "bf16_full_eval": True,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        # TODO: use tensor inputs instead of PIL, this is here just to leave the old expected_slices untouched
        # ensure determinism for the device-dependent torch.Generator on HPU
        # Device type HPU is not supported for torch.Generator() api
        device = "cpu"
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((64, 64))
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_inpaint(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = GaudiStableDiffusionInpaintPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4727, 0.5735, 0.3941, 0.5446, 0.5926, 0.4394, 0.5062, 0.4654, 0.4476])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)


class StableDiffusionInpaintPipelineIntegrationTests(TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()

    def create_inpaint_pipe(
        self,
        model_name="stabilityai/stable-diffusion-2-inpainting",
        scheduler=None,
        use_hpu_graphs=False,
        gaudi_config="Habana/stable-diffusion",
        disable_safety_checker=False,
        torch_dtype=torch.bfloat16,
    ):
        from optimum.habana.diffusers import GaudiDDIMScheduler

        if scheduler is None:
            scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

        kwargs = {
            "scheduler": scheduler,
            "use_habana": True,
            "use_hpu_graphs": use_hpu_graphs,
            "gaudi_config": gaudi_config,
        }

        if disable_safety_checker is True:
            kwargs["safety_checker"] = None

        sdi_pipe = GaudiStableDiffusionInpaintPipeline.from_pretrained(model_name, **kwargs).to(torch_dtype)

        sdi_pipe.set_progress_bar_config(disable=None)

        return sdi_pipe

    def test_stable_diffusion_inpaint_pipeline(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/sd2-inpaint/init_image.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-inpaint/mask.png"
        )
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-inpaint"
            "/yellow_cat_sitting_on_a_park_bench.npy"
        )

        model_id = "stabilityai/stable-diffusion-2-inpainting"
        init_kwargs = {
            "use_habana": True,
            "use_hpu_graphs": True,
            "gaudi_config": "Habana/stable-diffusion",
            "torch_dtype": torch.float,
        }

        pipe = GaudiStableDiffusionInpaintPipeline.from_pretrained(model_id, safety_checker=None, **init_kwargs)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

        generator = torch.manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        # There is no difference in the experimental results observed by the human eye.
        # np.abs(expected_image - image).max() = 0.31966144
        assert np.abs(expected_image - image).max() < 0.4

    def test_stable_diffusion_inpaint_pipeline_bf16(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/sd2-inpaint/init_image.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-inpaint/mask.png"
        )
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-inpaint"
            "/yellow_cat_sitting_on_a_park_bench_fp16.npy"
        )

        model_id = "stabilityai/stable-diffusion-2-inpainting"
        init_kwargs = {
            "use_habana": True,
            "use_hpu_graphs": True,
            "gaudi_config": "Habana/stable-diffusion-2",
            "torch_dtype": torch.bfloat16,
        }

        pipe = GaudiStableDiffusionInpaintPipeline.from_pretrained(model_id, safety_checker=None, **init_kwargs)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

        generator = torch.manual_seed(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            generator=generator,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)
        # The format of expected_image used for testing is only float16. There is no difference in the experimental results observed by the human eye.
        # np.abs(expected_image - image).max() = 0.9626465
        assert np.abs(expected_image - image).max() < 0.97

    @slow
    def test_stable_diffusion_inpaint_no_safety_checker(self):
        """Test that stable diffusion inpainting works without a saftey checker"""
        from diffusers.utils import load_image

        # Create test inpaint pipeline
        gaudi_config = GaudiConfig()
        scheduler = GaudiDDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        sdi_pipe = self.create_inpaint_pipe(
            gaudi_config=gaudi_config, scheduler=scheduler, disable_safety_checker=True
        )

        # Initialize inpaint parameters
        init_image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
        )

        self.assertIsInstance(sdi_pipe, GaudiStableDiffusionInpaintPipeline)
        self.assertIsInstance(sdi_pipe.scheduler, GaudiDDIMScheduler)
        self.assertIsNone(sdi_pipe.safety_checker)

        image = sdi_pipe("example prompt", image=init_image, mask_image=mask_image, num_inference_steps=2).images[0]
        self.assertIsNotNone(image)

        # Check that there's no error when saving a pipeline with one of the models being None
        with tempfile.TemporaryDirectory() as tmpdirname:
            sdi_pipe.save_pretrained(tmpdirname)
            sdi_pipe = GaudiStableDiffusionInpaintPipeline.from_pretrained(
                tmpdirname,
                use_habana=True,
                gaudi_config=tmpdirname,
            )

        # Sanity check that the pipeline still works
        self.assertIsNone(sdi_pipe.safety_checker)
        image = sdi_pipe("example prompt", image=init_image, mask_image=mask_image, num_inference_steps=2).images[0]
        self.assertIsNotNone(image)

    @slow
    def test_stable_diffusion_inpaint_enable_safety_checker(self):
        """Test that stable diffusion inpainting works with a saftey checker and it is loaded from_pretrained"""
        from diffusers.utils import load_image

        # Create test inpaint pipeline
        gaudi_config = GaudiConfig()
        scheduler = GaudiDDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        sdi_pipe = self.create_inpaint_pipe(
            gaudi_config=gaudi_config, scheduler=scheduler, disable_safety_checker=False
        )

        # Initialize inpaint parameters
        init_image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
        )

        self.assertIsInstance(sdi_pipe, GaudiStableDiffusionInpaintPipeline)
        self.assertIsInstance(sdi_pipe.scheduler, GaudiDDIMScheduler)
        # self.assertIsNotNone(sdi_pipe.safety_checker) <--- The safety checker is not being found.

        image = sdi_pipe("example prompt", image=init_image, mask_image=mask_image, num_inference_steps=2).images[0]
        self.assertIsNotNone(image)

        # Check that there's no error when saving a pipeline with one of the models being None
        with tempfile.TemporaryDirectory() as tmpdirname:
            sdi_pipe.save_pretrained(tmpdirname)
            sdi_pipe = GaudiStableDiffusionInpaintPipeline.from_pretrained(
                tmpdirname,
                use_habana=True,
                gaudi_config=tmpdirname,
            )

        # Sanity check that the pipeline still works
        self.assertIsNone(sdi_pipe.safety_checker)
        image = sdi_pipe("example prompt", image=init_image, mask_image=mask_image, num_inference_steps=2).images[0]
        self.assertIsNotNone(image)

    @slow
    def test_stable_diffusion_inpaint_no_throughput_regression(self):
        """Test that stable diffusion inpainting no throughput regression autocast"""
        from diffusers.utils import load_image

        # Initialize inpaint parameters
        init_image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
        )

        prompts = [
            "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k",
            "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k",
        ]
        num_images_per_prompt = 10
        num_inference_steps = 10
        model_name = "runwayml/stable-diffusion-inpainting"

        init_kwargs = {
            "use_habana": True,
            "use_hpu_graphs": True,
            "gaudi_config": "Habana/stable-diffusion",
            "torch_dtype": torch.bfloat16,
        }
        sdi_pipe = GaudiStableDiffusionInpaintPipeline.from_pretrained(model_name, **init_kwargs)

        set_seed(0)
        outputs = sdi_pipe(
            prompt=prompts,
            image=init_image,
            mask_image=mask_image,
            num_images_per_prompt=num_images_per_prompt,
            throughput_warmup_steps=3,
            num_inference_steps=num_inference_steps,
            batch_size=4,
        )

        self.assertEqual(len(outputs.images), num_images_per_prompt * len(prompts))
        self.assertGreaterEqual(outputs.throughput, 0.95 * INPAINT_THROUGHPUT_BASELINE_BF16)


"""
Copied from: https://github.com/huggingface/diffusers/blob/v0.26.3/tests/pipelines/stable_diffusion_xl/test_stable_diffusion_xl_inpaint.py
- Modified pipeline to Gaudi pipeline.
- Modified the get_dummy_components to add the Gaudi pipeline parameters: use_habana, use_hpu_graphs, gaudi_config, bf16_full_eval
- added test_stable_diffusion_xl_inpaint_no_throughput_regression
"""

class StableDiffusionXLInpaintPipelineFastTests(PipelineLatentTesterMixin, PipelineTesterMixin, TestCase):
    pipeline_class = GaudiStableDiffusionXLInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset([])
    # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess
    image_latents_params = frozenset([])
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union(
        {
            "add_text_embeds",
            "add_time_ids",
            "mask",
            "masked_image_latents",
        }
    )

    def get_dummy_components(self, skip_first_text_encoder=False, time_cond_proj_dim=None):
        torch.manual_seed(0)
        set_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            time_cond_proj_dim=time_cond_proj_dim,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=72,  # 5 * 8 + 32
            cross_attention_dim=64 if not skip_first_text_encoder else 32,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )
        torch.manual_seed(0)
        set_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(0)
        set_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        torch.manual_seed(0)
        set_seed(0)
        image_encoder_config = CLIPVisionConfig(
            hidden_size=32,
            image_size=224,
            projection_dim=32,
            intermediate_size=37,
            num_attention_heads=4,
            num_channels=3,
            num_hidden_layers=5,
            patch_size=14,
        )

        image_encoder = CLIPVisionModelWithProjection(image_encoder_config)

        feature_extractor = CLIPImageProcessor(
            crop_size=224,
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            resample=3,
            size=224,
        )

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder if not skip_first_text_encoder else None,
            "tokenizer": tokenizer if not skip_first_text_encoder else None,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "image_encoder": image_encoder,
            "feature_extractor": feature_extractor,
            "requires_aesthetics_score": True,
            "use_habana": True,
            "use_hpu_graphs": True,
            "gaudi_config": "Habana/stable-diffusion",
            "bf16_full_eval": True,
        }
        return components

    def get_dummy_inputs(self, device="cpu", seed=0):
        # TODO: use tensor inputs instead of PIL, this is here just to leave the old expected_slices untouched
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        # create mask
        image[8:, 8:, :] = 255
        mask_image = Image.fromarray(np.uint8(image)).convert("L").resize((64, 64))

        # Device type HPU is not supported for torch.Generator() api
        device = "cpu"
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "strength": 1.0,
            "output_type": "np",
        }
        return inputs

    def get_dummy_inputs_2images(self, device, seed=0, img_res=64):
        # Get random floats in [0, 1] as image with spatial size (img_res, img_res)
        image1 = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed)).to(device)
        image2 = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed + 22)).to(device)
        # Convert images to [-1, 1]
        init_image1 = 2.0 * image1 - 1.0
        init_image2 = 2.0 * image2 - 1.0

        # empty mask
        mask_image = torch.zeros((1, 1, img_res, img_res), device=device)

        # Device type HPU is not supported for torch.Generator() api
        device = "cpu"
        if str(device).startswith("mps"):
            generator1 = torch.manual_seed(seed)
            generator2 = torch.manual_seed(seed)
        else:
            generator1 = torch.Generator(device=device).manual_seed(seed)
            generator2 = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "prompt": ["A painting of a squirrel eating a burger"] * 2,
            "image": [init_image1, init_image2],
            "mask_image": [mask_image] * 2,
            "generator": [generator1, generator2],
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "batch_size": 2,
        }
        return inputs

    def test_components_function(self):
        init_components = self.get_dummy_components()
        init_components.pop("requires_aesthetics_score")
        init_components.pop("use_habana")
        init_components.pop("use_hpu_graphs")
        init_components.pop("bf16_full_eval")
        init_components.pop("gaudi_config")
        pipe = self.pipeline_class(**init_components)
        self.assertTrue(hasattr(pipe, "components"))
        self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))

    def test_stable_diffusion_xl_inpaint_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = GaudiStableDiffusionXLInpaintPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.8029, 0.5523, 0.5825, 0.6003, 0.6702, 0.7018, 0.6369, 0.5955, 0.5123])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_xl_inpaint_euler_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = GaudiStableDiffusionXLInpaintPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.6611, 0.5569, 0.5531, 0.5471, 0.5918, 0.6393, 0.5074, 0.5468, 0.5185])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_xl_inpaint_euler_lcm_custom_timesteps(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = GaudiStableDiffusionXLInpaintPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.6611, 0.5569, 0.5531, 0.5471, 0.5918, 0.6393, 0.5074, 0.5468, 0.5185])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_attention_slicing_forward_pass(self):
        super().test_attention_slicing_forward_pass(expected_max_diff=3e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    # TODO(Patrick, Sayak) - skip for now as this requires more refiner tests
    def test_save_load_optional_components(self):
        pass

    def test_stable_diffusion_xl_inpaint_negative_prompt_embeds(self):
        device = "cpu"
        components = self.get_dummy_components()
        sd_pipe = GaudiStableDiffusionXLInpaintPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        # forward without prompt embeds
        inputs = self.get_dummy_inputs(device)
        negative_prompt = 3 * ["this is a negative prompt"]
        inputs["negative_prompt"] = negative_prompt
        inputs["prompt"] = 3 * [inputs["prompt"]]

        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with prompt embeds
        inputs = self.get_dummy_inputs(device)
        negative_prompt = 3 * ["this is a negative prompt"]
        prompt = 3 * [inputs.pop("prompt")]

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = sd_pipe.encode_prompt(prompt, negative_prompt=negative_prompt)

        output = sd_pipe(
            **inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # make sure that it's equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

    def test_stable_diffusion_xl_refiner(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(skip_first_text_encoder=True)

        sd_pipe = self.pipeline_class(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.7045, 0.4838, 0.5454, 0.6270, 0.6168, 0.6717, 0.6484, 0.5681, 0.4922])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_two_xl_mixture_of_denoiser_fast(self):
        device = "cpu"
        components = self.get_dummy_components()
        pipe_1 = GaudiStableDiffusionXLInpaintPipeline(**components)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = GaudiStableDiffusionXLInpaintPipeline(**components)
        pipe_2.unet.set_default_attn_processor()

        def assert_run_mixture(
            num_steps, split, scheduler_cls_orig, num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps
        ):
            inputs = self.get_dummy_inputs(device)
            inputs["num_inference_steps"] = num_steps

            class scheduler_cls(scheduler_cls_orig):
                pass

            pipe_1.scheduler = scheduler_cls.from_config(pipe_1.scheduler.config)
            pipe_2.scheduler = scheduler_cls.from_config(pipe_2.scheduler.config)

            # Let's retrieve the number of timesteps we want to use
            pipe_1.scheduler.set_timesteps(num_steps)
            expected_steps = pipe_1.scheduler.timesteps.tolist()

            split_ts = num_train_timesteps - int(round(num_train_timesteps * split))

            if pipe_1.scheduler.order == 2:
                expected_steps_1 = list(filter(lambda ts: ts >= split_ts, expected_steps))
                expected_steps_2 = expected_steps_1[-1:] + list(filter(lambda ts: ts < split_ts, expected_steps))
                expected_steps = expected_steps_1 + expected_steps_2
            else:
                expected_steps_1 = list(filter(lambda ts: ts >= split_ts, expected_steps))
                expected_steps_2 = list(filter(lambda ts: ts < split_ts, expected_steps))

            # now we monkey patch step `done_steps`
            # list into the step function for testing
            done_steps = []
            old_step = copy.copy(scheduler_cls.step)

            def new_step(self, *args, **kwargs):
                done_steps.append(args[1].cpu().item())  # args[1] is always the passed `t`
                return old_step(self, *args, **kwargs)

            scheduler_cls.step = new_step

            inputs_1 = {**inputs, **{"denoising_end": split, "output_type": "latent"}}
            latents = pipe_1(**inputs_1).images[0]

            assert expected_steps_1 == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"

            inputs_2 = {**inputs, **{"denoising_start": split, "image": latents}}
            pipe_2(**inputs_2).images[0]

            assert expected_steps_2 == done_steps[len(expected_steps_1) :]
            assert expected_steps == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"

        for steps in [7, 20]:
            assert_run_mixture(steps, 0.33, EulerDiscreteScheduler)
            # Currently cannot support the default HeunDiscreteScheduler
            # assert_run_mixture(steps, 0.33, HeunDiscreteScheduler)

    @slow
    def test_stable_diffusion_two_xl_mixture_of_denoiser(self):
        components = self.get_dummy_components()
        pipe_1 = GaudiStableDiffusionXLInpaintPipeline(**components)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = GaudiStableDiffusionXLInpaintPipeline(**components)
        pipe_2.unet.set_default_attn_processor()

        def assert_run_mixture(
            num_steps, split, scheduler_cls_orig, num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps
        ):
            inputs = self.get_dummy_inputs()
            inputs["num_inference_steps"] = num_steps

            class scheduler_cls(scheduler_cls_orig):
                pass

            pipe_1.scheduler = scheduler_cls.from_config(pipe_1.scheduler.config)
            pipe_2.scheduler = scheduler_cls.from_config(pipe_2.scheduler.config)

            # Let's retrieve the number of timesteps we want to use
            pipe_1.scheduler.set_timesteps(num_steps)
            expected_steps = pipe_1.scheduler.timesteps.tolist()

            split_ts = num_train_timesteps - int(round(num_train_timesteps * split))

            if pipe_1.scheduler.order == 2:
                expected_steps_1 = list(filter(lambda ts: ts >= split_ts, expected_steps))
                expected_steps_2 = expected_steps_1[-1:] + list(filter(lambda ts: ts < split_ts, expected_steps))
                expected_steps = expected_steps_1 + expected_steps_2
            else:
                expected_steps_1 = list(filter(lambda ts: ts >= split_ts, expected_steps))
                expected_steps_2 = list(filter(lambda ts: ts < split_ts, expected_steps))

            # now we monkey patch step `done_steps`
            # list into the step function for testing
            done_steps = []
            old_step = copy.copy(scheduler_cls.step)

            def new_step(self, *args, **kwargs):
                done_steps.append(args[1].cpu().item())  # args[1] is always the passed `t`
                return old_step(self, *args, **kwargs)

            scheduler_cls.step = new_step

            inputs_1 = {**inputs, **{"denoising_end": split, "output_type": "latent"}}
            latents = pipe_1(**inputs_1).images[0]

            assert expected_steps_1 == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"

            inputs_2 = {**inputs, **{"denoising_start": split, "image": latents}}
            pipe_2(**inputs_2).images[0]

            assert expected_steps_2 == done_steps[len(expected_steps_1) :]
            assert expected_steps == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"

        for steps in [5, 8, 20]:
            for split in [0.33, 0.49, 0.71]:
                for scheduler_cls in [
                    GaudiDDIMScheduler,
                    GaudiEulerDiscreteScheduler,
                    GaudiEulerAncestralDiscreteScheduler,
                    DPMSolverMultistepScheduler,
                    UniPCMultistepScheduler,
                    # HeunDiscreteScheduler,
                ]:
                    assert_run_mixture(steps, split, scheduler_cls)

    @slow
    def test_stable_diffusion_three_xl_mixture_of_denoiser(self):
        components = self.get_dummy_components()
        pipe_1 = GaudiStableDiffusionXLInpaintPipeline(**components)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = GaudiStableDiffusionXLInpaintPipeline(**components)
        pipe_2.unet.set_default_attn_processor()
        pipe_3 = GaudiStableDiffusionXLInpaintPipeline(**components)
        pipe_3.unet.set_default_attn_processor()

        def assert_run_mixture(
            num_steps,
            split_1,
            split_2,
            scheduler_cls_orig,
            num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps,
        ):
            inputs = self.get_dummy_inputs()
            inputs["num_inference_steps"] = num_steps

            class scheduler_cls(scheduler_cls_orig):
                pass

            pipe_1.scheduler = scheduler_cls.from_config(pipe_1.scheduler.config)
            pipe_2.scheduler = scheduler_cls.from_config(pipe_2.scheduler.config)
            pipe_3.scheduler = scheduler_cls.from_config(pipe_3.scheduler.config)

            # Let's retrieve the number of timesteps we want to use
            pipe_1.scheduler.set_timesteps(num_steps)
            expected_steps = pipe_1.scheduler.timesteps.tolist()

            split_1_ts = num_train_timesteps - int(round(num_train_timesteps * split_1))
            split_2_ts = num_train_timesteps - int(round(num_train_timesteps * split_2))

            if pipe_1.scheduler.order == 2:
                expected_steps_1 = list(filter(lambda ts: ts >= split_1_ts, expected_steps))
                expected_steps_2 = expected_steps_1[-1:] + list(
                    filter(lambda ts: ts >= split_2_ts and ts < split_1_ts, expected_steps)
                )
                expected_steps_3 = expected_steps_2[-1:] + list(filter(lambda ts: ts < split_2_ts, expected_steps))
                expected_steps = expected_steps_1 + expected_steps_2 + expected_steps_3
            else:
                expected_steps_1 = list(filter(lambda ts: ts >= split_1_ts, expected_steps))
                expected_steps_2 = list(filter(lambda ts: ts >= split_2_ts and ts < split_1_ts, expected_steps))
                expected_steps_3 = list(filter(lambda ts: ts < split_2_ts, expected_steps))

            # now we monkey patch step `done_steps`
            # list into the step function for testing
            done_steps = []
            old_step = copy.copy(scheduler_cls.step)

            def new_step(self, *args, **kwargs):
                done_steps.append(args[1].cpu().item())  # args[1] is always the passed `t`
                return old_step(self, *args, **kwargs)

            scheduler_cls.step = new_step

            inputs_1 = {**inputs, **{"denoising_end": split_1, "output_type": "latent"}}
            latents = pipe_1(**inputs_1).images[0]

            assert (
                expected_steps_1 == done_steps
            ), f"Failure with {scheduler_cls.__name__} and {num_steps} and {split_1} and {split_2}"

            inputs_2 = {
                **inputs,
                **{"denoising_start": split_1, "denoising_end": split_2, "image": latents, "output_type": "latent"},
            }
            pipe_2(**inputs_2).images[0]

            assert expected_steps_2 == done_steps[len(expected_steps_1) :]

            inputs_3 = {**inputs, **{"denoising_start": split_2, "image": latents}}
            pipe_3(**inputs_3).images[0]

            assert expected_steps_3 == done_steps[len(expected_steps_1) + len(expected_steps_2) :]
            assert (
                expected_steps == done_steps
            ), f"Failure with {scheduler_cls.__name__} and {num_steps} and {split_1} and {split_2}"

        for steps in [7, 11, 20]:
            for split_1, split_2 in zip([0.19, 0.32], [0.81, 0.68]):
                for scheduler_cls in [
                    GaudiDDIMScheduler,
                    GaudiEulerDiscreteScheduler,
                    GaudiEulerAncestralDiscreteScheduler,
                    DPMSolverMultistepScheduler,
                    UniPCMultistepScheduler,
                    # HeunDiscreteScheduler,
                ]:
                    assert_run_mixture(steps, split_1, split_2, scheduler_cls)

    def test_stable_diffusion_xl_multi_prompts(self):
        device = "cpu"
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        # forward with single prompt
        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 5
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with same prompt duplicated
        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 5
        inputs["prompt_2"] = inputs["prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # ensure the results are equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

        # forward with different prompt
        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 5
        inputs["prompt_2"] = "different prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -3:, -3:, -1]

        # ensure the results are not equal
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

        # manually set a negative_prompt
        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with same negative_prompt duplicated
        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = inputs["negative_prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # ensure the results are equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

        # forward with different negative_prompt
        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = "different negative prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -3:, -3:, -1]

        # ensure the results are not equal
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

    def test_stable_diffusion_xl_img2img_negative_conditions(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice_with_no_neg_conditions = image[0, -3:, -3:, -1]

        image = sd_pipe(
            **inputs,
            negative_original_size=(512, 512),
            negative_crops_coords_top_left=(
                0,
                0,
            ),
            negative_target_size=(1024, 1024),
        ).images
        image_slice_with_neg_conditions = image[0, -3:, -3:, -1]

        assert (
            np.abs(image_slice_with_no_neg_conditions.flatten() - image_slice_with_neg_conditions.flatten()).max()
            > 1e-4
        )

    def test_stable_diffusion_xl_inpaint_mask_latents(self):
        device = "cpu"
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        # normal mask + normal image
        ##  `image`: pil, `mask_image``: pil, `masked_image_latents``: None
        inputs = self.get_dummy_inputs(device)
        inputs["strength"] = 0.9
        out_0 = sd_pipe(**inputs).images

        # image latents + mask latents
        inputs = self.get_dummy_inputs(device)
        image = sd_pipe.image_processor.preprocess(inputs["image"]).to(sd_pipe.device)
        mask = sd_pipe.mask_processor.preprocess(inputs["mask_image"]).to(sd_pipe.device)
        masked_image = image * (mask < 0.5)

        generator = torch.Generator(device=device).manual_seed(0)
        image_latents = sd_pipe._encode_vae_image(image, generator=generator)
        torch.randn((1, 4, 32, 32), generator=generator)
        mask_latents = sd_pipe._encode_vae_image(masked_image, generator=generator)
        inputs["image"] = image_latents
        inputs["masked_image_latents"] = mask_latents
        inputs["mask_image"] = mask
        inputs["strength"] = 0.9
        generator = torch.Generator(device=device).manual_seed(0)
        torch.randn((1, 4, 32, 32), generator=generator)
        inputs["generator"] = generator
        out_1 = sd_pipe(**inputs).images
        assert np.abs(out_0 - out_1).max() < 1e-2

    def test_stable_diffusion_xl_inpaint_2_images(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        # test to confirm if we pass two same image, we will get same output
        inputs = self.get_dummy_inputs(device)
        gen1 = torch.Generator(device=device).manual_seed(0)
        gen2 = torch.Generator(device=device).manual_seed(0)
        for name in ["prompt", "image", "mask_image"]:
            inputs[name] = [inputs[name]] * 2
        inputs["generator"] = [gen1, gen2]
        images = sd_pipe(**inputs).images

        assert images.shape == (2, 64, 64, 3)

        image_slice1 = images[0, -3:, -3:, -1]
        image_slice2 = images[1, -3:, -3:, -1]
        assert np.abs(image_slice1.flatten() - image_slice2.flatten()).max() < 1e-4

        # test to confirm that if we pass two different images, we will get different output
        inputs = self.get_dummy_inputs_2images(device)
        images = sd_pipe(**inputs).images
        assert images.shape == (2, 64, 64, 3)

        image_slice1 = images[0, -3:, -3:, -1]
        image_slice2 = images[1, -3:, -3:, -1]
        assert np.abs(image_slice1.flatten() - image_slice2.flatten()).max() > 1e-2

    def test_pipeline_interrupt(self):
        components = self.get_dummy_components()
        sd_pipe = GaudiStableDiffusionXLInpaintPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()

        prompt = "hey"
        num_inference_steps = 5

        # store intermediate latents from the generation process
        class PipelineState:
            def __init__(self):
                self.state = []

            def apply(self, pipe, i, t, callback_kwargs):
                self.state.append(callback_kwargs["latents"])
                return callback_kwargs

        pipe_state = PipelineState()
        sd_pipe(
            prompt,
            image=inputs["image"],
            mask_image=inputs["mask_image"],
            strength=0.8,
            num_inference_steps=num_inference_steps,
            output_type="np",
            generator=torch.Generator("cpu").manual_seed(0),
            callback_on_step_end=pipe_state.apply,
        ).images

        # interrupt generation at step index
        interrupt_step_idx = 1

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            if i == interrupt_step_idx:
                pipe._interrupt = True

            return callback_kwargs

        output_interrupted = sd_pipe(
            prompt,
            image=inputs["image"],
            mask_image=inputs["mask_image"],
            strength=0.8,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            generator=torch.Generator("cpu").manual_seed(0),
            callback_on_step_end=callback_on_step_end,
        ).images

        # fetch intermediate latents at the interrupted step
        # from the completed generation process
        intermediate_latent = pipe_state.state[interrupt_step_idx]

        # compare the intermediate latent to the output of the interrupted process
        # they should be the same
        assert torch.allclose(intermediate_latent, output_interrupted, atol=1e-4)

    @slow
    def test_stable_diffusion_xl_inpaint_no_throughput_regression(self):
        """Test that stable diffusion inpainting no throughput regression autocast"""
        from diffusers.utils import load_image

        # Initialize inpaint parameters
        init_image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
        )

        prompts = [
            "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k",
            "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k",
        ]
        model_name = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        num_images_per_prompt = 10
        num_inference_steps = 10
        init_kwargs = {
            "use_habana": True,
            "use_hpu_graphs": True,
            "gaudi_config": "Habana/stable-diffusion",
            "torch_dtype": torch.bfloat16,
        }
        sdi_pipe = GaudiStableDiffusionXLInpaintPipeline.from_pretrained(model_name, **init_kwargs)

        set_seed(0)
        outputs = sdi_pipe(
            prompt=prompts,
            image=init_image,
            mask_image=mask_image,
            num_images_per_prompt=num_images_per_prompt,
            throughput_warmup_steps=2,
            num_inference_steps=num_inference_steps,
            batch_size=4,
        )

        self.assertEqual(len(outputs.images), num_images_per_prompt * len(prompts))
        self.assertGreaterEqual(outputs.throughput, 0.95 * INPAINT_XL_THROUGHPUT_BASELINE_BF16)
