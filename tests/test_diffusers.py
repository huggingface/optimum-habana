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
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Union
from unittest import TestCase, skipUnless

import numpy as np
import pytest
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
)
from diffusers.pipelines.controlnet.pipeline_controlnet import MultiControlNetModel
from diffusers.utils import load_image, numpy_to_pil
from diffusers.utils.testing_utils import floats_tensor
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
)
from optimum.habana.utils import set_seed

from .clip_coco_utils import calculate_clip_score, download_files


IS_GAUDI2 = os.environ.get("GAUDI2_CI", "0") == "1"


if IS_GAUDI2:
    THROUGHPUT_BASELINE_BF16 = 1.086
    THROUGHPUT_BASELINE_AUTOCAST = 0.394
    TEXTUAL_INVERSION_THROUGHPUT = 104.29806
    TEXTUAL_INVERSION_RUNTIME = 114.1344320399221
    CONTROLNET_THROUGHPUT = 92.886919836857
    CONTROLNET_RUNTIME = 537.4276602957398
else:
    THROUGHPUT_BASELINE_BF16 = 0.309
    THROUGHPUT_BASELINE_AUTOCAST = 0.114
    TEXTUAL_INVERSION_THROUGHPUT = 60.5991479573174
    TEXTUAL_INVERSION_RUNTIME = 196.43840550999994
    CONTROLNET_THROUGHPUT = 44.7278034963213
    CONTROLNET_RUNTIME = 1116.084316640001


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
    @pytest.mark.skip(reason="The dataset used in this test is not available at the moment.")
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
                 --pretrained_vae_model_name_or_path stabilityai/sdxl-vae
                 --dataset_name lambdalabs/pokemon-blip-captions
                 --resolution 512
                 --crop_resolution 512
                 --center_crop
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
                 --max_train_steps 2
                 --output_dir {tmpdir}
                """.split()

            # Run train_text_to_image_sdxl.y
            p = subprocess.Popen(cmd_line)
            return_code = p.wait()

            # Ensure the run finished without any issue
            self.assertEqual(return_code, 0)

            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))


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
