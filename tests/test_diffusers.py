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

import contextlib
import copy
import gc
import inspect
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from io import BytesIO, StringIO
from pathlib import Path
from typing import Callable, Union
from unittest import TestCase, skipUnless

import diffusers
import numpy as np
import PIL
import pytest
import requests
import safetensors
import torch
from diffusers import (
    AutoencoderKL,
    AutoencoderKLTemporalDecoder,
    AutoencoderTiny,
    ControlNetModel,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
    I2VGenXLUNet,
    LCMScheduler,
    PNDMScheduler,
    SD3Transformer2DModel,
    StableDiffusionXLPipeline,
    StableVideoDiffusionPipeline,
    UNet2DConditionModel,
    UNet2DModel,
    UNet3DConditionModel,
    UNetSpatioTemporalConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.controlnet.pipeline_controlnet import MultiControlNetModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    numpy_cosine_similarity_distance,
    require_torch,
)
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from parameterized import parameterized
from PIL import Image
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    DPTConfig,
    DPTFeatureExtractor,
    DPTForDepthEstimation,
    T5EncoderModel,
)
from transformers.testing_utils import parse_flag_from_env, slow

from optimum.habana import GaudiConfig
from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiDDPMPipeline,
    GaudiDiffusionPipeline,
    GaudiEulerAncestralDiscreteScheduler,
    GaudiEulerDiscreteScheduler,
    GaudiFluxImg2ImgPipeline,
    GaudiFluxPipeline,
    GaudiI2VGenXLPipeline,
    GaudiStableDiffusion3Pipeline,
    GaudiStableDiffusionControlNetPipeline,
    GaudiStableDiffusionDepth2ImgPipeline,
    GaudiStableDiffusionImageVariationPipeline,
    GaudiStableDiffusionImg2ImgPipeline,
    GaudiStableDiffusionInpaintPipeline,
    GaudiStableDiffusionInstructPix2PixPipeline,
    GaudiStableDiffusionLDM3DPipeline,
    GaudiStableDiffusionPipeline,
    GaudiStableDiffusionUpscalePipeline,
    GaudiStableDiffusionXLImg2ImgPipeline,
    GaudiStableDiffusionXLInpaintPipeline,
    GaudiStableDiffusionXLPipeline,
    GaudiStableVideoDiffusionControlNetPipeline,
    GaudiStableVideoDiffusionPipeline,
    GaudiTextToVideoSDPipeline,
)
from optimum.habana.diffusers.models import (
    ControlNetSDVModel,
    UNetSpatioTemporalConditionControlNetModel,
)
from optimum.habana.utils import set_seed

from .clip_coco_utils import calculate_clip_score, download_files
from .utils import OH_DEVICE_CONTEXT


IS_GAUDI1 = bool("gaudi1" == OH_DEVICE_CONTEXT)


_run_custom_bf16_ops_test_ = parse_flag_from_env("CUSTOM_BF16_OPS", default=False)


def custom_bf16_ops(test_case):
    """
    Decorator marking a test as needing custom bf16 ops.
    Custom bf16 ops must be declared before `habana_frameworks.torch.core` is imported, which is not possible if some other tests are executed before.

    Such tests are skipped by default. Set the CUSTOM_BF16_OPS environment variable to a truthy value to run them.

    """
    return skipUnless(_run_custom_bf16_ops_test_, "test requires custom bf16 ops")(test_case)


def check_gated_model_access(model):
    """
    Skip test for a gated model if access is not granted; this occurs when an account
    with the required permissions is not logged into the HF Hub.
    """
    try:
        hf_hub_download(repo_id=model, filename=HfApi().model_info(model).siblings[0].rfilename)
        gated = False

    except HfHubHTTPError:
        gated = True

    return pytest.mark.skipif(gated, reason=f"{model} is gated, please log in with approved HF access token")


def check_8xhpu(test_case):
    """
    Decorator marking a test as it requires 8xHPU on system
    """
    from optimum.habana.utils import get_device_count

    if get_device_count() != 8:
        skip = True
    else:
        skip = False

    return pytest.mark.skipif(skip, reason="test requires 8xHPU multi-card system")(test_case)


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

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

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
        ]
        num_images_per_prompt = 28
        batch_size = 7
        model_name = "CompVis/stable-diffusion-v1-4"
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig.from_pretrained("Habana/stable-diffusion"),
            torch_dtype=torch.bfloat16,
            sdp_on_bf16=True,
        )
        pipeline.unet.set_default_attn_processor(pipeline.unet)
        set_seed(27)
        outputs = pipeline(
            prompt=prompts,
            num_images_per_prompt=num_images_per_prompt,
            batch_size=batch_size,
            output_type="np",
        )

        # Check expected number of output images
        self.assertEqual(len(outputs.images), num_images_per_prompt * len(prompts))

        # Throughput regression test
        self.baseline.assertRef(
            compare=lambda actual, ref: actual >= (0.95 * ref),
            context=[OH_DEVICE_CONTEXT],
            throughput=outputs.throughput,
        )

        n = 0
        clip_score_avg = 0.0
        for i in range(len(outputs.images)):
            # Check expected shape for each output image
            self.assertEqual(outputs.images[i].shape, (512, 512, 3))

            if np.any(outputs.images[i] != 0):
                clip_score = calculate_clip_score(np.expand_dims(outputs.images[i], axis=0), prompts)
                clip_score_avg += clip_score
                n += 1

        # Quality test (check that the average CLIP score of valid output images is well in the 30s range)
        clip_score_avg /= n
        CLIP_SCORE_THRESHOLD = 30.0
        self.assertGreaterEqual(clip_score_avg, CLIP_SCORE_THRESHOLD)

    @custom_bf16_ops
    @slow
    def test_no_throughput_regression_autocast(self):
        prompts = [
            "An image of a squirrel in Picasso style",
        ]
        num_images_per_prompt = 28
        batch_size = 7
        model_name = "stabilityai/stable-diffusion-2-1"
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig.from_pretrained("Habana/stable-diffusion-2"),
            sdp_on_bf16=True,
        )
        set_seed(27)
        outputs = pipeline(
            prompt=prompts,
            num_images_per_prompt=num_images_per_prompt,
            batch_size=batch_size,
            height=768,
            width=768,
        )

        # Check expected number of output images
        self.assertEqual(len(outputs.images), num_images_per_prompt * len(prompts))

        # Throughput regression test
        self.baseline.assertRef(
            compare=lambda actual, ref: actual >= (0.95 * ref),
            context=[OH_DEVICE_CONTEXT],
            throughput=outputs.throughput,
        )

    @custom_bf16_ops
    @slow
    def test_no_generation_regression_ldm3d(self):
        prompts = [
            "An image of a squirrel in Picasso style",
        ]
        num_images_per_prompt = 28
        batch_size = 7
        model_name = "Intel/ldm3d-4c"
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipeline = GaudiStableDiffusionLDM3DPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            safety_checker=None,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig.from_pretrained("Habana/stable-diffusion-2"),
            sdp_on_bf16=True,
        )
        set_seed(27)
        outputs = pipeline(
            prompt=prompts,
            num_images_per_prompt=num_images_per_prompt,
            batch_size=batch_size,
            output_type="np",
        )

        # Check expected number of output images
        self.assertEqual(len(outputs.rgb), num_images_per_prompt * len(prompts))
        self.assertEqual(len(outputs.depth), num_images_per_prompt * len(prompts))

        # Throughput regression test
        self.baseline.assertRef(
            compare=lambda actual, ref: actual >= (0.95 * ref),
            context=[OH_DEVICE_CONTEXT],
            throughput=outputs.throughput,
        )

        n = 0
        clip_score_avg = 0.0
        for i in range(len(outputs.rgb)):
            # Check expected shape for each output image
            self.assertEqual(outputs.rgb[i].shape, (512, 512, 3))
            self.assertEqual(outputs.depth[i].shape, (512, 512, 1))

            if np.any(outputs.rgb[i] != 0):
                clip_score = calculate_clip_score(np.expand_dims(outputs.rgb[i], axis=0), prompts)
                clip_score_avg += clip_score
                n += 1

        # Quality test (check that the average CLIP score of valid output images is well in the 30s range)
        clip_score_avg /= n
        CLIP_SCORE_THRESHOLD = 30.0
        self.assertGreaterEqual(clip_score_avg, CLIP_SCORE_THRESHOLD)

    @slow
    def test_no_generation_regression_upscale(self):
        model_name = "stabilityai/stable-diffusion-x4-upscaler"
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipeline = GaudiStableDiffusionUpscalePipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(use_torch_autocast=False),
            sdp_on_bf16=True,
        )
        set_seed(27)
        url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
        response = requests.get(url)
        low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
        low_res_img = low_res_img.resize((128, 128))
        prompt = "a white cat"
        upscaled_image = pipeline(prompt=prompt, image=low_res_img, output_type="np").images[0]

        # Check expected shape of the upscaled image
        self.assertEqual(upscaled_image.shape, (512, 512, 3))

        # Check expected upscaled values of a sample slice
        expected_slice = np.array(
            [
                0.16528079,
                0.16161581,
                0.15665841,
                0.16609294,
                0.15943781,
                0.14936810,
                0.15782778,
                0.15342544,
                0.14590860,
            ]
        )
        self.assertLess(np.abs(expected_slice - upscaled_image[-3:, -3:, -1].flatten()).max(), 5e-3)

    @slow
    @check_8xhpu
    def test_sd_textual_inversion(self):
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
            cache_dir = Path(data_dir, ".cache")
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir)
            with tempfile.TemporaryDirectory() as run_dir:
                cmd_line = [
                    "python3",
                    f"{path_to_script.parent.parent.parent / 'gaudi_spawn.py'}",
                    "--use_mpi",
                    "--world_size 8",
                    f"{path_to_script}",
                    "--pretrained_model_name_or_path CompVis/stable-diffusion-v1-4",
                    f"--train_data_dir {data_dir}",
                    '--learnable_property "object"',
                    '--placeholder_token "<cat-toy>"',
                    '--initializer_token "toy"',
                    "--resolution 256",
                    "--train_batch_size 4",
                    "--max_train_steps 10",
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

                self.baseline.assertRef(
                    compare=lambda actual, ref: actual >= (0.95 * ref),
                    context=[OH_DEVICE_CONTEXT],
                    train_samples_per_second=results["train_samples_per_second"],
                )
                self.baseline.assertRef(
                    compare=lambda actual, ref: actual <= (1.05 * ref),
                    context=[OH_DEVICE_CONTEXT],
                    train_runtime=results["train_runtime"],
                )

                # Assess generated image
                pipe = GaudiStableDiffusionPipeline.from_pretrained(
                    run_dir,
                    torch_dtype=torch.bfloat16,
                    use_habana=True,
                    use_hpu_graphs=True,
                    gaudi_config=GaudiConfig(use_habana_mixed_precision=False),
                    sdp_on_bf16=True,
                )
                prompt = "A <cat-toy> backpack"
                set_seed(27)
                image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, output_type="np").images[0]

                self.assertEqual(image.shape, (512, 512, 3))


class GaudiStableDiffusionXLPipelineTester(TestCase):
    """
    Tests the StableDiffusionXLPipeline for Gaudi.
    """

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

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
        sd_pipe_oh = GaudiStableDiffusionXLPipeline(use_habana=True, gaudi_config=gaudi_config, **components)
        sd_pipe_hf = StableDiffusionXLPipeline(**components)

        def _get_image_from_pipeline(pipeline, device=device):
            pipeline.set_progress_bar_config(disable=None)
            inputs = self.get_dummy_inputs(device)
            image = pipeline(**inputs).images[0]

            self.assertEqual(image.shape, (64, 64, 3))
            return image[-3:, -3:, -1]

        image_slice_oh = _get_image_from_pipeline(sd_pipe_oh)
        image_slice_hf = _get_image_from_pipeline(sd_pipe_hf)

        self.assertLess((np.abs(image_slice_oh.flatten() - image_slice_hf.flatten()).max()), 1e-2)

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
        expected_slice = np.array([0.4539, 0.5119, 0.4521, 0.4395, 0.5495, 0.49344, 0.5761, 0.5147, 0.4943])
        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-2)

    def test_stable_diffusion_xl_turbo_euler_ancestral(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(timestep_spacing="trailing")
        gaudi_config = GaudiConfig(use_torch_autocast=False)

        sd_pipe_oh = GaudiStableDiffusionXLPipeline(use_habana=True, gaudi_config=gaudi_config, **components)
        sd_pipe_oh.scheduler = GaudiEulerAncestralDiscreteScheduler.from_config(sd_pipe_oh.scheduler.config)
        sd_pipe_hf = StableDiffusionXLPipeline(**components)
        sd_pipe_hf.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipe_hf.scheduler.config)

        def _get_image_from_pipeline(pipeline, device=device):
            pipeline.set_progress_bar_config(disable=None)

            inputs = self.get_dummy_inputs(device)
            image = pipeline(**inputs).images[0]

            self.assertEqual(image.shape, (64, 64, 3))
            return image[-3:, -3:, -1]

        image_slice_oh = _get_image_from_pipeline(sd_pipe_oh)
        image_slice_hf = _get_image_from_pipeline(sd_pipe_hf)

        self.assertLess((np.abs(image_slice_oh.flatten() - image_slice_hf.flatten()).max()), 1e-2)

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

    @slow
    def test_sdxl_textual_inversion(self):
        path_to_script = (
            Path(os.path.dirname(__file__)).parent
            / "examples"
            / "stable-diffusion"
            / "training"
            / "textual_inversion_sdxl.py"
        )
        with tempfile.TemporaryDirectory() as data_dir:
            snapshot_download(
                "diffusers/cat_toy_example", local_dir=data_dir, repo_type="dataset", ignore_patterns=".gitattributes"
            )
            cache_dir = Path(data_dir, ".cache")
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir)
            with tempfile.TemporaryDirectory() as run_dir:
                cmd_line = [
                    "python3",
                    f"{path_to_script}",
                    "--pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0",
                    f"--train_data_dir {data_dir}",
                    "--learnable_property object",
                    "--placeholder_token <cat-toy>",
                    "--initializer_token toy",
                    "--resolution 64",
                    "--train_batch_size 1",
                    "--gradient_accumulation_steps 4",
                    "--max_train_steps 50",
                    "--learning_rate 5.0e-04",
                    "--scale_lr",
                    "--lr_scheduler constant",
                    "--lr_warmup_steps 0",
                    f"--output_dir {run_dir}",
                    "--save_as_full_pipeline",
                    "--gaudi_config_name Habana/stable-diffusion",
                    "--sdp_on_bf16",
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

                self.baseline.assertRef(
                    compare=lambda actual, ref: actual >= (0.95 * ref),
                    context=[OH_DEVICE_CONTEXT],
                    train_samples_per_second=results["train_samples_per_second"],
                )
                self.baseline.assertRef(
                    compare=lambda actual, ref: actual <= (1.05 * ref),
                    context=[OH_DEVICE_CONTEXT],
                    train_runtime=results["train_runtime"],
                )

                pipe = GaudiStableDiffusionXLPipeline.from_pretrained(
                    run_dir,
                    torch_dtype=torch.bfloat16,
                    use_habana=True,
                    use_hpu_graphs=True,
                    gaudi_config=GaudiConfig(use_habana_mixed_precision=False),
                    sdp_on_bf16=True,
                )

                set_seed(27)
                prompt_1 = "A <cat-toy> backpack"
                prompt_2 = "A <cat-toy> colored backpack"
                image = pipe(
                    prompt=prompt_1, prompt_2=prompt_2, num_inference_steps=50, guidance_scale=7.5, output_type="np"
                ).images[0]
                self.assertEqual(image.shape, (1024, 1024, 3))

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

    def test_stable_diffusion_xl_num_images_per_prompt_optimized(self):
        import habana_frameworks.torch.hpu as torch_hpu

        kwargs = {"timestep_spacing": "linspace"}
        scheduler = GaudiEulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", **kwargs
        )
        scheduler.hpu_opt = True
        kwargs = {
            "scheduler": scheduler,
            "use_habana": True,
            "use_hpu_graphs": True,
            "gaudi_config": "Habana/stable-diffusion",
            "torch_dtype": torch.bfloat16,
        }

        os.environ["PATCH_SDPA"] = "1"

        from optimum.habana.diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_mlperf import (
            StableDiffusionXLPipeline_HPU,
        )

        model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"

        sd_pipe = StableDiffusionXLPipeline_HPU.from_pretrained(
            model_name_or_path,
            **kwargs,
        )

        sd_pipe.unet.set_default_attn_processor(sd_pipe.unet)
        sd_pipe.to(torch.device("hpu"))
        sd_pipe.unet = torch_hpu.wrap_in_hpu_graph(sd_pipe.unet)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # Test num_images_per_prompt=1 (default)
        images = sd_pipe(prompt, num_inference_steps=2, output_type="np").images

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (1024, 1024, 3))

        # Test num_images_per_prompt=1 (default) for several prompts
        num_prompts = 3
        images = sd_pipe([prompt] * num_prompts, num_inference_steps=2, output_type="np").images

        self.assertEqual(len(images), num_prompts)
        self.assertEqual(images[-1].shape, (1024, 1024, 3))

        # Test num_images_per_prompt for single prompt
        num_images_per_prompt = 2
        images = sd_pipe(
            prompt, num_inference_steps=2, output_type="np", num_images_per_prompt=num_images_per_prompt
        ).images

        self.assertEqual(len(images), num_images_per_prompt)
        self.assertEqual(images[-1].shape, (1024, 1024, 3))

        # Test num_images_per_prompt for several prompts
        num_prompts = 2
        images = sd_pipe(
            [prompt] * num_prompts,
            num_inference_steps=2,
            output_type="np",
            num_images_per_prompt=num_images_per_prompt,
        ).images

        self.assertEqual(len(images), num_prompts * num_images_per_prompt)
        self.assertEqual(images[-1].shape, (1024, 1024, 3))

        os.environ.pop("PATCH_SDPA")

    def test_stable_diffusion_xl_optimized_fp8(self):
        import habana_frameworks.torch.hpu as torch_hpu

        kwargs = {"timestep_spacing": "linspace"}
        scheduler = GaudiEulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", **kwargs
        )
        scheduler.hpu_opt = True
        kwargs = {
            "scheduler": scheduler,
            "use_habana": True,
            "use_hpu_graphs": True,
            "gaudi_config": "Habana/stable-diffusion",
            "torch_dtype": torch.bfloat16,
        }

        os.environ["PATCH_SDPA"] = "1"
        # Set QUANT_CONFIG environment variable
        os.environ["QUANT_CONFIG"] = "./quantization/stable-diffusion-xl/quantize_config.json"

        from optimum.habana.diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_mlperf import (
            StableDiffusionXLPipeline_HPU,
        )

        model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"

        sd_pipe = StableDiffusionXLPipeline_HPU.from_pretrained(
            model_name_or_path,
            **kwargs,
        )
        sd_pipe.unet.set_default_attn_processor(sd_pipe.unet)
        sd_pipe.to(torch.device("hpu"))

        quant_config_path = os.getenv("QUANT_CONFIG")

        original_dir = os.getcwd()
        config_dir = Path(os.path.dirname(__file__)).parent / "examples" / "stable-diffusion"
        os.chdir(config_dir)

        if quant_config_path:
            import habana_frameworks.torch.core as htcore
            from neural_compressor.torch.quantization import FP8Config, convert, prepare

            htcore.hpu_set_env()

            config = FP8Config.from_json_file(quant_config_path)

            if config.measure:
                print("Running measurements")
                sd_pipe.unet = prepare(sd_pipe.unet, config)
            elif config.quantize:
                print("Running quantization")
                sd_pipe.unet = convert(sd_pipe.unet, config)
            htcore.hpu_initialize(sd_pipe.unet, mark_only_scales_as_const=True)

        sd_pipe.unet = torch_hpu.wrap_in_hpu_graph(sd_pipe.unet)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # Test using quantization configuration
        images = sd_pipe(prompt, num_inference_steps=2, output_type="np").images

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (1024, 1024, 3))
        os.chdir(original_dir)

        os.environ.pop("PATCH_SDPA")

    @slow
    def test_stable_diffusion_xl_generation_throughput(self):
        prompts = [
            "Sailing ship painting by Van Gogh",
        ]
        num_images_per_prompt = 28
        batch_size = 7
        model_name = "stabilityai/stable-diffusion-xl-base-1.0"
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

        pipeline = GaudiStableDiffusionXLPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig.from_pretrained("Habana/stable-diffusion"),
            sdp_on_bf16=True,
        )
        set_seed(27)
        outputs = pipeline(
            prompt=prompts,
            num_images_per_prompt=num_images_per_prompt,
            batch_size=batch_size,
            num_inference_steps=30,
        )

        # Check expected number of output images
        self.assertEqual(len(outputs.images), num_images_per_prompt * len(prompts))

        # Throughput regression test
        self.baseline.assertRef(
            compare=lambda actual, ref: actual >= (0.95 * ref),
            context=[OH_DEVICE_CONTEXT],
            throughput=outputs.throughput,
        )


class GaudiStableDiffusion3PipelineTester(TestCase):
    """
    Tests the GaudiStableDiffusion3Pipeline for Gaudi.
    """

    pipeline_class = GaudiStableDiffusion3Pipeline
    params = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    batch_params = frozenset(["prompt", "negative_prompt"])

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = SD3Transformer2DModel(
            sample_size=32,
            patch_size=1,
            in_channels=4,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=4,
            caption_projection_dim=32,
            joint_attention_dim=32,
            pooled_projection_dim=64,
            out_channels=4,
        )
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )

        torch.manual_seed(0)
        text_encoder = CLIPTextModelWithProjection(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = CLIPTextModelWithProjection(clip_text_encoder_config)

        # HF issue with T5EncoderModel from tiny-random-t5
        # text_encoder_3 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder_3 = None

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_3 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "text_encoder_3": text_encoder_3,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "tokenizer_3": tokenizer_3,
            "transformer": transformer,
            "vae": vae,
        }

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_3_different_prompts(self):
        pipe = self.pipeline_class(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            **self.get_dummy_components(),
        )

        inputs = self.get_dummy_inputs(torch_device)
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt_2"] = "a different prompt"
        inputs["prompt_3"] = "another different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = np.abs(output_same_prompt - output_different_prompts).max()

        # Outputs should be different here
        assert max_diff > 1e-2

    def test_stable_diffusion_3_different_negative_prompts(self):
        pipe = self.pipeline_class(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            **self.get_dummy_components(),
        )

        inputs = self.get_dummy_inputs(torch_device)
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["negative_prompt_2"] = "deformed"
        inputs["negative_prompt_3"] = "blurry"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = np.abs(output_same_prompt - output_different_prompts).max()

        # Outputs should be different here
        assert max_diff > 1e-2

    def test_stable_diffusion_3_prompt_embeds(self):
        pipe = self.pipeline_class(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            **self.get_dummy_components(),
        )
        inputs = self.get_dummy_inputs(torch_device)

        output_with_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = inputs.pop("prompt")

        do_classifier_free_guidance = inputs["guidance_scale"] > 1
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt,
            prompt_2=None,
            prompt_3=None,
            do_classifier_free_guidance=do_classifier_free_guidance,
            device=torch_device,
        )
        output_with_embeds = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            **inputs,
        ).images[0]

        max_diff = np.abs(output_with_prompt - output_with_embeds).max()
        assert max_diff < 1e-1

    def test_fused_qkv_projections(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = self.pipeline_class(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            **components,
        )
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        original_image_slice = image[0, -3:, -3:, -1]

        pipe.transformer.fuse_qkv_projections()
        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice_fused = image[0, -3:, -3:, -1]

        pipe.transformer.unfuse_qkv_projections()
        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice_disabled = image[0, -3:, -3:, -1]

        assert np.allclose(original_image_slice, image_slice_fused, atol=1e-3, rtol=1e-3), (
            "Fusion of QKV projections shouldn't affect the outputs."
        )
        assert np.allclose(image_slice_fused, image_slice_disabled, atol=1e-3, rtol=1e-3), (
            "Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled."
        )
        assert np.allclose(original_image_slice, image_slice_disabled, atol=1e-2, rtol=1e-2), (
            "Original outputs should match when fused QKV projections are disabled."
        )

    @slow
    @check_gated_model_access("stabilityai/stable-diffusion-3-medium-diffusers")
    @pytest.mark.skipif(IS_GAUDI1, reason="does not fit into Gaudi1 memory")
    def test_sd3_inference(self):
        repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"

        pipe = self.pipeline_class.from_pretrained(
            repo_id,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            torch_dtype=torch.bfloat16,
            sdp_on_bf16=True,
        )

        outputs = pipe(
            prompt="Sailing ship painting by Van Gogh",
            num_inference_steps=28,
            batch_size=1,
            num_images_per_prompt=10,
            output_type="np",
        )

        # Check expected performance of FLUX.1 dev img-to-img model
        self.baseline.assertRef(
            compare=lambda actual, ref: actual >= (0.95 * ref),
            context=[OH_DEVICE_CONTEXT],
            throughput=outputs.throughput,
        )


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


class GaudiStableDiffusionDepth2ImgPipelineTester(TestCase):
    """
    Tests for depth to image generation
    """

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=5,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            attention_head_dim=(2, 4),
            use_linear_projection=True,
        )
        scheduler = PNDMScheduler(skip_prk_steps=True)
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
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

        backbone_config = {
            "global_padding": "same",
            "layer_type": "bottleneck",
            "depths": [3, 4, 9],
            "out_features": ["stage1", "stage2", "stage3"],
            "embedding_dynamic_padding": True,
            "hidden_sizes": [96, 192, 384, 768],
            "num_groups": 2,
        }
        depth_estimator_config = DPTConfig(
            image_size=32,
            patch_size=16,
            num_channels=3,
            hidden_size=32,
            num_hidden_layers=4,
            backbone_out_indices=(0, 1, 2, 3),
            num_attention_heads=4,
            intermediate_size=37,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=False,
            initializer_range=0.02,
            is_hybrid=True,
            backbone_config=backbone_config,
            backbone_featmap_shape=[1, 384, 24, 24],
        )
        depth_estimator = DPTForDepthEstimation(depth_estimator_config).eval()
        feature_extractor = DPTFeatureExtractor.from_pretrained(
            "hf-internal-testing/tiny-random-DPTForDepthEstimation"
        )

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "depth_estimator": depth_estimator,
            "feature_extractor": feature_extractor,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB").resize((32, 32))
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def get_dummy_image(self, shape=(1, 3, 32, 32), seed=0):
        image = floats_tensor(shape, rng=random.Random(seed))
        image = image.cpu().permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB").resize(shape[-2:])
        return image

    def test_depth2img_pipeline_default(self):
        components = self.get_dummy_components()
        inputs = self.get_dummy_inputs("cpu")
        gaudi_config = GaudiConfig(use_torch_autocast=False)

        pipe = GaudiStableDiffusionDepth2ImgPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        pipe.set_progress_bar_config(disable=None)

        outputs = pipe(**inputs)
        image = outputs.images[0]
        image = np.array(image)
        image_slice = image[-3:, -3:, -1]
        expected_slice = np.array(
            [0.42007083, 0.44642246, 0.44746736, 0.4038852, 0.560547, 0.5513845, 0.5325784, 0.5170926, 0.46997207]
        )

        assert image.shape == (32, 32, 3)
        assert np.allclose(image_slice.flatten(), expected_slice)

    def test_depth2img_pipeline_batch(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig(use_torch_autocast=False)

        pipe = GaudiStableDiffusionDepth2ImgPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        pipe.set_progress_bar_config(disable=None)

        outputs = pipe(
            prompt=["A painting of a squirrel eating a burger", "A painting of a squirrel eating a burger"],
            image=self.get_dummy_image(),
            generator=torch.Generator("cpu").manual_seed(0),
            num_inference_steps=2,
            output_type="np",
        )
        images = outputs.images

        assert len(images) == 2
        assert images[-1].shape == (32, 32, 3)

    def test_depth2img_pipeline_bf16(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig(use_torch_autocast=True)

        pipe = GaudiStableDiffusionDepth2ImgPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        pipe.set_progress_bar_config(disable=None)

        outputs = pipe(
            prompt="A painting of a squirrel eating a burger",
            image=self.get_dummy_image(),
            generator=torch.Generator("cpu").manual_seed(0),
            num_inference_steps=2,
            output_type="np",
        )
        images = outputs.images

        assert len(images) == 1
        assert images[0].shape == (32, 32, 3)

    def test_depth2img_pipeline_hpu_graphs(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig(use_torch_autocast=False)

        pipe = GaudiStableDiffusionDepth2ImgPipeline(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=gaudi_config,
            **components,
        )
        pipe.set_progress_bar_config(disable=None)

        outputs = pipe(
            prompt="A painting of a squirrel eating a burger",
            image=self.get_dummy_image(),
            generator=torch.Generator("cpu").manual_seed(0),
            num_inference_steps=2,
            output_type="np",
        )
        images = outputs.images

        assert len(images) == 1
        assert images[0].shape == (32, 32, 3)

    @slow
    def test_depth2img_pipeline(self):
        gaudi_config = GaudiConfig(use_torch_autocast=True)
        model_name = "stabilityai/stable-diffusion-2-depth"
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

        pipe = GaudiStableDiffusionDepth2ImgPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=gaudi_config,
            sdp_on_bf16=True,
        )
        image = Image.open(
            requests.get(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png",
                stream=True,
            ).raw
        )
        prompt = "A fancy meal with soup and pancakes"

        outputs = pipe(
            prompt=prompt,
            image=image,
            generator=torch.Generator("cpu").manual_seed(0),
            num_inference_steps=50,
            output_type="np",
        )

        images = outputs.images
        clip_score = calculate_clip_score(np.expand_dims(image, axis=0), [prompt])
        target_score = 22.76

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (512, 512, 3))
        self.assertGreaterEqual(clip_score, 0.95 * target_score)


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
                 --sdp_on_bf16
                 --bf16
                 --adjust_throughput
                 --center_crop
                 --max_train_steps 2
                 --checkpointing_steps 2
                 --output_dir {tmpdir}
                """.split()

            # Run train_text_to_image_sdxl.py
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

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

    def test_script_train_controlnet(self):
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
    @check_8xhpu
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
                    {path_to_script.parent.parent.parent / "gaudi_spawn.py"}
                    --use_mpi
                    --world_size 8
                    {path_to_script}
                    --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4
                    --dataset_name fusing/fill50k
                    --resolution 256
                    --train_batch_size 4
                    --learning_rate 1e-05
                    --validation_steps 1000
                    --validation_image "{tmpdir}/conditioning_image_1.png" "{tmpdir}/conditioning_image_2.png"
                    --validation_prompt "red circle with blue background" "cyan circle with brown floral background"
                    --checkpointing_steps 1000
                    --throughput_warmup_steps 3
                    --use_hpu_graphs
                    --sdp_on_bf16
                    --bf16
                    --max_train_steps 10
                    --output_dir {tmpdir}
                    --trust_remote_code
                """.split()

            # Run train_controlnet.y
            p = subprocess.Popen(cmd_line)
            return_code = p.wait()

            # Ensure the run finished without any issue
            self.assertEqual(return_code, 0)

            # Assess throughput
            with open(Path(tmpdir) / "speed_metrics.json") as fp:
                results = json.load(fp)

            self.baseline.assertRef(
                compare=lambda actual, ref: actual >= (0.95 * ref),
                context=[OH_DEVICE_CONTEXT],
                train_samples_per_second=results["train_samples_per_second"],
            )
            self.baseline.assertRef(
                compare=lambda actual, ref: actual <= (1.05 * ref),
                context=[OH_DEVICE_CONTEXT],
                train_runtime=results["train_runtime"],
            )

            # Assess generated image
            controlnet = ControlNetModel.from_pretrained(tmpdir, torch_dtype=torch.bfloat16)
            pipe = GaudiStableDiffusionControlNetPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                controlnet=controlnet,
                torch_dtype=torch.bfloat16,
                use_habana=True,
                use_hpu_graphs=True,
                gaudi_config=GaudiConfig(use_habana_mixed_precision=False),
                sdp_on_bf16=True,
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
                --instance_data_dir {Path(os.path.dirname(__file__)) / "resource/img"}
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

    @slow
    def test_dreambooth_full(self):
        self._test_dreambooth("full")

    @slow
    def test_dreambooth_full_with_text_encoder(self):
        self._test_dreambooth("full", train_text_encoder=True)

    @slow
    def test_dreambooth_lora(self):
        self._test_dreambooth("lora")

    @slow
    def test_dreambooth_lora_with_text_encoder(self):
        self._test_dreambooth("lora", train_text_encoder=True)

    @slow
    def test_dreambooth_lokr(self):
        self._test_dreambooth("lokr")

    @slow
    def test_dreambooth_lokr_with_text_encoder(self):
        self._test_dreambooth("lokr", train_text_encoder=True)

    @slow
    def test_dreambooth_loha(self):
        self._test_dreambooth("loha")

    @slow
    def test_dreambooth_loha_with_text_encoder(self):
        self._test_dreambooth("loha", train_text_encoder=True)

    @slow
    def test_dreambooth_oft(self):
        self._test_dreambooth("oft")

    @slow
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
                --instance_data_dir {Path(os.path.dirname(__file__)) / "resource/img"}
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

    @slow
    def test_dreambooth_lora_sdxl_with_text_encoder(self):
        self._test_dreambooth_lora_sdxl(train_text_encoder=True)

    @slow
    def test_dreambooth_lora_sdxl(self):
        self._test_dreambooth_lora_sdxl(train_text_encoder=False)


class GaudiStableVideoDiffusionPipelineTester(TestCase):
    """
    Tests the StableVideoDiffusionPipeline for Gaudi.
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.24.0-release/tests/pipelines/stable_video_diffusion/test_stable_video_diffusion.py
    """

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

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
        sd_pipe_oh = GaudiStableVideoDiffusionPipeline(use_habana=True, gaudi_config=gaudi_config, **components)
        components2 = self.get_dummy_components()
        sd_pipe_hf = StableVideoDiffusionPipeline(**components2)

        def _get_image_from_pipeline(pipeline, device=device):
            pipeline.to(device)
            pipeline.set_progress_bar_config(disable=None)

            outputs = pipeline(
                **self.get_dummy_inputs(device),
            ).frames
            image = outputs[0]

            self.assertEqual(len(outputs), 1)
            self.assertEqual(image.shape, (2, 3, 32, 32))
            return image[0, -3:, -3:, -1]

        image_slice_oh = _get_image_from_pipeline(sd_pipe_oh, device="hpu").cpu()
        image_slice_hf = _get_image_from_pipeline(sd_pipe_hf)

        self.assertLess(np.abs(image_slice_oh.flatten() - image_slice_hf.flatten()).max(), 1e-2)

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
            sdp_on_bf16=True,
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

        # Check expected number of output frames
        self.assertEqual(len(outputs.frames[0]), 25)

        # Throughput regression test
        self.baseline.assertRef(
            compare=lambda actual, ref: actual >= (0.95 * ref),
            context=[OH_DEVICE_CONTEXT],
            throughput=outputs.throughput,
        )


class GaudiStableVideoDiffusionControlNetPipelineTester(TestCase):
    """
    Tests the StableVideoDiffusionControlNetPipeline for Gaudi.
    """

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNetSpatioTemporalConditionControlNetModel(
            sample_size=None,
            in_channels=8,
            out_channels=4,
            down_block_types=(
                "CrossAttnDownBlockSpatioTemporal",
                "CrossAttnDownBlockSpatioTemporal",
                "CrossAttnDownBlockSpatioTemporal",
                "DownBlockSpatioTemporal",
            ),
            up_block_types=(
                "UpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
            ),
            block_out_channels=(320, 640, 1280, 1280),
            addition_time_embed_dim=256,
            projection_class_embeddings_input_dim=768,
            layers_per_block=2,
            cross_attention_dim=1024,
            transformer_layers_per_block=1,
            num_attention_heads=(5, 10, 20, 20),
            num_frames=14,
        )
        torch.manual_seed(0)
        controlnet = ControlNetSDVModel(
            sample_size=96,  #
            in_channels=8,
            out_channels=4,
            down_block_types=(
                "CrossAttnDownBlockSpatioTemporal",
                "CrossAttnDownBlockSpatioTemporal",
                "CrossAttnDownBlockSpatioTemporal",
                "DownBlockSpatioTemporal",
            ),
            up_block_types=(
                "UpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
            ),
            block_out_channels=(320, 640, 1280, 1280),
            addition_time_embed_dim=256,
            projection_class_embeddings_input_dim=768,
            layers_per_block=2,
            cross_attention_dim=1024,
            transformer_layers_per_block=1,
            num_attention_heads=(5, 10, 20, 20),  #
            num_frames=14,
            conditioning_channels=3,
            conditioning_embedding_out_channels=(16, 32, 96, 256),
        )
        scheduler = GaudiEulerDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="v_prediction",
            interpolation_type="linear",
            use_karras_sigmas=True,
            sigma_min=0.002,
            sigma_max=700,
            timestep_spacing="leading",
            timestep_type="continuous",  # can be "discrete" or "continuous"
            steps_offset=1,
            rescale_betas_zero_snr=False,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLTemporalDecoder(
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            latent_channels=4,
            sample_size=768,
            scaling_factor=0.18215,
            force_upcast=True,
        )

        torch.manual_seed(0)
        config = CLIPVisionConfig(
            hidden_size=1280,
            intermediate_size=5120,
            projection_dim=1024,
            num_hidden_layers=32,
            num_attention_heads=16,
            num_channels=3,
            image_size=224,
            patch_size=14,
            hidden_act="gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
        )
        image_encoder = CLIPVisionModelWithProjection(config)

        torch.manual_seed(0)
        feature_extractor = CLIPImageProcessor(
            crop_size=64,
            size=64,
        )
        components = {
            "unet": unet,
            "image_encoder": image_encoder,
            "controlnet": controlnet,
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

        image = floats_tensor((1, 3, 64, 64), rng=random.Random(0)).to(device)
        controlnet_condition = floats_tensor((14, 3, 64, 64), rng=random.Random(0)).to(device)
        inputs = {
            "generator": generator,
            "image": image,
            "controlnet_condition": controlnet_condition,
            "num_inference_steps": 2,
            "output_type": "pt",
            "min_guidance_scale": 1.0,
            "max_guidance_scale": 2.5,
            "num_frames": 14,
            "height": 64,
            "width": 64,
        }
        return inputs

    def test_stable_video_diffusion_single_video(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig(use_torch_autocast=True)
        sd_pipe = GaudiStableVideoDiffusionControlNetPipeline(use_habana=True, gaudi_config=gaudi_config, **components)

        def _get_image_from_pipeline(pipeline, device=device):
            for component in pipeline.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()
            pipeline.to(device)
            pipeline.set_progress_bar_config(disable=None)
            outputs = pipeline(
                **self.get_dummy_inputs(device),
            ).frames
            image = outputs[0]
            self.assertEqual(len(outputs), 1)
            self.assertEqual(image.shape, (14, 3, 64, 64))
            return image[0::6, -3:, -3:, -1]

        expected_slice = np.array(
            [
                0.5463,
                0.5432,
                0.5576,
                0.4749,
                0.4736,
                0.4888,
                0.3865,
                0.3912,
                0.3869,
                0.5402,
                0.5389,
                0.5326,
                0.4747,
                0.4920,
                0.4760,
                0.4115,
                0.4248,
                0.4222,
                0.5359,
                0.5335,
                0.5221,
                0.4835,
                0.5044,
                0.4884,
                0.4266,
                0.4409,
                0.4295,
            ]
        )
        image_slice = _get_image_from_pipeline(sd_pipe)

        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-1)


class GaudiStableDiffusionInstructPix2PixPipelineTests(TestCase):
    """
    Tests the class StableDiffusionInstructPix2PixPipeline for Gaudi.
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.26.3/tests/pipelines/stable_diffusion/test_stable_diffusion_instruction_pix2pix.py
    """

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=8,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
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
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
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
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB")
        generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "image_guidance_scale": 1,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_pix2pix_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = GaudiStableDiffusionInstructPix2PixPipeline(
            **components,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(use_torch_autocast=False),
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        self.assertEqual(image.shape, (1, 32, 32, 3))
        expected_slice = np.array([0.7526, 0.3750, 0.4547, 0.6117, 0.5866, 0.5016, 0.4327, 0.5642, 0.4815])

        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-1)

    def test_stable_diffusion_pix2pix_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = GaudiStableDiffusionInstructPix2PixPipeline(
            **components,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(use_torch_autocast=False),
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(image.shape, (1, 32, 32, 3))
        expected_slice = np.array([0.7511, 0.3642, 0.4553, 0.6236, 0.5797, 0.5013, 0.4343, 0.5611, 0.4831])

        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-1)

    def test_stable_diffusion_pix2pix_multiple_init_images(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = GaudiStableDiffusionInstructPix2PixPipeline(
            **components,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(use_torch_autocast=False),
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * 2

        image = np.array(inputs["image"]).astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        image = image / 2 + 0.5
        image = image.permute(0, 3, 1, 2)
        inputs["image"] = image.repeat(2, 1, 1, 1)

        image = sd_pipe(**inputs).images
        image_slice = image[-1, -3:, -3:, -1]

        self.assertEqual(image.shape, (2, 32, 32, 3))
        expected_slice = np.array([0.5812, 0.5748, 0.5222, 0.5908, 0.5695, 0.7174, 0.6804, 0.5523, 0.5579])

        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-1)

    def test_stable_diffusion_pix2pix_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["scheduler"] = GaudiEulerAncestralDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        sd_pipe = GaudiStableDiffusionInstructPix2PixPipeline(
            **components,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(use_torch_autocast=False),
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        slice = [round(x, 4) for x in image_slice.flatten().tolist()]
        print(",".join([str(x) for x in slice]))

        self.assertEqual(image.shape, (1, 32, 32, 3))
        expected_slice = np.array([0.7417, 0.3842, 0.4732, 0.5776, 0.5891, 0.5139, 0.4052, 0.5673, 0.4986])

        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-3)


class GaudiStableDiffusionImg2ImgPipelineTests(TestCase):
    """
    Tests the class StableDiffusionImg2ImgPipeline for Gaudi.
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.26.3/tests/pipelines/stable_diffusion/test_stable_diffusion_img2img.py
    """

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            time_cond_proj_dim=time_cond_proj_dim,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = PNDMScheduler(skip_prk_steps=True, steps_offset=1)
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
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
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
            "use_habana": True,
            "use_hpu_graphs": True,
            "gaudi_config": GaudiConfig(use_torch_autocast=False),
        }
        return components

    def get_dummy_tiny_autoencoder(self):
        return AutoencoderTiny(in_channels=3, out_channels=3, latent_channels=4)

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image / 2 + 0.5
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_img2img_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = GaudiStableDiffusionImg2ImgPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [0.50006074, 0.49048987, 0.51323986, 0.5654023, 0.5470734, 0.6720333, 0.6559875, 0.5050407, 0.5401596]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img2img_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = GaudiStableDiffusionImg2ImgPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array(
            [0.5165765, 0.49377573, 0.5040854, 0.5882658, 0.574415, 0.67791325, 0.66678274, 0.51392066, 0.544225]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_img2img_multiple_init_images(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = GaudiStableDiffusionImg2ImgPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * 2
        inputs["image"] = inputs["image"].repeat(2, 1, 1, 1)
        image = sd_pipe(**inputs).images
        image_slice = image[-1, -3:, -3:, -1]

        assert image.shape == (2, 32, 32, 3)
        expected_slice = np.array(
            [0.3323526, 0.44501957, 0.51663095, 0.32356155, 0.40758416, 0.6448872, 0.44775, 0.5695873, 0.5541928]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3


class GaudiStableDiffusionImageVariationPipelineTests(TestCase):
    """
    Tests the class StableDiffusionImageVariationPipeline for Gaudi.
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.26.3/tests/pipelines/stable_diffusion_image_variation/test_stable_diffusion_image_variation.py
    """

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
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
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        torch.manual_seed(0)
        image_encoder_config = CLIPVisionConfig(
            hidden_size=32,
            projection_dim=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            image_size=32,
            patch_size=4,
        )
        image_encoder = CLIPVisionModelWithProjection(image_encoder_config)
        feature_extractor = CLIPImageProcessor(crop_size=32, size=32)

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "image_encoder": image_encoder,
            "feature_extractor": feature_extractor,
            "safety_checker": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB").resize((32, 32))
        generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_img_variation_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = GaudiStableDiffusionImageVariationPipeline(
            **components,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(use_torch_autocast=False),
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(image.shape, (1, 64, 64, 3))
        expected_slice = np.array([0.5239, 0.5723, 0.4796, 0.5049, 0.5550, 0.4685, 0.5329, 0.4891, 0.4921])
        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-1)

    def test_stable_diffusion_img_variation_multiple_images(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = GaudiStableDiffusionImageVariationPipeline(
            **components,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(use_torch_autocast=False),
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["image"] = 2 * [inputs["image"]]
        output = sd_pipe(**inputs)

        image = output.images

        image_slice = image[-1, -3:, -3:, -1]

        self.assertEqual(image.shape, (2, 64, 64, 3))
        expected_slice = np.array([0.6892, 0.5637, 0.5836, 0.5771, 0.6254, 0.6409, 0.5580, 0.5569, 0.5289])

        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-1)


class GaudiStableDiffusionXLImg2ImgPipelineTests(TestCase):
    def get_dummy_components(self, skip_first_text_encoder=False, time_cond_proj_dim=None):
        torch.manual_seed(0)
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
        scheduler = GaudiEulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
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
            "text_encoder": text_encoder if not skip_first_text_encoder else None,
            "tokenizer": tokenizer if not skip_first_text_encoder else None,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "requires_aesthetics_score": True,
            "image_encoder": image_encoder,
            "feature_extractor": feature_extractor,
        }
        return components

    def get_dummy_tiny_autoencoder(self):
        return AutoencoderTiny(in_channels=3, out_channels=3, latent_channels=4)

    def test_components_function(self):
        init_components = self.get_dummy_components()
        init_components.pop("requires_aesthetics_score")
        pipe = GaudiStableDiffusionXLImg2ImgPipeline(
            **init_components,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(use_torch_autocast=False),
        )

        self.assertTrue(hasattr(pipe, "components"))
        self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image / 2 + 0.5
        generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "strength": 0.8,
        }
        return inputs

    def test_stable_diffusion_xl_img2img_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = GaudiStableDiffusionXLImg2ImgPipeline(
            **components,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(use_torch_autocast=False),
        )
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(image.shape, (1, 32, 32, 3))

        expected_slice = np.array([0.4664, 0.4886, 0.4403, 0.6902, 0.5592, 0.4534, 0.5931, 0.5951, 0.5224])
        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-2)


class GaudiTextToVideoSDPipelineTester(TestCase):
    """
    Tests the TextToVideoSDPipeline for Gaudi.
    Adapted from https://github.com/huggingface/diffusers/blob/v0.24.0-release/tests/pipelines/text_to_video_synthesis/test_text_to_video.py
    """

    def get_dummy_components(self):
        set_seed(0)
        unet = UNet3DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock3D", "DownBlock3D"),
            up_block_types=("UpBlock3D", "CrossAttnUpBlock3D"),
            cross_attention_dim=4,
            attention_head_dim=4,
            norm_num_groups=2,
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
        set_seed(0)
        vae = AutoencoderKL(
            block_out_channels=(8,),
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            latent_channels=4,
            sample_size=32,
            norm_num_groups=2,
        )
        set_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=4,
            intermediate_size=16,
            layer_norm_eps=1e-05,
            num_attention_heads=2,
            num_hidden_layers=2,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
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

    def test_text_to_video_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig(use_torch_autocast=False)
        sd_pipe = GaudiTextToVideoSDPipeline(use_habana=True, gaudi_config=gaudi_config, **components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["output_type"] = "np"
        video = sd_pipe(**inputs).videos[0]
        image_slice = video[0][-3:, -3:, -1]

        assert video[0].shape == (32, 32, 3)
        expected_slice = np.array(
            [0.32823694, 0.5277065, 0.5257378, 0.51532686, 0.62792695, 0.5966803, 0.55225205, 0.6153607, 0.60387087]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    @slow
    def test_stable_video_diffusion_no_latency_regression_bf16(self):
        prompts = [
            "An astronaut riding a horse",
        ]
        model_name = "ali-vilab/text-to-video-ms-1.7b"
        pipeline = GaudiTextToVideoSDPipeline.from_pretrained(
            model_name,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig.from_pretrained("Habana/stable-diffusion"),
            torch_dtype=torch.bfloat16,
            sdp_to_bf16=True,
        )

        set_seed(27)
        outputs = pipeline(
            prompt=prompts,
        )

        # Check expected number of output frames
        self.assertEqual(len(outputs.videos[0]), 16)


"""
Copied from: https://github.com/huggingface/diffusers/blob/v0.26.3/tests/pipelines/test_pipelines_common.py
- Remove PipelinePushToHubTester testcase.
- Remove test_multi_vae testcase.
- Remove test_save_load_local.
- Remove  test_save_load_optional_components.
- Modified the get_dummy_components to add the Gaudi pipeline parameters: use_habana, use_hpu_graphs, gaudi_config, bf16_full_eval
"""


torch_device = "hpu"


def to_np(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    return tensor


def check_same_shape(tensor_list):
    shapes = [tensor.shape for tensor in tensor_list]
    return all(shape == shapes[0] for shape in shapes[1:])


class PipelineLatentTesterMixin:
    """
    This mixin is designed to be used with PipelineTesterMixin and unittest.TestCase classes.
    It provides a set of common tests for PyTorch pipeline that has vae, e.g.
    equivalence of different input and output types, etc.
    """

    @property
    def image_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `image_params` in the child test class. "
            "`image_params` are tested for if all accepted input image types (i.e. `pt`,`pil`,`np`) are producing same results"
        )

    @property
    def image_latents_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `image_latents_params` in the child test class. "
            "`image_latents_params` are tested for if passing latents directly are producing same results"
        )

    def get_dummy_inputs_by_type(self, device, seed=0, input_image_type="pt", output_type="np"):
        inputs = self.get_dummy_inputs(device, seed)

        def convert_to_pt(image):
            if isinstance(image, torch.Tensor):
                input_image = image
            elif isinstance(image, np.ndarray):
                input_image = VaeImageProcessor.numpy_to_pt(image)
            elif isinstance(image, Image.Image):
                input_image = VaeImageProcessor.pil_to_numpy(image)
                input_image = VaeImageProcessor.numpy_to_pt(input_image)
            else:
                raise ValueError(f"unsupported input_image_type {type(image)}")
            return input_image

        def convert_pt_to_type(image, input_image_type):
            if input_image_type == "pt":
                input_image = image
            elif input_image_type == "np":
                input_image = VaeImageProcessor.pt_to_numpy(image)
            elif input_image_type == "pil":
                input_image = VaeImageProcessor.pt_to_numpy(image)
                input_image = VaeImageProcessor.numpy_to_pil(input_image)
            else:
                raise ValueError(f"unsupported input_image_type {input_image_type}.")
            return input_image

        for image_param in self.image_params:
            if image_param in inputs.keys():
                inputs[image_param] = convert_pt_to_type(
                    convert_to_pt(inputs[image_param]).to(device), input_image_type
                )

        inputs["output_type"] = output_type

        return inputs

    def test_pt_np_pil_outputs_equivalent(self, expected_max_diff=1e-4):
        self._test_pt_np_pil_outputs_equivalent(expected_max_diff=expected_max_diff)

    def _test_pt_np_pil_outputs_equivalent(self, expected_max_diff=1e-4, input_image_type="pt"):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        output_pt = pipe(
            **self.get_dummy_inputs_by_type(torch_device, input_image_type=input_image_type, output_type="pt")
        )[0]
        output_np = pipe(
            **self.get_dummy_inputs_by_type(torch_device, input_image_type=input_image_type, output_type="np")
        )[0]
        output_pil = pipe(
            **self.get_dummy_inputs_by_type(torch_device, input_image_type=input_image_type, output_type="pil")
        )[0]

        max_diff = np.abs(output_pt.cpu().numpy().transpose(0, 2, 3, 1) - output_np).max()
        self.assertLess(
            max_diff, expected_max_diff, "`output_type=='pt'` generate different results from `output_type=='np'`"
        )

        max_diff = np.abs(np.array(output_pil[0]) - (output_np * 255).round()).max()
        self.assertLess(max_diff, 2.0, "`output_type=='pil'` generate different results from `output_type=='np'`")

    def test_pt_np_pil_inputs_equivalent(self):
        if len(self.image_params) == 0:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        out_input_pt = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="pt"))[0]
        out_input_np = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="np"))[0]
        out_input_pil = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="pil"))[0]

        max_diff = np.abs(out_input_pt - out_input_np).max()
        self.assertLess(max_diff, 1e-4, "`input_type=='pt'` generate different result from `input_type=='np'`")
        max_diff = np.abs(out_input_pil - out_input_np).max()
        self.assertLess(max_diff, 1e-2, "`input_type=='pt'` generate different result from `input_type=='np'`")

    def test_latents_input(self):
        if len(self.image_latents_params) == 0:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)
        pipe.set_progress_bar_config(disable=None)

        out = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="pt"))[0]

        vae = components["vae"]
        inputs = self.get_dummy_inputs_by_type(torch_device, input_image_type="pt")
        generator = inputs["generator"]
        for image_param in self.image_latents_params:
            if image_param in inputs.keys():
                inputs[image_param] = (
                    vae.encode(inputs[image_param]).latent_dist.sample(generator) * vae.config.scaling_factor
                )
        out_latents_inputs = pipe(**inputs)[0]

        max_diff = np.abs(out - out_latents_inputs).max()
        self.assertLess(max_diff, 1e-4, "passing latents as image input generate different result from passing image")


@require_torch
class PipelineKarrasSchedulerTesterMixin:
    """
    This mixin is designed to be used with unittest.TestCase classes.
    It provides a set of common tests for each PyTorch pipeline that makes use of KarrasDiffusionSchedulers
    equivalence of dict and tuple outputs, etc.
    """

    def test_karras_schedulers_shape(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        # make sure that PNDM does not need warm-up
        pipe.scheduler.register_to_config(skip_prk_steps=True)

        pipe.to("cpu")
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = 2

        if "strength" in inputs:
            inputs["num_inference_steps"] = 4
            inputs["strength"] = 0.5

        outputs = []
        for scheduler_enum in KarrasDiffusionSchedulers:
            if "KDPM2" in scheduler_enum.name:
                inputs["num_inference_steps"] = 5

            scheduler_cls = getattr(diffusers, scheduler_enum.name)
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
            output = pipe(**inputs)[0]
            outputs.append(output)

            if "KDPM2" in scheduler_enum.name:
                inputs["num_inference_steps"] = 2

        assert check_same_shape(outputs)


@require_torch
class PipelineTesterMixin:
    """
    This mixin is designed to be used with unittest.TestCase classes.
    It provides a set of common tests for each PyTorch pipeline, e.g. saving and loading the pipeline,
    equivalence of dict and tuple outputs, etc.
    """

    # Canonical parameters that are passed to `__call__` regardless
    # of the type of pipeline. They are always optional and have common
    # sense default values.
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "num_images_per_prompt",
            "generator",
            "latents",
            "output_type",
            "return_dict",
        ]
    )

    # set these parameters to False in the child class if the pipeline does not support the corresponding functionality
    test_attention_slicing = True

    test_xformers_attention = True

    def get_generator(self, seed):
        device = "cpu"
        generator = torch.Generator(device).manual_seed(seed)
        return generator

    @property
    def pipeline_class(self) -> Union[Callable, DiffusionPipeline]:
        raise NotImplementedError(
            "You need to set the attribute `pipeline_class = ClassNameOfPipeline` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_dummy_components(self):
        raise NotImplementedError(
            "You need to implement `get_dummy_components(self)` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_dummy_inputs(self, device, seed=0):
        raise NotImplementedError(
            "You need to implement `get_dummy_inputs(self, device, seed)` in the child test class. "
            "See existing pipeline tests for reference."
        )

    @property
    def params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `params` in the child test class. "
            "`params` are checked for if all values are present in `__call__`'s signature."
            " You can set `params` using one of the common set of parameters defined in `pipeline_params.py`"
            " e.g., `TEXT_TO_IMAGE_PARAMS` defines the common parameters used in text to  "
            "image pipelines, including prompts and prompt embedding overrides."
            "If your pipeline's set of arguments has minor changes from one of the common sets of arguments, "
            "do not make modifications to the existing common sets of arguments. I.e. a text to image pipeline "
            "with non-configurable height and width arguments should set the attribute as "
            "`params = TEXT_TO_IMAGE_PARAMS - {'height', 'width'}`. "
            "See existing pipeline tests for reference."
        )

    @property
    def batch_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `batch_params` in the child test class. "
            "`batch_params` are the parameters required to be batched when passed to the pipeline's "
            "`__call__` method. `pipeline_params.py` provides some common sets of parameters such as "
            "`TEXT_TO_IMAGE_BATCH_PARAMS`, `IMAGE_VARIATION_BATCH_PARAMS`, etc... If your pipeline's "
            "set of batch arguments has minor changes from one of the common sets of batch arguments, "
            "do not make modifications to the existing common sets of batch arguments. I.e. a text to "
            "image pipeline `negative_prompt` is not batched should set the attribute as "
            "`batch_params = TEXT_TO_IMAGE_BATCH_PARAMS - {'negative_prompt'}`. "
            "See existing pipeline tests for reference."
        )

    @property
    def callback_cfg_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `callback_cfg_params` in the child test class that requires to run test_callback_cfg. "
            "`callback_cfg_params` are the parameters that needs to be passed to the pipeline's callback "
            "function when dynamically adjusting `guidance_scale`. They are variables that require special"
            "treatment when `do_classifier_free_guidance` is `True`. `pipeline_params.py` provides some common"
            " sets of parameters such as `TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS`. If your pipeline's "
            "set of cfg arguments has minor changes from one of the common sets of cfg arguments, "
            "do not make modifications to the existing common sets of cfg arguments. I.e. for inpaint pipeine, you "
            " need to adjust batch size of `mask` and `masked_image_latents` so should set the attribute as"
            "`callback_cfg_params = TEXT_TO_IMAGE_CFG_PARAMS.union({'mask', 'masked_image_latents'})`"
        )

    def tearDown(self):
        # clean up the VRAM after each test in case of CUDA runtime errors
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_pipeline_call_signature(self):
        self.assertTrue(
            hasattr(self.pipeline_class, "__call__"), f"{self.pipeline_class} should have a `__call__` method"
        )

        parameters = inspect.signature(self.pipeline_class.__call__).parameters

        optional_parameters = set()

        for k, v in parameters.items():
            if v.default != inspect._empty:
                optional_parameters.add(k)

        parameters = set(parameters.keys())
        parameters.remove("self")
        parameters.discard("kwargs")  # kwargs can be added if arguments of pipeline call function are deprecated

        remaining_required_parameters = set()

        for param in self.params:
            if param not in parameters:
                remaining_required_parameters.add(param)

        self.assertTrue(
            len(remaining_required_parameters) == 0,
            f"Required parameters not present: {remaining_required_parameters}",
        )

        remaining_required_optional_parameters = set()

        for param in self.required_optional_params:
            if param not in optional_parameters:
                remaining_required_optional_parameters.add(param)

        self.assertTrue(
            len(remaining_required_optional_parameters) == 0,
            f"Required optional parameters not present: {remaining_required_optional_parameters}",
        )

    def test_inference_batch_consistent(self, batch_sizes=[2]):
        self._test_inference_batch_consistent(batch_sizes=batch_sizes)

    def _test_inference_batch_consistent(
        self, batch_sizes=[2], additional_params_copy_to_batched_inputs=["num_inference_steps"], batch_generator=True
    ):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["generator"] = self.get_generator(0)

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # prepare batched inputs
        batched_inputs = []
        for batch_size in batch_sizes:
            batched_input = {}
            batched_input.update(inputs)

            for name in self.batch_params:
                if name not in inputs:
                    continue

                value = inputs[name]
                if name == "prompt":
                    len_prompt = len(value)
                    # make unequal batch sizes
                    batched_input[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]

                    # make last batch super long
                    batched_input[name][-1] = 100 * "very long"

                else:
                    batched_input[name] = batch_size * [value]

            if batch_generator and "generator" in inputs:
                batched_input["generator"] = [self.get_generator(i) for i in range(batch_size)]

            if "batch_size" in inputs:
                batched_input["batch_size"] = batch_size

            batched_inputs.append(batched_input)
        logger.setLevel(level=diffusers.logging.WARNING)
        for batch_size, batched_input in zip(batch_sizes, batched_inputs):
            output = pipe(**batched_input)
            assert len(output[0]) == batch_size

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-4):
        self._test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def _test_inference_batch_single_identical(
        self,
        batch_size=2,
        expected_max_diff=1e-4,
        additional_params_copy_to_batched_inputs=["num_inference_steps"],
    ):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for components in pipe.components.values():
            if hasattr(components, "set_default_attn_processor"):
                components.set_default_attn_processor()

        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs(torch_device)
        # Reset generator in case it is has been used in self.get_dummy_inputs
        inputs["generator"] = self.get_generator(0)

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # batchify inputs
        batched_inputs = {}
        batched_inputs.update(inputs)

        for name in self.batch_params:
            if name not in inputs:
                continue

            value = inputs[name]
            if name == "prompt":
                len_prompt = len(value)
                batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]
                batched_inputs[name][-1] = 100 * "very long"

            else:
                batched_inputs[name] = batch_size * [value]

        if "generator" in inputs:
            batched_inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]

        if "batch_size" in inputs:
            batched_inputs["batch_size"] = batch_size

        for arg in additional_params_copy_to_batched_inputs:
            batched_inputs[arg] = inputs[arg]

        output = pipe(**inputs)
        output_batch = pipe(**batched_inputs)

        assert output_batch[0].shape[0] == batch_size

        max_diff = np.abs(to_np(output_batch[0][0]) - to_np(output[0][0])).max()
        assert max_diff < expected_max_diff

    def test_dict_tuple_outputs_equivalent(self, expected_max_difference=1e-4):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        pipe.set_progress_bar_config(disable=None)

        generator_device = "cpu"
        output = pipe(**self.get_dummy_inputs(generator_device))[0]
        output_tuple = pipe(**self.get_dummy_inputs(generator_device), return_dict=False)[0]

        max_diff = np.abs(to_np(output) - to_np(output_tuple)).max()
        self.assertLess(max_diff, expected_max_difference)

    def test_components_function(self):
        init_components = self.get_dummy_components()

        pipe = self.pipeline_class(**init_components)
        init_components.pop("use_habana")
        init_components.pop("use_hpu_graphs")
        init_components.pop("bf16_full_eval")
        init_components.pop("gaudi_config")

        self.assertTrue(hasattr(pipe, "components"))
        self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))

    def test_to_dtype(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        model_dtypes = [component.dtype for component in components.values() if hasattr(component, "dtype")]
        self.assertTrue(all(dtype == torch.float32 for dtype in model_dtypes))

        pipe.to(dtype=torch.bfloat16)
        model_dtypes = [component.dtype for component in components.values() if hasattr(component, "dtype")]
        self.assertTrue(all(dtype == torch.bfloat16 for dtype in model_dtypes))

    def test_attention_slicing_forward_pass(self, expected_max_diff=1e-3):
        self._test_attention_slicing_forward_pass(expected_max_diff=expected_max_diff)

    def _test_attention_slicing_forward_pass(
        self, test_max_difference=True, test_mean_pixel_difference=True, expected_max_diff=1e-3
    ):
        if not self.test_attention_slicing:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.set_progress_bar_config(disable=None)

        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        output_without_slicing = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=1)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_slicing = pipe(**inputs)[0]

        if test_max_difference:
            max_diff = np.abs(to_np(output_with_slicing) - to_np(output_without_slicing)).max()
            self.assertLess(max_diff, expected_max_diff, "Attention slicing should not affect the inference results")

        if test_mean_pixel_difference:
            assert_mean_pixel_difference(to_np(output_with_slicing[0]), to_np(output_without_slicing[0]))

    def _test_xformers_attention_forwardGenerator_pass(
        self, test_max_difference=True, test_mean_pixel_difference=True, expected_max_diff=1e-4
    ):
        if not self.test_xformers_attention:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_without_offload = pipe(**inputs)[0]
        output_without_offload = (
            output_without_offload.cpu() if torch.is_tensor(output_without_offload) else output_without_offload
        )

        pipe.enable_xformers_memory_efficient_attention()
        inputs = self.get_dummy_inputs(torch_device)
        output_with_offload = pipe(**inputs)[0]
        output_with_offload = (
            output_with_offload.cpu() if torch.is_tensor(output_with_offload) else output_without_offload
        )

        if test_max_difference:
            max_diff = np.abs(to_np(output_with_offload) - to_np(output_without_offload)).max()
            self.assertLess(max_diff, expected_max_diff, "XFormers attention should not affect the inference results")

        if test_mean_pixel_difference:
            assert_mean_pixel_difference(output_with_offload[0], output_without_offload[0])

    def test_progress_bar(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        inputs = self.get_dummy_inputs(torch_device)
        with StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            stderr = stderr.getvalue()
            # we can't calculate the number of progress steps beforehand e.g. for strength-dependent img2img,
            # so we just match "5" in "#####| 1/5 [00:01<00:00]"
            max_steps = re.search("/(.*?) ", stderr).group(1)
            self.assertTrue(max_steps is not None and len(max_steps) > 0)
            self.assertTrue(
                f"{max_steps}/{max_steps}" in stderr, "Progress bar should be enabled and stopped at the max step"
            )

        pipe.set_progress_bar_config(disable=True)
        with StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            self.assertTrue(stderr.getvalue() == "", "Progress bar should be disabled")

    def test_num_images_per_prompt(self):
        sig = inspect.signature(self.pipeline_class.__call__)

        if "num_images_per_prompt" not in sig.parameters:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        batch_sizes = [1, 2]
        num_images_per_prompts = [1, 2]

        for batch_size in batch_sizes:
            for num_images_per_prompt in num_images_per_prompts:
                inputs = self.get_dummy_inputs(torch_device)

                for key in inputs.keys():
                    if key in self.batch_params:
                        inputs[key] = batch_size * [inputs[key]]

                images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt)[0]

                assert images.shape[0] == batch_size * num_images_per_prompt

    def test_cfg(self):
        sig = inspect.signature(self.pipeline_class.__call__)

        if "guidance_scale" not in sig.parameters:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)

        inputs["guidance_scale"] = 1.0
        out_no_cfg = pipe(**inputs)[0]

        inputs["guidance_scale"] = 7.5
        out_cfg = pipe(**inputs)[0]

        assert out_cfg.shape == out_no_cfg.shape

    def test_callback_inputs(self):
        sig = inspect.signature(self.pipeline_class.__call__)
        has_callback_tensor_inputs = "callback_on_step_end_tensor_inputs" in sig.parameters
        has_callback_step_end = "callback_on_step_end" in sig.parameters

        if not (has_callback_tensor_inputs and has_callback_step_end):
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        self.assertTrue(
            hasattr(pipe, "_callback_tensor_inputs"),
            f" {self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables its callback function can use as inputs",
        )

        def callback_inputs_subset(pipe, i, t, callback_kwargs):
            # interate over callback args
            for tensor_name, tensor_value in callback_kwargs.items():
                # check that we're only passing in allowed tensor inputs
                assert tensor_name in pipe._callback_tensor_inputs

            return callback_kwargs

        def callback_inputs_all(pipe, i, t, callback_kwargs):
            for tensor_name in pipe._callback_tensor_inputs:
                assert tensor_name in callback_kwargs

            # interate over callback args
            for tensor_name, tensor_value in callback_kwargs.items():
                # check that we're only passing in allowed tensor inputs
                assert tensor_name in pipe._callback_tensor_inputs

            return callback_kwargs

        inputs = self.get_dummy_inputs(torch_device)

        # Test passing in a subset
        inputs["callback_on_step_end"] = callback_inputs_subset
        inputs["callback_on_step_end_tensor_inputs"] = ["latents"]
        inputs["output_type"] = "latent"
        output = pipe(**inputs)[0]

        # Test passing in a everything
        inputs["callback_on_step_end"] = callback_inputs_all
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["output_type"] = "latent"
        output = pipe(**inputs)[0]

        def callback_inputs_change_tensor(pipe, i, t, callback_kwargs):
            is_last = i == (pipe.num_timesteps - 1)
            if is_last:
                callback_kwargs["latents"] = torch.zeros_like(callback_kwargs["latents"])
            return callback_kwargs

        inputs["callback_on_step_end"] = callback_inputs_change_tensor
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["output_type"] = "latent"
        output = pipe(**inputs)[0]
        assert output.abs().sum() == 0

    def test_callback_cfg(self):
        sig = inspect.signature(self.pipeline_class.__call__)
        has_callback_tensor_inputs = "callback_on_step_end_tensor_inputs" in sig.parameters
        has_callback_step_end = "callback_on_step_end" in sig.parameters

        if not (has_callback_tensor_inputs and has_callback_step_end):
            return

        if "guidance_scale" not in sig.parameters:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        self.assertTrue(
            hasattr(pipe, "_callback_tensor_inputs"),
            f" {self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables its callback function can use as inputs",
        )

        def callback_increase_guidance(pipe, i, t, callback_kwargs):
            pipe._guidance_scale += 1.0
            return callback_kwargs

        inputs = self.get_dummy_inputs(torch_device)

        # use cfg guidance because some pipelines modify the shape of the latents
        # outside of the denoising loop
        inputs["guidance_scale"] = 2.0
        inputs["callback_on_step_end"] = callback_increase_guidance
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        _ = pipe(**inputs)[0]

        # we increase the guidance scale by 1.0 at every step
        # check that the guidance scale is increased by the number of scheduler timesteps
        # accounts for models that modify the number of inference steps based on strength
        assert pipe.guidance_scale == (inputs["guidance_scale"] + pipe.num_timesteps)


# For SDXL and its derivative pipelines (such as ControlNet), we have the text encoders
# and the tokenizers as optional components. So, we need to override the `test_save_load_optional_components()`
# test for all such pipelines. This requires us to use a custom `encode_prompt()` function.
class SDXLOptionalComponentsTesterMixin:
    def encode_prompt(
        self, tokenizers, text_encoders, prompt: str, num_images_per_prompt: int = 1, negative_prompt: str = None
    ):
        device = text_encoders[0].device

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids

            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        if negative_prompt is None:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        else:
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            negative_prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(device), output_hidden_states=True)
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        bs_embed, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # for classifier-free guidance
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        # for classifier-free guidance
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _test_save_load_optional_components(self, expected_max_difference=1e-4):
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        pipe.set_progress_bar_config(disable=None)

        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)

        tokenizer = components.pop("tokenizer")
        tokenizer_2 = components.pop("tokenizer_2")
        text_encoder = components.pop("text_encoder")
        text_encoder_2 = components.pop("text_encoder_2")

        tokenizers = [tokenizer, tokenizer_2] if tokenizer is not None else [tokenizer_2]
        text_encoders = [text_encoder, text_encoder_2] if text_encoder is not None else [text_encoder_2]
        prompt = inputs.pop("prompt")
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(tokenizers, text_encoders, prompt)
        inputs["prompt_embeds"] = prompt_embeds
        inputs["negative_prompt_embeds"] = negative_prompt_embeds
        inputs["pooled_prompt_embeds"] = pooled_prompt_embeds
        inputs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            for component in pipe_loaded.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            self.assertTrue(
                getattr(pipe_loaded, optional_component) is None,
                f"`{optional_component}` did not stay set to None after loading.",
            )

        inputs = self.get_dummy_inputs(generator_device)
        _ = inputs.pop("prompt")
        inputs["prompt_embeds"] = prompt_embeds
        inputs["negative_prompt_embeds"] = negative_prompt_embeds
        inputs["pooled_prompt_embeds"] = pooled_prompt_embeds
        inputs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, expected_max_difference)


# Some models (e.g. unCLIP) are extremely likely to significantly deviate depending on which hardware is used.
# This helper function is used to check that the image doesn't deviate on average more than 10 pixels from a
# reference image.
def assert_mean_pixel_difference(image, expected_image, expected_max_diff=10):
    image = np.asarray(DiffusionPipeline.numpy_to_pil(image)[0], dtype=np.float32)
    expected_image = np.asarray(DiffusionPipeline.numpy_to_pil(expected_image)[0], dtype=np.float32)
    avg_diff = np.abs(image - expected_image).mean()
    assert avg_diff < expected_max_diff, f"Error image deviates {avg_diff} pixels on average"


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


enable_full_determinism()


class StableDiffusionInpaintPipelineTests(
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

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

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

    @slow
    def test_stable_diffusion_inpaint_no_throughput_regression(self):
        """Test that stable diffusion inpainting no throughput regression autocast"""

        # Initialize inpaint parameters
        init_image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
        )

        prompts = [
            "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k",
        ]
        model_name = "stabilityai/stable-diffusion-2-inpainting"
        num_images_per_prompt = 12
        batch_size = 4
        pipeline = GaudiStableDiffusionInpaintPipeline.from_pretrained(
            model_name,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            torch_dtype=torch.bfloat16,
            sdp_on_bf16=True,
        )

        set_seed(27)
        outputs = pipeline(
            prompt=prompts,
            image=init_image,
            mask_image=mask_image,
            num_images_per_prompt=num_images_per_prompt,
            batch_size=batch_size,
            throughput_warmup_steps=3,
        )

        # Check expected number of output images
        self.assertEqual(len(outputs.images), num_images_per_prompt * len(prompts))

        # Throughput regression test
        self.baseline.assertRef(
            compare=lambda actual, ref: actual >= (0.95 * ref),
            context=[OH_DEVICE_CONTEXT],
            throughput=outputs.throughput,
        )


class StableDiffusionXLInpaintPipelineTests(PipelineLatentTesterMixin, PipelineTesterMixin, TestCase):
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

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

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

        expected_slice = np.array([0.8279, 0.5673, 0.6088, 0.6156, 0.6923, 0.7347, 0.6547, 0.6108, 0.5198])

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

        expected_slice = np.array([0.7540, 0.5231, 0.5833, 0.6217, 0.6339, 0.7067, 0.6507, 0.5672, 0.5030])

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

        # Initialize inpaint parameters
        init_image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
        )

        prompts = [
            "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k",
        ]
        model_name = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        num_images_per_prompt = 12
        batch_size = 4
        pipeline = GaudiStableDiffusionXLInpaintPipeline.from_pretrained(
            model_name,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            torch_dtype=torch.bfloat16,
            sdp_on_bf16=True,
        )

        set_seed(27)
        outputs = pipeline(
            prompt=prompts,
            image=init_image,
            mask_image=mask_image,
            num_images_per_prompt=num_images_per_prompt,
            batch_size=batch_size,
            throughput_warmup_steps=3,
        )

        # Check expected number of output images
        self.assertEqual(len(outputs.images), num_images_per_prompt * len(prompts))

        # Throughput regression test
        self.baseline.assertRef(
            compare=lambda actual, ref: actual >= (0.95 * ref),
            context=[OH_DEVICE_CONTEXT],
            throughput=outputs.throughput,
        )


class GaudiDDPMPipelineTester(TestCase):
    """
    Tests for unconditional image generation
    """

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DModel(
            sample_size=256,
            in_channels=3,
            out_channels=3,
            center_input_sample=False,
            time_embedding_type="positional",
            freq_shift=1,
            flip_sin_to_cos=False,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=(128, 128, 256, 256, 512, 512),
            downsample_padding=1,
            norm_eps=1e-6,
        )
        scheduler = GaudiDDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        components = {
            "unet": unet,
            "scheduler": scheduler,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "generator": generator,
            "num_inference_steps": 2,
            "batch_size": 8,
        }
        return inputs

    def test_ddpmpipline_default(self):
        device = "cpu"

        components = self.get_dummy_components()
        gaudi_config = GaudiConfig(use_torch_autocast=False)

        pipe = GaudiDDPMPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = pipe(**inputs)
        image = output.images[0]
        image = np.array(image)
        image_slice = image[-3:, -3:, -1]

        self.assertEqual(image.shape, (256, 256, 3))
        expected_slice = np.array([255, 0, 138, 139, 255, 36, 164, 0, 255])
        self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1)

    def test_ddpmpipline_batch_sizes(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        pipe = GaudiDDPMPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        pipe.set_progress_bar_config(disable=None)

        batch_size = 2
        images = pipe(
            num_inference_steps=2,
            batch_size=batch_size,
        ).images

        self.assertEqual(len(images), batch_size)
        self.assertEqual(np.array(images[-1]).shape, (256, 256, 3))

    def test_ddpmpipline_bf16(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig(use_torch_autocast=True)

        pipe = GaudiDDPMPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
            **components,
        )
        pipe.set_progress_bar_config(disable=None)
        generator = torch.Generator(device="cpu").manual_seed(0)
        image = pipe(generator=generator, num_inference_steps=2, batch_size=1).images[0]

        self.assertEqual(np.array(image).shape, (256, 256, 3))

    def test_ddpmpipline_hpu_graphs(self):
        components = self.get_dummy_components()
        gaudi_config = GaudiConfig()

        pipe = GaudiDDPMPipeline(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=gaudi_config,
            **components,
        )
        pipe.set_progress_bar_config(disable=None)
        generator = torch.Generator(device="cpu").manual_seed(0)
        images = pipe(
            generator=generator,
            num_inference_steps=2,
            batch_size=1,
        ).images

        self.assertEqual(len(images), 1)
        self.assertEqual(np.array(images[-1]).shape, (256, 256, 3))

    @slow
    def test_no_throughput_regression_bf16(self):
        batch_size = 16  # use batch size 16 as the baseline
        model_name = "google/ddpm-ema-celebahq-256"
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name)
        gaudi_config = GaudiConfig(use_torch_autocast=True)

        pipe = GaudiDDPMPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=gaudi_config,
            sdp_on_bf16=True,
        )
        outputs = pipe(batch_size=batch_size)
        self.baseline.assertRef(
            compare=lambda actual, ref: actual >= (0.95 * ref),
            context=[OH_DEVICE_CONTEXT],
            throughput=outputs.throughput,
        )


class GaudiFluxPipelineTester(TestCase):
    """
    Tests the Flux pipeline for Gaudi.
    """

    pipeline_class = GaudiFluxPipeline
    params = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "prompt_embeds",
            "pooled_prompt_embeds",
        ]
    )
    batch_params = frozenset(["prompt"])

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
        )
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )

        torch.manual_seed(0)
        text_encoder = CLIPTextModel(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        # HF issue with T5EncoderModel from tiny-random-t5
        # text_encoder_3 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        # text_encoder_3 = None

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "transformer": transformer,
            "vae": vae,
        }

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 48,
            "output_type": "np",
        }
        return inputs

    def test_flux_different_prompts(self):
        pipe = self.pipeline_class(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            **self.get_dummy_components(),
        ).to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt_2"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = np.abs(output_same_prompt - output_different_prompts).max()

        # Outputs should be different here
        # For some reasons, they don't show large differences
        assert max_diff > 1e-6

    def test_flux_prompt_embeds(self):
        pipe = self.pipeline_class(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            **self.get_dummy_components(),
        ).to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        output_with_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = inputs.pop("prompt")

        (prompt_embeds, pooled_prompt_embeds, text_ids) = pipe.encode_prompt(
            prompt,
            prompt_2=None,
            device=torch_device,
            max_sequence_length=inputs["max_sequence_length"],
        )
        output_with_embeds = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            **inputs,
        ).images[0]

        max_diff = np.abs(output_with_prompt - output_with_embeds).max()
        assert max_diff < 1e-4

    @slow
    @pytest.mark.skipif(IS_GAUDI1, reason="does not fit into Gaudi1 memory")
    def test_flux_inference(self):
        prompts = [
            "A cat holding a sign that says hello world",
        ]
        num_images_per_prompt = 10
        batch_size = 1
        model_name = "black-forest-labs/FLUX.1-schnell"
        pipeline = GaudiFluxPipeline.from_pretrained(
            model_name,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            sdp_on_bf16=True,
        )
        set_seed(27)
        outputs = pipeline(
            prompt=prompts,
            num_images_per_prompt=num_images_per_prompt,
            batch_size=batch_size,
            num_inference_steps=4,
        )

        # Check expected number of output images
        self.assertEqual(len(outputs.images), num_images_per_prompt * len(prompts))

        # Check expected performance of FLUX.1 schnell model
        self.baseline.assertRef(
            compare=lambda actual, ref: actual >= (0.95 * ref),
            context=[OH_DEVICE_CONTEXT],
            throughput=outputs.throughput,
        )


class GaudiFluxImg2ImgPipelineTester(TestCase):
    """
    Tests the Flux image-to-image pipeline for Gaudi.
    """

    pipeline_class = GaudiFluxImg2ImgPipeline
    params = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "prompt_embeds",
            "pooled_prompt_embeds",
        ]
    )
    batch_params = frozenset(["prompt"])

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
        )
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )

        torch.manual_seed(0)
        text_encoder = CLIPTextModel(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "transformer": transformer,
            "vae": vae,
        }

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 48,
            "output_type": "np",
        }
        return inputs

    def test_flux_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt_2"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = np.abs(output_same_prompt - output_different_prompts).max()

        # Outputs should be different here
        # For some reasons, they don't show large differences
        assert max_diff > 1e-6

    def test_flux_prompt_embeds(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        output_with_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = inputs.pop("prompt")

        (prompt_embeds, pooled_prompt_embeds, text_ids) = pipe.encode_prompt(
            prompt,
            prompt_2=None,
            device=torch_device,
            max_sequence_length=inputs["max_sequence_length"],
        )
        output_with_embeds = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            **inputs,
        ).images[0]

        max_diff = np.abs(output_with_prompt - output_with_embeds).max()
        assert max_diff < 1e-4

    @slow
    @check_gated_model_access("black-forest-labs/FLUX.1-dev")
    @pytest.mark.skipif(IS_GAUDI1, reason="does not fit into Gaudi1 memory")
    def test_flux_img2img_inference(self):
        repo_id = "black-forest-labs/FLUX.1-dev"
        image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
        image = PIL.Image.open(requests.get(image_path, stream=True).raw)
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        pipe = self.pipeline_class.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
            sdp_on_bf16=True,
        )

        outputs = pipe(
            image=image,
            prompt="cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",
            num_inference_steps=30,
            guidance_scale=3.5,
            strength=0.9,
            batch_size=1,
            num_images_per_prompt=10,
            output_type="np",
        )

        # Check expected performance of FLUX.1 dev img-to-img model
        self.baseline.assertRef(
            compare=lambda actual, ref: actual >= (0.95 * ref),
            context=[OH_DEVICE_CONTEXT],
            throughput=outputs.throughput,
        )


class I2VGenXLPipelineTests(TestCase):
    pipeline_class = GaudiI2VGenXLPipeline
    params = frozenset(["prompt", "negative_prompt", "image"])
    batch_params = frozenset(["prompt", "negative_prompt", "image", "generator"])
    # No `output_type`.
    required_optional_params = frozenset(["num_inference_steps", "generator", "latents", "return_dict"])

    supports_dduf = False
    test_layerwise_casting = True

    @pytest.fixture(autouse=True)
    def _use_(self, baseline):
        """
        https://docs.pytest.org/en/stable/how-to/unittest.html#using-autouse-fixtures-and-accessing-other-fixtures
        """
        self.baseline = baseline

    def get_dummy_components(self):
        torch.manual_seed(0)
        scheduler = GaudiDDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

        torch.manual_seed(0)
        unet = I2VGenXLUNet(
            block_out_channels=(4, 8),
            layers_per_block=1,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock3D", "DownBlock3D"),
            up_block_types=("UpBlock3D", "CrossAttnUpBlock3D"),
            cross_attention_dim=4,
            attention_head_dim=4,
            num_attention_heads=None,
            norm_num_groups=2,
        )

        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=(8,),
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=32,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=4,
            intermediate_size=16,
            layer_norm_eps=1e-05,
            num_attention_heads=2,
            num_hidden_layers=2,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        torch.manual_seed(0)
        vision_encoder_config = CLIPVisionConfig(
            hidden_size=4,
            projection_dim=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=32,
            intermediate_size=16,
            patch_size=1,
        )
        image_encoder = CLIPVisionModelWithProjection(vision_encoder_config)

        torch.manual_seed(0)
        feature_extractor = CLIPImageProcessor(crop_size=32, size=32)

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "image_encoder": image_encoder,
            "tokenizer": tokenizer,
            "feature_extractor": feature_extractor,
            "use_habana": True,
            "use_hpu_graphs": True,
            "gaudi_config": "Habana/stable-diffusion",
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        input_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": input_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "pt",
            "num_frames": 4,
            "width": 32,
            "height": 32,
        }
        return inputs

    def test_cfg(self):
        sig = inspect.signature(self.pipeline_class.__call__)

        if "guidance_scale" not in sig.parameters:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device="cpu")

        inputs["guidance_scale"] = 1.0
        out_no_cfg = pipe(**inputs)[0]

        inputs["guidance_scale"] = 7.5
        out_cfg = pipe(**inputs)[0]

        assert out_cfg[0].shape == out_no_cfg[0].shape

    def test_text_to_video_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["output_type"] = "np"
        frames = pipe(**inputs).frames

        image_slice = frames[0][0][-3:, -3:, -1]

        assert frames[0][0].shape == (32, 32, 3)
        expected_slice = np.array([0.5146, 0.6525, 0.6032, 0.5204, 0.5675, 0.4125, 0.3016, 0.5172, 0.4095])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_num_videos_per_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["output_type"] = "np"
        frames = pipe(**inputs, num_videos_per_prompt=2).frames

        assert frames.shape == (2, 4, 32, 32, 3)
        assert frames[0][0].shape == (32, 32, 3)

        image_slice = frames[0][0][-3:, -3:, -1]
        expected_slice = np.array([0.5146, 0.6525, 0.6032, 0.5204, 0.5675, 0.4125, 0.3016, 0.5172, 0.4095])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @slow
    def test_i2vgen_xl_bf16(self):
        pipe = GaudiI2VGenXLPipeline.from_pretrained(
            "ali-vilab/i2vgen-xl",
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig.from_pretrained("Habana/stable-diffusion"),
            sdp_on_bf16=True,
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png?download=true"
        )

        generator = torch.Generator("cpu").manual_seed(0)
        num_frames = 3

        output = pipe(
            image=image,
            prompt="my cat",
            num_frames=num_frames,
            generator=generator,
            num_inference_steps=50,
            output_type="np",
        )

        image = output.frames[0]
        assert image.shape == (num_frames, 704, 1280, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array(
            [0.44921875, 0.3642578, 0.38671875, 0.46484375, 0.41210938, 0.45874023, 0.49536133, 0.4387207, 0.48242188]
        )
        assert numpy_cosine_similarity_distance(image_slice.flatten(), expected_slice.flatten()) < 1e-3

        self.baseline.assertRef(
            compare=lambda actual, ref: actual >= (0.95 * ref),
            context=[OH_DEVICE_CONTEXT],
            throughput=output.throughput,
        )
