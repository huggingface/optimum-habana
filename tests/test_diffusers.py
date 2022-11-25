import os
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest
import torch

from diffusers import AutoencoderKL, UNet2DConditionModel, UNet2DModel, VQModel
from habana_frameworks.torch.hpex import hmp
from optimum.habana import GaudiConfig
from optimum.habana.diffusers import GaudiDiffusionPipeline, GaudiStableDiffusionPipeline
from optimum.habana.diffusers.schedulers import GaudiDDIMScheduler
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer


class GaudiPipelineUtilsTester(TestCase):
    """
    Tests the features added on top of diffusers/pipeline_utils.py.
    """

    def test_use_lazy_mode_is_false_without_habana(self):
        pipeline = GaudiDiffusionPipeline(
            use_habana=False,
            use_lazy_mode=True,
        )
        self.assertFalse(pipeline.use_lazy_mode)

    def test_use_hpu_graphs_is_false_without_habana(self):
        pipeline = GaudiDiffusionPipeline(
            use_habana=False,
            use_hpu_graphs=True,
        )
        self.assertFalse(pipeline.use_hpu_graphs)

    def test_gaudi_config_is_none_without_habana(self):
        pipeline = GaudiDiffusionPipeline(
            use_habana=False,
            gaudi_config=GaudiConfig(),
        )
        self.assertIsNone(pipeline.gaudi_config)

    def test_use_lazy_mode_and_use_hpu_graphs_are_exclusive(self):
        with self.assertRaises(ValueError) as e:
            _ = GaudiDiffusionPipeline(
                use_habana=True,
                use_lazy_mode=True,
                use_hpu_graphs=True,
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

    def test_use_hpu_graphs(self):
        pipeline = GaudiDiffusionPipeline(
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=GaudiConfig(),
        )

        self.assertTrue(hasattr(pipeline, "ht"))
        self.assertTrue(hasattr(pipeline, "hpu_graph"))
        self.assertTrue(hasattr(pipeline, "hpu_stream"))
        self.assertTrue(hasattr(pipeline, "static_inputs"))
        self.assertTrue(hasattr(pipeline, "static_outputs"))

    def test_use_lazy_mode(self):
        pipeline = GaudiDiffusionPipeline(
            use_habana=True,
            use_lazy_mode=True,
            gaudi_config=GaudiConfig(),
        )

        self.assertTrue(hasattr(pipeline, "htcore"))

    def test_habana_mixed_precision(self):
        gaudi_config = GaudiConfig(
            use_habana_mixed_precision=True,
        )

        pipeline = GaudiDiffusionPipeline(
            use_habana=True,
            gaudi_config=gaudi_config,
        )

        self.assertTrue(hasattr(pipeline, "hmp"))

    def test_save_pretrained(self):
        model_name = "hf-internal-testing/tiny-stable-diffusion-lms-pipe"
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            use_habana=True,
            use_lazy_mode=False,
            use_hpu_graphs=False,
            gaudi_config=GaudiConfig(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)
            self.assertTrue(Path(tmp_dir, "gaudi_config.json").is_file())


class GaudiStableDiffusionPipelineTester(TestCase):
    """
    Tests the StableDiffusionPipeline for Gaudi.
    """

    @property
    def dummy_uncond_unet(self):
        torch.manual_seed(0)
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        return model

    @property
    def dummy_cond_unet(self):
        torch.manual_seed(0)
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        return model

    @property
    def dummy_cond_unet_inpaint(self):
        torch.manual_seed(0)
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=9,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        return model

    @property
    def dummy_vq_model(self):
        torch.manual_seed(0)
        model = VQModel(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=3,
        )
        return model

    @property
    def dummy_vae(self):
        torch.manual_seed(0)
        model = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        return model

    @property
    def dummy_text_encoder(self):
        torch.manual_seed(0)
        config = CLIPTextConfig(
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
        return CLIPTextModel(config)

    @property
    def dummy_extractor(self):
        def extract(*args, **kwargs):
            class Out:
                def __init__(self):
                    self.pixel_values = torch.ones([0])

                def to(self, device):
                    self.pixel_values.to(device)
                    return self

            return Out()

        return extract

    def test_stable_diffusion_ddim(self):
        with hmp.disable_casts():
            device = "cpu"
            unet = self.dummy_cond_unet
            scheduler = GaudiDDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
            vae = self.dummy_vae
            bert = self.dummy_text_encoder
            tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

            gaudi_config = GaudiConfig(use_habana_mixed_precision=False)

            sd_pipe = GaudiStableDiffusionPipeline(
                unet=unet,
                scheduler=scheduler,
                vae=vae,
                text_encoder=bert,
                tokenizer=tokenizer,
                safety_checker=None,
                feature_extractor=self.dummy_extractor,
                use_habana=True,
                use_lazy_mode=False,
                use_hpu_graphs=False,
                gaudi_config=gaudi_config,
            )
            sd_pipe.set_progress_bar_config(disable=None)

            prompt = "A painting of a squirrel eating a burger"

            generator = torch.Generator(device=device).manual_seed(0)
            output = sd_pipe(
                [prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="np"
            )
            image = output.images[0]

            generator = torch.Generator(device=device).manual_seed(0)
            image_from_tuple = sd_pipe(
                [prompt],
                generator=generator,
                guidance_scale=6.0,
                num_inference_steps=2,
                output_type="np",
                return_dict=False,
            )[0][0]

            image_slice = image[-3:, -3:, -1]
            image_from_tuple_slice = image_from_tuple[-3:, -3:, -1]

            self.assertEqual(image.shape, (64, 64, 3))
            expected_slice = np.array(
                [
                    0.5643956661224365,
                    0.6017904281616211,
                    0.4799129366874695,
                    0.5267305374145508,
                    0.5584856271743774,
                    0.46413588523864746,
                    0.5159522294998169,
                    0.4963662028312683,
                    0.47919973731040955,
                ]
            )

            self.assertLess(np.abs(image_slice.flatten() - expected_slice).max(), 1e-2)
            self.assertLess(np.abs(image_from_tuple_slice.flatten() - expected_slice).max(), 1e-2)

    def test_stable_diffusion_no_safety_checker(self):
        gaudi_config = GaudiConfig(use_habana_mixed_precision=False)
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
            use_lazy_mode=False,
            use_hpu_graphs=False,
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
                use_lazy_mode=False,
                use_hpu_graphs=False,
                gaudi_config=tmpdirname,
            )

        # Sanity check that the pipeline still works
        self.assertIsNone(pipe.safety_checker)
        image = pipe("example prompt", num_inference_steps=2).images[0]
        self.assertIsNotNone(image)

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
        unet = self.dummy_cond_unet
        scheduler = GaudiDDIMScheduler()
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
            use_habana=True,
            use_lazy_mode=False,
            use_hpu_graphs=False,
            gaudi_config=gaudi_config,
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
        unet = self.dummy_cond_unet
        scheduler = GaudiDDIMScheduler()
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
            use_habana=True,
            use_lazy_mode=False,
            use_hpu_graphs=False,
            gaudi_config=gaudi_config,
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

        # Test batch_size when it is not a divider of the toal number of generated images for a single prompt
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
        unet = self.dummy_cond_unet
        scheduler = GaudiDDIMScheduler()
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        gaudi_config = GaudiConfig()

        sd_pipe = GaudiStableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
            use_habana=True,
            use_lazy_mode=False,
            use_hpu_graphs=False,
            gaudi_config=gaudi_config,
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.Generator(device="cpu").manual_seed(0)
        image = sd_pipe([prompt], generator=generator, num_inference_steps=2, output_type="np").images[0]

        self.assertEqual(image.shape, (64, 64, 3))

    def test_stable_diffusion_lazy_mode(self):
        # Skip this test if PT_HPU_LAZY_MODE=2
        if os.environ.get("PT_HPU_LAZY_MODE", 1) == "2":
            pytest.skip(
                "Skipping this test because the environment variable `PT_HPU_LAZY_MODE` has already been declared and"
                " is equal to '2', which is not compatible with lazy mode and HPU graphs."
            )

        unet = self.dummy_cond_unet
        scheduler = GaudiDDIMScheduler()
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        sd_pipe = GaudiStableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
            use_habana=True,
            use_lazy_mode=True,
            use_hpu_graphs=False,
            gaudi_config="Habana/stable-diffusion",
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
        # Skip this test if PT_HPU_LAZY_MODE=2
        if os.environ.get("PT_HPU_LAZY_MODE", 1) == "2":
            pytest.skip(
                "Skipping this test because the environment variable `PT_HPU_LAZY_MODE` has already been declared and"
                " is equal to '2', which is not compatible with lazy mode and HPU graphs."
            )

        unet = self.dummy_cond_unet
        scheduler = GaudiDDIMScheduler()
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        sd_pipe = GaudiStableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
            use_habana=True,
            use_lazy_mode=False,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion",
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
