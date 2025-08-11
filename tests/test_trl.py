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

import gc
import importlib.metadata
import tempfile
import unittest

import torch
from datasets import load_dataset
from packaging import version
from parameterized import parameterized
from transformers.testing_utils import require_peft, slow
from transformers.utils import is_peft_available
from trl import DDPOConfig

from optimum.habana import GaudiConfig


trl_version = importlib.metadata.version("trl")
if version.parse(trl_version) < version.parse("0.17.0"):
    from optimum.habana.trl import (
        GaudiDDPOTrainer,
        GaudiDefaultDDPOStableDiffusionPipeline,
    )
else:
    from optimum.habana.trl import (
        GaudiGRPOConfig,
        GaudiGRPOTrainer,
    )

if is_peft_available():
    from peft import LoraConfig, PeftModel


def scorer_function(images, prompts, metadata):
    return torch.randn(1) * 3.0, {}


def prompt_function():
    return ("cabbages", {})


class GaudiDDPOTrainerTester(unittest.TestCase):
    """
    Test the GaudiDDPOTrainer class.

    Adapted from https://github.com/huggingface/trl/blob/main/tests/test_ddpo_trainer.py
    The main changes are:
     - use GaudiDefaultDDPOStableDiffusionPipeline instead of DefaultDDPOStableDiffusionPipeline
     - use GaudiDDPOTrainer instead of DDPOTrainer
     - use bf16 instead of fp32
     - combine test_generate_samples and test_calculate_loss in single test
    """

    def setUp(self):
        self.ddpo_config = DDPOConfig(
            num_epochs=2,
            train_gradient_accumulation_steps=1,
            per_prompt_stat_tracking_buffer_size=32,
            sample_num_batches_per_epoch=2,
            sample_batch_size=2,
            mixed_precision=None,
            save_freq=1000000,
        )
        pretrained_model = "hf-internal-testing/tiny-stable-diffusion-torch"
        pretrained_revision = "main"

        gaudi_config = GaudiConfig()
        pipeline = GaudiDefaultDDPOStableDiffusionPipeline(
            pretrained_model,
            pretrained_model_revision=pretrained_revision,
            use_lora=True,
            gaudi_config=gaudi_config,
            use_habana=True,
            use_hpu_graphs=False,
        )

        self.trainer = GaudiDDPOTrainer(
            self.ddpo_config,
            scorer_function,
            prompt_function,
            pipeline,
            gaudi_config=gaudi_config,
            use_habana=True,
            use_hpu_graphs=False,
        )

        return super().setUp()

    def tearDown(self) -> None:
        gc.collect()

    def test_loss(self):
        advantage = torch.tensor([-1.0])
        clip_range = 0.0001
        ratio = torch.tensor([1.0])
        loss = self.trainer.loss(advantage, clip_range, ratio)
        assert loss.item() == 1.0

    @slow
    def test_calculate_loss(self):
        samples, output_pairs = self.trainer._generate_samples(1, 2)
        assert len(samples) == 1
        assert len(output_pairs) == 1
        assert len(output_pairs[0][0]) == 2

        sample = samples[0]
        latents = sample["latents"][0, 0].unsqueeze(0)
        next_latents = sample["next_latents"][0, 0].unsqueeze(0)
        log_probs = sample["log_probs"][0, 0].unsqueeze(0)
        timesteps = sample["timesteps"][0, 0].unsqueeze(0)
        prompt_embeds = sample["prompt_embeds"]
        advantage = torch.tensor([1.0], device=prompt_embeds.device)

        assert latents.shape == (1, 4, 64, 64)
        assert next_latents.shape == (1, 4, 64, 64)
        assert log_probs.shape == (1,)
        assert timesteps.shape == (1,)
        assert prompt_embeds.shape == (2, 77, 32)
        loss, approx_kl, clipfrac = self.trainer.calculate_loss(
            latents, timesteps, next_latents, log_probs, advantage, prompt_embeds
        )

        assert torch.isfinite(loss.cpu())


class GaudiDDPOTrainerWithLoRATester(GaudiDDPOTrainerTester):
    """
    Test the GaudiDDPOTrainer class.
    """

    def setUp(self):
        self.ddpo_config = DDPOConfig(
            num_epochs=2,
            train_gradient_accumulation_steps=1,
            per_prompt_stat_tracking_buffer_size=32,
            sample_num_batches_per_epoch=2,
            sample_batch_size=2,
            mixed_precision=None,
            save_freq=1000000,
        )
        pretrained_model = "hf-internal-testing/tiny-stable-diffusion-torch"
        pretrained_revision = "main"

        gaudi_config = GaudiConfig()
        pipeline = GaudiDefaultDDPOStableDiffusionPipeline(
            pretrained_model,
            pretrained_model_revision=pretrained_revision,
            use_lora=True,
            gaudi_config=gaudi_config,
            use_habana=True,
            use_hpu_graphs=False,
        )

        self.trainer = GaudiDDPOTrainer(
            self.ddpo_config,
            scorer_function,
            prompt_function,
            pipeline,
            gaudi_config=gaudi_config,
            use_habana=True,
            use_hpu_graphs=False,
        )

        return super().setUp()


class GaudiGRPOTrainerTester(unittest.TestCase):
    """
    Test the GaudiGRPOTrainer class.

    Adapted from https://github.com/huggingface/trl/blob/main/tests/test_grpo_trainer.py#L216
    The main changes are:
     - use GaudiGRPOConfig and GaudiGRPOTrainer instead of GRPOConfig and GRPOTrainer
     - add GaudiConfig
    """

    def test_init_minimal(self):
        # Test that GRPOTrainer can be instantiated with only model, reward_model and train_dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GaudiGRPOConfig(
            use_habana=True,
            use_lazy_mode=True,
        )
        gaudi_config = GaudiConfig()

        GaudiGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            gaudi_config=gaudi_config,
        )

    @parameterized.expand([("standard_prompt_only",), ("conversational_prompt_only",)])
    def test_training(self, config_name):
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        gaudi_config = GaudiConfig()

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GaudiGRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
                use_habana=True,
                use_lazy_mode=True,
            )
            trainer = GaudiGRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
                gaudi_config=gaudi_config,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_peft
    def test_training_peft(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        gaudi_config = GaudiConfig()

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GaudiGRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
                use_habana=True,
                use_lazy_mode=True,
            )
            trainer = GaudiGRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
                peft_config=LoraConfig(),
                gaudi_config=gaudi_config,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the peft params have changed and the base model params have not changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if "lora" in n.lower():  # We expect the lora params to be different
                    self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed.")
                else:  # We expect the rest of params to be the same
                    self.assertTrue(torch.allclose(param, new_param), f"Parameter {n} has changed.")

    @require_peft
    def test_training_peft_with_gradient_checkpointing(self):
        """Test that training works with PEFT and gradient checkpointing enabled."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        gaudi_config = GaudiConfig()

        lora_config = LoraConfig(
            r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GaudiGRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                gradient_checkpointing=True,  # Enable gradient checkpointing
                report_to="none",
                use_habana=True,
                use_lazy_mode=True,
            )
            trainer = GaudiGRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
                peft_config=lora_config,
                gaudi_config=gaudi_config,
            )

            # Verify gradient checkpointing is enabled
            self.assertIsInstance(trainer.model, PeftModel)

            # Store initial parameters to check which ones change
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that only LoRA parameters have changed, base model parameters remain unchanged
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if "lora" in n.lower():  # LoRA parameters should change
                    self.assertFalse(torch.equal(param, new_param), f"LoRA parameter {n} has not changed.")
                else:  # Base model parameters should not change
                    self.assertTrue(torch.equal(param, new_param), f"Base parameter {n} has changed.")
