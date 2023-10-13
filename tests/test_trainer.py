# coding=utf-8
# Copyright 2022 the HuggingFace Inc. team.
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

import dataclasses
import json
import math
import os
import random
import re
import subprocess
import tempfile
import unittest
from itertools import product
from pathlib import Path
from typing import Optional, Union

import numpy as np
from huggingface_hub import HfFolder, delete_repo, list_repo_commits
from parameterized import parameterized
from requests.exceptions import HTTPError
from transformers import IntervalStrategy, PretrainedConfig, is_torch_available
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS
from transformers.testing_utils import (
    ENDPOINT_STAGING,
    TOKEN,
    USER,
    CaptureLogger,
    TestCasePlus,
    get_gpu_count,
    get_tests_dir,
    is_staging_test,
    require_optuna,
    require_safetensors,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, HPSearchBackend
from transformers.training_args import OptimizerNames
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    is_safetensors_available,
)
from transformers.utils.hp_naming import TrialShortNamer

from optimum.habana import GaudiConfig, GaudiTrainingArguments
from optimum.utils import logging


if is_torch_available():
    import torch
    import transformers.optimization
    from torch import nn
    from torch.utils.data import IterableDataset
    from transformers import EarlyStoppingCallback, GPT2Config, GPT2LMHeadModel, PreTrainedModel, TrainerState
    from transformers.modeling_utils import unwrap_model

    from optimum.habana import GaudiTrainer

    if is_safetensors_available():
        import safetensors.torch


PATH_SAMPLE_TEXT = f"{get_tests_dir()}/fixtures/sample_text.txt"


class RegressionDataset:
    def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
        np.random.seed(seed)
        self.label_names = ["labels"] if label_names is None else label_names
        self.length = length
        self.x = np.random.normal(size=(length,)).astype(np.float32)
        self.ys = [a * self.x + b + np.random.normal(scale=0.1, size=(length,)) for _ in self.label_names]
        self.ys = [y.astype(np.float32) for y in self.ys]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        result = {name: y[i] for name, y in zip(self.label_names, self.ys)}
        result["input_x"] = self.x[i]
        return result


@dataclasses.dataclass
class RegressionGaudiTrainingArguments(GaudiTrainingArguments):
    a: float = 0.0
    b: float = 0.0

    def __post_init__(self):
        # save resources not dealing with reporting (also avoids the warning when it's not set)
        self.report_to = []
        super().__post_init__()


class RepeatDataset:
    def __init__(self, x, length=64):
        self.x = x
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"input_ids": self.x, "labels": self.x}


class AlmostAccuracy:
    def __init__(self, thresh=0.25):
        self.thresh = thresh

    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        true = np.abs(predictions - labels) <= self.thresh
        return {"accuracy": true.astype(np.float32).mean().item()}


class RegressionModelConfig(PretrainedConfig):
    def __init__(self, a=0, b=0, double_output=False, random_torch=True, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.double_output = double_output
        self.random_torch = random_torch
        self.hidden_size = 1


if is_torch_available():

    class SampleIterableDataset(IterableDataset):
        def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
            self.dataset = RegressionDataset(a=a, b=b, length=length, seed=seed, label_names=label_names)

        def __iter__(self):
            for i in range(len(self.dataset)):
                yield self.dataset[i]

    class FiniteIterableDataset(SampleIterableDataset):
        def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
            super().__init__(a, b, length, seed, label_names)
            self.current_sample = 0

        def __iter__(self):
            while self.current_sample < len(self.dataset):
                yield self.dataset[self.current_sample]
                self.current_sample += 1

    class MultiLoader:
        def __init__(self, loaders):
            self.loaders = loaders

        def __len__(self):
            return sum(len(loader) for loader in self.loaders)

        def __iter__(self):
            for loader in self.loaders:
                yield from loader

    class CustomDataloaderTrainer(GaudiTrainer):
        def get_train_dataloader(self):
            dataloaders = [super().get_train_dataloader(), super().get_train_dataloader()]
            return MultiLoader(dataloaders)

        def get_eval_dataloader(self, eval_dataset):
            dataloaders = [super().get_eval_dataloader(eval_dataset), super().get_eval_dataloader(eval_dataset)]
            return MultiLoader(dataloaders)

    class RegressionModel(nn.Module):
        def __init__(self, a=0, b=0, double_output=False):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(a).float())
            self.b = nn.Parameter(torch.tensor(b).float())
            self.double_output = double_output
            self.config = None

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            if labels is None:
                return (y, y) if self.double_output else (y,)
            loss = nn.functional.mse_loss(y, labels)
            return (loss, y, y) if self.double_output else (loss, y)

    class RegressionDictModel(nn.Module):
        def __init__(self, a=0, b=0):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(a).float())
            self.b = nn.Parameter(torch.tensor(b).float())
            self.config = None

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            result = {"output": y}
            if labels is not None:
                result["loss"] = nn.functional.mse_loss(y, labels)
            return result

    class RegressionPreTrainedModel(PreTrainedModel):
        config_class = RegressionModelConfig
        base_model_prefix = "regression"

        def __init__(self, config):
            super().__init__(config)
            self.a = nn.Parameter(torch.tensor(config.a).float(), requires_grad=True)
            self.b = nn.Parameter(torch.tensor(config.b).float(), requires_grad=True)
            self.double_output = config.double_output

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            if labels is None:
                return (y, y) if self.double_output else (y,)
            loss = nn.functional.mse_loss(y, labels)
            return (loss, y, y) if self.double_output else (loss, y)

    class RegressionRandomPreTrainedModel(PreTrainedModel):
        config_class = RegressionModelConfig
        base_model_prefix = "regression"

        def __init__(self, config):
            super().__init__(config)
            self.a = nn.Parameter(torch.tensor(config.a).float())
            self.b = nn.Parameter(torch.tensor(config.b).float())
            self.random_torch = config.random_torch

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x * self.a + self.b
            if self.random_torch:
                torch_rand = torch.randn(1).squeeze()
            np_rand = np.random.rand()
            rand_rand = random.random()

            if self.random_torch:
                y += 0.05 * torch_rand
            y += 0.05 * torch.tensor(np_rand + rand_rand)

            if labels is None:
                return (y,)
            loss = nn.functional.mse_loss(y, labels)
            return (loss, y)

    class TstLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear1 = nn.Linear(hidden_size, hidden_size)
            self.ln1 = nn.LayerNorm(hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.ln2 = nn.LayerNorm(hidden_size)
            self.bias = nn.Parameter(torch.zeros(hidden_size))

        def forward(self, x):
            h = self.ln1(nn.functional.relu(self.linear1(x)))
            h = nn.functional.relu(self.linear2(x))
            return self.ln2(x + h + self.bias)

    def get_gaudi_config(gaudi_config_name_or_path: Optional[Union[str, Path]] = None) -> GaudiConfig:
        if gaudi_config_name_or_path is None:
            gaudi_config_name_or_path = Path(__file__).parent.resolve() / Path(
                "configs/gaudi_config_trainer_test.json"
            )
        return GaudiConfig.from_pretrained(gaudi_config_name_or_path)

    def get_regression_trainer(a=0, b=0, double_output=False, train_len=64, eval_len=64, pretrained=True, **kwargs):
        label_names = kwargs.get("label_names", None)
        train_dataset = RegressionDataset(length=train_len, label_names=label_names)
        eval_dataset = RegressionDataset(length=eval_len, label_names=label_names)

        model_init = kwargs.pop("model_init", None)
        if model_init is not None:
            model = None
        else:
            if pretrained:
                config = RegressionModelConfig(a=a, b=b, double_output=double_output)
                model = RegressionPreTrainedModel(config)
            else:
                model = RegressionModel(a=a, b=b, double_output=double_output)

        gaudi_config = get_gaudi_config()

        compute_metrics = kwargs.pop("compute_metrics", None)
        data_collator = kwargs.pop("data_collator", None)
        optimizers = kwargs.pop("optimizers", (None, None))
        output_dir = kwargs.pop("output_dir", "./regression")
        preprocess_logits_for_metrics = kwargs.pop("preprocess_logits_for_metrics", None)

        args = RegressionGaudiTrainingArguments(output_dir, use_habana=True, use_lazy_mode=True, a=a, b=b, **kwargs)

        return GaudiTrainer(
            model,
            gaudi_config,
            args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            model_init=model_init,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )


class GaudiTrainerIntegrationCommon:
    def check_saved_checkpoints(self, output_dir, freq, total, is_pretrained=True, safe_weights=False):
        weights_file = WEIGHTS_NAME if not safe_weights else SAFE_WEIGHTS_NAME
        file_list = [weights_file, "training_args.bin", "optimizer.pt", "scheduler.pt", "trainer_state.json"]
        if is_pretrained:
            file_list.append("config.json")
            file_list.append("gaudi_config.json")
        for step in range(freq, total, freq):
            checkpoint = os.path.join(output_dir, f"checkpoint-{step}")
            self.assertTrue(os.path.isdir(checkpoint))
            for filename in file_list:
                self.assertTrue(os.path.isfile(os.path.join(checkpoint, filename)))

    def check_best_model_has_been_loaded(
        self, output_dir, freq, total, trainer, metric, greater_is_better=False, is_pretrained=True, safe_weights=False
    ):
        checkpoint = os.path.join(output_dir, f"checkpoint-{(total // freq) * freq}")
        log_history = TrainerState.load_from_json(os.path.join(checkpoint, "trainer_state.json")).log_history

        values = [d[metric] for d in log_history]
        best_value = max(values) if greater_is_better else min(values)
        best_checkpoint = (values.index(best_value) + 1) * freq
        checkpoint = os.path.join(output_dir, f"checkpoint-{best_checkpoint}")
        if is_pretrained:
            best_model = RegressionPreTrainedModel.from_pretrained(checkpoint)
            best_model.to(trainer.args.device)
        else:
            best_model = RegressionModel()
            if not safe_weights:
                state_dict = torch.load(os.path.join(checkpoint, WEIGHTS_NAME))
            else:
                state_dict = safetensors.torch.load_file(os.path.join(checkpoint, SAFE_WEIGHTS_NAME))
            best_model.load_state_dict(state_dict)
            best_model.to(trainer.args.device)
        self.assertTrue(torch.allclose(best_model.a, trainer.model.a))
        self.assertTrue(torch.allclose(best_model.b, trainer.model.b))

        metrics = trainer.evaluate()
        self.assertEqual(metrics[metric], best_value)

    def check_trainer_state_are_the_same(self, trainer_state, trainer_state1):
        # We'll pop things so operate on copies.
        state = trainer_state.copy()
        state1 = trainer_state1.copy()
        # Log history main contain different logs for the time metrics (after resuming a training).
        log_history = state.pop("log_history", None)
        log_history1 = state1.pop("log_history", None)
        self.assertEqual(state, state1)
        skip_log_keys = ["train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss"]
        for log, log1 in zip(log_history, log_history1):
            for key in skip_log_keys:
                _ = log.pop(key, None)
                _ = log1.pop(key, None)
            self.assertEqual(log, log1)

    def convert_to_sharded_checkpoint(self, folder, save_safe=False, load_safe=False):
        # Converts a checkpoint of a regression model to a sharded checkpoint.
        if load_safe:
            loader = safetensors.torch.load_file
            weights_file = os.path.join(folder, SAFE_WEIGHTS_NAME)
        else:
            loader = torch.load
            weights_file = os.path.join(folder, WEIGHTS_NAME)

        if save_safe:
            extension = "safetensors"
            saver = safetensors.torch.save_file
            index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)
            shard_name = SAFE_WEIGHTS_NAME
        else:
            extension = "bin"
            saver = torch.save
            index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
            shard_name = WEIGHTS_NAME

        state_dict = loader(weights_file)

        os.remove(weights_file)
        keys = list(state_dict.keys())

        shard_files = [
            shard_name.replace(f".{extension}", f"-{idx+1:05d}-of-{len(keys):05d}.{extension}")
            for idx in range(len(keys))
        ]
        index = {"metadata": {}, "weight_map": {key: shard_files[i] for i, key in enumerate(keys)}}

        with open(index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)

        for param_name, shard_file in zip(keys, shard_files):
            saver({param_name: state_dict[param_name]}, os.path.join(folder, shard_file))


@require_torch
@require_sentencepiece
@require_tokenizers
class GaudiTrainerIntegrationPrerunTest(TestCasePlus, GaudiTrainerIntegrationCommon):
    """
    Only tests that want to tap into the auto-pre-run 2 trainings:
    - self.default_trained_model
    - self.alternate_trained_model
    directly, or via check_trained_model
    """

    def setUp(self):
        super().setUp()
        args = GaudiTrainingArguments("..", use_habana=True, use_lazy_mode=True)
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size
        trainer = get_regression_trainer(learning_rate=0.1)
        trainer.train()
        self.default_trained_model = (trainer.model.a, trainer.model.b)

        trainer = get_regression_trainer(learning_rate=0.1, seed=314)
        trainer.train()
        self.alternate_trained_model = (trainer.model.a, trainer.model.b)

    def check_trained_model(self, model, alternate_seed=False, bf16=False):
        # Checks a training seeded with learning_rate = 0.1
        (a, b) = self.alternate_trained_model if alternate_seed else self.default_trained_model
        if not bf16:
            self.assertTrue(torch.allclose(model.a, a))
            self.assertTrue(torch.allclose(model.b, b))
        else:
            self.assertTrue(torch.allclose(model.a, a, atol=1e-03, rtol=0))
            self.assertTrue(torch.allclose(model.b, b, atol=1e-03, rtol=0))

    def test_reproducible_training(self):
        # Checks that training worked, model trained and seed made a reproducible training.
        trainer = get_regression_trainer(learning_rate=0.1)
        trainer.train()
        self.check_trained_model(trainer.model)

        # Checks that a different seed gets different (reproducible) results.
        trainer = get_regression_trainer(learning_rate=0.1, seed=314)
        trainer.train()
        self.check_trained_model(trainer.model, alternate_seed=True)

    def test_trainer_with_datasets(self):
        import datasets

        np.random.seed(42)
        x = np.random.normal(size=(64,)).astype(np.float32)
        y = 2.0 * x + 3.0 + np.random.normal(scale=0.1, size=(64,))
        train_dataset = datasets.Dataset.from_dict({"input_x": x, "label": y})

        gaudi_config = get_gaudi_config()

        # Base training. Should have the same results as test_reproducible_training
        model = RegressionModel()
        args = GaudiTrainingArguments("./regression", learning_rate=0.1, use_habana=True, use_lazy_mode=True)
        trainer = GaudiTrainer(model, gaudi_config, args, train_dataset=train_dataset)
        trainer.train()
        self.check_trained_model(trainer.model)

        # Can return tensors.
        train_dataset.set_format(type="torch", dtype=torch.float32)
        model = RegressionModel()
        trainer = GaudiTrainer(model, gaudi_config, args, train_dataset=train_dataset)
        trainer.train()
        self.check_trained_model(trainer.model)

        # Adding one column not used by the model should have no impact
        z = np.random.normal(size=(64,)).astype(np.float32)
        train_dataset = datasets.Dataset.from_dict({"input_x": x, "label": y, "extra": z})
        model = RegressionModel()
        trainer = GaudiTrainer(model, gaudi_config, args, train_dataset=train_dataset)
        trainer.train()
        self.check_trained_model(trainer.model)

    def test_model_init(self):
        train_dataset = RegressionDataset()
        gaudi_config = get_gaudi_config()
        args = GaudiTrainingArguments("./regression", learning_rate=0.1, use_habana=True, use_lazy_mode=True)
        trainer = GaudiTrainer(
            gaudi_config=gaudi_config, args=args, train_dataset=train_dataset, model_init=lambda: RegressionModel()
        )
        trainer.train()
        self.check_trained_model(trainer.model)

        # Re-training should restart from scratch, thus lead the same results.
        trainer.train()
        self.check_trained_model(trainer.model)

        # Re-training should restart from scratch, thus lead the same results and new seed should be used.
        trainer.args.seed = 314
        trainer.train()
        self.check_trained_model(trainer.model, alternate_seed=True)

    def test_gradient_accumulation(self):
        # Training with half the batch size but accumulation steps as 2 should give the same results.
        trainer = get_regression_trainer(
            gradient_accumulation_steps=2, per_device_train_batch_size=4, learning_rate=0.1
        )
        trainer.train()
        self.check_trained_model(trainer.model)

    def test_training_loss(self):
        n_gpus = max(1, get_gpu_count())

        # With even logs
        trainer = get_regression_trainer(logging_steps=64 / (8 * n_gpus))
        trainer.train()
        log_history = trainer.state.log_history

        losses = [log["loss"] for log in log_history if "loss" in log]
        train_loss = log_history[-1]["train_loss"]
        self.assertAlmostEqual(sum(losses) / len(losses), train_loss, places=4)

        # With uneven logs
        trainer = get_regression_trainer(logging_steps=5)
        trainer.train()
        log_history = trainer.state.log_history

        # Training loss should be the same as before
        new_train_loss = log_history[-1]["train_loss"]
        self.assertAlmostEqual(train_loss, new_train_loss, places=4)

    def test_custom_optimizer(self):
        train_dataset = RegressionDataset()
        gaudi_config = get_gaudi_config()
        gaudi_config.use_fused_adam = False
        args = GaudiTrainingArguments("./regression", use_habana=True, use_lazy_mode=True)
        model = RegressionModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0)
        trainer = GaudiTrainer(
            model, gaudi_config, args, train_dataset=train_dataset, optimizers=(optimizer, lr_scheduler)
        )
        trainer.train()

        (a, b) = self.default_trained_model
        self.assertFalse(torch.allclose(trainer.model.a, a))
        self.assertFalse(torch.allclose(trainer.model.b, b))
        self.assertEqual(trainer.optimizer.state_dict()["param_groups"][0]["lr"], 1.0)

    def test_reduce_lr_on_plateau_args(self):
        # test passed arguments for a custom ReduceLROnPlateau scheduler
        train_dataset = RegressionDataset(length=64)
        eval_dataset = RegressionDataset(length=64)
        gaudi_config = get_gaudi_config()
        gaudi_config.use_fused_adam = False
        args = GaudiTrainingArguments(
            "./regression",
            evaluation_strategy="epoch",
            metric_for_best_model="eval_loss",
            use_habana=True,
            use_lazy_mode=True,
        )
        model = RegressionModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, cooldown=2)
        trainer = GaudiTrainer(
            model,
            gaudi_config,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizers=(optimizer, lr_scheduler),
        )
        trainer.train()

        self.assertIsInstance(trainer.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(trainer.lr_scheduler.factor, 0.2)
        self.assertEqual(trainer.lr_scheduler.patience, 5)
        self.assertEqual(trainer.lr_scheduler.cooldown, 2)

    def test_reduce_lr_on_plateau(self):
        # test the ReduceLROnPlateau scheduler

        class TrainerWithLRLogs(GaudiTrainer):
            def log(self, logs):
                # the LR is computed after metrics and does not exist for the first epoch
                if hasattr(self.lr_scheduler, "_last_lr"):
                    logs["learning_rate"] = self.lr_scheduler._last_lr
                super().log(logs)

        train_dataset = RegressionDataset(length=64)
        eval_dataset = RegressionDataset(length=64)
        gaudi_config = get_gaudi_config()
        gaudi_config.use_fused_adam = False

        args = GaudiTrainingArguments(
            "./regression",
            lr_scheduler_type="reduce_lr_on_plateau",
            evaluation_strategy="epoch",
            metric_for_best_model="eval_loss",
            num_train_epochs=10,
            learning_rate=0.2,
            use_habana=True,
            use_lazy_mode=True,
        )
        model = RegressionModel()
        trainer = TrainerWithLRLogs(model, gaudi_config, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        trainer.train()

        self.assertIsInstance(trainer.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        patience = trainer.lr_scheduler.patience

        logs = trainer.state.log_history[1:]
        best_loss = logs[0]["eval_loss"]
        bad_epochs = 0
        for i, log in enumerate(logs[:-1]):  # Compare learning rate to next epoch's
            loss = log["eval_loss"]
            just_decreased = False
            if loss > best_loss:
                bad_epochs += 1
                if bad_epochs > patience:
                    self.assertLess(logs[i + 1]["learning_rate"][0], log["learning_rate"][0])
                    just_decreased = True
                    bad_epochs = 0
            else:
                best_loss = loss
                bad_epochs = 0
            if not just_decreased:
                self.assertEqual(logs[i + 1]["learning_rate"][0], log["learning_rate"][0])

    def test_adafactor_lr_none(self):
        # test the special case where lr=None, since Trainer can't not have lr_scheduler

        from transformers.optimization import Adafactor, AdafactorSchedule

        train_dataset = RegressionDataset()
        args = GaudiTrainingArguments("./regression", use_habana=True, use_lazy_mode=True)
        gaudi_config = get_gaudi_config()
        gaudi_config.use_fused_adam = False
        model = RegressionModel().to("hpu")
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        lr_scheduler = AdafactorSchedule(optimizer)
        trainer = GaudiTrainer(
            model, gaudi_config, args, train_dataset=train_dataset, optimizers=(optimizer, lr_scheduler)
        )
        trainer.train()

        (a, b) = self.default_trained_model
        self.assertFalse(torch.allclose(trainer.model.a, a))
        self.assertFalse(torch.allclose(trainer.model.b, b))
        self.assertGreater(trainer.optimizer.state_dict()["param_groups"][0]["lr"], 0)

    def test_mixed_bf16(self):
        # very basic test
        trainer = get_regression_trainer(learning_rate=0.1, bf16=True)
        self.assertTrue(trainer.use_hpu_amp)
        trainer.train()
        self.check_trained_model(trainer.model, bf16=True)


@require_torch
@require_sentencepiece
@require_tokenizers
class GaudiTrainerIntegrationTest(TestCasePlus, GaudiTrainerIntegrationCommon):
    def setUp(self):
        super().setUp()
        args = GaudiTrainingArguments("..", use_habana=True, use_lazy_mode=True)
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_eager_mode(self):
        train_dataset = RegressionDataset()
        eval_dataset = RegressionDataset()
        model = RegressionModel()
        gaudi_config = get_gaudi_config()
        args = GaudiTrainingArguments("./regression", use_habana=True, use_lazy_mode=False)
        trainer = GaudiTrainer(model, gaudi_config, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        trainer.train()
        _ = trainer.evaluate()
        _ = trainer.predict(eval_dataset)

    def test_hpu_graphs(self):
        train_dataset = RegressionDataset()
        eval_dataset = RegressionDataset()
        model = RegressionModel()
        gaudi_config = get_gaudi_config()
        args = GaudiTrainingArguments(
            "./regression",
            use_habana=True,
            use_lazy_mode=True,
            use_hpu_graphs_for_training=True,
            use_hpu_graphs_for_inference=True,
        )
        trainer = GaudiTrainer(model, gaudi_config, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        trainer.train()
        _ = trainer.evaluate()
        _ = trainer.predict(eval_dataset)

    def test_trainer_works_with_dict(self):
        train_dataset = RegressionDataset()
        eval_dataset = RegressionDataset()
        model = RegressionDictModel()
        gaudi_config = get_gaudi_config()
        args = GaudiTrainingArguments("./regression", use_habana=True, use_lazy_mode=True)
        trainer = GaudiTrainer(model, gaudi_config, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        trainer.train()
        _ = trainer.evaluate()
        _ = trainer.predict(eval_dataset)

    def test_evaluation_with_keys_to_drop(self):
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GPT2LMHeadModel(config)
        x = torch.randint(0, 100, (128,))
        eval_dataset = RepeatDataset(x)
        args = GaudiTrainingArguments("./test", use_habana=True, use_lazy_mode=True)
        gaudi_config = get_gaudi_config()
        trainer = GaudiTrainer(tiny_gpt2, gaudi_config, args, eval_dataset=eval_dataset)
        # By default the past_key_values are removed
        result = trainer.predict(eval_dataset)
        self.assertTrue(isinstance(result.predictions, np.ndarray))
        # We can still get them by setting ignore_keys to []
        result = trainer.predict(eval_dataset, ignore_keys=[])
        self.assertTrue(isinstance(result.predictions, tuple))
        self.assertEqual(len(result.predictions), 2)

    def test_training_arguments_are_left_untouched(self):
        trainer = get_regression_trainer()
        trainer.train()
        args = GaudiTrainingArguments("./regression", use_habana=True, use_lazy_mode=True, report_to=[])
        dict1, dict2 = args.to_dict(), trainer.args.to_dict()
        for key in dict1.keys():
            # Logging dir can be slightly different as they default to something with the time.
            if key != "logging_dir":
                self.assertEqual(dict1[key], dict2[key])

    def test_number_of_steps_in_training(self):
        # Regular training has n_epochs * len(train_dl) steps
        trainer = get_regression_trainer(learning_rate=0.1)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, self.n_epochs * 64 / self.batch_size)

        # Check passing num_train_epochs works (and a float version too):
        trainer = get_regression_trainer(learning_rate=0.1, num_train_epochs=1.5)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, int(1.5 * 64 / self.batch_size))

        # If we pass a max_steps, num_train_epochs is ignored
        trainer = get_regression_trainer(learning_rate=0.1, max_steps=10)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, 10)

    def test_logging_inf_nan_filter(self):
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GPT2LMHeadModel(config)
        x = torch.randint(0, 100, (128,))
        train_dataset = RepeatDataset(x)

        # GaudiTrainer without inf/nan filter
        gaudi_config = get_gaudi_config()
        args = GaudiTrainingArguments(
            "./test",
            learning_rate=1e9,
            logging_steps=5,
            logging_nan_inf_filter=False,
            use_habana=True,
            use_lazy_mode=True,
        )
        trainer = GaudiTrainer(tiny_gpt2, gaudi_config, args, train_dataset=train_dataset)
        trainer.train()
        log_history_no_filter = trainer.state.log_history

        # GaudiTrainer with inf/nan filter
        args = GaudiTrainingArguments(
            "./test",
            learning_rate=1e9,
            logging_steps=5,
            logging_nan_inf_filter=True,
            use_habana=True,
            use_lazy_mode=True,
        )
        trainer = GaudiTrainer(tiny_gpt2, gaudi_config, args, train_dataset=train_dataset)
        trainer.train()
        log_history_filter = trainer.state.log_history

        def is_any_loss_nan_or_inf(log_history):
            losses = [l["loss"] for l in log_history[:-1]]
            return any(math.isnan(x) for x in losses) or any(math.isinf(x) for x in losses)

        self.assertTrue(is_any_loss_nan_or_inf(log_history_no_filter))
        self.assertFalse(is_any_loss_nan_or_inf(log_history_filter))

    def test_train_and_eval_dataloaders(self):
        trainer = get_regression_trainer(learning_rate=0.1, per_device_train_batch_size=16)
        self.assertEqual(trainer.get_train_dataloader().total_batch_size, 16)
        trainer = get_regression_trainer(learning_rate=0.1, per_device_eval_batch_size=16)
        self.assertEqual(trainer.get_eval_dataloader().total_batch_size, 16)

        # Check drop_last works
        trainer = get_regression_trainer(
            train_len=66, eval_len=74, learning_rate=0.1, per_device_train_batch_size=16, per_device_eval_batch_size=32
        )
        self.assertEqual(len(trainer.get_train_dataloader()), 66 // (16) + 1)
        self.assertEqual(len(trainer.get_eval_dataloader()), 74 // (32) + 1)

        trainer = get_regression_trainer(
            train_len=66,
            eval_len=74,
            learning_rate=0.1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            dataloader_drop_last=True,
        )
        self.assertEqual(len(trainer.get_train_dataloader()), 66 // (16))
        self.assertEqual(len(trainer.get_eval_dataloader()), 74 // (32))

        # Check passing a new dataset for evaluation works
        new_eval_dataset = RegressionDataset(length=128)
        self.assertEqual(len(trainer.get_eval_dataloader(new_eval_dataset)), 128 // (32))

    # tests that we do not require dataloader to have a .dataset attribute
    def test_dataloader_without_dataset(self):
        train_dataset = RegressionDataset(length=128)
        args = GaudiTrainingArguments(output_dir="tmp_trainer", use_habana=True, use_lazy_mode=True)
        trainer = CustomDataloaderTrainer(
            model=RegressionModel(),
            gaudi_config=get_gaudi_config(),
            args=args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )
        trainer.train()
        trainer.evaluate()

    def test_data_is_not_parallelized_when_model_is_parallel(self):
        model = RegressionModel()
        # Make the Trainer believe it's a parallelized model
        model.is_parallelizable = True
        model.model_parallel = True
        args = GaudiTrainingArguments(
            "./regression",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            use_habana=True,
            use_lazy_mode=True,
        )
        gaudi_config = get_gaudi_config()
        trainer = GaudiTrainer(
            model, gaudi_config, args, train_dataset=RegressionDataset(), eval_dataset=RegressionDataset()
        )
        # Check the Trainer was fooled
        self.assertTrue(trainer.is_model_parallel)
        self.assertEqual(trainer.args.n_gpu, 1)

        # The batch size of the training and evaluation dataloaders should be 16, not 16 * n_gpu
        self.assertEqual(trainer.get_train_dataloader().total_batch_size, 16)
        self.assertEqual(len(trainer.get_train_dataloader()), 64 // 16)
        self.assertEqual(trainer.get_eval_dataloader().total_batch_size, 16)
        self.assertEqual(len(trainer.get_eval_dataloader()), 64 // 16)

    def test_evaluate(self):
        trainer = get_regression_trainer(a=1.5, b=2.5, compute_metrics=AlmostAccuracy())
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

        # With a number of elements not a round multiple of the batch size
        trainer = get_regression_trainer(a=1.5, b=2.5, eval_len=66, compute_metrics=AlmostAccuracy())
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

        # With logits preprocess
        trainer = get_regression_trainer(
            a=1.5,
            b=2.5,
            compute_metrics=AlmostAccuracy(),
            preprocess_logits_for_metrics=lambda logits, labels: logits + 1,
        )
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred + 1, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

    def test_predict(self):
        trainer = get_regression_trainer(a=1.5, b=2.5)
        preds = trainer.predict(trainer.eval_dataset).predictions
        x = trainer.eval_dataset.x
        self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

        # With a number of elements not a round multiple of the batch size
        trainer = get_regression_trainer(a=1.5, b=2.5, eval_len=66)
        preds = trainer.predict(trainer.eval_dataset).predictions
        x = trainer.eval_dataset.x
        self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

        # With more than one output of the model
        trainer = get_regression_trainer(a=1.5, b=2.5, double_output=True)
        preds = trainer.predict(trainer.eval_dataset).predictions
        x = trainer.eval_dataset.x
        self.assertEqual(len(preds), 2)
        self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
        self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))

        # With more than one output/label of the model
        trainer = get_regression_trainer(a=1.5, b=2.5, double_output=True, label_names=["labels", "labels_2"])
        outputs = trainer.predict(trainer.eval_dataset)
        preds = outputs.predictions
        labels = outputs.label_ids
        x = trainer.eval_dataset.x
        self.assertEqual(len(preds), 2)
        self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
        self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))
        self.assertTrue(np.array_equal(labels[0], trainer.eval_dataset.ys[0]))
        self.assertTrue(np.array_equal(labels[1], trainer.eval_dataset.ys[1]))

    # def test_dynamic_shapes(self):
    #     eval_dataset = DynamicShapesDataset(batch_size=self.batch_size)
    #     model = RegressionModel(a=2, b=1)
    #     args = TrainingArguments("./regression")
    #     trainer = Trainer(model, args, eval_dataset=eval_dataset)

    #     # Check evaluation can run to completion
    #     _ = trainer.evaluate()

    #     # Check predictions
    #     preds = trainer.predict(eval_dataset)
    #     for expected, seen in zip(eval_dataset.ys, preds.label_ids):
    #         self.assertTrue(np.array_equal(expected, seen[: expected.shape[0]]))
    #         self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

    #     for expected, seen in zip(eval_dataset.xs, preds.predictions):
    #         self.assertTrue(np.array_equal(2 * expected + 1, seen[: expected.shape[0]]))
    #         self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

    #     # Same tests with eval accumulation
    #     args = TrainingArguments("./regression", eval_accumulation_steps=2)
    #     trainer = Trainer(model, args, eval_dataset=eval_dataset)

    #     # Check evaluation can run to completion
    #     _ = trainer.evaluate()

    #     # Check predictions
    #     preds = trainer.predict(eval_dataset)
    #     for expected, seen in zip(eval_dataset.ys, preds.label_ids):
    #         self.assertTrue(np.array_equal(expected, seen[: expected.shape[0]]))
    #         self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

    #     for expected, seen in zip(eval_dataset.xs, preds.predictions):
    #         self.assertTrue(np.array_equal(2 * expected + 1, seen[: expected.shape[0]]))
    #         self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

    def test_log_level(self):
        # testing only --log_level (--log_level_replica requires multiple gpus and DDP and is tested elsewhere)
        logger = logging.get_logger()
        log_info_string = "Running training"

        # test with the default log_level - should be the same as before and thus we test depending on is_info
        is_info = logging.get_verbosity() <= 20
        with CaptureLogger(logger) as cl:
            trainer = get_regression_trainer()
            trainer.train()
        if is_info:
            self.assertIn(log_info_string, cl.out)
        else:
            self.assertNotIn(log_info_string, cl.out)

        # test with low log_level - lower than info
        with CaptureLogger(logger) as cl:
            trainer = get_regression_trainer(log_level="debug")
            trainer.train()
        self.assertIn(log_info_string, cl.out)

        # test with high log_level - should be quiet
        with CaptureLogger(logger) as cl:
            trainer = get_regression_trainer(log_level="error")
            trainer.train()
        self.assertNotIn(log_info_string, cl.out)

    def test_save_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(output_dir=tmpdir, save_steps=5)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, int(self.n_epochs * 64 / self.batch_size))

        # With a regular model that is not a PreTrainedModel
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(output_dir=tmpdir, save_steps=5, pretrained=False)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, int(self.n_epochs * 64 / self.batch_size), False)

    @require_safetensors
    def test_safe_checkpoints(self):
        for save_safetensors in [True, False]:
            with tempfile.TemporaryDirectory() as tmpdir:
                trainer = get_regression_trainer(output_dir=tmpdir, save_steps=5, save_safetensors=save_safetensors)
                trainer.train()
                self.check_saved_checkpoints(
                    tmpdir, 5, int(self.n_epochs * 64 / self.batch_size), safe_weights=save_safetensors
                )

            # With a regular model that is not a PreTrainedModel
            with tempfile.TemporaryDirectory() as tmpdir:
                trainer = get_regression_trainer(
                    output_dir=tmpdir, save_steps=5, pretrained=False, save_safetensors=save_safetensors
                )
                trainer.train()
                self.check_saved_checkpoints(
                    tmpdir, 5, int(self.n_epochs * 64 / self.batch_size), False, safe_weights=save_safetensors
                )

    def test_can_resume_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = {
                "output_dir": tmpdir,
                "train_len": 128,
                "save_steps": 5,
                "learning_rate": 0.1,
                "logging_steps": 5,
            }
            trainer = get_regression_trainer(**kwargs)
            # Disable FusedClipNorm because it makes the test fail
            trainer.gaudi_config.use_fused_clip_norm = False
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(tmpdir, "checkpoint-5")

            # Reinitialize trainer
            trainer = get_regression_trainer(**kwargs)
            # Disable FusedClipNorm because it makes the test fail
            trainer.gaudi_config.use_fused_clip_norm = False

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

            # Now check with a later checkpoint that it also works when we span over one epoch
            checkpoint = os.path.join(tmpdir, "checkpoint-15")

            # Reinitialize trainer and load model
            trainer = get_regression_trainer(**kwargs)
            # Disable FusedClipNorm because it makes the test fail
            trainer.gaudi_config.use_fused_clip_norm = False

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

        # With a regular model that is not a PreTrainedModel
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = {
                "output_dir": tmpdir,
                "train_len": 128,
                "save_steps": 5,
                "learning_rate": 0.1,
                "pretrained": False,
            }

            trainer = get_regression_trainer(**kwargs)
            # Disable FusedClipNorm because it makes the test fail
            trainer.gaudi_config.use_fused_clip_norm = False
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(tmpdir, "checkpoint-5")

            # Reinitialize trainer and load model
            trainer = get_regression_trainer(**kwargs)
            # Disable FusedClipNorm because it makes the test fail
            trainer.gaudi_config.use_fused_clip_norm = False

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

            # Now check with a later checkpoint that it also works when we span over one epoch
            checkpoint = os.path.join(tmpdir, "checkpoint-15")

            # Reinitialize trainer and load model
            trainer = get_regression_trainer(**kwargs)
            # Disable FusedClipNorm because it makes the test fail
            trainer.gaudi_config.use_fused_clip_norm = False

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

        # Now check failures

        # 1. fail to find a bogus checkpoint
        trainer = get_regression_trainer()
        with self.assertRaises(Exception) as context:
            trainer.train(resume_from_checkpoint=f"{checkpoint}-bogus")
        self.assertTrue("Can't find a valid checkpoint at" in str(context.exception))

        # 2. fail to find any checkpoint - due a fresh output_dir
        output_dir2 = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=output_dir2)
        with self.assertRaises(Exception) as context:
            trainer.train(resume_from_checkpoint=True)
        self.assertTrue("No valid checkpoint found in output directory" in str(context.exception))

    def test_resume_training_with_randomness(self):
        train_dataset = RegressionDataset(length=128)
        eval_dataset = RegressionDataset()

        config = RegressionModelConfig(a=0, b=2)
        model = RegressionRandomPreTrainedModel(config)

        tmp_dir = self.get_auto_remove_tmp_dir()
        args = RegressionGaudiTrainingArguments(
            tmp_dir, save_steps=5, learning_rate=0.1, use_habana=True, use_lazy_mode=True
        )
        gaudi_config = get_gaudi_config()
        # Disable FusedClipNorm because it makes the test fail
        gaudi_config.use_fused_clip_norm = False
        trainer = GaudiTrainer(model, gaudi_config, args, train_dataset=train_dataset, eval_dataset=eval_dataset)

        trainer.train()
        (a, b) = trainer.model.a.item(), trainer.model.b.item()

        model = RegressionRandomPreTrainedModel(config)
        trainer = GaudiTrainer(model, gaudi_config, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        trainer.train(resume_from_checkpoint=os.path.join(tmp_dir, "checkpoint-15"))
        (a1, b1) = trainer.model.a.item(), trainer.model.b.item()

        self.assertAlmostEqual(a, a1, delta=1e-5)
        self.assertAlmostEqual(b, b1, delta=1e-5)

    # regression for this issue: https://github.com/huggingface/transformers/issues/12970
    def test_training_with_resume_from_checkpoint_false(self):
        train_dataset = RegressionDataset(length=128)
        eval_dataset = RegressionDataset()

        config = RegressionModelConfig(a=0, b=2)
        model = RegressionRandomPreTrainedModel(config)

        tmp_dir = self.get_auto_remove_tmp_dir()
        args = RegressionGaudiTrainingArguments(
            tmp_dir, save_steps=5, learning_rate=0.1, use_habana=True, use_lazy_mode=True
        )
        gaudi_config = get_gaudi_config()
        trainer = GaudiTrainer(model, gaudi_config, args, train_dataset=train_dataset, eval_dataset=eval_dataset)

        trainer.train(resume_from_checkpoint=False)

    @require_safetensors
    def test_resume_training_with_safe_checkpoint(self):
        # This test will fail for more than 2 GPUs since the batch size will get bigger and with the number of
        # save_steps, the checkpoint will resume training at epoch 2 or more (so the data seen by the model
        # won't be the same since the training dataloader is shuffled).

        for initial_safe in [False, True]:
            for loaded_safe in [False, True]:
                with tempfile.TemporaryDirectory() as tmpdir:
                    trainer = get_regression_trainer(
                        output_dir=tmpdir,
                        train_len=128,
                        save_steps=5,
                        learning_rate=0.1,
                        save_safetensors=initial_safe,
                    )
                    trainer.train()
                    (a, b) = trainer.model.a.item(), trainer.model.b.item()
                    state = dataclasses.asdict(trainer.state)

                    checkpoint = os.path.join(tmpdir, "checkpoint-5")
                    self.convert_to_sharded_checkpoint(checkpoint, load_safe=initial_safe, save_safe=loaded_safe)

                    # Reinitialize trainer
                    trainer = get_regression_trainer(
                        output_dir=tmpdir, train_len=128, save_steps=5, learning_rate=0.1, save_safetensors=loaded_safe
                    )

                    trainer.train(resume_from_checkpoint=checkpoint)
                    (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
                    state1 = dataclasses.asdict(trainer.state)
                    self.assertEqual(a, a1)
                    self.assertEqual(b, b1)
                    self.check_trainer_state_are_the_same(state, state1)

    def test_resume_training_with_gradient_accumulation(self):
        # This test will fail for more than 2 GPUs since the batch size will get bigger and with the number of
        # save_steps, the checkpoint will resume training at epoch 2 or more (so the data seen by the model
        # won't be the same since the training dataloader is shuffled).

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                train_len=128,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=4,
                save_steps=5,
                learning_rate=0.1,
            )
            # Disable FusedClipNorm because it makes this test fail
            # TODO: investigate why
            trainer.gaudi_config.use_fused_clip_norm = False
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(tmpdir, "checkpoint-5")

            # Reinitialize trainer
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                train_len=128,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=4,
                save_steps=5,
                learning_rate=0.1,
            )
            # Disable FusedClipNorm because it makes this test fail
            # TODO: investigate why
            trainer.gaudi_config.use_fused_clip_norm = False
            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)

            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

    def test_resume_training_with_frozen_params(self):
        # This test will fail for more than 2 GPUs since the batch size will get bigger and with the number of
        # save_steps, the checkpoint will resume training at epoch 2 or more (so the data seen by the model
        # won't be the same since the training dataloader is shuffled).

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                train_len=128,
                per_device_train_batch_size=4,
                save_steps=5,
                learning_rate=0.1,
            )
            trainer.model.a.requires_grad_(False)
            # Disable FusedClipNorm because it makes this test fail
            # TODO: investigate why
            trainer.gaudi_config.use_fused_clip_norm = False
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(tmpdir, "checkpoint-5")

            # Reinitialize trainer
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                train_len=128,
                per_device_train_batch_size=4,
                save_steps=5,
                learning_rate=0.1,
            )
            trainer.model.a.requires_grad_(False)
            # Disable FusedClipNorm because it makes this test fail
            # TODO: investigate why
            trainer.gaudi_config.use_fused_clip_norm = False
            trainer.train(resume_from_checkpoint=checkpoint)
            self.assertFalse(trainer.model.a.requires_grad)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)

            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

    def test_load_best_model_at_end(self):
        total = int(self.n_epochs * 64 / self.batch_size)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_steps=5,
                evaluation_strategy="steps",
                save_steps=5,
                load_best_model_at_end=True,
            )
            self.assertFalse(trainer.args.greater_is_better)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, total)
            self.check_best_model_has_been_loaded(tmpdir, 5, total, trainer, "eval_loss")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_steps=5,
                evaluation_strategy="steps",
                save_steps=5,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
            )
            self.assertTrue(trainer.args.greater_is_better)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, total)
            self.check_best_model_has_been_loaded(tmpdir, 5, total, trainer, "eval_accuracy", greater_is_better=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
            )
            self.assertTrue(trainer.args.greater_is_better)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 64 // self.batch_size, total)
            self.check_best_model_has_been_loaded(
                tmpdir, 64 // self.batch_size, total, trainer, "eval_accuracy", greater_is_better=True
            )

        # Test this works with a non PreTrainedModel
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_steps=5,
                evaluation_strategy="steps",
                save_steps=5,
                load_best_model_at_end=True,
                pretrained=False,
            )
            self.assertFalse(trainer.args.greater_is_better)
            trainer.train()
            self.check_saved_checkpoints(tmpdir, 5, total, is_pretrained=False)
            self.check_best_model_has_been_loaded(tmpdir, 5, total, trainer, "eval_loss", is_pretrained=False)

    @require_safetensors
    def test_load_best_model_from_safetensors(self):
        total = int(self.n_epochs * 64 / self.batch_size)
        for save_safetensors, pretrained in product([False, True], [False, True]):
            with tempfile.TemporaryDirectory() as tmpdir:
                trainer = get_regression_trainer(
                    a=1.5,
                    b=2.5,
                    output_dir=tmpdir,
                    learning_rate=0.1,
                    eval_steps=5,
                    evaluation_strategy="steps",
                    save_steps=5,
                    load_best_model_at_end=True,
                    save_safetensors=save_safetensors,
                    pretrained=pretrained,
                )
                self.assertFalse(trainer.args.greater_is_better)
                trainer.train()
                self.check_saved_checkpoints(tmpdir, 5, total, is_pretrained=pretrained, safe_weights=save_safetensors)
                self.check_best_model_has_been_loaded(
                    tmpdir, 5, total, trainer, "eval_loss", is_pretrained=pretrained, safe_weights=save_safetensors
                )

    def test_training_iterable_dataset(self):
        config = RegressionModelConfig()
        model = RegressionPreTrainedModel(config)
        # Adding one column not used by the model should have no impact
        train_dataset = SampleIterableDataset(label_names=["labels", "extra"])

        args = RegressionGaudiTrainingArguments(
            output_dir="./examples", max_steps=4, use_habana=True, use_lazy_mode=True
        )
        gaudi_config = get_gaudi_config()
        trainer = GaudiTrainer(model=model, gaudi_config=gaudi_config, args=args, train_dataset=train_dataset)
        trainer.train()
        self.assertEqual(trainer.state.global_step, 4)

        loader = trainer.get_train_dataloader()
        self.assertIsInstance(loader, torch.utils.data.DataLoader)
        self.assertIsInstance(loader.sampler, torch.utils.data.dataloader._InfiniteConstantSampler)

    def test_evaluation_iterable_dataset(self):
        config = RegressionModelConfig(a=1.5, b=2.5)
        model = RegressionPreTrainedModel(config)
        # Adding one column not used by the model should have no impact
        eval_dataset = SampleIterableDataset(label_names=["labels", "extra"])

        args = RegressionGaudiTrainingArguments(output_dir="./examples", use_habana=True, use_lazy_mode=True)
        gaudi_config = get_gaudi_config()
        trainer = GaudiTrainer(
            model=model,
            gaudi_config=gaudi_config,
            args=args,
            eval_dataset=eval_dataset,
            compute_metrics=AlmostAccuracy(),
        )
        results = trainer.evaluate()

        x, y = trainer.eval_dataset.dataset.x, trainer.eval_dataset.dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

        # With a number of elements not a round multiple of the batch size
        eval_dataset = SampleIterableDataset(length=66)
        results = trainer.evaluate(eval_dataset)

        x, y = eval_dataset.dataset.x, eval_dataset.dataset.ys[0]
        pred = 1.5 * x + 2.5
        expected_loss = ((pred - y) ** 2).mean()
        self.assertAlmostEqual(results["eval_loss"], expected_loss)
        expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
        self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

    def test_predict_iterable_dataset(self):
        config = RegressionModelConfig(a=1.5, b=2.5)
        model = RegressionPreTrainedModel(config)
        eval_dataset = SampleIterableDataset()

        args = RegressionGaudiTrainingArguments(output_dir="./examples", use_habana=True, use_lazy_mode=True)
        gaudi_config = get_gaudi_config()
        trainer = GaudiTrainer(
            model=model,
            gaudi_config=gaudi_config,
            args=args,
            eval_dataset=eval_dataset,
            compute_metrics=AlmostAccuracy(),
        )

        preds = trainer.predict(trainer.eval_dataset).predictions
        x = eval_dataset.dataset.x
        self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

        # With a number of elements not a round multiple of the batch size
        # Adding one column not used by the model should have no impact
        test_dataset = SampleIterableDataset(length=66, label_names=["labels", "extra"])
        preds = trainer.predict(test_dataset).predictions
        x = test_dataset.dataset.x
        self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

    def test_num_train_epochs_in_training(self):
        # len(train_dl) < gradient_accumulation_steps shouldn't give ``ZeroDivisionError`` when ``max_steps`` is given.
        # It should give 1 update step for each epoch.
        trainer = get_regression_trainer(
            max_steps=3, train_len=64, per_device_train_batch_size=16, gradient_accumulation_steps=5
        )
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, 3)

        # Even ``max_steps`` is not specified, we still expect 1 update step for each epoch if
        # len(train_dl) < gradient_accumulation_steps.
        trainer = get_regression_trainer(train_len=64, per_device_train_batch_size=16, gradient_accumulation_steps=5)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, int(self.n_epochs))

    def test_early_stopping_callback(self):
        # early stopping stops training before num_training_epochs
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                num_train_epochs=20,
                gradient_accumulation_steps=1,
                per_device_train_batch_size=16,
                load_best_model_at_end=True,
                evaluation_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                compute_metrics=AlmostAccuracy(),
                metric_for_best_model="accuracy",
            )
            trainer.add_callback(EarlyStoppingCallback(1, 0.0001))
            train_output = trainer.train()
            self.assertLess(train_output.global_step, 20 * 64 / 16)

        # Invalid inputs to trainer with early stopping callback result in assertion error
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                num_train_epochs=20,
                gradient_accumulation_steps=1,
                per_device_train_batch_size=16,
                evaluation_strategy=IntervalStrategy.EPOCH,
                compute_metrics=AlmostAccuracy(),
                metric_for_best_model="accuracy",
            )
            trainer.add_callback(EarlyStoppingCallback(1))
            self.assertEqual(trainer.state.global_step, 0)
            try:
                trainer.train()
            except AssertionError:
                self.assertEqual(trainer.state.global_step, 0)

    def test_flos_extraction(self):
        trainer = get_regression_trainer(learning_rate=0.1)

        def assert_flos_extraction(trainer, wrapped_model_to_check):
            self.assertEqual(trainer.model, unwrap_model(wrapped_model_to_check))
            self.assertGreaterEqual(getattr(unwrap_model(wrapped_model_to_check).config, "total_flos", 0), 0)

        # with plain model
        assert_flos_extraction(trainer, trainer.model)

        # # with enforced DataParallel
        # assert_flos_extraction(trainer, nn.DataParallel(trainer.model))

        trainer.train()
        self.assertTrue(isinstance(trainer.state.total_flos, float))

    def check_checkpoint_deletion(self, trainer, output_dir, expected):
        # Make fake checkpoints
        for n in [5, 10, 15, 20, 25]:
            os.makedirs(os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-{n}"), exist_ok=True)
        trainer._rotate_checkpoints(output_dir=output_dir)
        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*")]
        values = [int(re.match(f".*{PREFIX_CHECKPOINT_DIR}-([0-9]+)", d).groups()[0]) for d in glob_checkpoints]
        self.assertSetEqual(set(values), set(expected))

    def test_checkpoint_rotation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Without best model at end
            trainer = get_regression_trainer(output_dir=tmp_dir, save_total_limit=2)
            self.check_checkpoint_deletion(trainer, tmp_dir, [20, 25])

            # With best model at end
            trainer = get_regression_trainer(
                output_dir=tmp_dir, evaluation_strategy="steps", load_best_model_at_end=True, save_total_limit=2
            )
            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-5")
            self.check_checkpoint_deletion(trainer, tmp_dir, [5, 25])

            # Edge case: we don't always honor save_total_limit=1 if load_best_model_at_end=True to be able to resume
            # from checkpoint
            trainer = get_regression_trainer(
                output_dir=tmp_dir, evaluation_strategy="steps", load_best_model_at_end=True, save_total_limit=1
            )
            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-25")
            self.check_checkpoint_deletion(trainer, tmp_dir, [25])

            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-5")
            self.check_checkpoint_deletion(trainer, tmp_dir, [5, 25])

    def check_mem_metrics(self, trainer, check_func):
        metrics = trainer.train().metrics
        check_func("init_mem_cpu_alloc_delta", metrics)
        check_func("train_mem_cpu_alloc_delta", metrics)
        if torch.cuda.device_count() > 0:
            check_func("init_mem_gpu_alloc_delta", metrics)
            check_func("train_mem_gpu_alloc_delta", metrics)

        metrics = trainer.evaluate()
        check_func("eval_mem_cpu_alloc_delta", metrics)
        if torch.cuda.device_count() > 0:
            check_func("eval_mem_gpu_alloc_delta", metrics)

        metrics = trainer.predict(RegressionDataset()).metrics
        check_func("test_mem_cpu_alloc_delta", metrics)
        if torch.cuda.device_count() > 0:
            check_func("test_mem_gpu_alloc_delta", metrics)

    def test_mem_metrics(self):
        # with mem metrics enabled
        trainer = get_regression_trainer(skip_memory_metrics=False)
        self.check_mem_metrics(trainer, self.assertIn)

        # with mem metrics disabled
        trainer = get_regression_trainer(skip_memory_metrics=True)
        self.check_mem_metrics(trainer, self.assertNotIn)

    def test_no_wd_param_group(self):
        model = nn.Sequential(TstLayer(128), nn.ModuleList([TstLayer(128), TstLayer(128)]))
        gaudi_config = get_gaudi_config()
        args = GaudiTrainingArguments(output_dir="./test", use_habana=True, use_lazy_mode=True)
        trainer = GaudiTrainer(model=model, gaudi_config=gaudi_config, args=args)
        trainer.create_optimizer_and_scheduler(10)
        # fmt: off
        wd_names = ['0.linear1.weight', '0.linear2.weight', '1.0.linear1.weight', '1.0.linear2.weight', '1.1.linear1.weight', '1.1.linear2.weight']
        # fmt: on
        wd_params = [p for n, p in model.named_parameters() if n in wd_names]
        no_wd_params = [p for n, p in model.named_parameters() if n not in wd_names]
        self.assertListEqual(trainer.optimizer.param_groups[0]["params"], wd_params)
        self.assertListEqual(trainer.optimizer.param_groups[1]["params"], no_wd_params)

    def test_profiling(self):
        # 24 total steps and compilation takes place during the 1st three steps
        trainer = get_regression_trainer(profiling_warmup_steps=3, profiling_steps=21)
        trainer.train()


@require_torch
@is_staging_test
class GaudiTrainerIntegrationWithHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    @classmethod
    def tearDownClass(cls):
        for model in ["test-trainer", "test-trainer-epoch", "test-trainer-step"]:
            try:
                delete_repo(token=cls._token, repo_id=model)
            except HTTPError:
                pass

        try:
            delete_repo(token=cls._token, repo_id="valid_org/test-trainer-org")
        except HTTPError:
            pass

    def test_push_to_hub(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=os.path.join(tmp_dir, "test-trainer"),
                push_to_hub=True,
                hub_token=self._token,
            )
            url = trainer.push_to_hub()

            # Extract repo_name from the url
            re_search = re.search(ENDPOINT_STAGING + r"/([^/]+/[^/]+)/", url)
            self.assertTrue(re_search is not None)
            repo_name = re_search.groups()[0]

            self.assertEqual(repo_name, f"{USER}/test-trainer")

            model = RegressionPreTrainedModel.from_pretrained(repo_name)
            self.assertEqual(model.a.item(), trainer.model.a.item())
            self.assertEqual(model.b.item(), trainer.model.b.item())

    def test_push_to_hub_in_organization(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(output_dir=tmp_dir)
            trainer.save_model()
            trainer = get_regression_trainer(
                output_dir=os.path.join(tmp_dir, "test-trainer-org"),
                push_to_hub=True,
                hub_model_id="valid_org/test-trainer-org",
                hub_token=self._token,
            )
            url = trainer.push_to_hub()

            # Extract repo_name from the url
            re_search = re.search(ENDPOINT_STAGING + r"/([^/]+/[^/]+)/", url)
            self.assertTrue(re_search is not None)
            repo_name = re_search.groups()[0]
            self.assertEqual(repo_name, "valid_org/test-trainer-org")

            model = RegressionPreTrainedModel.from_pretrained("valid_org/test-trainer-org")
            self.assertEqual(model.a.item(), trainer.model.a.item())
            self.assertEqual(model.b.item(), trainer.model.b.item())

    def get_commit_history(self, repo):
        commit_logs = subprocess.run(
            "git log".split(),
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=repo,
        ).stdout
        commits = commit_logs.split("\n\n")[1::2]
        return [commit.strip() for commit in commits]

    def test_push_to_hub_with_saves_each_epoch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=os.path.join(tmp_dir, "test-trainer-epoch"),
                push_to_hub=True,
                hub_token=self._token,
                # To avoid any flakiness if the training goes faster than the uploads.
                hub_always_push=True,
                save_strategy="epoch",
            )
            trainer.train()

        commits = list_repo_commits(f"{USER}/test-trainer-epoch", token=self._token)
        commits = [c.title for c in commits]
        self.assertIn("initial commit", commits)
        for i in range(1, 4):
            self.assertIn(f"Training in progress, epoch {i}", commits)

    def test_push_to_hub_with_saves_each_n_steps(self):
        num_gpus = max(1, get_gpu_count())
        if num_gpus > 2:
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=os.path.join(tmp_dir, "test-trainer-step"),
                push_to_hub=True,
                hub_token=self._token,
                # To avoid any flakiness if the training goes faster than the uploads.
                hub_always_push=True,
                save_strategy="steps",
                save_steps=5,
            )
            trainer.train()

        commits = list_repo_commits(f"{USER}/test-trainer-step", token=self._token)
        commits = [c.title for c in commits]
        self.assertIn("initial commit", commits)

        # max_steps depend on the number of available GPUs
        max_steps = math.ceil(trainer.args.num_train_epochs * len(trainer.get_train_dataloader()))
        for i in range(5, max_steps, 5):
            self.assertIn(f"Training in progress, step {i}", commits)


@require_torch
@require_optuna
class GaudiTrainerHyperParameterOptunaIntegrationTest(unittest.TestCase):
    def setUp(self):
        args = GaudiTrainingArguments("..", use_habana=True, use_lazy_mode=True)
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_hyperparameter_search(self):
        class MyTrialShortNamer(TrialShortNamer):
            DEFAULTS = {"a": 0, "b": 0}

        def hp_space(trial):
            return {}

        def model_init(trial):
            if trial is not None:
                a = trial.suggest_int("a", -4, 4)
                b = trial.suggest_int("b", -4, 4)
            else:
                a = 0
                b = 0
            config = RegressionModelConfig(a=a, b=b, double_output=False)

            return RegressionPreTrainedModel(config)

        def hp_name(trial):
            return MyTrialShortNamer.shortname(trial.params)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                learning_rate=0.1,
                logging_steps=1,
                evaluation_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                num_train_epochs=4,
                disable_tqdm=True,
                load_best_model_at_end=True,
                logging_dir="runs",
                run_name="test",
                model_init=model_init,
            )
            trainer.hyperparameter_search(direction="minimize", hp_space=hp_space, hp_name=hp_name, n_trials=4)


# TODO: crashes because `TypeError: cannot pickle 'PyCapsule' object`
# @require_torch
# @require_ray
# class GaudiTrainerHyperParameterRayIntegrationTest(unittest.TestCase):
#     def setUp(self):
#         args = GaudiTrainingArguments("..", use_habana=True, use_lazy_mode=True)
#         self.n_epochs = args.num_train_epochs
#         self.batch_size = args.train_batch_size

#     def ray_hyperparameter_search(self):
#         class MyTrialShortNamer(TrialShortNamer):
#             DEFAULTS = {"a": 0, "b": 0}

#         def hp_space(trial):
#             from ray import tune

#             return {
#                 "a": tune.randint(-4, 4),
#                 "b": tune.randint(-4, 4),
#             }

#         def model_init(config):
#             if config is None:
#                 a = 0
#                 b = 0
#             else:
#                 a = config["a"]
#                 b = config["b"]
#             model_config = RegressionModelConfig(a=a, b=b, double_output=False)

#             return RegressionPreTrainedModel(model_config)

#         def hp_name(params):
#             return MyTrialShortNamer.shortname(params)

#         with tempfile.TemporaryDirectory() as tmp_dir:
#             trainer = get_regression_trainer(
#                 output_dir=tmp_dir,
#                 learning_rate=0.1,
#                 logging_steps=1,
#                 evaluation_strategy=IntervalStrategy.EPOCH,
#                 save_strategy=IntervalStrategy.EPOCH,
#                 num_train_epochs=4,
#                 disable_tqdm=True,
#                 load_best_model_at_end=True,
#                 logging_dir="runs",
#                 run_name="test",
#                 model_init=model_init,
#             )
#             trainer.hyperparameter_search(
#                 direction="minimize", hp_space=hp_space, hp_name=hp_name, backend="ray", n_trials=4
#             )

#     def test_hyperparameter_search(self):
#         self.ray_hyperparameter_search()

#     def test_hyperparameter_search_ray_client(self):
#         import ray
#         from ray.util.client.ray_client_helpers import ray_start_client_server

#         with ray_start_client_server():
#             assert ray.util.client.ray.is_connected()
#             self.ray_hyperparameter_search()


# TODO: enable this test when a SIGOPT_API_TOKEN is added to Github Actions secrets
# @require_torch
# @require_sigopt
# class GaudiTrainerHyperParameterSigOptIntegrationTest(unittest.TestCase):
#     def setUp(self):
#         args = GaudiTrainingArguments("..", use_habana=True, use_lazy_mode=True)
#         self.n_epochs = args.num_train_epochs
#         self.batch_size = args.train_batch_size

#     def test_hyperparameter_search(self):
#         class MyTrialShortNamer(TrialShortNamer):
#             DEFAULTS = {"a": 0, "b": 0}

#         def hp_space(trial):
#             return [
#                 {"bounds": {"min": -4, "max": 4}, "name": "a", "type": "int"},
#                 {"bounds": {"min": -4, "max": 4}, "name": "b", "type": "int"},
#             ]

#         def model_init(trial):
#             if trial is not None:
#                 a = trial.assignments["a"]
#                 b = trial.assignments["b"]
#             else:
#                 a = 0
#                 b = 0
#             config = RegressionModelConfig(a=a, b=b, double_output=False)

#             return RegressionPreTrainedModel(config)

#         def hp_name(trial):
#             return MyTrialShortNamer.shortname(trial.assignments)

#         with tempfile.TemporaryDirectory() as tmp_dir:
#             trainer = get_regression_trainer(
#                 output_dir=tmp_dir,
#                 learning_rate=0.1,
#                 logging_steps=1,
#                 evaluation_strategy=IntervalStrategy.EPOCH,
#                 save_strategy=IntervalStrategy.EPOCH,
#                 num_train_epochs=4,
#                 disable_tqdm=True,
#                 load_best_model_at_end=True,
#                 logging_dir="runs",
#                 run_name="test",
#                 model_init=model_init,
#             )
#             trainer.hyperparameter_search(
#                 direction="minimize", hp_space=hp_space, hp_name=hp_name, backend="sigopt", n_trials=4
#             )


optim_test_params = []
if is_torch_available():
    default_adam_kwargs = {
        "betas": (GaudiTrainingArguments.adam_beta1, GaudiTrainingArguments.adam_beta2),
        "eps": GaudiTrainingArguments.adam_epsilon,
        "lr": GaudiTrainingArguments.learning_rate,
    }

    optim_test_params = [
        (
            OptimizerNames.ADAMW_HF,
            transformers.optimization.AdamW,
            default_adam_kwargs,
        ),
        (
            OptimizerNames.ADAMW_HF.value,
            transformers.optimization.AdamW,
            default_adam_kwargs,
        ),
        (
            OptimizerNames.ADAMW_TORCH,
            torch.optim.AdamW,
            default_adam_kwargs,
        ),
        (
            OptimizerNames.ADAFACTOR,
            transformers.optimization.Adafactor,
            {
                "scale_parameter": False,
                "relative_step": False,
                "lr": GaudiTrainingArguments.learning_rate,
            },
        ),
    ]


@require_torch
class GaudiTrainerOptimizerChoiceTest(unittest.TestCase):
    def check_optim_and_kwargs(self, optim: OptimizerNames, mandatory_kwargs, expected_cls):
        args = GaudiTrainingArguments(optim=optim, output_dir="None", use_habana=True, use_lazy_mode=True)
        actual_cls, optim_kwargs = GaudiTrainer.get_optimizer_cls_and_kwargs(args)
        self.assertEqual(expected_cls, actual_cls)
        self.assertIsNotNone(optim_kwargs)

        for p, v in mandatory_kwargs.items():
            self.assertTrue(p in optim_kwargs)
            actual_v = optim_kwargs[p]
            self.assertTrue(actual_v == v, f"Failed check for {p}. Expected {v}, but got {actual_v}.")

    @parameterized.expand(optim_test_params, skip_on_empty=True)
    def test_optim_supported(self, name: str, expected_cls, mandatory_kwargs):
        # exercises all the valid --optim options
        self.check_optim_and_kwargs(name, mandatory_kwargs, expected_cls)

        trainer = get_regression_trainer(optim=name)
        trainer.gaudi_config.use_fused_adam = False
        trainer.train()


# TODO: solve the Git error returned by this test
# @require_torch
# @require_wandb
# class GaudiTrainerHyperParameterWandbIntegrationTest(unittest.TestCase):
#     def setUp(self):
#         args = GaudiTrainingArguments("..", use_habana=True, use_lazy_mode=True)
#         self.n_epochs = args.num_train_epochs
#         self.batch_size = args.train_batch_size

#     def test_hyperparameter_search(self):
#         class MyTrialShortNamer(TrialShortNamer):
#             DEFAULTS = {"a": 0, "b": 0}

#         def hp_space(trial):
#             return {
#                 "method": "random",
#                 "metric": {},
#                 "parameters": {
#                     "a": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
#                     "b": {"distribution": "int_uniform", "min": 1, "max": 6},
#                 },
#             }

#         def model_init(config):
#             if config is None:
#                 a = 0
#                 b = 0
#             else:
#                 a = config["a"]
#                 b = config["b"]
#             model_config = RegressionModelConfig(a=a, b=b, double_output=False)

#             return RegressionPreTrainedModel(model_config)

#         def hp_name(params):
#             return MyTrialShortNamer.shortname(params)

#         with tempfile.TemporaryDirectory() as tmp_dir:
#             trainer = get_regression_trainer(
#                 output_dir=tmp_dir,
#                 learning_rate=0.1,
#                 logging_steps=1,
#                 evaluation_strategy=IntervalStrategy.EPOCH,
#                 save_strategy=IntervalStrategy.EPOCH,
#                 num_train_epochs=4,
#                 disable_tqdm=True,
#                 load_best_model_at_end=True,
#                 logging_dir="runs",
#                 run_name="test",
#                 model_init=model_init,
#             )
#             trainer.hyperparameter_search(
#                 direction="minimize", hp_space=hp_space, hp_name=hp_name, backend="wandb", n_trials=4, anonymous="must"
#             )


class HyperParameterSearchBackendsTest(unittest.TestCase):
    def test_hyperparameter_search_backends(self):
        self.assertEqual(
            list(ALL_HYPERPARAMETER_SEARCH_BACKENDS.keys()),
            list(HPSearchBackend),
        )
