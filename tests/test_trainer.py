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
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from accelerate.state import AcceleratorState
from huggingface_hub import HfFolder, ModelCard, create_branch, list_repo_commits, list_repo_files
from parameterized import parameterized
from pytest import mark
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    IntervalStrategy,
    LineByLineTextDataset,
    PretrainedConfig,
    TrainerCallback,
    default_data_collator,
    get_polynomial_decay_schedule_with_warmup,
    is_torch_available,
)
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS
from transformers.testing_utils import (
    ENDPOINT_STAGING,
    TOKEN,
    USER,
    CaptureLogger,
    LoggingLevel,
    TemporaryHubRepo,
    TestCasePlus,
    evaluate_side_effect_factory,
    get_gpu_count,
    get_steps_per_epoch,
    get_tests_dir,
    is_staging_test,
    require_accelerate,
    require_optuna,
    require_peft,
    require_safetensors,
    require_sentencepiece,
    require_tensorboard,
    require_tokenizers,
    require_torch,
    require_vision,
)
from transformers.trainer_pt_utils import AcceleratorConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, HPSearchBackend
from transformers.training_args import OptimizerNames
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    is_accelerate_available,
    is_safetensors_available,
)
from transformers.utils.hp_naming import TrialShortNamer

from optimum.habana import GaudiConfig, GaudiTrainingArguments
from optimum.habana.accelerate import GaudiAccelerator
from optimum.habana.utils import set_seed
from optimum.utils import logging


if is_torch_available():
    import torch
    import transformers.optimization
    from torch import nn
    from torch.utils.data import IterableDataset
    from transformers import EarlyStoppingCallback, GPT2Config, PreTrainedModel, TrainerState

    from optimum.habana import GaudiTrainer
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    from optimum.habana.transformers.models.gpt2 import GaudiGPT2LMHeadModel

    if is_safetensors_available():
        import safetensors.torch


# for version specific tests in TrainerIntegrationTest
require_accelerate_version_min_0_28 = partial(require_accelerate, min_version="0.28")
require_accelerate_version_min_0_30 = partial(require_accelerate, min_version="0.30")
GRAD_ACCUM_KWARGS_VERSION_AVAILABLE = is_accelerate_available("0.28")


PATH_SAMPLE_TEXT = f"{get_tests_dir()}/resource/sample_text.txt"


adapt_transformers_to_gaudi()


class StoreLossCallback(TrainerCallback):
    """
    Simple callback to store the loss.
    """

    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])


class MockOOMCallback(TrainerCallback):
    """
    Simple callback to simulate CUDA OOM error if
    the batch size is >= to `batch_size_limit`.
    """

    def __init__(self, batch_size_limit=16):
        self.batch_size_limit = batch_size_limit

    def on_step_end(self, args, state, control, **kwargs):
        # simulate OOM on the first step
        if state.train_batch_size >= self.batch_size_limit:
            raise RuntimeError("Out of memory.")


def ForCausalLMLoss(logits, labels, vocab_size, num_items_in_batch, disable_num_items_in_batch=False):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    if num_items_in_batch is None or disable_num_items_in_batch:
        loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="mean")
    else:
        loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="sum")
        loss = loss / num_items_in_batch
    return loss


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


class RegressionDatasetDynamic:
    def __init__(self, a=2, b=3, length=128, seed=42, label_names=None):
        np.random.seed(seed)
        self.label_names = ["labels"] if label_names is None else label_names
        self.length = length
        self.a = a
        self.b = b

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        self.x = np.random.normal(size=(self.length + i,)).astype(np.float32)
        self.ys = self.a * self.x + self.b + np.random.normal(scale=0.1, size=(self.length + i,)).astype(np.float32)
        result = {}
        result["labels"] = self.ys
        result["input_x"] = self.x
        return result


@dataclasses.dataclass
class RegressionGaudiTrainingArguments(GaudiTrainingArguments):
    a: float = 0.0
    b: float = 0.0
    keep_report_to: bool = False

    def __post_init__(self):
        super().__post_init__()
        # save resources not dealing with reporting unless specified (also avoids the warning when it's not set)
        # can be explicitly disabled via `keep_report_to`
        if not self.keep_report_to:
            self.report_to = []


class RepeatDataset:
    def __init__(self, x, length=64):
        self.x = x
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"input_ids": self.x, "labels": self.x}


class DynamicShapesDataset:
    def __init__(self, length=64, seed=42, batch_size=8):
        self.length = length
        np.random.seed(seed)
        sizes = np.random.randint(1, 20, (length // batch_size,))
        # For easy batching, we make every batch_size consecutive samples the same size.
        self.xs = [np.random.normal(size=(s,)).astype(np.float32) for s in sizes.repeat(batch_size)]
        self.ys = [np.random.normal(size=(s,)).astype(np.float32) for s in sizes.repeat(batch_size)]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"input_x": self.xs[i], "labels": self.ys[i]}


class AlmostAccuracy:
    def __init__(self, thresh=0.25):
        self.thresh = thresh

    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        true = np.abs(predictions - labels) <= self.thresh
        return {"accuracy": true.astype(np.float32).mean().item()}


class AlmostAccuracyBatched:
    def __init__(self, thresh=0.25):
        self.thresh = thresh
        self.batch_acc = []

    def __call__(self, eval_pred, compute_result):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if isinstance(labels, tuple):
            labels = labels[0]
        batch_size = len(predictions)
        true = torch.abs(predictions - labels) <= self.thresh
        acc = true.type(torch.FloatTensor).mean().item()
        self.batch_acc.extend([acc] * batch_size)
        if compute_result:
            result = {"accuracy": np.mean(self.batch_acc).item()}
            self.batch_acc = []
            return result


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

    class RegressionPreTrainedModelWithGradientCheckpointing(PreTrainedModel):
        config_class = RegressionModelConfig
        base_model_prefix = "regression"
        supports_gradient_checkpointing = True

        def __init__(self, config):
            super().__init__(config)
            self.layers = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(4)])
            self.head = nn.Linear(config.hidden_size, 1)
            self.gradient_checkpointing = False
            self.double_output = config.double_output

        def forward(self, input_x, labels=None, **kwargs):
            y = input_x.unsqueeze(0)

            for layer in self.layers:
                if self.training and self.gradient_checkpointing:
                    outputs = self._gradient_checkpointing_func(layer.__call__, y)
                else:
                    outputs = layer(y)

                y = outputs * 3

            logits = self.head(y)

            if labels is None:
                return (logits, logits) if self.double_output else (logits,)

            loss = nn.functional.mse_loss(logits, labels)

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

    def get_regression_trainer(
        a=0,
        b=0,
        double_output=False,
        train_len=64,
        eval_len=64,
        pretrained=True,
        keep_report_to=False,
        output_dir=None,
        **kwargs,
    ):
        label_names = kwargs.get("label_names", None)
        gradient_checkpointing = kwargs.get("gradient_checkpointing", False)
        train_dataset = RegressionDataset(length=train_len, label_names=label_names)
        eval_dataset = RegressionDataset(length=eval_len, label_names=label_names)

        model_init = kwargs.pop("model_init", None)
        if model_init is not None:
            model = None
        else:
            if pretrained:
                config = RegressionModelConfig(a=a, b=b, double_output=double_output)
                # We infer the correct model class if one uses gradient_checkpointing or not
                target_cls = (
                    RegressionPreTrainedModel
                    if not gradient_checkpointing
                    else RegressionPreTrainedModelWithGradientCheckpointing
                )
                model = target_cls(config)
            else:
                model = RegressionModel(a=a, b=b, double_output=double_output)

        gaudi_config = get_gaudi_config()

        compute_metrics = kwargs.pop("compute_metrics", None)
        data_collator = kwargs.pop("data_collator", None)
        optimizers = kwargs.pop("optimizers", (None, None))
        preprocess_logits_for_metrics = kwargs.pop("preprocess_logits_for_metrics", None)
        assert output_dir is not None, "output_dir should be specified for testing"

        args = RegressionGaudiTrainingArguments(
            output_dir, use_habana=True, use_lazy_mode=True, a=a, b=b, keep_report_to=keep_report_to, **kwargs
        )

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

    def get_language_model_trainer(**kwargs):
        import datasets

        dataset = datasets.load_dataset("fka/awesome-chatgpt-prompts")
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        def _tokenize_function(examples):
            model_inputs = tokenizer(examples["prompt"], padding="max_length", truncation=True)
            model_inputs["labels"] = np.array(model_inputs["input_ids"]).astype(np.int64)
            return model_inputs

        tokenized_datasets = dataset.map(_tokenize_function, batched=True)
        training_args = GaudiTrainingArguments(use_habana=True, use_lazy_mode=True, **kwargs)
        gaudi_config = get_gaudi_config()

        trainer = GaudiTrainer(
            model=model,
            gaudi_config=gaudi_config,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
        )

        return trainer


class GaudiTrainerIntegrationCommon:
    def check_saved_checkpoints(
        self, output_dir, freq, total, is_pretrained=True, safe_weights=True, use_scaler=False
    ):
        weights_file = WEIGHTS_NAME if not safe_weights else SAFE_WEIGHTS_NAME
        file_list = [weights_file, "training_args.bin", "optimizer.pt", "scheduler.pt", "trainer_state.json"]
        if is_pretrained:
            file_list.append("config.json")
            file_list.append("gaudi_config.json")
        if use_scaler:
            file_list.append("scaler.pt")
        for step in range(freq, total, freq):
            checkpoint = os.path.join(output_dir, f"checkpoint-{step}")
            self.assertTrue(os.path.isdir(checkpoint))
            for filename in file_list:
                self.assertTrue(os.path.isfile(os.path.join(checkpoint, filename)))

    def check_best_model_has_been_loaded(
        self, output_dir, freq, total, trainer, metric, greater_is_better=False, is_pretrained=True, safe_weights=True
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
                state_dict = torch.load(os.path.join(checkpoint, WEIGHTS_NAME), weights_only=True)
            else:
                state_dict = safetensors.torch.load_file(os.path.join(checkpoint, SAFE_WEIGHTS_NAME))
            best_model.load_state_dict(state_dict)
            best_model.to(trainer.args.device)
        torch.testing.assert_close(best_model.a, trainer.model.a)
        torch.testing.assert_close(best_model.b, trainer.model.b)

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

    def convert_to_sharded_checkpoint(self, folder, save_safe=True, load_safe=True):
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
            shard_name.replace(f".{extension}", f"-{idx + 1:05d}-of-{len(keys):05d}.{extension}")
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
        with tempfile.TemporaryDirectory() as tmpdir:
            args = GaudiTrainingArguments(tmpdir, use_habana=True, use_lazy_mode=True)
            self.n_epochs = args.num_train_epochs
            self.batch_size = args.train_batch_size
            trainer = get_regression_trainer(output_dir=tmpdir, learning_rate=0.1)
            trainer.train()
            self.default_trained_model = (trainer.model.a, trainer.model.b)

            trainer = get_regression_trainer(output_dir=tmpdir, learning_rate=0.1, seed=314)
            trainer.train()
            self.alternate_trained_model = (trainer.model.a, trainer.model.b)

    def check_trained_model(self, model, alternate_seed=False, bf16=False, **kwargs):
        # Checks a training seeded with learning_rate = 0.1
        (a, b) = self.alternate_trained_model if alternate_seed else self.default_trained_model
        if not bf16:
            torch.testing.assert_close(model.a, a, **kwargs)
            torch.testing.assert_close(model.b, b, **kwargs)
        else:
            self.assertTrue(torch.allclose(model.a, a, atol=1e-03, rtol=0))
            self.assertTrue(torch.allclose(model.b, b, atol=1e-03, rtol=0))

    def test_reproducible_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Checks that training worked, model trained and seed made a reproducible training.
            trainer = get_regression_trainer(output_dir=tmpdir, learning_rate=0.1)
            trainer.train()
            self.check_trained_model(trainer.model)

            # Checks that a different seed gets different (reproducible) results.
            trainer = get_regression_trainer(output_dir=tmpdir, learning_rate=0.1, seed=314)
            trainer.train()
            self.check_trained_model(trainer.model, alternate_seed=True)

    def test_trainer_with_datasets(self):
        import datasets

        np.random.seed(42)
        x = np.random.normal(size=(64,)).astype(np.float32)
        y = 2.0 * x + 3.0 + np.random.normal(scale=0.1, size=(64,)).astype(np.float32)
        train_dataset = datasets.Dataset.from_dict({"input_x": x, "label": y})

        gaudi_config = get_gaudi_config()

        # Base training. Should have the same results as test_reproducible_training
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = GaudiTrainingArguments(
                tmpdir, learning_rate=0.1, use_habana=True, use_lazy_mode=True, report_to="none"
            )
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
        with tempfile.TemporaryDirectory() as tmpdir:
            args = GaudiTrainingArguments(
                tmpdir, learning_rate=0.1, use_habana=True, use_lazy_mode=True, report_to="none"
            )
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

    def test_gradient_accumulation_loss_alignment_with_model_loss(self):
        set_seed(42)
        import datasets

        model_name = "nickypro/tinyllama-15M"
        dataset_name = "wikitext"
        dataset_config = "wikitext-2-raw-v1"
        dataset = datasets.load_dataset(dataset_name, dataset_config, split="train[:40]")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            return tokenizer(examples["text"], max_length=16, padding="max_length", truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        args_kwargs = {
            "report_to": "none",
            "logging_steps": 1,
            "max_steps": 5,
            "learning_rate": 3e-4,
            "disable_tqdm": True,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = GaudiTrainingArguments(
                tmp_dir,
                use_habana=True,
                use_lazy_mode=True,
                **args_kwargs,
            )
            # train with base loss
            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            base_loss_callback = StoreLossCallback()
            gaudi_config = get_gaudi_config()
            trainer = GaudiTrainer(
                model,
                gaudi_config,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[base_loss_callback],
                data_collator=data_collator,
            )
            assert trainer.model_accepts_loss_kwargs
            trainer.train()

            args = GaudiTrainingArguments(
                tmp_dir,
                **args_kwargs,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=4,
                use_habana=True,
                use_lazy_mode=True,
            )

            # train with gradient accumulation
            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            grad_accum_loss_callback = StoreLossCallback()
            trainer = GaudiTrainer(
                model,
                gaudi_config,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[grad_accum_loss_callback],
                data_collator=data_collator,
            )
            assert trainer.model_accepts_loss_kwargs
            trainer.train()

            # train with broken loss
            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            broken_loss_callback = StoreLossCallback()
            trainer = GaudiTrainer(
                model,
                gaudi_config,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[broken_loss_callback],
                data_collator=data_collator,
            )
            # disable model_accepts_loss_kwargs so that "num_items_in_batch" is not passed to the model
            trainer.model_accepts_loss_kwargs = False
            trainer.train()

        # Calculate the difference between the base loss and the grad_accum loss
        diff_truth = [
            abs(base - grad) for base, grad in zip(base_loss_callback.losses, grad_accum_loss_callback.losses)
        ]
        diff_broken = [
            abs(base - grad) for base, grad in zip(base_loss_callback.losses, broken_loss_callback.losses)
        ]

        # all diff truth should be quite close
        self.assertLess(max(diff_truth), 0.01, f"Difference {max(diff_truth)} is not within 0.01")
        # max diff broken should be very off ("very off" is arbitrary, but as long as it's bigger than 0.1, it's fine)
        # updated target value compared original implementation https://github.com/huggingface/transformers/blob/v4.49.0/tests/trainer/test_trainer.py#L888
        self.assertGreater(max(diff_broken), 1.0, f"Difference {max(diff_broken)} is not greater than 1.0")

        loss_base = sum(base_loss_callback.losses)
        loss_broken = sum(broken_loss_callback.losses)

        # mean/sum loss should not vary too much.
        relative_diff = abs(loss_base - loss_broken) / max(loss_base, loss_broken)
        self.assertLess(relative_diff, 0.2, f"Relative difference {relative_diff} is not within 0.2")

    def test_gradient_accumulation_loss_alignment_with_loss_func(self):
        set_seed(42)
        import datasets

        model_name = "roneneldan/TinyStories-33M"
        dataset_name = "wikitext"
        dataset_config = "wikitext-2-raw-v1"
        dataset = datasets.load_dataset(dataset_name, dataset_config, split="train[:40]")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            return tokenizer(examples["text"], max_length=16, padding="max_length", truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        model = AutoModelForCausalLM.from_pretrained(model_name)

        def compute_loss(logits, labels, vocab_size, num_items_in_batch, disable_num_items_in_batch=False):
            return ForCausalLMLoss(
                logits["logits"], labels, vocab_size, num_items_in_batch, disable_num_items_in_batch
            )

        loss_fn = partial(compute_loss, vocab_size=model.config.vocab_size, disable_num_items_in_batch=False)

        base_loss_callback = StoreLossCallback()

        args_kwargs = {
            "report_to": "none",
            "logging_steps": 1,
            "max_steps": 5,
            "learning_rate": 3e-4,
            "disable_tqdm": True,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = GaudiTrainingArguments(
                tmp_dir,
                use_habana=True,
                use_lazy_mode=True,
                **args_kwargs,
            )
            gaudi_config = get_gaudi_config()
            trainer = GaudiTrainer(
                model,
                gaudi_config,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[base_loss_callback],
                compute_loss_func=loss_fn,
                data_collator=data_collator,
            )
            trainer.train()

        grad_accum_loss_callback = StoreLossCallback()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = GaudiTrainingArguments(
                tmp_dir,
                **args_kwargs,
                gradient_accumulation_steps=2,
                per_device_train_batch_size=4,
                use_habana=True,
                use_lazy_mode=True,
            )
            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            trainer = GaudiTrainer(
                model,
                gaudi_config,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[grad_accum_loss_callback],
                compute_loss_func=loss_fn,
                data_collator=data_collator,
            )
            trainer.train()

            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            broken_loss_callback = StoreLossCallback()
            loss_fn = partial(compute_loss, vocab_size=model.config.vocab_size, disable_num_items_in_batch=True)
            trainer = GaudiTrainer(
                model,
                gaudi_config,
                args,
                train_dataset=tokenized_dataset,
                callbacks=[broken_loss_callback],
                compute_loss_func=loss_fn,
                data_collator=data_collator,
            )
            trainer.train()

            # Calculate the difference between the base loss and the grad_accum loss
            diff_truth = [
                abs(base - grad) for base, grad in zip(base_loss_callback.losses, grad_accum_loss_callback.losses)
            ]
            diff_broken = [
                abs(base - grad) for base, grad in zip(base_loss_callback.losses, broken_loss_callback.losses)
            ]

            # all diff truth should be quite close
            self.assertLess(max(diff_truth), 0.01, f"Difference {max(diff_truth)} is not within 0.01")

            # max diff broken should be very off
            self.assertGreater(max(diff_broken), 3, f"Difference {max(diff_broken)} is not greater than 3")

    def test_gradient_accumulation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Training with half the batch size but accumulation steps as 2 should give the same training losses.
            trainer = get_regression_trainer(
                output_dir=tmpdir, gradient_accumulation_steps=2, per_device_train_batch_size=4, learning_rate=0.1
            )
            trainer.train()
            self.check_trained_model(trainer.model)

    # The test below is commented because it leads to a core dumped error
    # when it is run with all other tests. It passes when run alone.
    # It seems to be caused by setting `use_reentrant` to False in
    # gradient checkpointing.
    # def test_gradient_checkpointing(self):
    #     trainer = get_regression_trainer(
    #         per_device_train_batch_size=1,
    #         learning_rate=0.1,
    #         gradient_checkpointing=True,
    #         gradient_checkpointing_kwargs={"use_reentrant": False},
    #     )
    #     previous_params = {k: v.detach().clone() for k, v in trainer.model.named_parameters()}

    #     trainer.train()

    #     # Check if model weights have been updated
    #     for k, v in trainer.model.named_parameters():
    #         self.assertFalse(
    #             torch.allclose(previous_params[k], v, rtol=1e-4, atol=1e-4),
    #             f"Model weights for {k} have not been updated",
    #         )

    def test_training_loss(self):
        n_gpus = max(1, get_gpu_count())

        with tempfile.TemporaryDirectory() as tmpdir:
            # With even logs
            trainer = get_regression_trainer(output_dir=tmpdir, logging_steps=64 / (8 * n_gpus))
            trainer.train()
            log_history = trainer.state.log_history

            losses = [log["loss"] for log in log_history if "loss" in log]
            train_loss = log_history[-1]["train_loss"]
            self.assertAlmostEqual(sum(losses) / len(losses), train_loss, places=4)

            # With uneven logs
            trainer = get_regression_trainer(output_dir=tmpdir, logging_steps=5)
            trainer.train()
            log_history = trainer.state.log_history

            # Training loss should be the same as before
            new_train_loss = log_history[-1]["train_loss"]
            self.assertAlmostEqual(train_loss, new_train_loss, places=4)

    def test_custom_optimizer(self):
        train_dataset = RegressionDataset()
        gaudi_config = get_gaudi_config()
        gaudi_config.use_fused_adam = False
        with tempfile.TemporaryDirectory() as tmpdir:
            args = GaudiTrainingArguments(tmpdir, use_habana=True, use_lazy_mode=True, report_to="none")
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

    def test_lr_scheduler_kwargs(self):
        # test scheduler kwargs passed via TrainingArguments
        train_dataset = RegressionDataset()
        model = RegressionModel()
        num_steps, num_warmup_steps = 10, 2
        extra_kwargs = {"power": 5.0, "lr_end": 1e-5}  # Non-default arguments
        with tempfile.TemporaryDirectory() as tmpdir:
            args = GaudiTrainingArguments(
                tmpdir,
                lr_scheduler_type="polynomial",
                lr_scheduler_kwargs=extra_kwargs,
                learning_rate=0.2,
                warmup_steps=num_warmup_steps,
                use_habana=True,
                use_lazy_mode=True,
                report_to="none",
            )
            gaudi_config = get_gaudi_config()
            trainer = GaudiTrainer(model, gaudi_config, args, train_dataset=train_dataset)
            trainer.create_optimizer_and_scheduler(num_training_steps=num_steps)

            # Checking that the scheduler was created
            self.assertIsNotNone(trainer.lr_scheduler)

            # Checking that the correct args were passed
            sched1 = trainer.lr_scheduler
            sched2 = get_polynomial_decay_schedule_with_warmup(
                trainer.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_steps, **extra_kwargs
            )
            self.assertEqual(sched1.lr_lambdas[0].args, sched2.lr_lambdas[0].args)
            self.assertEqual(sched1.lr_lambdas[0].keywords, sched2.lr_lambdas[0].keywords)

    def test_cosine_with_min_lr_scheduler(self):
        train_dataset = RegressionDataset()
        model = RegressionModel()
        num_steps, num_warmup_steps = 10, 2
        extra_kwargs = {"min_lr": 1e-5}  # Non-default arguments
        with tempfile.TemporaryDirectory() as tmpdir:
            args = GaudiTrainingArguments(
                tmpdir,
                lr_scheduler_type="cosine_with_min_lr",
                lr_scheduler_kwargs=extra_kwargs,
                learning_rate=0.2,
                warmup_steps=num_warmup_steps,
                use_habana=True,
                use_lazy_mode=True,
                report_to="none",
            )
            trainer = GaudiTrainer(model, gaudi_config=get_gaudi_config(), args=args, train_dataset=train_dataset)
            trainer.create_optimizer_and_scheduler(num_training_steps=num_steps)

            # Checking that the scheduler was created
            self.assertIsNotNone(trainer.lr_scheduler)

            # Check the last learning rate
            for _ in range(num_steps):
                trainer.lr_scheduler.step()
            self.assertEqual(trainer.lr_scheduler.get_last_lr()[0], 1e-5)

    def test_reduce_lr_on_plateau_args(self):
        # test passed arguments for a custom ReduceLROnPlateau scheduler
        train_dataset = RegressionDataset(length=64)
        eval_dataset = RegressionDataset(length=64)
        gaudi_config = get_gaudi_config()
        gaudi_config.use_fused_adam = False
        with tempfile.TemporaryDirectory() as tmpdir:
            args = GaudiTrainingArguments(
                tmpdir,
                eval_strategy="epoch",
                metric_for_best_model="eval_loss",
                use_habana=True,
                use_lazy_mode=True,
                report_to="none",
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
                    logs["learning_rate"] = self.lr_scheduler._last_lr[0]
                super().log(logs)

        train_dataset = RegressionDataset(length=64)
        eval_dataset = RegressionDataset(length=64)
        gaudi_config = get_gaudi_config()
        gaudi_config.use_fused_adam = False

        with tempfile.TemporaryDirectory() as tmpdir:
            args = GaudiTrainingArguments(
                tmpdir,
                lr_scheduler_type="reduce_lr_on_plateau",
                eval_strategy="epoch",
                metric_for_best_model="eval_loss",
                num_train_epochs=10,
                learning_rate=0.2,
                report_to="none",
                use_habana=True,
                use_lazy_mode=True,
            )
            model = RegressionModel()
            trainer = TrainerWithLRLogs(
                model, gaudi_config, args, train_dataset=train_dataset, eval_dataset=eval_dataset
            )
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
                        self.assertLess(logs[i + 1]["learning_rate"], log["learning_rate"])
                        just_decreased = True
                        bad_epochs = 0
                else:
                    best_loss = loss
                    bad_epochs = 0
                if not just_decreased:
                    self.assertEqual(logs[i + 1]["learning_rate"], log["learning_rate"])

    def test_adafactor_lr_none(self):
        # test the special case where lr=None, since Trainer can't not have lr_scheduler

        from transformers.optimization import Adafactor, AdafactorSchedule

        train_dataset = RegressionDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = GaudiTrainingArguments(tmpdir, use_habana=True, use_lazy_mode=True, report_to="none")
            gaudi_config = get_gaudi_config()
            gaudi_config.use_fused_adam = False
            model = RegressionModel().to("hpu")
            optimizer = Adafactor(
                model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None
            )
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
        with tempfile.TemporaryDirectory() as tmpdir:
            # very basic test
            trainer = get_regression_trainer(output_dir=tmpdir, learning_rate=0.1, bf16=True)
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

    @mark.skip("Skip this test until PT_HPU_LAZY_MODE=0 is set as default for all tests")
    def test_eager_mode(self):
        train_dataset = RegressionDataset()
        eval_dataset = RegressionDataset()
        model = RegressionModel()
        gaudi_config = get_gaudi_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = GaudiTrainingArguments(tmpdir, use_habana=True, use_lazy_mode=False)
            trainer = GaudiTrainer(model, gaudi_config, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
            trainer.train()
            _ = trainer.evaluate()
            _ = trainer.predict(eval_dataset)

    def test_hpu_graphs(self):
        train_dataset = RegressionDataset()
        eval_dataset = RegressionDataset()
        model = RegressionModel()
        gaudi_config = get_gaudi_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = GaudiTrainingArguments(
                tmpdir,
                use_habana=True,
                use_lazy_mode=True,
                use_hpu_graphs_for_training=True,
                use_hpu_graphs_for_inference=True,
                disable_tensor_cache_hpu_graphs=True,
                max_hpu_graphs=1,
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
        args = GaudiTrainingArguments(
            self.get_auto_remove_tmp_dir(), use_habana=True, use_lazy_mode=True, report_to="none"
        )
        trainer = GaudiTrainer(model, gaudi_config, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        trainer.train()
        _ = trainer.evaluate()
        _ = trainer.predict(eval_dataset)

    def test_evaluation_with_keys_to_drop(self):
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GaudiGPT2LMHeadModel(config)
        x = torch.randint(0, 100, (128,))
        eval_dataset = RepeatDataset(x)
        args = GaudiTrainingArguments(
            self.get_auto_remove_tmp_dir(), use_habana=True, use_lazy_mode=True, report_to="none"
        )
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
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir)
        trainer.train()
        args = GaudiTrainingArguments(tmp_dir, use_habana=True, use_lazy_mode=True, report_to=[])
        dict1, dict2 = args.to_dict(), trainer.args.to_dict()
        for key in dict1.keys():
            # Logging dir can be slightly different as they default to something with the time.
            if key != "logging_dir":
                self.assertEqual(dict1[key], dict2[key])

    def test_number_of_steps_in_training(self):
        # Regular training has n_epochs * len(train_dl) steps
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir, learning_rate=0.1)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, self.n_epochs * 64 / self.batch_size)

        # Check passing num_train_epochs works (and a float version too):
        trainer = get_regression_trainer(output_dir=tmp_dir, learning_rate=0.1, num_train_epochs=1.5)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, int(1.5 * 64 / self.batch_size))

        # If we pass a max_steps, num_train_epochs is ignored
        trainer = get_regression_trainer(output_dir=tmp_dir, learning_rate=0.1, max_steps=10)
        train_output = trainer.train()
        self.assertEqual(train_output.global_step, 10)

    # TODO: enable this test when torch.compile becomes the default on Gaudi
    # def test_torch_compile_loss_func_compatibility(self):
    #     config = LlamaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=3, num_attention_heads=4)
    #     tiny_llama = LlamaForCausalLM(config)

    #     x = torch.randint(0, 100, (128,))
    #     train_dataset = RepeatDataset(x)

    #     args = GaudiTrainingArguments(
    #         self.get_auto_remove_tmp_dir(),
    #         per_device_train_batch_size=2,
    #         torch_compile=True,
    #         max_steps=1,  # compile happens on the first step
    #         use_habana=True,
    #         use_lazy_mode=True,
    #     )
    #     gaudi_config = get_gaudi_config()
    #     trainer = GaudiTrainer(model=tiny_llama, gaudi_config=gaudi_config, args=args, train_dataset=train_dataset)  # noqa
    #     trainer.train()

    @require_peft
    def test_multiple_peft_adapters(self):
        from peft import LoraConfig, get_peft_model

        # Tests if resuming from checkpoint works if the model has multiple adapters

        MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tiny_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

        peft_config = LoraConfig(
            r=4,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        tiny_model = get_peft_model(tiny_model, peft_config, "adapter1")
        tiny_model.add_adapter("adapter2", peft_config)

        train_dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=PATH_SAMPLE_TEXT,
            block_size=tokenizer.max_len_single_sentence,
        )
        for example in train_dataset.examples:
            example["labels"] = example["input_ids"]

        tokenizer.pad_token = tokenizer.eos_token

        tmp_dir = self.get_auto_remove_tmp_dir()
        args = GaudiTrainingArguments(
            tmp_dir,
            per_device_train_batch_size=1,
            learning_rate=1e-9,
            save_steps=5,
            logging_steps=5,
            max_steps=10,
            use_habana=True,
            use_lazy_mode=True,
        )
        gaudi_config = get_gaudi_config()
        trainer = GaudiTrainer(tiny_model, gaudi_config, args, processing_class=tokenizer, train_dataset=train_dataset)

        trainer.train()
        parameters = dict(tiny_model.named_parameters())
        state = dataclasses.asdict(trainer.state)

        # Reinitialize trainer
        trainer = GaudiTrainer(tiny_model, gaudi_config, args, processing_class=tokenizer, train_dataset=train_dataset)

        checkpoint = os.path.join(tmp_dir, "checkpoint-5")

        trainer.train(resume_from_checkpoint=checkpoint)
        parameters1 = dict(tiny_model.named_parameters())
        state1 = dataclasses.asdict(trainer.state)
        self.assertEqual(parameters, parameters1)
        self.check_trainer_state_are_the_same(state, state1)

    # TODO: investigate why this test fails
    # def test_neftune(self):
    #     config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
    #     tiny_gpt2 = GPT2LMHeadModel(config)
    #     x = torch.randint(0, 100, (128,))
    #     train_dataset = RepeatDataset(x)

    #     # Trainer without inf/nan filter
    #     args = GaudiTrainingArguments(
    #         "./test", learning_rate=1e-9, logging_steps=5, logging_nan_inf_filter=False, neftune_noise_alpha=0.4, use_habana=True, use_lazy_mode=True, report_to="none"
    #     )
    #     gaudi_config = get_gaudi_config()
    #     trainer = GaudiTrainer(tiny_gpt2, gaudi_config, args, train_dataset=train_dataset)

    #     trainer.model = trainer._activate_neftune(trainer.model)

    #     dummy_input = torch.LongTensor([[1, 0, 1]]).to("hpu")

    #     emb1 = trainer.model.get_input_embeddings()(dummy_input)
    #     emb2 = trainer.model.get_input_embeddings()(dummy_input)

    #     self.assertFalse(torch.allclose(emb1, emb2), "Neftune noise is not applied!")

    #     # redefine the model
    #     tiny_gpt2 = GPT2LMHeadModel(config)
    #     # Trainer without inf/nan filter
    #     args = GaudiTrainingArguments(
    #         "./test", learning_rate=1e-9, logging_steps=5, logging_nan_inf_filter=False, neftune_noise_alpha=0.4, use_habana=True, use_lazy_mode=True, report_to="none"
    #     )
    #     trainer = GaudiTrainer(tiny_gpt2, gaudi_config, args, train_dataset=train_dataset)

    #     # Check that it trains without errors
    #     trainer.train()

    #     # Make sure forward pass works fine
    #     _ = trainer.model(dummy_input)
    #     self.assertTrue(len(trainer.model.get_input_embeddings()._forward_hooks) == 0)

    #     trainer.model.eval()

    #     # Check that we get identical embeddings just in case
    #     emb1 = trainer.model.get_input_embeddings()(dummy_input)
    #     emb2 = trainer.model.get_input_embeddings()(dummy_input)

    #     self.assertTrue(torch.allclose(emb1, emb2), "Neftune noise is still applied!")

    def test_logging_inf_nan_filter(self):
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GaudiGPT2LMHeadModel(config)
        x = torch.randint(0, 100, (128,))
        train_dataset = RepeatDataset(x)

        # GaudiTrainer without inf/nan filter
        gaudi_config = get_gaudi_config()
        args = GaudiTrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e9,
            logging_steps=5,
            logging_nan_inf_filter=False,
            use_habana=True,
            use_lazy_mode=True,
            report_to="none",
        )
        trainer = GaudiTrainer(tiny_gpt2, gaudi_config, args, train_dataset=train_dataset)
        trainer.train()
        log_history_no_filter = trainer.state.log_history

        # GaudiTrainer with inf/nan filter
        args = GaudiTrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e9,
            logging_steps=5,
            logging_nan_inf_filter=True,
            use_habana=True,
            use_lazy_mode=True,
            report_to="none",
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
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir, learning_rate=0.1, per_device_train_batch_size=16)
        self.assertEqual(trainer.get_train_dataloader().total_batch_size, 16)
        trainer = get_regression_trainer(output_dir=tmp_dir, learning_rate=0.1, per_device_eval_batch_size=16)
        self.assertEqual(trainer.get_eval_dataloader().total_batch_size, 16)

        # Check drop_last works
        trainer = get_regression_trainer(
            output_dir=tmp_dir,
            train_len=66,
            eval_len=74,
            learning_rate=0.1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
        )
        self.assertEqual(len(trainer.get_train_dataloader()), 66 // (16) + 1)
        self.assertEqual(len(trainer.get_eval_dataloader()), 74 // (32) + 1)

        trainer = get_regression_trainer(
            output_dir=tmp_dir,
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
        args = GaudiTrainingArguments(
            output_dir=self.get_auto_remove_tmp_dir(), use_habana=True, use_lazy_mode=True, report_to="none"
        )
        trainer = CustomDataloaderTrainer(
            model=RegressionModel(),
            gaudi_config=get_gaudi_config(),
            args=args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )
        trainer.train()
        trainer.evaluate()

    def test_get_eval_dataloader_without_persistent_workers(self):
        train_dataset = RegressionDataset()
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GPT2LMHeadModel(config)
        args = GaudiTrainingArguments(
            self.get_auto_remove_tmp_dir(),
            report_to="none",
            dataloader_persistent_workers=False,
            use_habana=True,
            use_lazy_mode=True,
        )

        # Single evaluation dataset
        eval_dataset = RegressionDataset()
        gaudi_config = get_gaudi_config()
        trainer = GaudiTrainer(tiny_gpt2, gaudi_config, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        # Mocking the prepare method to avoid the dataloader changing with each call to get_eval_dataloader
        trainer.accelerator.prepare = lambda x: x

        default_dataloader = trainer.get_eval_dataloader()
        dataloader_with_dataset = trainer.get_eval_dataloader(eval_dataset)

        self.assertEqual(default_dataloader.dataset, eval_dataset)
        self.assertEqual(dataloader_with_dataset.dataset, eval_dataset)
        self.assertNotEqual(default_dataloader, dataloader_with_dataset)

        # Multiple evaluation datasets
        first_dataset = RegressionDataset()
        second_dataset = RegressionDataset()
        trainer = GaudiTrainer(
            tiny_gpt2,
            gaudi_config,
            args,
            train_dataset=train_dataset,
            eval_dataset={"first": first_dataset, "second": second_dataset},
        )
        # Mocking the prepare method to avoid the dataloader changing with each call to get_eval_dataloader
        trainer.accelerator.prepare = lambda x: x

        first_dataloader = trainer.get_eval_dataloader("first")
        first_dataloader_repeated = trainer.get_eval_dataloader("first")
        second_dataloader = trainer.get_eval_dataloader("second")
        second_dataloader_repeated = trainer.get_eval_dataloader("second")

        self.assertEqual(first_dataset, first_dataloader.dataset)
        self.assertEqual(first_dataloader.dataset, first_dataloader_repeated.dataset)
        self.assertEqual(second_dataset, second_dataloader.dataset)
        self.assertEqual(second_dataloader.dataset, second_dataloader_repeated.dataset)
        self.assertNotEqual(first_dataloader, first_dataloader_repeated)
        self.assertNotEqual(second_dataloader, second_dataloader_repeated)

    def test_get_eval_dataloader_with_persistent_workers(self):
        train_dataset = RegressionDataset()
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GPT2LMHeadModel(config)
        args = GaudiTrainingArguments(
            self.get_auto_remove_tmp_dir(),
            report_to="none",
            dataloader_persistent_workers=True,
            dataloader_num_workers=2,
            use_habana=True,
            use_lazy_mode=True,
        )

        # Single evaluation dataset
        eval_dataset = RegressionDataset()
        gaudi_config = get_gaudi_config()
        trainer = GaudiTrainer(tiny_gpt2, gaudi_config, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        # Mocking the prepare method to avoid the dataloader changing with each call to get_eval_dataloader
        trainer.accelerator.prepare = lambda x: x

        default_dataloader = trainer.get_eval_dataloader()
        dataloader_with_dataset = trainer.get_eval_dataloader(eval_dataset)

        self.assertEqual(default_dataloader.dataset, eval_dataset)
        self.assertEqual(dataloader_with_dataset.dataset, eval_dataset)
        self.assertEqual(default_dataloader, dataloader_with_dataset)

        # Multiple evaluation datasets
        first_dataset = RegressionDataset()
        second_dataset = RegressionDataset()
        trainer = GaudiTrainer(
            tiny_gpt2,
            gaudi_config,
            args,
            train_dataset=train_dataset,
            eval_dataset={"first": first_dataset, "second": second_dataset},
        )
        # Mocking the prepare method to avoid the dataloader changing with each call to get_eval_dataloader
        trainer.accelerator.prepare = lambda x: x

        first_dataloader = trainer.get_eval_dataloader("first")
        first_dataloader_repeated = trainer.get_eval_dataloader("first")
        second_dataloader = trainer.get_eval_dataloader("second")
        second_dataloader_repeated = trainer.get_eval_dataloader("second")

        self.assertEqual(first_dataset, first_dataloader.dataset)
        self.assertEqual(first_dataloader.dataset, first_dataloader_repeated.dataset)
        self.assertEqual(second_dataset, second_dataloader.dataset)
        self.assertEqual(second_dataloader.dataset, second_dataloader_repeated.dataset)
        self.assertEqual(first_dataloader, first_dataloader_repeated)
        self.assertEqual(second_dataloader, second_dataloader_repeated)

    def test_data_is_not_parallelized_when_model_is_parallel(self):
        model = RegressionModel()
        # Make the Trainer believe it's a parallelized model
        model.is_parallelizable = True
        model.model_parallel = True
        with tempfile.TemporaryDirectory() as tmpdir:
            args = GaudiTrainingArguments(
                tmpdir,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                use_habana=True,
                use_lazy_mode=True,
                report_to="none",
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
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(output_dir=tmpdir, a=1.5, b=2.5, compute_metrics=AlmostAccuracy())
            results = trainer.evaluate()

            x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
            pred = 1.5 * x + 2.5
            expected_loss = ((pred - y) ** 2).mean()
            self.assertAlmostEqual(results["eval_loss"], expected_loss)
            expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
            self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

            # With a number of elements not a round multiple of the batch size
            trainer = get_regression_trainer(
                output_dir=tmpdir, a=1.5, b=2.5, eval_len=66, compute_metrics=AlmostAccuracy()
            )
            results = trainer.evaluate()

            x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
            pred = 1.5 * x + 2.5
            expected_loss = ((pred - y) ** 2).mean()
            self.assertAlmostEqual(results["eval_loss"], expected_loss)
            expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
            self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

            # With logits preprocess
            trainer = get_regression_trainer(
                output_dir=tmpdir,
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

    def test_evaluate_with_batch_eval_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir, a=1.5, b=2.5, compute_metrics=AlmostAccuracyBatched(), batch_eval_metrics=True
            )
            results = trainer.evaluate()

            x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
            pred = 1.5 * x + 2.5
            expected_loss = ((pred - y) ** 2).mean()
            self.assertAlmostEqual(results["eval_loss"], expected_loss)
            expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
            self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

            # With a number of elements not a round multiple of the batch size
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                a=1.5,
                b=2.5,
                eval_len=66,
                compute_metrics=AlmostAccuracyBatched(),
                batch_eval_metrics=True,
            )
            results = trainer.evaluate()

            x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
            pred = 1.5 * x + 2.5
            expected_loss = ((pred - y) ** 2).mean()
            self.assertAlmostEqual(results["eval_loss"], expected_loss)
            expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
            self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

            # With logits preprocess
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                a=1.5,
                b=2.5,
                compute_metrics=AlmostAccuracyBatched(),
                batch_eval_metrics=True,
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
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(output_dir=tmpdir, a=1.5, b=2.5)
            preds = trainer.predict(trainer.eval_dataset).predictions
            x = trainer.eval_dataset.x
            self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

            # With a number of elements not a round multiple of the batch size
            trainer = get_regression_trainer(output_dir=tmpdir, a=1.5, b=2.5, eval_len=66)
            preds = trainer.predict(trainer.eval_dataset).predictions
            x = trainer.eval_dataset.x
            self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

            # With more than one output of the model
            trainer = get_regression_trainer(output_dir=tmpdir, a=1.5, b=2.5, double_output=True)
            preds = trainer.predict(trainer.eval_dataset).predictions
            x = trainer.eval_dataset.x
            self.assertEqual(len(preds), 2)
            self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
            self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))

            # With more than one output/label of the model
            trainer = get_regression_trainer(
                output_dir=tmpdir, a=1.5, b=2.5, double_output=True, label_names=["labels", "labels_2"]
            )
            outputs = trainer.predict(trainer.eval_dataset)
            preds = outputs.predictions
            labels = outputs.label_ids
            x = trainer.eval_dataset.x
            self.assertEqual(len(preds), 2)
            self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
            self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))
            self.assertTrue(np.array_equal(labels[0], trainer.eval_dataset.ys[0]))
            self.assertTrue(np.array_equal(labels[1], trainer.eval_dataset.ys[1]))

    def test_predict_with_batch_eval_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir, a=1.5, b=2.5, compute_metrics=AlmostAccuracyBatched(), batch_eval_metrics=True
            )
            results = trainer.predict(trainer.eval_dataset)
            preds = results.predictions
            x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
            gt = 1.5 * x + 2.5
            self.assertTrue(np.allclose(preds, gt))
            expected_acc = AlmostAccuracy()((preds, y))["accuracy"]
            self.assertAlmostEqual(results.metrics["test_accuracy"], expected_acc)

            # With a number of elements not a round multiple of the batch size
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                a=1.5,
                b=2.5,
                eval_len=66,
                compute_metrics=AlmostAccuracyBatched(),
                batch_eval_metrics=True,
            )
            results = trainer.predict(trainer.eval_dataset)
            preds = results.predictions
            x, y = trainer.eval_dataset.x, trainer.eval_dataset.ys[0]
            self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))
            expected_acc = AlmostAccuracy()((preds, y))["accuracy"]
            self.assertAlmostEqual(results.metrics["test_accuracy"], expected_acc)

            # With more than one output of the model
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                a=1.5,
                b=2.5,
                double_output=True,
                compute_metrics=AlmostAccuracyBatched(),
                batch_eval_metrics=True,
            )
            preds = trainer.predict(trainer.eval_dataset).predictions
            x = trainer.eval_dataset.x
            self.assertEqual(len(preds), 2)
            self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
            self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))

            # With more than one output/label of the model
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                a=1.5,
                b=2.5,
                double_output=True,
                label_names=["labels", "labels_2"],
                compute_metrics=AlmostAccuracyBatched(),
                batch_eval_metrics=True,
            )
            outputs = trainer.predict(trainer.eval_dataset)
            preds = outputs.predictions
            labels = outputs.label_ids
            x = trainer.eval_dataset.x
            self.assertEqual(len(preds), 2)
            self.assertTrue(np.allclose(preds[0], 1.5 * x + 2.5))
            self.assertTrue(np.allclose(preds[1], 1.5 * x + 2.5))
            self.assertTrue(np.array_equal(labels[0], trainer.eval_dataset.ys[0]))
            self.assertTrue(np.array_equal(labels[1], trainer.eval_dataset.ys[1]))

    def test_dynamic_shapes(self):
        eval_dataset = DynamicShapesDataset(batch_size=self.batch_size)
        model = RegressionModel(a=2, b=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = GaudiTrainingArguments(tmpdir, use_habana=True, use_lazy_mode=True)
            gaudi_config = get_gaudi_config()
            gaudi_config.use_dynamic_shapes = True
            trainer = GaudiTrainer(model, gaudi_config, args, eval_dataset=eval_dataset)

            # Check evaluation can run to completion
            _ = trainer.evaluate()

            # Check predictions
            preds = trainer.predict(eval_dataset)
            for expected, seen in zip(eval_dataset.ys, preds.label_ids):
                self.assertTrue(np.allclose(expected, seen[: expected.shape[0]]))
                self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

            for expected, seen in zip(eval_dataset.xs, preds.predictions):
                self.assertTrue(np.allclose(2 * expected + 1, seen[: expected.shape[0]]))
                self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

            # Same tests with eval accumulation
            args = GaudiTrainingArguments(tmpdir, use_habana=True, use_lazy_mode=True, eval_accumulation_steps=2)
            trainer = GaudiTrainer(model, gaudi_config, args, eval_dataset=eval_dataset)

            # Check evaluation can run to completion
            _ = trainer.evaluate()

            # Check predictions
            preds = trainer.predict(eval_dataset)
            for expected, seen in zip(eval_dataset.ys, preds.label_ids):
                self.assertTrue(np.allclose(expected, seen[: expected.shape[0]]))
                self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

            for expected, seen in zip(eval_dataset.xs, preds.predictions):
                self.assertTrue(np.allclose(2 * expected + 1, seen[: expected.shape[0]]))
                self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

    def test_dynamic_shape_feature(self):
        # Run training with variable length inputs and enable dynamic shapes support
        train_dataset = RegressionDatasetDynamic(length=256)
        gaudi_config = get_gaudi_config()
        gaudi_config.use_dynamic_shapes = True
        with tempfile.TemporaryDirectory() as tmpdir:
            args = GaudiTrainingArguments(
                tmpdir,
                use_habana=True,
                use_lazy_mode=True,
                per_device_train_batch_size=1,
                num_train_epochs=1,
                report_to="none",
            )
            model = RegressionModel()
            trainer = GaudiTrainer(
                model,
                gaudi_config,
                args,
                train_dataset=train_dataset,
            )
            train_output_ds = trainer.train()

            # Run training again with variable length inputs and disable dynamic shapes support
            train_dataset = RegressionDatasetDynamic(length=256)
            gaudi_config = get_gaudi_config()
            gaudi_config.use_dynamic_shapes = False
            args = GaudiTrainingArguments(
                tmpdir,
                use_habana=True,
                use_lazy_mode=True,
                per_device_train_batch_size=1,
                num_train_epochs=1,
                report_to="none",
            )
            model = RegressionModel()
            trainer = GaudiTrainer(
                model,
                gaudi_config,
                args,
                train_dataset=train_dataset,
            )
            train_output_static = trainer.train()

            # Check if performance with dynamic shapes support is at least 5 times that without dynamic shapes
            # Note "5x" number is not applicable across models, it is tuned for this particular dummy model
            self.assertGreaterEqual(
                train_output_ds.metrics["train_samples_per_second"],
                5 * train_output_static.metrics["train_samples_per_second"],
            )

    def test_log_level(self):
        # testing only --log_level (--log_level_replica requires multiple gpus and DDP and is tested elsewhere)
        logger = logging.get_logger()
        log_info_string = "Running training"

        with tempfile.TemporaryDirectory() as tmpdir:
            # test with the default log_level - should be the same as before and thus we test depending on is_info
            is_info = logging.get_verbosity() <= 20
            with CaptureLogger(logger) as cl:
                trainer = get_regression_trainer(output_dir=tmpdir)
                trainer.train()
            if is_info:
                self.assertIn(log_info_string, cl.out)
            else:
                self.assertNotIn(log_info_string, cl.out)

            with LoggingLevel(logging.INFO):
                # test with low log_level - lower than info
                with CaptureLogger(logger) as cl:
                    trainer = get_regression_trainer(output_dir=tmpdir, log_level="debug")
                    trainer.train()
                self.assertIn(log_info_string, cl.out)

            with LoggingLevel(logging.INFO):
                # test with high log_level - should be quiet
                with CaptureLogger(logger) as cl:
                    trainer = get_regression_trainer(output_dir=tmpdir, log_level="error")
                    trainer.train()
                self.assertNotIn(log_info_string, cl.out)

    def test_save_checkpoints(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir, save_steps=5)
        trainer.train()
        self.check_saved_checkpoints(tmp_dir, 5, int(self.n_epochs * 64 / self.batch_size))

        # With a regular model that is not a PreTrainedModel
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir, save_steps=5, pretrained=False)
        trainer.train()
        self.check_saved_checkpoints(tmp_dir, 5, int(self.n_epochs * 64 / self.batch_size), False)

    @require_safetensors
    def test_safe_checkpoints(self):
        for save_safetensors in [True, False]:
            tmp_dir = self.get_auto_remove_tmp_dir()
            trainer = get_regression_trainer(output_dir=tmp_dir, save_steps=5, save_safetensors=save_safetensors)
            trainer.train()
            self.check_saved_checkpoints(
                tmp_dir, 5, int(self.n_epochs * 64 / self.batch_size), safe_weights=save_safetensors
            )

            # With a regular model that is not a PreTrainedModel
            tmp_dir = self.get_auto_remove_tmp_dir()
            trainer = get_regression_trainer(
                output_dir=tmp_dir, save_steps=5, pretrained=False, save_safetensors=save_safetensors
            )
            trainer.train()
            self.check_saved_checkpoints(
                tmp_dir, 5, int(self.n_epochs * 64 / self.batch_size), False, safe_weights=save_safetensors
            )

    def test_save_collator_tokenizer_by_default(self):
        class FakeCollator:
            def __init__(self):
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.tokenizer.add_tokens(["<NEW_TOKEN1>", "<NEW_TOKEN2>"])

            def __call__(self, features: list[Any], return_tensors="pt") -> dict[str, Any]:
                return default_data_collator(features, return_tensors)

        data_collator = FakeCollator()
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(
            output_dir=tmp_dir, save_steps=5, save_safetensors=True, data_collator=data_collator
        )
        trainer.train()
        loaded_tokenizer = AutoTokenizer.from_pretrained(os.path.join(tmp_dir, os.listdir(tmp_dir)[0]))
        assert len(loaded_tokenizer) == len(trainer.data_collator.tokenizer), "Failed to load updated tokenizer"

    def test_load_best_model_with_save(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(
            output_dir=tmp_dir,
            save_steps=5,
            eval_strategy="steps",
            eval_steps=5,
            max_steps=9,
        )
        trainer.train()
        # Check that we have the last known step:
        assert os.path.exists(os.path.join(tmp_dir, f"checkpoint-{trainer.state.max_steps}")), (
            f"Could not find checkpoint-{trainer.state.max_steps}"
        )
        # And then check the last step
        assert os.path.exists(os.path.join(tmp_dir, "checkpoint-9")), "Could not find checkpoint-9"

        # Now test that using a limit works
        # Should result in:
        # - save at step 5 (but is deleted)
        # - save at step 10 (loaded in at the end when `load_best_model=True`)
        # - save at step 11
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(
            output_dir=tmp_dir,
            save_steps=5,
            eval_strategy="steps",
            eval_steps=5,
            load_best_model_at_end=True,
            save_total_limit=2,
            max_steps=11,
        )
        trainer.train()
        # Check that we have the last known step:
        assert os.path.exists(os.path.join(tmp_dir, "checkpoint-11")), "Could not find checkpoint-11"
        # And then check the last multiple
        assert os.path.exists(os.path.join(tmp_dir, "checkpoint-10")), "Could not find checkpoint-10"
        # Finally check that we don't have an old one
        assert not os.path.exists(os.path.join(tmp_dir, "checkpoint-5")), "Found checkpoint-5, limit not respected"

        # Finally check that the right model was loaded in, checkpoint-10
        # this goes by the last `eval` step check to do so, so it won't be
        # the last model *saved*
        model_state = trainer.model.state_dict()
        final_model_weights = safetensors.torch.load_file(os.path.join(tmp_dir, "checkpoint-10", "model.safetensors"))
        for k, v in model_state.items():
            assert torch.allclose(v, final_model_weights[k]), f"{k} is not the same"

    def test_can_resume_training(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        kwargs = {
            "output_dir": tmp_dir,
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

        checkpoint = os.path.join(tmp_dir, "checkpoint-5")

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
        checkpoint = os.path.join(tmp_dir, "checkpoint-15")

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
        tmp_dir = self.get_auto_remove_tmp_dir()
        kwargs = {
            "output_dir": tmp_dir,
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

        checkpoint = os.path.join(tmp_dir, "checkpoint-5")

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
        checkpoint = os.path.join(tmp_dir, "checkpoint-15")

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
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir)
        with self.assertRaises(Exception) as context:
            trainer.train(resume_from_checkpoint=f"{checkpoint}-bogus")
        self.assertTrue("Can't find a valid checkpoint at" in str(context.exception))

        # 2. fail to find any checkpoint - due a fresh output_dir
        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(output_dir=tmp_dir)
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

    # @require_deepspeed
    # def test_auto_batch_size_with_deepspeed(self):
    #     train_dataset = RegressionDataset(length=128)

    #     config = RegressionModelConfig(a=0, b=2)
    #     model = RegressionRandomPreTrainedModel(config)

    #     tmp_dir = self.get_auto_remove_tmp_dir()

    #     for stage in [1, 2]:
    #         deepspeed = {
    #             "zero_optimization": {
    #                 "stage": stage,
    #             },
    #             "train_batch_size": "auto",
    #             "train_micro_batch_size_per_gpu": "auto",
    #         }

    #     args = RegressionGaudiTrainingArguments(
    #         tmp_dir,
    #         do_train=True,
    #         max_steps=2,
    #         save_strategy="no",
    #         per_device_train_batch_size=16,
    #         auto_find_batch_size=True,
    #         deepspeed=deepspeed,
    #         use_habana=True,
    #         use_lazy_mode=True,
    #     )
    #     gaudi_config = get_gaudi_config()
    #     trainer = GaudiTrainer(model, gaudi_config, args, train_dataset=train_dataset, callbacks=[MockOOMCallback()])
    #     trainer.train()
    #     self.assertEqual(trainer._train_batch_size, 8)

    # def test_auto_batch_size_with_resume_from_checkpoint(self):
    #     train_dataset = RegressionDataset(length=128)

    #     config = RegressionModelConfig(a=0, b=2)
    #     model = RegressionRandomPreTrainedModel(config)

    #     tmp_dir = self.get_auto_remove_tmp_dir()

    #     args = RegressionGaudiTrainingArguments(
    #         tmp_dir,
    #         do_train=True,
    #         max_steps=2,
    #         save_steps=1,
    #         per_device_train_batch_size=16,
    #         auto_find_batch_size=True,
    #         use_habana=True,
    #         use_lazy_mode=True,
    #     )
    #     gaudi_config = get_gaudi_config()
    #     trainer = GaudiTrainer(
    #         model, gaudi_config, args, train_dataset=train_dataset, callbacks=[MockOOMCallback()]
    #     )
    #     trainer.train()
    #     # After `auto_find_batch_size` is ran we should now be at 8
    #     self.assertEqual(trainer._train_batch_size, 8)

    #     # We can then make a new Trainer
    #     trainer = GaudiTrainer(model, gaudi_config, args, train_dataset=train_dataset)
    #     # Check we are at 16 to start
    #     self.assertEqual(trainer._train_batch_size, 16 * max(trainer.args.n_gpu, 1))
    #     trainer.train(resume_from_checkpoint=True)
    #     # We should be back to 8 again, picking up based upon the last ran Trainer
    #     self.assertEqual(trainer._train_batch_size, 8)

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
                eval_strategy="steps",
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
                eval_strategy="steps",
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
                eval_strategy="epoch",
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
                eval_strategy="steps",
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
                    eval_strategy="steps",
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
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RegressionModelConfig()
            model = RegressionPreTrainedModel(config)
            # Adding one column not used by the model should have no impact
            train_dataset = SampleIterableDataset(label_names=["labels", "extra"])

            args = RegressionGaudiTrainingArguments(
                output_dir=tmpdir, max_steps=4, use_habana=True, use_lazy_mode=True
            )
            gaudi_config = get_gaudi_config()
            trainer = GaudiTrainer(model=model, gaudi_config=gaudi_config, args=args, train_dataset=train_dataset)
            trainer.train()
            self.assertEqual(trainer.state.global_step, 4)

            loader = trainer.get_train_dataloader()
            self.assertIsInstance(loader, torch.utils.data.DataLoader)
            self.assertIsInstance(loader.sampler, torch.utils.data.dataloader._InfiniteConstantSampler)

    def test_evaluation_iterable_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            model = RegressionPreTrainedModel(config)
            # Adding one column not used by the model should have no impact
            eval_dataset = SampleIterableDataset(label_names=["labels", "extra"])

            args = RegressionGaudiTrainingArguments(output_dir=tmpdir, use_habana=True, use_lazy_mode=True)
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
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            model = RegressionPreTrainedModel(config)
            eval_dataset = SampleIterableDataset()

            args = RegressionGaudiTrainingArguments(output_dir=tmpdir, use_habana=True, use_lazy_mode=True)
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
        with tempfile.TemporaryDirectory() as tmpdir:
            # len(train_dl) < gradient_accumulation_steps shouldn't give ``ZeroDivisionError`` when ``max_steps`` is given.
            # It should give 1 update step for each epoch.
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                max_steps=3,
                train_len=64,
                per_device_train_batch_size=16,
                gradient_accumulation_steps=5,
            )
            train_output = trainer.train()
            self.assertEqual(train_output.global_step, 3)

            # Even ``max_steps`` is not specified, we still expect 1 update step for each epoch if
            # len(train_dl) < gradient_accumulation_steps.
            trainer = get_regression_trainer(
                output_dir=tmpdir, train_len=64, per_device_train_batch_size=16, gradient_accumulation_steps=5
            )
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
                eval_strategy=IntervalStrategy.EPOCH,
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
                eval_strategy=IntervalStrategy.EPOCH,
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
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(output_dir=tmp_dir, learning_rate=0.1)

            def assert_flos_extraction(trainer, wrapped_model_to_check):
                self.assertEqual(trainer.model, trainer.accelerator.unwrap_model(wrapped_model_to_check))
                self.assertGreaterEqual(
                    getattr(trainer.accelerator.unwrap_model(wrapped_model_to_check).config, "total_flos", 0), 0
                )

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
                output_dir=tmp_dir, eval_strategy="steps", load_best_model_at_end=True, save_total_limit=2
            )
            trainer.state.best_model_checkpoint = os.path.join(tmp_dir, "checkpoint-5")
            self.check_checkpoint_deletion(trainer, tmp_dir, [5, 25])

            # Edge case: we don't always honor save_total_limit=1 if load_best_model_at_end=True to be able to resume
            # from checkpoint
            trainer = get_regression_trainer(
                output_dir=tmp_dir, eval_strategy="steps", load_best_model_at_end=True, save_total_limit=1
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
        with tempfile.TemporaryDirectory() as tmp_dir:
            # with mem metrics enabled
            trainer = get_regression_trainer(output_dir=tmp_dir, skip_memory_metrics=False)
            self.check_mem_metrics(trainer, self.assertIn)

            # with mem metrics disabled
            trainer = get_regression_trainer(output_dir=tmp_dir, skip_memory_metrics=True)
            self.check_mem_metrics(trainer, self.assertNotIn)

    def test_no_wd_param_group(self):
        model = nn.Sequential(TstLayer(128), nn.ModuleList([TstLayer(128), TstLayer(128)]))
        gaudi_config = get_gaudi_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = GaudiTrainingArguments(output_dir=tmp_dir, use_habana=True, use_lazy_mode=True, report_to="none")
            trainer = GaudiTrainer(model=model, gaudi_config=gaudi_config, args=args)
            trainer.create_optimizer_and_scheduler(10)
            wd_names = ['0.linear1.weight', '0.linear2.weight', '1.0.linear1.weight', '1.0.linear2.weight', '1.1.linear1.weight', '1.1.linear2.weight']  # fmt: skip
            wd_params = [p for n, p in model.named_parameters() if n in wd_names]
            no_wd_params = [p for n, p in model.named_parameters() if n not in wd_names]
            self.assertListEqual(trainer.optimizer.param_groups[0]["params"], wd_params)
            self.assertListEqual(trainer.optimizer.param_groups[1]["params"], no_wd_params)

    def test_accelerator_config_empty(self):
        # Checks that a config can be made with the defaults if not passed
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            model = RegressionPreTrainedModel(config)
            eval_dataset = SampleIterableDataset()

            # Leaves one option as something *not* basic
            gaudi_config = get_gaudi_config()
            args = RegressionGaudiTrainingArguments(output_dir=tmp_dir, use_habana=True)
            trainer = GaudiTrainer(model=model, gaudi_config=gaudi_config, args=args, eval_dataset=eval_dataset)
            self.assertEqual(trainer.accelerator.split_batches, False)
            self.assertEqual(trainer.accelerator.dispatch_batches, None)
            self.assertEqual(trainer.accelerator.even_batches, True)
            self.assertEqual(trainer.accelerator.use_seedable_sampler, True)

            if GRAD_ACCUM_KWARGS_VERSION_AVAILABLE:
                # gradient accumulation kwargs configures gradient_state
                self.assertNotIn("sync_each_batch", trainer.accelerator.gradient_state.plugin_kwargs)

    def test_accelerator_config_from_dict(self):
        # Checks that accelerator kwargs can be passed through
        # and the accelerator is initialized respectively
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            model = RegressionPreTrainedModel(config)
            eval_dataset = SampleIterableDataset()

            accelerator_config: dict[str, Any] = {
                "split_batches": True,
                "dispatch_batches": True,
                "even_batches": False,
                "use_seedable_sampler": True,
            }
            if GRAD_ACCUM_KWARGS_VERSION_AVAILABLE:
                accelerator_config["gradient_accumulation_kwargs"] = {"sync_each_batch": True}

            # Leaves all options as something *not* basic
            gaudi_config = get_gaudi_config()
            args = RegressionGaudiTrainingArguments(
                output_dir=tmp_dir,
                accelerator_config=accelerator_config,
                use_habana=True,
            )
            trainer = GaudiTrainer(model=model, gaudi_config=gaudi_config, args=args, eval_dataset=eval_dataset)
            self.assertEqual(trainer.accelerator.split_batches, True)
            self.assertEqual(trainer.accelerator.dispatch_batches, True)
            self.assertEqual(trainer.accelerator.even_batches, False)
            self.assertEqual(trainer.accelerator.use_seedable_sampler, True)

    def test_accelerator_config_from_yaml(self):
        # Checks that accelerator kwargs can be passed through
        # and the accelerator is initialized respectively
        with tempfile.TemporaryDirectory() as tmp_dir:
            path_file = Path(tmp_dir) / "accelerator_config.json"
            with open(path_file, "w") as f:
                accelerator_config = {
                    "split_batches": True,
                    "dispatch_batches": True,
                    "even_batches": False,
                    "use_seedable_sampler": False,
                }
                json.dump(accelerator_config, f)
            config = RegressionModelConfig(a=1.5, b=2.5)
            model = RegressionPreTrainedModel(config)
            eval_dataset = SampleIterableDataset()

            # Leaves all options as something *not* basic
            gaudi_config = get_gaudi_config()
            args = RegressionGaudiTrainingArguments(output_dir=tmp_dir, accelerator_config=path_file, use_habana=True)
            trainer = GaudiTrainer(model=model, gaudi_config=gaudi_config, args=args, eval_dataset=eval_dataset)
            self.assertEqual(trainer.accelerator.split_batches, True)
            self.assertEqual(trainer.accelerator.dispatch_batches, True)
            self.assertEqual(trainer.accelerator.even_batches, False)
            self.assertEqual(trainer.accelerator.use_seedable_sampler, False)

    def test_accelerator_config_from_dataclass(self):
        # Checks that accelerator kwargs can be passed through
        # and the accelerator is initialized respectively

        accelerator_config = AcceleratorConfig(
            split_batches=True,
            dispatch_batches=True,
            even_batches=False,
            use_seedable_sampler=False,
        )
        config = RegressionModelConfig(a=1.5, b=2.5)
        model = RegressionPreTrainedModel(config)
        eval_dataset = SampleIterableDataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            gaudi_config = get_gaudi_config()
            args = RegressionGaudiTrainingArguments(
                output_dir=tmp_dir, accelerator_config=accelerator_config, use_habana=True
            )
            trainer = GaudiTrainer(model=model, gaudi_config=gaudi_config, args=args, eval_dataset=eval_dataset)
            self.assertEqual(trainer.accelerator.split_batches, True)
            self.assertEqual(trainer.accelerator.dispatch_batches, True)
            self.assertEqual(trainer.accelerator.even_batches, False)
            self.assertEqual(trainer.accelerator.use_seedable_sampler, False)

    @require_accelerate_version_min_0_28
    def test_accelerate_config_from_dataclass_grad_accum(self):
        # Checks that accelerator kwargs can be passed through
        # and the accelerator is initialized respectively

        grad_acc_kwargs = {
            "num_steps": 10,
            "adjust_scheduler": False,
            "sync_with_dataloader": False,
            "sync_each_batch": True,
        }
        accelerator_config = AcceleratorConfig(
            split_batches=True,
            dispatch_batches=True,
            even_batches=False,
            use_seedable_sampler=False,
            gradient_accumulation_kwargs=grad_acc_kwargs,
        )
        config = RegressionModelConfig(a=1.5, b=2.5)
        model = RegressionPreTrainedModel(config)
        eval_dataset = SampleIterableDataset()
        gaudi_config = get_gaudi_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = RegressionGaudiTrainingArguments(
                output_dir=tmp_dir, accelerator_config=accelerator_config, use_habana=True
            )
            trainer = GaudiTrainer(model=model, gaudi_config=gaudi_config, args=args, eval_dataset=eval_dataset)
            self.assertEqual(trainer.args.gradient_accumulation_steps, 10)

    def test_accelerator_config_from_partial(self):
        # Checks that accelerator kwargs can be passed through
        # and the accelerator is initialized respectively
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            model = RegressionPreTrainedModel(config)
            eval_dataset = SampleIterableDataset()

            # Leaves one option as something *not* basic
            gaudi_config = get_gaudi_config()
            args = RegressionGaudiTrainingArguments(
                output_dir=tmp_dir,
                accelerator_config={
                    "split_batches": True,
                },
                use_habana=True,
            )
            trainer = GaudiTrainer(model=model, gaudi_config=gaudi_config, args=args, eval_dataset=eval_dataset)
            self.assertEqual(trainer.accelerator.split_batches, True)
            self.assertEqual(trainer.accelerator.dispatch_batches, None)
            self.assertEqual(trainer.accelerator.even_batches, True)
            self.assertEqual(trainer.accelerator.use_seedable_sampler, True)

    def test_accelerator_custom_state(self):
        AcceleratorState._reset_state(reset_partial_state=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError) as cm:
                _ = RegressionGaudiTrainingArguments(
                    output_dir=tmp_dir, use_habana=True, accelerator_config={"use_configured_state": True}
                )
                self.assertIn("Please define this beforehand", str(cm.warnings[0].message))
            _ = GaudiAccelerator()
            _ = RegressionGaudiTrainingArguments(
                output_dir=tmp_dir, use_habana=True, accelerator_config={"use_configured_state": True}
            )
        AcceleratorState._reset_state(reset_partial_state=True)

    @require_accelerate_version_min_0_28
    def test_accelerator_config_from_dict_grad_accum_num_steps(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            model = RegressionPreTrainedModel(config)
            eval_dataset = SampleIterableDataset()
            gaudi_config = get_gaudi_config()

            # case - TrainingArguments.gradient_accumulation_steps == 1
            #      - gradient_accumulation_kwargs['num_steps] == 1
            # results in grad accum set to 1
            args = RegressionGaudiTrainingArguments(
                output_dir=tmp_dir,
                gradient_accumulation_steps=1,
                accelerator_config={
                    "gradient_accumulation_kwargs": {
                        "num_steps": 1,
                    }
                },
                use_habana=True,
            )
            trainer = GaudiTrainer(model=model, gaudi_config=gaudi_config, args=args, eval_dataset=eval_dataset)
            self.assertEqual(trainer.accelerator.gradient_state.plugin_kwargs["num_steps"], 1)

            # case - TrainingArguments.gradient_accumulation_steps > 1
            #      - gradient_accumulation_kwargs['num_steps] specified
            # results in exception raised
            args = RegressionGaudiTrainingArguments(
                output_dir=tmp_dir,
                gradient_accumulation_steps=2,
                accelerator_config={
                    "gradient_accumulation_kwargs": {
                        "num_steps": 10,
                    }
                },
                use_habana=True,
            )
            with self.assertRaises(Exception) as context:
                trainer = GaudiTrainer(model=model, gaudi_config=gaudi_config, args=args, eval_dataset=eval_dataset)
            self.assertTrue("The `AcceleratorConfig`'s `num_steps` is set but" in str(context.exception))

    def test_accelerator_config_not_instantiated(self):
        # Checks that accelerator kwargs can be passed through
        # and the accelerator is initialized respectively
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(NotImplementedError) as context:
                _ = RegressionGaudiTrainingArguments(
                    output_dir=tmp_dir,
                    accelerator_config=AcceleratorConfig,
                    use_habana=True,
                    use_lazy_mode=True,
                )
            self.assertTrue("Tried passing in a callable to `accelerator_config`" in str(context.exception))

        # Now test with a custom subclass
        @dataclasses.dataclass
        class CustomAcceleratorConfig(AcceleratorConfig):
            pass

        @dataclasses.dataclass
        class CustomTrainingArguments(GaudiTrainingArguments):
            accelerator_config: dict = dataclasses.field(
                default=CustomAcceleratorConfig,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(NotImplementedError) as context:
                _ = CustomTrainingArguments(
                    output_dir=tmp_dir,
                    use_habana=True,
                    use_lazy_mode=True,
                )
            self.assertTrue("Tried passing in a callable to `accelerator_config`" in str(context.exception))

    def test_torch_dtype_to_json(self):
        @dataclasses.dataclass
        class TorchDtypeTrainingArguments(GaudiTrainingArguments):
            torch_dtype: torch.dtype = dataclasses.field(
                default=torch.float32,
            )

        for dtype in [
            "float32",
            "float64",
            "complex64",
            "complex128",
            "bfloat16",
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
            "bool",
        ]:
            torch_dtype = getattr(torch, dtype)
            with tempfile.TemporaryDirectory() as tmp_dir:
                args = TorchDtypeTrainingArguments(output_dir=tmp_dir, torch_dtype=torch_dtype, use_habana=True)

                args_dict = args.to_dict()
                self.assertIn("torch_dtype", args_dict)
                self.assertEqual(args_dict["torch_dtype"], dtype)

    @require_accelerate_version_min_0_30
    def test_eval_use_gather_object(self):
        train_dataset = RegressionDataset()
        eval_dataset = RegressionDataset()
        model = RegressionDictModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = GaudiTrainingArguments(
                tmpdir, use_habana=True, use_lazy_mode=True, report_to="none", eval_use_gather_object=True
            )
            gaudi_config = get_gaudi_config()
            trainer = GaudiTrainer(model, gaudi_config, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
            trainer.train()
            _ = trainer.evaluate()
            _ = trainer.predict(eval_dataset)

    def test_trainer_saves_tokenizer(self):
        MODEL_ID = "google-bert/bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            gaudi_config = get_gaudi_config()
            trainer = GaudiTrainer(
                model=RegressionPreTrainedModel(config),
                args=GaudiTrainingArguments(output_dir=tmp_dir, use_habana=True, use_lazy_mode=True),
                gaudi_config=gaudi_config,
                processing_class=tokenizer,
            )
            trainer.save_model()

            reloaded_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)

        # For tokenizers, there isn't a direct to_dict method and the properties stored in the configs e.g.
        # saved tokens change overtime, so we check that two tokenizers are equal by comparing their encoded outputs
        test_sentence = "This is a test sentence"
        self.assertListEqual(
            tokenizer(test_sentence, padding="max_length").input_ids,
            reloaded_tokenizer(test_sentence, padding="max_length").input_ids,
        )

    @require_vision
    def test_trainer_saves_image_processor(self):
        MODEL_ID = "openai/clip-vit-base-patch32"
        image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            gaudi_config = get_gaudi_config()
            trainer = GaudiTrainer(
                model=RegressionPreTrainedModel(config),
                args=GaudiTrainingArguments(output_dir=tmp_dir, use_habana=True, use_lazy_mode=True),
                gaudi_config=gaudi_config,
                processing_class=image_processor,
            )
            trainer.save_model()
            reloaded_image_processor = AutoImageProcessor.from_pretrained(tmp_dir)

        self.assertDictEqual(image_processor.to_dict(), reloaded_image_processor.to_dict())

    def test_trainer_saves_feature_extractor(self):
        MODEL_ID = "facebook/wav2vec2-base-960h"
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            gaudi_config = get_gaudi_config()
            trainer = GaudiTrainer(
                model=RegressionPreTrainedModel(config),
                args=GaudiTrainingArguments(output_dir=tmp_dir, use_habana=True, use_lazy_mode=True),
                gaudi_config=gaudi_config,
                processing_class=feature_extractor,
            )
            trainer.save_model()

            reloaded_feature_extractor = AutoFeatureExtractor.from_pretrained(tmp_dir)

        self.assertDictEqual(feature_extractor.to_dict(), reloaded_feature_extractor.to_dict())

    @require_vision
    def test_trainer_saves_processor(self):
        MODEL_ID = "openai/clip-vit-base-patch32"
        image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
        processor = AutoProcessor.from_pretrained(MODEL_ID)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = RegressionModelConfig(a=1.5, b=2.5)
            gaudi_config = get_gaudi_config()
            trainer = GaudiTrainer(
                model=RegressionPreTrainedModel(config),
                args=GaudiTrainingArguments(output_dir=tmp_dir, use_habana=True, use_lazy_mode=True),
                gaudi_config=gaudi_config,
                processing_class=processor,
            )
            trainer.save_model()

            reloaded_processor = AutoProcessor.from_pretrained(tmp_dir)
            reloaded_image_processor = AutoImageProcessor.from_pretrained(tmp_dir)
            reloaded_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)

        self.assertDictEqual(reloaded_processor.to_dict(), processor.to_dict())

        image_processor_dict = image_processor.to_dict()
        reloaded_image_processor_dict = reloaded_image_processor.to_dict()
        # When the processor is saved in the trainer, the _processor_class gets set in the reload_image_processor dict
        image_processor_dict.pop("_processor_class")
        reloaded_image_processor_dict.pop("_processor_class")
        self.assertDictEqual(image_processor_dict, reloaded_image_processor_dict)

        # For tokenizers, there isn't a direct to_dict method and the properties stored in the configs e.g.
        # saved tokens change overtime, so we check that two tokenizers are equal by comparing their encoded outputs
        test_sentence = "This is a test sentence"
        self.assertListEqual(
            tokenizer(test_sentence, padding="max_length").input_ids,
            reloaded_tokenizer(test_sentence, padding="max_length").input_ids,
        )

    def test_save_best_checkpoint(self):
        freq = int(64 / self.batch_size)
        total = int(self.n_epochs * 64 / self.batch_size)

        # Case 1: args.metric_for_best_model == "accuracy".
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_strategy="epoch",
                save_strategy="best",
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
            )
            self.assertTrue(trainer.args.metric_for_best_model == "accuracy")

            with unittest.mock.patch.object(
                trainer,
                "_evaluate",
                side_effect=[
                    {"eval_loss": 0.03, "eval_accuracy": 0.60, "epoch": 1.0},
                    {"eval_loss": 0.02, "eval_accuracy": 0.65, "epoch": 2.0},
                    {"eval_loss": 0.01, "eval_accuracy": 0.64, "epoch": 3.0},
                ],
            ):
                trainer.train()

                self.assertEqual(len(os.listdir(tmpdir)), 2)
                self.check_saved_checkpoints(
                    output_dir=tmpdir,
                    freq=freq,
                    total=total,
                )

        # Case 2: args.metric_for_best_model == "loss".
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_strategy="epoch",
                save_strategy="best",
                metric_for_best_model="loss",
                compute_metrics=AlmostAccuracy(),
            )
            self.assertTrue(trainer.args.metric_for_best_model == "loss")

            with unittest.mock.patch.object(
                trainer,
                "_evaluate",
                side_effect=[
                    {"eval_loss": 0.03, "eval_accuracy": 0.60, "epoch": 1.0},
                    {"eval_loss": 0.02, "eval_accuracy": 0.65, "epoch": 2.0},
                    {"eval_loss": 0.03, "eval_accuracy": 0.66, "epoch": 3.0},
                ],
            ):
                trainer.train()

                self.assertEqual(len(os.listdir(tmpdir)), 2)
                self.check_saved_checkpoints(
                    output_dir=tmpdir,
                    freq=freq,
                    total=total,
                )

    def test_metric_for_best_model_behavior(self):
        # Case 1: Metric name not provided when `save_strategy == "best"`.
        # Should raise ValueError.
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as context:
                trainer = get_regression_trainer(
                    a=1.5,
                    b=2.5,
                    output_dir=tmpdir,
                    learning_rate=0.1,
                    eval_strategy="epoch",
                    save_strategy="best",
                    compute_metrics=AlmostAccuracy(),
                )
            self.assertIn("`args.metric_for_best_model` must be provided", str(context.exception))

        # Case 2: Metric name not provided when `load_best_model_at_end == True`.
        # `metric_for_best_model` should be set to `"loss"` by default.
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                a=1.5,
                b=2.5,
                output_dir=tmpdir,
                learning_rate=0.1,
                eval_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
            )
            self.assertTrue(trainer.args.metric_for_best_model == "loss")

    def test_best_model_checkpoint_behavior(self):
        # Case 1. Never evaluated, save_total_limit > 1 and save_steps == 1.
        # Both best_metric and best_model_checkpoint should be None.
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                eval_strategy="steps",
                save_strategy="steps",
                save_steps=1,
                metric_for_best_model="accuracy",
                greater_is_better=True,
            )
            trainer.train()

            assert trainer.state.best_metric is None
            assert trainer.state.best_model_checkpoint is None
            assert len(os.listdir(tmpdir)) == trainer.state.global_step

        # Case 2. Never evaluated and save_total_limit == 1.
        # Both best_metric and best_model_checkpoint should be None.
        # Only the last checkpoint should remain.
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                eval_strategy="steps",
                save_strategy="steps",
                save_steps=1,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                save_total_limit=1,
            )
            trainer.train()

            num_steps = trainer.state.global_step

            assert trainer.state.best_metric is None
            assert trainer.state.best_model_checkpoint is None
            assert len(os.listdir(tmpdir)) == 1

            ckpt = os.path.join(tmpdir, f"{PREFIX_CHECKPOINT_DIR}-{num_steps}")
            assert os.path.isdir(ckpt)
            assert os.listdir(tmpdir)[0] == f"{PREFIX_CHECKPOINT_DIR}-{num_steps}"

        # Case 3. eval_strategy == save_strategy.
        # best_model_checkpoint should be at epoch 1.
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                eval_strategy="epoch",
                save_strategy="epoch",
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
                greater_is_better=True,
                load_best_model_at_end=False,
            )

            with unittest.mock.patch.object(
                trainer,
                "_evaluate",
                side_effect=evaluate_side_effect_factory(
                    [
                        {"eval_accuracy": 0.59},
                        {"eval_accuracy": 0.57},
                        {"eval_accuracy": 0.55},
                    ]
                ),
            ):
                trainer.train()

            steps_per_epoch = get_steps_per_epoch(trainer)

            assert trainer.state.best_metric == 0.59
            assert trainer.state.best_global_step == steps_per_epoch

            best_ckpt = os.path.join(tmpdir, f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.best_global_step}")
            assert trainer.state.best_model_checkpoint == best_ckpt

            assert len(os.listdir(tmpdir)) == trainer.state.num_train_epochs

        # Case 4. eval_strategy != save_strategy.
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                eval_strategy="epoch",
                save_strategy="steps",
                save_steps=1,
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
                greater_is_better=True,
                load_best_model_at_end=False,
            )

            with unittest.mock.patch.object(
                trainer,
                "_evaluate",
                side_effect=evaluate_side_effect_factory(
                    [
                        {"eval_accuracy": 0.59},
                        {"eval_accuracy": 0.57},
                        {"eval_accuracy": 0.55},
                    ]
                ),
            ):
                trainer.train()

            steps_per_epoch = get_steps_per_epoch(trainer)

            assert trainer.state.best_metric == 0.59
            assert trainer.state.best_global_step == steps_per_epoch

            best_ckpt = os.path.join(tmpdir, f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.best_global_step}")
            assert trainer.state.best_model_checkpoint == best_ckpt

            assert len(os.listdir(tmpdir)) == trainer.state.global_step

        # Case 5. Multiple checkpoints, save_total_limit == 1.
        # Best metric is found at step 1 and that checkpoint should be saved.
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                eval_strategy="steps",
                eval_steps=1,
                save_strategy="steps",
                save_steps=1,
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
                greater_is_better=True,
                save_total_limit=1,
            )

            with unittest.mock.patch.object(
                trainer,
                "_evaluate",
                side_effect=evaluate_side_effect_factory(
                    [
                        {"eval_accuracy": 0.90},
                        {"eval_accuracy": 0.80},
                        {"eval_accuracy": 0.70},
                    ]
                ),
            ):
                trainer.train()

            assert trainer.state.best_metric == 0.90
            assert trainer.state.best_global_step == 1

            best_ckpt = os.path.join(tmpdir, f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.best_global_step}")
            assert trainer.state.best_model_checkpoint == best_ckpt

            assert len(os.listdir(tmpdir)) == 1

        # Case 6. Saving happens more often and eval/save mismatch.
        # `best_model_checkpoint` should be None due to a step mismatch.
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = get_regression_trainer(
                output_dir=tmpdir,
                eval_strategy="steps",
                eval_steps=3,
                save_strategy="steps",
                save_steps=2,
                metric_for_best_model="accuracy",
                compute_metrics=AlmostAccuracy(),
                greater_is_better=True,
            )

            with unittest.mock.patch.object(
                trainer,
                "_evaluate",
                side_effect=evaluate_side_effect_factory(
                    [
                        {"eval_accuracy": 0.90},
                        {"eval_accuracy": 0.80},
                        {"eval_accuracy": 0.70},
                    ]
                ),
            ):
                trainer.train()

            assert trainer.state.best_metric == 0.90
            assert trainer.state.best_global_step == 3

            assert trainer.state.best_model_checkpoint is None

            assert len(os.listdir(tmpdir)) == trainer.state.global_step // 2

    def test_profiling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 24 total steps and compilation takes place during the 1st three steps
            trainer = get_regression_trainer(output_dir=tmp_dir, profiling_warmup_steps=3, profiling_steps=21)
            trainer.train()


@require_torch
@is_staging_test
class GaudiTrainerIntegrationWithHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    def test_push_to_hub(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            output_dir_name = tmp_repo.repo_name
            with tempfile.TemporaryDirectory() as tmp_dir:
                trainer = get_regression_trainer(
                    output_dir=os.path.join(tmp_dir, output_dir_name),
                    push_to_hub=True,
                    hub_token=self._token,
                )
                url = trainer.push_to_hub()

            # Extract repo_name from the url
            re_search = re.search(ENDPOINT_STAGING + r"/([^/]+/[^/]+)/", url)
            self.assertTrue(re_search is not None)
            repo_name = re_search.groups()[0]

            self.assertEqual(repo_name, f"{USER}/{output_dir_name}")

            model = RegressionPreTrainedModel.from_pretrained(repo_name)
            self.assertEqual(model.a.item(), trainer.model.a.item())
            self.assertEqual(model.b.item(), trainer.model.b.item())

    def test_push_to_hub_in_organization(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            with tempfile.TemporaryDirectory() as tmp_dir:
                trainer = get_regression_trainer(output_dir=tmp_dir)
                trainer.save_model()
                output_dir_name = tmp_repo.repo_name
                trainer = get_regression_trainer(
                    output_dir=os.path.join(tmp_dir, output_dir_name),
                    push_to_hub=True,
                    hub_model_id=f"valid_org/{output_dir_name}",
                    hub_token=self._token,
                )
                url = trainer.push_to_hub()

            # Extract repo_name from the url
            re_search = re.search(ENDPOINT_STAGING + r"/([^/]+/[^/]+)/", url)
            self.assertTrue(re_search is not None)
            repo_name = re_search.groups()[0]
            self.assertEqual(repo_name, f"valid_org/{output_dir_name}")

            model = RegressionPreTrainedModel.from_pretrained(f"valid_org/{output_dir_name}")
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
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            with tempfile.TemporaryDirectory() as tmp_dir:
                with self.assertLogs(level="WARNING") as logs:
                    output_dir_name = tmp_repo.repo_name
                    trainer = get_regression_trainer(
                        output_dir=os.path.join(tmp_dir, output_dir_name),
                        push_to_hub=True,
                        hub_token=self._token,
                        # To avoid any flakiness if the training goes faster than the uploads.
                        hub_always_push=True,
                        save_strategy="epoch",
                    )
                    trainer.train()

            commits = list_repo_commits(f"{USER}/{output_dir_name}", token=self._token)
            commits = [c.title for c in commits]
            self.assertIn("initial commit", commits)
            self.assertIn("Training in progress, epoch 1", commits)
            self.assertIn("Training in progress, epoch 2", commits)
            # Epochs 3 and 4 are not guaranteed to be present (empty commits)
            self.assertTrue(any("Skipping to prevent empty commit." in record.message for record in logs.records))

    def test_push_to_hub_with_saves_each_n_steps(self):
        num_gpus = max(1, get_gpu_count())
        if num_gpus > 2:
            self.skipTest(reason="More than 2 GPUs available")

        with TemporaryHubRepo(token=self._token) as tmp_repo:
            with tempfile.TemporaryDirectory() as tmp_dir:
                with self.assertLogs(level="WARNING") as logs:
                    output_dir_name = tmp_repo.repo_name
                    trainer = get_regression_trainer(
                        output_dir=os.path.join(tmp_dir, output_dir_name),
                        push_to_hub=True,
                        hub_token=self._token,
                        # To avoid any flakiness if the training goes faster than the uploads.
                        hub_always_push=True,
                        save_strategy="steps",
                        save_steps=5,
                    )
                    trainer.train()

            commits = list_repo_commits(f"{USER}/{output_dir_name}", token=self._token)
            commits = [c.title for c in commits]
            self.assertIn("initial commit", commits)

            # Some commits are skipped if nothing has changed
            # We expect 1 commit per 5 epochs + 1 commit at the end
            nb_empty_commits = len(
                [record for record in logs.records if "Skipping to prevent empty commit." in record.message]
            )
            nb_epoch_commits = len([commit for commit in commits if "Training in progress, step" in commit])

            # max_steps depend on the number of available GPUs
            max_steps = math.ceil(trainer.args.num_train_epochs * len(trainer.get_train_dataloader()))
            nb_expected_commits = len(range(5, max_steps, 5))

            # '>=' since final commit might be an empty commit as well (not deterministic)
            self.assertGreaterEqual(nb_empty_commits + nb_epoch_commits, nb_expected_commits)

    @require_tensorboard
    def test_push_to_hub_with_tensorboard_logs(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_dir_name = tmp_repo.repo_name
                trainer = get_regression_trainer(
                    output_dir=os.path.join(tmp_dir, output_dir_name),
                    hub_token=self._token,
                    save_strategy="epoch",
                    report_to=["tensorboard"],
                    keep_report_to=True,
                )
                trainer.train()
                # Push the runs via `push_to_hub()`
                trainer.push_to_hub()

            files = list_repo_files(f"{USER}/{output_dir_name}", token=self._token)
            found_log = False
            for f in files:
                if len(f.split("runs")) > 1 and "events.out.tfevents" in f:
                    found_log = True

            assert found_log is True, "No tensorboard log found in repo"

    def test_push_to_hub_tags(self):
        # Checks if `trainer.push_to_hub()` works correctly by adding the desired
        # tag without having to pass `tags` in `push_to_hub`
        # see:
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_dir_name = tmp_repo.repo_name
                trainer = get_regression_trainer(
                    output_dir=os.path.join(tmp_dir, output_dir_name),
                    push_to_hub=True,
                    hub_token=self._token,
                )

                trainer.model.add_model_tags(["test-trainer-tags"])

                url = trainer.push_to_hub()

            # Extract repo_name from the url
            re_search = re.search(ENDPOINT_STAGING + r"/([^/]+/[^/]+)/", url)
            self.assertTrue(re_search is not None)
            repo_name = re_search.groups()[0]

            self.assertEqual(repo_name, f"{USER}/{output_dir_name}")

            model_card = ModelCard.load(repo_name)
            self.assertTrue("test-trainer-tags" in model_card.data.tags)

    def test_push_to_hub_with_revision(self):
        # Checks if `trainer.push_to_hub()` works correctly by adding revision
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_dir_name = tmp_repo.repo_name
                trainer = get_regression_trainer(
                    output_dir=os.path.join(tmp_dir, output_dir_name),
                    push_to_hub=True,
                    hub_token=self._token,
                )
                branch = "v1.0"
                create_branch(repo_id=trainer.hub_model_id, branch=branch, token=self._token, exist_ok=True)
                url = trainer.push_to_hub(revision=branch)

            # Extract branch from the url
            re_search = re.search(r"tree/([^/]+)/", url)
            self.assertIsNotNone(re_search)

            branch_name = re_search.groups()[0]
            self.assertEqual(branch_name, branch)


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
                eval_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                num_train_epochs=4,
                disable_tqdm=True,
                load_best_model_at_end=True,
                logging_dir="runs",
                run_name="test",
                model_init=model_init,
            )
            trainer.hyperparameter_search(direction="minimize", hp_space=hp_space, hp_name=hp_name, n_trials=4)


@require_torch
@require_optuna
class TrainerHyperParameterMultiObjectOptunaIntegrationTest(unittest.TestCase):
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

        def compute_objective(metrics: dict[str, float]) -> list[float]:
            return metrics["eval_loss"], metrics["eval_accuracy"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                learning_rate=0.1,
                logging_steps=1,
                eval_strategy=IntervalStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
                num_train_epochs=10,
                disable_tqdm=True,
                load_best_model_at_end=True,
                logging_dir="runs",
                run_name="test",
                model_init=model_init,
                compute_metrics=AlmostAccuracy(),
            )
            trainer.hyperparameter_search(
                direction=["minimize", "maximize"],
                hp_space=hp_space,
                hp_name=hp_name,
                n_trials=4,
                compute_objective=compute_objective,
            )


@require_torch
@require_optuna
class TrainerHyperParameterOptunaIntegrationTestWithFullEval(unittest.TestCase):
    def test_hyperparameter_search(self):
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = get_regression_trainer(
                output_dir=tmp_dir,
                disable_tqdm=True,
                model_init=model_init,
            )
            trainer.hyperparameter_search(
                direction="minimize",
                hp_space=hp_space,
                n_trials=2,
            )


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
#                 eval_strategy=IntervalStrategy.EPOCH,
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
#                 eval_strategy=IntervalStrategy.EPOCH,
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
        with tempfile.TemporaryDirectory() as tmp_dir:
            # exercises all the valid --optim options
            self.check_optim_and_kwargs(name, mandatory_kwargs, expected_cls)

            trainer = get_regression_trainer(output_dir=tmp_dir, optim=name)
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

#         with tempfile.TemporaryDirectory() as tmp_dir:
#             trainer = get_regression_trainer(
#                 output_dir=tmp_dir,
#                 learning_rate=0.1,
#                 logging_steps=1,
#                 eval_strategy=IntervalStrategy.EPOCH,
#                 save_strategy=IntervalStrategy.EPOCH,
#                 num_train_epochs=4,
#                 disable_tqdm=True,
#                 load_best_model_at_end=True,
#                 logging_dir="runs",
#                 run_name="test",
#                 model_init=model_init,
#             )
#             sweep_kwargs = {
#                "direction": "minimize",
#                "hp_space": hp_space,
#                "backend": "wandb",
#                "n_trials": 4,
#            }
#            best_run = trainer.hyperparameter_search(**sweep_kwargs)

#            self.assertIsNotNone(best_run.run_id)
#            self.assertIsNotNone(best_run.run_summary)
#            hp_keys = set(best_run.hyperparameters.keys())
#            self.assertSetEqual(hp_keys, {"a", "b", "assignments", "metric"})

#            # pretend restarting the process purged the environ
#            import os

#            del os.environ["WANDB_ENTITY"]
#            del os.environ["WANDB_PROJECT"]
#            sweep_kwargs["sweep_id"] = best_run.run_summary
#            updated_best_run = trainer.hyperparameter_search(**sweep_kwargs)

#            self.assertIsNotNone(updated_best_run.run_id)
#            self.assertEqual(updated_best_run.run_summary, best_run.run_summary)
#            updated_hp_keys = set(updated_best_run.hyperparameters.keys())
#            self.assertSetEqual(updated_hp_keys, {"a", "b", "assignments", "metric"})


class HyperParameterSearchBackendsTest(unittest.TestCase):
    def test_hyperparameter_search_backends(self):
        self.assertEqual(
            list(ALL_HYPERPARAMETER_SEARCH_BACKENDS.keys()),
            list(HPSearchBackend),
        )


@require_torch
class OptimizerAndModelInspectionTest(unittest.TestCase):
    def test_get_num_trainable_parameters(self):
        model = nn.Sequential(nn.Linear(128, 64), nn.Linear(64, 32))
        # in_features * out_features + bias
        layer_1 = 128 * 64 + 64
        layer_2 = 64 * 32 + 32
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = GaudiTrainingArguments(
                output_dir=tmp_dir,
                use_habana=True,
                use_lazy_mode=True,
                report_to="none",
            )
            trainer = GaudiTrainer(model=model, gaudi_config=get_gaudi_config(), args=args)
            self.assertEqual(trainer.get_num_trainable_parameters(), layer_1 + layer_2)
            # Freeze the last layer
            for param in model[-1].parameters():
                param.requires_grad = False
            self.assertEqual(trainer.get_num_trainable_parameters(), layer_1)

    def test_get_learning_rates(self):
        model = nn.Sequential(nn.Linear(128, 64))
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = GaudiTrainingArguments(
                output_dir=tmp_dir,
                use_habana=True,
                use_lazy_mode=True,
                report_to="none",
            )
            trainer = GaudiTrainer(model=model, gaudi_config=get_gaudi_config(), args=args)
            with self.assertRaises(ValueError):
                trainer.get_learning_rates()
            trainer.create_optimizer()
            self.assertEqual(trainer.get_learning_rates(), [5e-05, 5e-05])

    def test_get_optimizer_group(self):
        model = nn.Sequential(nn.Linear(128, 64))
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = GaudiTrainingArguments(
                output_dir=tmp_dir,
                use_habana=True,
                use_lazy_mode=True,
                report_to="none",
            )
            trainer = GaudiTrainer(model=model, gaudi_config=get_gaudi_config(), args=args)
            # ValueError is raised if optimizer is None
            with self.assertRaises(ValueError):
                trainer.get_optimizer_group()
            trainer.create_optimizer()
            # Get groups
            num_groups = len(trainer.get_optimizer_group())
            self.assertEqual(num_groups, 2)
            # Get group of parameter
            param = next(model.parameters())
            group = trainer.get_optimizer_group(param)
            self.assertIn(param, group["params"])
