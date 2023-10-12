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

from pathlib import Path
from typing import Dict

from transformers import EvalPrediction, HfArgumentParser, is_torch_available
from transformers.testing_utils import TestCasePlus

from optimum.habana import GaudiConfig, GaudiTrainingArguments
from optimum.habana.distributed import DistributedRunner
from optimum.utils import logging


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data import Dataset

    from optimum.habana import GaudiTrainer

    class DummyDataset(Dataset):
        def __init__(self, length: int = 101):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, i) -> int:
            return i

    class DummyDataCollator:
        def __call__(self, features):
            return {"input_ids": torch.tensor(features), "labels": torch.tensor(features)}

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Add some (unused) params otherwise DDP will complain.
            self.fc = nn.Linear(120, 80)

        def forward(self, input_ids, labels=None):
            if labels is not None:
                return torch.tensor(0.0, device=input_ids.device), input_ids
            else:
                return input_ids


class TestGaudiTrainerDistributed(TestCasePlus):
    def _test_gaudi_trainer_distributed(self, kwargs={}):
        output_dir = self.get_auto_remove_tmp_dir()

        command_list = [f"{self.test_file_dir}/test_trainer_distributed.py"]
        command_list += ["--output_dir"]
        command_list += [output_dir]
        command_list += ["--use_habana"]
        command_list += ["--use_lazy_mode"]
        for key, value in kwargs.items():
            command_list += [f"--{key} {value}"]
        command = [" ".join(command_list)]

        distributed_runner = DistributedRunner(
            command_list=command,
            world_size=8,
            use_mpi=True,
        )

        ret_code = distributed_runner.run()

        # ret_code equals 0 or None if successful run
        self.assertTrue(ret_code == 0 or ret_code is None)

    def test_gaudi_trainer_distributed(self):
        self._test_gaudi_trainer_distributed()

    def test_gaudi_trainer_distributed_hpu_graphs(self):
        self._test_gaudi_trainer_distributed(
            {
                "use_hpu_graphs_for_training": "",
                "use_hpu_graphs_for_inference": "",
                "distribution_strategy": "fast_ddp",
            }
        )


if __name__ == "__main__":
    # The script below is meant to be run under mpirun, on a machine with multiple HPUs:
    #
    # PYTHONPATH="src" python optimum-habana/examples/gaudi_spawn.py --world_size 8 --use_mpi --output_dir output_dir ./tests/test_trainer_distributed.py

    parser = HfArgumentParser((GaudiTrainingArguments,))
    training_args = parser.parse_args_into_dataclasses()[0]

    gaudi_config_file = Path(__file__).parent.resolve() / Path("configs/gaudi_config_trainer_test.json")
    gaudi_config = GaudiConfig.from_pretrained(gaudi_config_file)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_hpu: {training_args.world_size},"
        f" distributed training: {training_args.local_rank != -1}"
    )

    # Essentially, what we want to verify in the distributed case is that we get all samples back,
    # in the right order. (this is crucial for prediction for instance)
    for dataset_length in [101, 40, 7]:
        dataset = DummyDataset(dataset_length)

        def compute_metrics(p: EvalPrediction) -> Dict:
            sequential = list(range(len(dataset)))
            success = p.predictions.tolist() == sequential and p.label_ids.tolist() == sequential
            if not success and training_args.local_rank == 0:
                logger.warning(
                    "Predictions and/or labels do not match expected results:\n  - predictions: "
                    f"{p.predictions.tolist()}\n  - labels: {p.label_ids.tolist()}\n  - expected: {sequential}"
                )
            return {"success": success}

        trainer = GaudiTrainer(
            model=DummyModel(),
            gaudi_config=gaudi_config,
            args=training_args,
            data_collator=DummyDataCollator(),
            eval_dataset=dataset,
            compute_metrics=compute_metrics,
        )

        metrics = trainer.evaluate()
        logger.info(metrics)
        if metrics["eval_success"] is not True:
            logger.error(metrics)
            exit(1)

        p = trainer.predict(dataset)
        logger.info(p.metrics)
        if p.metrics["test_success"] is not True:
            logger.error(p.metrics)
            exit(1)

        trainer.args.eval_accumulation_steps = 2

        metrics = trainer.evaluate()
        logger.info(metrics)
        if metrics["eval_success"] is not True:
            logger.error(metrics)
            exit(1)

        p = trainer.predict(dataset)
        logger.info(p.metrics)
        if p.metrics["test_success"] is not True:
            logger.error(p.metrics)
            exit(1)

        trainer.args.eval_accumulation_steps = None
