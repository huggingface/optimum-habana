#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from optimum.utils import logging
from transformers.file_utils import cached_property, is_torch_available, torch_required
from transformers.training_args import TrainingArguments


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


# List of arguments that are not supported by optimum-habana
UNSUPPORTED_ARGUMENTS = [
    "bf16",  # bf16 for CUDA devices
    "bf16_full_eval",  # bf16 for CUDA devices
    "deepspeed",
    "fp16",
    "fp16_backend",
    "fp16_full_eval",
    "fp16_opt_level",
    "half_precision_backend",  # not supported, Habana Mixed Precision should be used and specified in Gaudi configuration
    "mp_parameters",
    "sharded_ddp",
    "tf32",
    "tpu_metrics_debug",
    "tpu_num_cores",
]


@dataclass
class GaudiTrainingArguments(TrainingArguments):
    """
    GaudiTrainingArguments is built on top of the tranformers' TrainingArguments
    to enable deployment on Habana's Gaudi.
    """

    use_habana: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use Habana's HPU for training the model."}
    )

    gaudi_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained Gaudi config name or path if not the same as model_name."}
    )

    use_lazy_mode: bool = field(
        default=False,
        metadata={"help": "Whether to use lazy mode for training the model."},
    )

    adam_epsilon: float = field(default=1e-6, metadata={"help": "Epsilon for AdamW optimizer."})

    def __post_init__(self):
        # Raise errors for arguments that are not supported by optimum-habana
        if self.bf16 or self.bf16_full_eval:
            raise ValueError(
                "--bf16 and --bf16_full_eval are not supported by optimum-habana. You should turn on Habana Mixed Precision in your Gaudi configuration to enable bf16."
            )
        if self.fp16 or self.fp16_full_eval:
            raise ValueError(
                "--fp16, --fp16_backend, --fp16_full_eval, --fp16_opt_level and --half_precision_backend are not supported by optimum-habana. Mixed-precision training can be enabled in your Gaudi configuration."
            )
        if self.tpu_num_cores or self.tpu_metrics_debug:
            raise ValueError("TPUs are not supported by optimum-habana.")
        if self.deepspeed:
            raise ValueError("--deepspeed is not supported by optimum-habana.")
        if self.mp_parameters:
            raise ValueError("--mp_parameters is not supported by optimum-habana.")
        if self.sharded_ddp:
            raise ValueError("--sharded_ddp is not supported by optimum-habana.")
        if self.tf32:
            raise ValueError("--tf32 is not supported by optimum-habana.")

        super().__post_init__()

    def __str__(self):
        self_as_dict = asdict(self)

        # Remove deprecated arguments. That code should be removed once
        # those deprecated arguments are removed from TrainingArguments. (TODO: transformers v5)
        del self_as_dict["per_gpu_train_batch_size"]
        del self_as_dict["per_gpu_eval_batch_size"]
        # Remove arguments that are unsupported by optimum-habana
        for key in UNSUPPORTED_ARGUMENTS:
            del self_as_dict[key]

        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
            if self.local_rank != -1:
                # Initializes distributed backend for cpu
                if self.xpu_backend not in ("mpi", "ccl"):
                    raise ValueError(
                        "CPU distributed training backend is not properly set. "
                        "Please set '--xpu_backend' to either 'mpi' or 'ccl'."
                    )
                torch.distributed.init_process_group(backend=self.xpu_backend)
        elif self.use_habana:
            logger.info("Habana is enabled.")

            if self.use_lazy_mode:
                logger.info("Enabled lazy mode.")
            else:
                os.environ["PT_HPU_LAZY_MODE"] = "2"
                logger.info("Enabled eager mode because use_lazy_mode=False.")
            from habana_frameworks.torch.utils.library_loader import load_habana_module

            load_habana_module()
            device = torch.device("hpu")
            self._n_gpu = 1
            world_size = 1
            rank = -1

            if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
                world_size = int(os.environ["WORLD_SIZE"])
                rank = int(os.environ["RANK"])
                if "LOCAL_RANK" in os.environ:
                    self.local_rank = int(os.environ["LOCAL_RANK"])
                logger.info("Torch distributed launch used")
            elif (
                "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ
                and "OMPI_COMM_WORLD_SIZE" in os.environ
                and "OMPI_COMM_WORLD_RANK" in os.environ
            ):
                self.local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
                world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
                rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
                logger.info("MPI environment variables set")
            else:
                try:
                    global mpi_comm
                    from mpi4py import MPI

                    mpi_comm = MPI.COMM_WORLD
                    world_size = mpi_comm.Get_size()
                    if world_size > 1:
                        rank = mpi_comm.Get_rank()
                        self.local_rank = rank
                    else:
                        raise ("Single MPI process")
                except Exception as e:
                    logger.info("Single node run")

            if self.local_rank != -1:
                try:
                    import habana_frameworks.torch.core.hccl
                except ImportError as error:
                    error.msg = f"Could not import habana_frameworks.torch.core.hccl. {error.msg}."
                    raise error
                os.environ["ID"] = str(self.local_rank)
                torch.distributed.init_process_group(backend="hccl", rank=self.local_rank, world_size=world_size)
                logger.info("Enabled distributed run.")

        return device
