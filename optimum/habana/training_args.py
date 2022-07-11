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
from transformers.training_args import TrainingArguments, get_int_from_env


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
        default=False,
        metadata={"help": "Whether to use Habana's HPU for training the model."},
    )

    gaudi_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained Gaudi config name or path if not the same as model_name."},
    )

    use_lazy_mode: bool = field(
        default=False,
        metadata={"help": "Whether to use lazy mode for training the model."},
    )

    throughput_warmup_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Number of steps to ignore for throughput calculation. For example, with throughput_warmup_steps=N,"
                " the first N steps will not be considered in the calculation of the throughput. This is especially"
                " useful in lazy mode."
            )
        },
    )

    # Override the default value of epsilon to be consistent with Habana FusedAdamW
    adam_epsilon: float = field(
        default=1e-6,
        metadata={"help": "Epsilon for AdamW optimizer."},
    )

    # Override logging_nan_inf_filter to make False the default value
    logging_nan_inf_filter: bool = field(
        default=False,
        metadata={"help": "Filter nan and inf losses for logging."},
    )

    def __post_init__(self):
        if (self.use_lazy_mode or self.gaudi_config_name) and not self.use_habana:
            raise ValueError("--use_lazy_mode and --gaudi_config_name cannot be used without --use_habana")

        # Raise errors for arguments that are not supported by optimum-habana
        if self.bf16 or self.bf16_full_eval:
            raise ValueError(
                "--bf16 and --bf16_full_eval are not supported by optimum-habana. You should turn on Habana Mixed"
                " Precision in your Gaudi configuration to enable bf16."
            )
        if self.fp16 or self.fp16_full_eval:
            raise ValueError(
                "--fp16, --fp16_backend, --fp16_full_eval, --fp16_opt_level and --half_precision_backend are not"
                " supported by optimum-habana. Mixed-precision training can be enabled in your Gaudi configuration."
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

        if self.throughput_warmup_steps < 0:
            raise ValueError("--throughput_warmup_steps must be positive.")

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
        # Set the log level here for optimum.utils.logging
        # otherwise logs are not sent in this method.
        log_level = self.get_process_log_level()
        logging.set_verbosity(log_level)

        logger.info("PyTorch: setting up devices")
        if torch.distributed.is_available() and torch.distributed.is_initialized() and self.local_rank == -1:
            logger.warning("torch.distributed process group is initialized, but local_rank == -1. ")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
            self.local_rank = get_int_from_env(
                ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"],
                self.local_rank,
            )
            if self.local_rank != -1 and not torch.distributed.is_initialized():
                # Initializes distributed backend for cpu
                if self.xpu_backend not in ("mpi", "ccl"):
                    raise ValueError(
                        "CPU distributed training backend is not properly set. "
                        "Please set '--xpu_backend' to either 'mpi' or 'ccl'."
                    )
                if self.xpu_backend == "ccl" and int(os.environ.get("CCL_WORKER_COUNT", 0)) < 1:
                    raise ValueError(
                        "CPU distributed training backend is ccl. but CCL_WORKER_COUNT is not correctly set. "
                        "Please use like 'export CCL_WORKER_COUNT = 1' to set."
                    )

                # Try to get launch configuration from environment variables set by MPI launcher - works for Intel MPI, OpenMPI and MVAPICH
                rank = get_int_from_env(["RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"], 0)
                size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], 1)
                local_size = get_int_from_env(
                    ["MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"], 1
                )
                os.environ["RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = str(size)
                os.environ["LOCAL_RANK"] = str(self.local_rank)
                if not os.environ.get("MASTER_PORT", None):
                    os.environ["MASTER_PORT"] = "29500"
                if not os.environ.get("MASTER_ADDR", None):
                    if local_size != size or self.xpu_backend != "mpi":
                        raise ValueError(
                            "Looks like distributed multinode run but MASTER_ADDR env not set, "
                            "please try exporting rank 0's hostname as MASTER_ADDR"
                        )
                torch.distributed.init_process_group(backend=self.xpu_backend, rank=rank, world_size=size)
        elif self.use_habana:
            import habana_frameworks.torch.hpu as hthpu

            if hthpu.is_available():
                logger.info("Habana is enabled.")
            else:
                raise RuntimeError("No HPU is currently available.")

            if self.use_lazy_mode:
                logger.info("Enabled lazy mode.")
            else:
                os.environ["PT_HPU_LAZY_MODE"] = "2"
                logger.info("Enabled eager mode because use_lazy_mode=False.")

            device = torch.device("hpu")
            self._n_gpu = 1
            from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

            world_size, rank, self.local_rank = initialize_distributed_hpu()

            if self.local_rank != -1:
                if world_size > hthpu.device_count():
                    raise RuntimeError(
                        f"world_size is equal to {world_size} but there are only {hthpu.device_count()} devices."
                    )
                torch.distributed.init_process_group(backend="hccl", rank=self.local_rank, world_size=world_size)
                logger.info("Enabled distributed run.")
            else:
                logger.info("Single node run.")
        else:
            raise ValueError(
                "No device has been set. Use either --use_habana to run on HPU or --no_cuda to run on CPU."
            )

        return device
