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
from dataclasses import dataclass, field
from typing import Optional

from optimum.utils import logging
from transformers.file_utils import (
    cached_property,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_tpu_available,
    torch_required,
)
from transformers.training_args import TrainingArguments


if is_torch_available():
    import torch

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as sm_dist

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    smp.init()


logger = logging.get_logger(__name__)


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
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif is_sagemaker_mp_enabled():
            local_rank = smp.local_rank()
            device = torch.device("cuda", local_rank)
            self._n_gpu = 1
        elif is_sagemaker_dp_enabled():
            sm_dist.init_process_group()
            self.local_rank = sm_dist.get_local_rank()
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.deepspeed:
            # deepspeed inits torch.distributed internally
            from transformers.deepspeed import is_deepspeed_available

            if not is_deepspeed_available():
                raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
            import deepspeed

            deepspeed.init_distributed()

            # workaround for setups like notebooks where the launcher can't be used,
            # but deepspeed requires a dist env.
            # env LOCAL_RANK could be set manually by the user, or via init_distributed if mpi4py is installed
            self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.use_habana:
            try:
                from habana_frameworks.torch.utils.library_loader import is_habana_avaialble
            except ImportError as error:
                error.msg = (
                    f"Could not import is_habana_avaialble from habana_frameworks.utils.library_loader. {error.msg}."
                )
                raise error
            if not is_habana_avaialble():
                raise RuntimeError("Habana is not available.")
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
                import habana_frameworks.torch.core.hccl

                os.environ["ID"] = str(self.local_rank)
                torch.distributed.init_process_group(backend="hccl", rank=self.local_rank, world_size=world_size)
                logger.info("Enabled distributed run.")

        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device
