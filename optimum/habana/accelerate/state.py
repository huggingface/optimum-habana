# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import os

import accelerate
import torch
from accelerate import DistributedType
from accelerate.state import PartialState
from accelerate.utils import is_deepspeed_available, parse_flag_from_env

from optimum.utils import logging

from ..distributed import parallel_state


logger = logging.get_logger()


# TODO: Remove when minimize_memory is supported in upstream accelerate
# and sequence/context parallelism is managed in GaudiTrainer or supported in upstream accelerate
class GaudiPartialState(PartialState):
    """
    Adapted from: https://github.com/huggingface/accelerate/blob/8514c35192ac9762920f1ab052e5cea4c0e46eeb/src/accelerate/state.py#L96
    """

    def __init__(self, cpu: bool = False, **kwargs):
        self.__dict__ = self._shared_state
        if not self.initialized:
            self._cpu = cpu
            self.backend = None
            env_device = os.environ.get("ACCELERATE_TORCH_DEVICE", None)
            self.device = torch.device(env_device) if env_device is not None else None
            self.debug = parse_flag_from_env("ACCELERATE_DEBUG_MODE")

            # initialize_distributed_hpu is already called in the __init__ of
            # habana_frameworks.torch.distributed.hccl
            # It is necessary so that the env variable LOCAL_RANK is set before the
            # conditional statement right below
            from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

            if int(os.environ.get("LOCAL_RANK", -1)) != -1 and not cpu:
                world_size, rank, local_rank = initialize_distributed_hpu()
                self.backend = kwargs.pop("backend", "hccl")
                context_parallel_size = kwargs.pop("context_parallel_size", 1)
                self.minimize_memory = kwargs.pop("minimize_memory", False)
                if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true":
                    if not is_deepspeed_available():
                        raise ImportError(
                            "DeepSpeed is not available, install it with: `pip install"
                            " git+https://github.com/HabanaAI/DeepSpeed.git@1.20.0`."
                        )
                    self.distributed_type = DistributedType.DEEPSPEED
                    import deepspeed

                    if world_size > 1:
                        # override HLS_MODULE_ID only if it's not previously set by bridge
                        if "HLS_MODULE_ID" not in os.environ:
                            os.environ["HLS_MODULE_ID"] = str(local_rank)
                        os.environ["ID"] = str(rank)

                    deepspeed.init_distributed(dist_backend=self.backend, **kwargs)
                    logger.info("DeepSpeed is enabled.")
                    self._mixed_precision = "no"  # deepspeed handles mixed_precision using deepspeed_config
                elif os.environ.get("ACCELERATE_USE_FSDP", "false") == "true":
                    self.distributed_type = DistributedType.FSDP
                    if not torch.distributed.is_initialized():
                        torch.distributed.init_process_group(backend=self.backend, rank=rank, world_size=world_size)
                        logger.info("Enabled distributed run.")
                else:
                    self.distributed_type = DistributedType.MULTI_HPU
                    if not torch.distributed.is_initialized():
                        torch.distributed.init_process_group(backend=self.backend, rank=rank, world_size=world_size)
                        logger.info("Enabled distributed run.")
                self.num_processes = world_size
                self.process_index = rank
                self.local_process_index = local_rank
                if self.device is None:
                    # TODO: replace by `torch.device("hpu", self.local_process_index)` when hpu:x is supported
                    self.device = torch.device("hpu")
                if not is_deepspeed_available():
                    context_parallel_size = 1
                if parallel_state.is_unitialized():
                    parallel_state.initialize_model_parallel(
                        sequence_parallel_size=context_parallel_size, use_fp8=False
                    )
                else:
                    if parallel_state.get_sequence_parallel_world_size() != context_parallel_size:
                        raise ValueError(
                            "The initialized sequence parallel world size does not match the context parallel size."
                        )
                    if parallel_state.amax_reduction_is_initialized():
                        logger.info("FP8 amax reduction group is already initialized.")
            else:
                self.distributed_type = (
                    DistributedType.NO
                    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "false"
                    else DistributedType.DEEPSPEED
                )
                self.num_processes = 1
                self.process_index = self.local_process_index = 0
                logger.info("Single-device run.")

                if self.device is None:
                    self.device = torch.device("cpu") if cpu else self.default_device

        self.fork_launched = parse_flag_from_env("FORK_LAUNCHED", 0)


# monkey patching
accelerate.PartialState = GaudiPartialState
