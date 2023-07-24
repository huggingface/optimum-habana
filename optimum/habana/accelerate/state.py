import os

import torch
from accelerate.state import AcceleratorState, PartialState
from accelerate.utils import is_deepspeed_available, parse_choice_from_env, parse_flag_from_env

from optimum.utils import logging

from .utils import GaudiDistributedType


logger = logging.get_logger()


class GaudiPartialState(PartialState):
    """
    Singleton class that has information about the current training environment and functions to help with process
    control. Designed to be used when only process control and device execution states are needed. Does *not* need to
    be initialized from `Accelerator`.

    **Available attributes:**

        - **device** (`torch.device`) -- The device to use.
        - **distributed_type** ([`GaudiDistributedType`]) -- The type of distributed environment currently
          in use.
        - **local_process_index** (`int`) -- The index of the current process on the current server.
        - **mixed_precision** (`str`) -- Whether or not the current script will use mixed precision, and if so the type
          of mixed precision being performed.
        - **num_processes** (`int`) -- The number of processes currently launched in parallel.
        - **process_index** (`int`) -- The index of the current process.
        - **is_last_process** (`bool`) -- Whether or not the current process is the last one.
        - **is_main_process** (`bool`) -- Whether or not the current process is the main one.
        - **is_local_main_process** (`bool`) -- Whether or not the current process is the main one on the local node.
    """

    def __init__(self, cpu: bool = False, **kwargs):
        self.__dict__ = self._shared_state
        if not self.initialized:
            self._cpu = cpu
            self.backend = None
            env_device = os.environ.get("ACCELERATE_TORCH_DEVICE", None)
            self.device = torch.device(env_device) if env_device is not None else None

            if int(os.environ.get("LOCAL_RANK", -1)) != -1 and not cpu:
                from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

                world_size, rank, local_rank = initialize_distributed_hpu()
                self.backend = kwargs.pop("backend", "hccl")

                if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true":
                    if not is_deepspeed_available():
                        raise ImportError(
                            "DeepSpeed is not available, install it with: `pip install"
                            " git+https://github.com/HabanaAI/DeepSpeed.git@1.10.0`."
                        )
                    self.distributed_type = GaudiDistributedType.DEEPSPEED
                    if not torch.distributed.is_initialized():
                        import deepspeed

                        if world_size > 1:
                            os.environ["HLS_MODULE_ID"] = str(local_rank)
                            os.environ["ID"] = str(rank)

                        deepspeed.init_distributed(dist_backend=self.backend, **kwargs)
                        logger.info("DeepSpeed is enabled.")
                    self._mixed_precision = "no"  # deepspeed handles mixed_precision using deepspeed_config
                else:
                    self.distributed_type = GaudiDistributedType.MULTI_HPU
                    if not torch.distributed.is_initialized():
                        torch.distributed.init_process_group(backend=self.backend, rank=rank, world_size=world_size)
                        logger.info("Enabled distributed run.")
                self.num_processes = world_size
                self.process_index = rank
                self.local_process_index = local_rank
                if self.device is None:
                    self.device = torch.device("hpu", self.local_process_index)
            else:
                self.distributed_type = GaudiDistributedType.NO
                self.num_processes = 1
                self.process_index = self.local_process_index = 0
                logger.info("Single-device run.")

                if self.device is None:
                    self.device = torch.device("cpu") if cpu else self.default_device

        self.fork_launched = parse_flag_from_env("FORK_LAUNCHED", 0)

    def wait_for_everyone(self):
        """
        Will stop the execution of the current process until every other process has reached that point (so this does
        nothing when the script is only run in one process). Useful to do before saving a model.

        Example:

        ```python
        >>> # Assuming two GPU processes
        >>> import time
        >>> from accelerate.state import PartialState

        >>> state = PartialState()
        >>> if state.is_main_process:
        ...     time.sleep(2)
        >>> else:
        ...     print("I'm waiting for the main process to finish its sleep...")
        >>> state.wait_for_everyone()
        >>> # Should print on every process at the same time
        >>> print("Everyone is here")
        ```
        """
        if self.distributed_type in (
            GaudiDistributedType.MULTI_CPU,
            GaudiDistributedType.DEEPSPEED,
            GaudiDistributedType.MULTI_HPU,
        ):
            torch.distributed.barrier()

    @property
    def default_device(self) -> torch.device:
        """
        Returns the default device which is:
        - NPU if `is_npu_available()`
        - CPU otherwise
        """
        import habana_frameworks.torch.hpu as hthpu

        if hthpu.is_available():
            return torch.device("hpu")
        else:
            return torch.device("cpu")


class GaudiAcceleratorState(AcceleratorState):
    """
    Singleton class that has information about the current training environment.

    **Available attributes:**

        - **device** (`torch.device`) -- The device to use.
        - **distributed_type** ([`GaudiDistributedType`]) -- The type of distributed environment currently
          in use.
        - **initialized** (`bool`) -- Whether or not the `AcceleratorState` has been initialized from `Accelerator`.
        - **local_process_index** (`int`) -- The index of the current process on the current server.
        - **mixed_precision** (`str`) -- Whether or not the current script will use mixed precision, and if so the type
          of mixed precision being performed.
        - **num_processes** (`int`) -- The number of processes currently launched in parallel.
        - **process_index** (`int`) -- The index of the current process.
        - **is_last_process** (`bool`) -- Whether or not the current process is the last one.
        - **is_main_process** (`bool`) -- Whether or not the current process is the main one.
        - **is_local_main_process** (`bool`) -- Whether or not the current process is the main one on the local node.
    """

    def __init__(
        self,
        mixed_precision: str = None,
        cpu: bool = False,
        dynamo_plugin=None,
        deepspeed_plugin=None,
        fsdp_plugin=None,
        megatron_lm_plugin=None,
        _from_accelerator: bool = False,
        **kwargs,
    ):
        self.__dict__ = self._shared_state
        if parse_flag_from_env("ACCELERATE_USE_CPU"):
            cpu = True
        if GaudiPartialState._shared_state == {}:
            GaudiPartialState(cpu, **kwargs)
        self.__dict__.update(GaudiPartialState._shared_state)
        self._check_initialized(mixed_precision, cpu)
        if not self.initialized:
            self.deepspeed_plugin = None
            mixed_precision = (
                parse_choice_from_env("ACCELERATE_MIXED_PRECISION", "no")
                if mixed_precision is None
                else mixed_precision.lower()
            )
            self.dynamo_plugin = dynamo_plugin
            # deepspeed handles mixed_precision using deepspeed_config
            self._mixed_precision = (
                "no" if self.distributed_type == GaudiDistributedType.DEEPSPEED else mixed_precision
            )
            if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true" and not cpu:
                self.deepspeed_plugin = deepspeed_plugin
            GaudiPartialState._shared_state["distributed_type"] = self.distributed_type

    @property
    def mixed_precision(self):
        if self.distributed_type == GaudiDistributedType.DEEPSPEED:
            config = self.deepspeed_plugin.deepspeed_config
            if config.get("fp16", {}).get("enabled", False):
                mixed_precision = "fp16"
            elif config.get("bf16", {}).get("enabled", False):
                mixed_precision = "bf16"
            else:
                mixed_precision = "no"
        else:
            mixed_precision = self._mixed_precision

        if mixed_precision == "fp16":
            raise ValueError("fp16 is not supported on Habana Gaudi.")

        return mixed_precision
