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

import random
import subprocess
import time
from typing import Any, Dict

import numpy as np
import torch
from packaging import version
from transformers.utils import is_torch_available

from optimum.utils import logging

from .version import __version__


logger = logging.get_logger(__name__)


CURRENTLY_VALIDATED_SYNAPSE_VERSION = version.parse("1.12.0")


def to_device_dtype(my_input: Any, target_device: torch.device = None, target_dtype: torch.dtype = None):
    """
    Move a state_dict to the target device and convert it into target_dtype.

    Args:
        my_input : input to transform
        target_device (torch.device, optional): target_device to move the input on. Defaults to None.
        target_dtype (torch.dtype, optional): target dtype to convert the input into. Defaults to None.

    Returns:
        : transformed input
    """
    if isinstance(my_input, torch.Tensor):
        if target_device is None:
            target_device = my_input.device
        if target_dtype is None:
            target_dtype = my_input.dtype
        return my_input.to(device=target_device, dtype=target_dtype)
    elif isinstance(my_input, list):
        return [to_device_dtype(i, target_device, target_dtype) for i in my_input]
    elif isinstance(my_input, tuple):
        return tuple(to_device_dtype(i, target_device, target_dtype) for i in my_input)
    elif isinstance(my_input, dict):
        return {k: to_device_dtype(v, target_device, target_dtype) for k, v in my_input.items()}
    else:
        return my_input


def speed_metrics(
    split: str,
    start_time: float,
    num_samples: int = None,
    num_steps: int = None,
    start_time_after_warmup: float = None,
    log_evaluate_save_time: float = None,
) -> Dict[str, float]:
    """
    Measure and return speed performance metrics.
    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
        split (str): name to prefix metric (like train, eval, test...)
        start_time (float): operation start time
        num_samples (int, optional): number of samples processed. Defaults to None.
        num_steps (int, optional): number of steps performed. Defaults to None.
        start_time_after_warmup (float, optional): time after warmup steps have been performed. Defaults to None.
        log_evaluate_save_time (float, optional): time spent to log, evaluate and save. Defaults to None.

    Returns:
        Dict[str, float]: dictionary with performance metrics.
    """

    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if runtime == 0:
        return result

    # Adjust runtime if log_evaluate_save_time should not be included
    if log_evaluate_save_time is not None:
        runtime = runtime - log_evaluate_save_time

    # Adjust runtime if there were warmup steps
    if start_time_after_warmup is not None:
        runtime = runtime + start_time - start_time_after_warmup

    # Compute throughputs
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)

    return result


def to_gb_rounded(mem: float) -> float:
    """
    Rounds and converts to GB.

    Args:
        mem (float): memory in bytes

    Returns:
        float: memory in GB rounded to the second decimal
    """
    return np.round(mem / 1024**3, 2)


def get_hpu_memory_stats(device=None) -> Dict[str, float]:
    """
    Returns memory stats of HPU as a dictionary:
    - current memory allocated (GB)
    - maximum memory allocated (GB)
    - total memory available (GB)

    Returns:
        Dict[str, float]: memory stats.
    """
    from habana_frameworks.torch.hpu import memory_stats

    mem_stats = memory_stats(device)

    mem_dict = {
        "memory_allocated (GB)": to_gb_rounded(mem_stats["InUse"]),
        "max_memory_allocated (GB)": to_gb_rounded(mem_stats["MaxInUse"]),
        "total_memory_available (GB)": to_gb_rounded(mem_stats["Limit"]),
    }

    return mem_dict


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy` and `torch`.
    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        from habana_frameworks.torch.hpu import random as hpu_random

        torch.manual_seed(seed)
        hpu_random.manual_seed_all(seed)


def check_synapse_version():
    """
    Checks whether the versions of SynapseAI and drivers have been validated for the current version of Optimum Habana.
    """
    # Change the logging format
    logging.enable_default_handler()
    logging.enable_explicit_format()

    # Check the version of habana_frameworks
    habana_frameworks_version_number = get_habana_frameworks_version()
    if (
        habana_frameworks_version_number.major != CURRENTLY_VALIDATED_SYNAPSE_VERSION.major
        or habana_frameworks_version_number.minor != CURRENTLY_VALIDATED_SYNAPSE_VERSION.minor
    ):
        logger.warning(
            f"optimum-habana v{__version__} has been validated for SynapseAI v{CURRENTLY_VALIDATED_SYNAPSE_VERSION} but habana-frameworks v{habana_frameworks_version_number} was found, this could lead to undefined behavior!"
        )

    # Check driver version
    driver_version = get_driver_version()
    # This check is needed to make sure an error is not raised while building the documentation
    # Because the doc is built on an instance that does not have `hl-smi`
    if driver_version is not None:
        if (
            driver_version.major != CURRENTLY_VALIDATED_SYNAPSE_VERSION.major
            or driver_version.minor != CURRENTLY_VALIDATED_SYNAPSE_VERSION.minor
        ):
            logger.warning(
                f"optimum-habana v{__version__} has been validated for SynapseAI v{CURRENTLY_VALIDATED_SYNAPSE_VERSION} but the driver version is v{driver_version}, this could lead to undefined behavior!"
            )
    else:
        logger.warning(
            "Could not run `hl-smi`, please follow the installation guide: https://docs.habana.ai/en/latest/Installation_Guide/index.html."
        )


def get_habana_frameworks_version():
    """
    Returns the installed version of SynapseAI.
    """
    output = subprocess.run(
        "pip list | grep habana-torch-plugin",
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return version.parse(output.stdout.split("\n")[0].split(" ")[-1])


def get_driver_version():
    """
    Returns the driver version.
    """
    output = subprocess.run(
        "hl-smi",
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if output.returncode == 0:
        return version.parse(output.stdout.split("\n")[2].replace(" ", "").split(":")[1][:-1].split("-")[0])
    return None


class HabanaProfile(object):
    """
    HPU profiler only could be run once, so HABANA_PROFILE_ENABLED, a class static variable shared by all the instances of HabanaProfile, is used to control which part will be captured.
    """

    HABANA_PROFILE_ENABLED = True

    def __init__(
        self,
        warmup: int = 0,
        active: int = 0,
        record_shapes: bool = True,
        output_dir: str = "./hpu_profile",
        wait: int = 0,
    ):
        if active <= 0 or warmup <= 0 or not HabanaProfile.HABANA_PROFILE_ENABLED:

            def noop():
                pass

            self.start = noop
            self.stop = noop
            self.step = noop
        else:
            HabanaProfile.HABANA_PROFILE_ENABLED = False
            schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1)
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]

            profiler = torch.profiler.profile(
                schedule=schedule,
                activities=activities,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
                record_shapes=record_shapes,
                with_stack=True,
            )
            self.start = profiler.start
            self.stop = profiler.stop
            self.step = profiler.step
            HabanaProfile.enable.invalid = True
            HabanaProfile.disable.invalid = True

    def stop(self):
        self.stop()

    def start(self):
        self.start()

    def step(self):
        self.step()

    @staticmethod
    def disable():
        """
        Runs only once and must happen before doing profiling.
        """
        if hasattr(HabanaProfile.disable, "invalid"):
            if not HabanaProfile.disable.invalid:
                HabanaProfile.HABANA_PROFILE_ENABLED = False
        else:
            HabanaProfile.HABANA_PROFILE_ENABLED = False

    @staticmethod
    def enable():
        """
        Runs only once and must happen before doing profiling.
        """
        if hasattr(HabanaProfile.enable, "invalid"):
            if not HabanaProfile.enable.invalid:
                HabanaProfile.HABANA_PROFILE_ENABLED = True
        else:
            HabanaProfile.HABANA_PROFILE_ENABLED = True


def check_optimum_habana_min_version(min_version):
    """
    Checks if the installed version of `optimum-habana` is larger than or equal to `min_version`.

    Copied from: https://github.com/huggingface/transformers/blob/c41291965f078070c5c832412f5d4a5f633fcdc4/src/transformers/utils/__init__.py#L212
    """
    if version.parse(__version__) < version.parse(min_version):
        error_message = (
            f"This example requires `optimum-habana` to have a minimum version of {min_version},"
            f" but the version found is {__version__}.\n"
        )
        if "dev" in min_version:
            error_message += (
                "You can install it from source with: "
                "`pip install git+https://github.com/huggingface/optimum-habana.git`."
            )
        raise ImportError(error_message)


def get_device_name():
    """
    Returns the name of the current device: Gaudi or Gaudi2.

    Inspired from: https://github.com/HabanaAI/Model-References/blob/a87c21f14f13b70ffc77617b9e80d1ec989a3442/PyTorch/computer_vision/classification/torchvision/utils.py#L274
    """
    import habana_frameworks.torch.utils.experimental as htexp

    device_type = htexp._get_device_type()

    if device_type == htexp.synDeviceType.synDeviceGaudi:
        return "gaudi"
    elif device_type == htexp.synDeviceType.synDeviceGaudi2:
        return "gaudi2"
    else:
        raise ValueError(f"Unsupported device: the device type is {device_type}.")
