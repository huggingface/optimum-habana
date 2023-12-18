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

###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Union

from optimum.utils import logging


logger = logging.get_logger(__name__)


class DistributedRunner:
    """
    Set up training/inference hardware configurations and run distributed commands.
    """

    def __init__(
        self,
        command_list: List = [],
        world_size: int = 1,
        hostfile: Union[str, Path] = None,
        use_mpi: bool = False,
        use_deepspeed: bool = False,
        use_env: bool = False,
        map_by: bool = "socket",
        multi_hls=None,
    ):
        """
        The `DistributedRunner` enables to exectute a command in a distributed way:
        - On one Gaudi server with MPI, DeepSpeed or `torch.distributed`
        - On several nodes with DeepSpeed.

        Args:
            command_list (List, optional): The list of commands to execute. Defaults to [].
            world_size (int, optional): The number of devices to use. This is only used for single-node runs. Defaults to 1.
            hostfile (Union[str, Path], optional): The path to the hostfile specifying the IP addresses and the number of devices to use for each node. This is only used for multi-node runs. Defaults to None.
            use_mpi (bool, optional): Whether to use OpenMPI for the communication between devices. Defaults to False.
            use_deepspeed (bool, optional): Wheter to use DeepSpeed. Defaults to False.
            use_env (bool, optional): Whether to use `--use_env` with `torch.distributed`. Defaults to False.
            map_by (bool, optional): The mapping unit used for assigning processes with MPI. Defaults to "socket".
        """

        logging.set_verbosity(logging.INFO)
        logging.enable_default_handler()
        logging.enable_explicit_format()

        self._commands = command_list
        self._world_size = world_size
        self._hostfile = hostfile
        self._map_by = map_by
        self._use_env = use_env
        self._interpreter = f"{sys.executable} "

        self._model_env_vars = {}

        # TODO: remove multi_hls
        if multi_hls is not None:
            logger.warning("`multi_hls` is deprecated and will be removed in a future version.")

        if use_deepspeed and use_mpi:
            raise ValueError("`use_mpi` and `use_deepspeed` cannot be both True.")

        if hostfile is not None:
            if isinstance(self._hostfile, str):
                self._hostfile = Path(self._hostfile)
            # Multi-node run
            if use_deepspeed:
                self.create_multi_node_setup()
            else:
                raise ValueError(
                    "A hostfile is specified to perform a multi-node run. This requires to enable DeepSpeed with"
                    " `use_deepspeed=True`."
                )
        elif self._world_size > 1:
            # Distributed run
            if use_deepspeed:
                # Single-node multi-card run with DeepSpeed
                self.create_single_node_setup_deepspeed()
            elif use_mpi:
                # Single-node multi-card run with MPI
                self._model_env_vars["MASTER_ADDR"] = "localhost"
                self._model_env_vars["MASTER_PORT"] = "12345"
                self.create_single_node_setup_mpirun()
            else:
                # Single-node multi-card run with torch.distributed
                self.create_single_node_setup()
        else:
            # Single-card run
            logger.warning(
                "The run will be executed on one device only. Specify `world_size` > 1 or `hostfile` to perform a"
                " distributed run."
            )
            self.create_single_card_setup(use_deepspeed)

    def get_peval(self):
        cmd1 = r"lscpu 2>/dev/null | awk '/Socket\(s\)/  { print $2 }'"
        cmd2 = r"lscpu 2>/dev/null | awk '/Core\(s\) per socket/  { print $4 }'"
        with subprocess.Popen(
            cmd1, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as proc:
            lscpu_output1 = proc.stdout.read()
        with subprocess.Popen(
            cmd2, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as proc:
            lscpu_output2 = proc.stdout.read()
        sockets = int(lscpu_output1)
        corespsocket = int(lscpu_output2)
        if corespsocket == 1:  # running inside VM?
            logger.warning(f"Cores per socket is {corespsocket}. Running inside a VM?")
            logger.warning("Mapping by slot instead of socket")
            self._map_by = "slot"
        if self._hostfile:
            _hls_list = str(os.getenv("MULTI_HLS_IPS", "")).split(",")
            _world_size = 8
            _per_node_processes = int(_world_size / len(_hls_list))
            peval = (sockets * corespsocket) // _per_node_processes
        else:
            peval = (sockets * corespsocket) // self._world_size
        return peval, sockets, corespsocket

    def setup_config_env_mpirun(self):
        peval, _, _ = self.get_peval()
        return f"--map-by {self._map_by}:PE={peval}"

    def create_single_card_setup(self, use_deepspeed=False):
        """
        Single-card setup.
        """

        if use_deepspeed:
            self._interpreter = "deepspeed --num_gpus 1 "
        else:
            self._interpreter = f"{sys.executable} "

    def create_single_node_setup_mpirun(self):
        """
        Single-node multi-card configuration setup for mpirun.
        """

        mpi_cmd = self.setup_config_env_mpirun()
        self._interpreter = (
            f"mpirun -n {self._world_size} --bind-to core {mpi_cmd} --rank-by core --report-bindings"
            f" --allow-run-as-root {sys.executable} "
        )

    def create_single_node_setup_deepspeed(self):
        """
        Single-node multi-card configuration setup for DeepSpeed.
        """

        self._interpreter = f"deepspeed --num_nodes 1 --num_gpus {self._world_size} --no_local_rank "

    def create_single_node_setup(self):
        """
        Single-node multi-card configuration setup.
        """

        use_env_param = "--use_env" if self._use_env else ""

        self._interpreter = (
            f"{sys.executable} -um torch.distributed.run --nproc_per_node={self._world_size} {use_env_param} "
        )

    def create_multi_node_setup(self):
        """
        Multi-node configuration setup for DeepSpeed.
        """

        master_addr = self.process_hostfile()
        self._interpreter = f"deepspeed --hostfile {self._hostfile} --master_addr {master_addr} --no_local_rank "

    def run(self):
        """
        Runs the desired command with configuration specified by the user.
        """

        try:
            if self._model_env_vars:
                print("Running with the following model specific env vars: ")
                for env_name, env_val in [
                    *self._model_env_vars.items()
                ]:  # iterate key value pairs of self._model_env_vars
                    print(f"{env_name}={env_val}")
                    if "LD_PRELOAD" in str(env_name) and os.environ.get(str(env_name), None):
                        os.environ[str(env_name)] = str(env_val) + ":" + os.environ.get(str(env_name), None)
                    else:
                        os.environ[str(env_name)] = str(env_val)
            for command in self._commands:
                command = self._interpreter + command
                print(f"{self.__class__.__name__} run(): command = {command}")
                sys.stdout.flush()
                sys.stderr.flush()
                with subprocess.Popen(command, shell=True, executable="/bin/bash") as proc:
                    proc.wait()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    if proc.returncode != 0:
                        logger.error(f"{command}  exited with status = {proc.returncode}")
                        return proc.returncode
            if self._model_env_vars:
                for env_name in [*self._model_env_vars.keys()]:  # iterate keys of self._model_env_vars
                    del os.environ[str(env_name)]
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} run()") from exc

    def process_hostfile(self) -> str:
        """
        Returns the master address to use for multi-node runs with DeepSpeed.
        Directly inspired from https://github.com/microsoft/DeepSpeed/blob/316c4a43e0802a979951ee17f735daf77ea9780f/deepspeed/autotuning/utils.py#L145.

        Returns:
            str: address of the master node.
        """
        if not self._hostfile.is_file():
            raise ValueError(f"Unable to find hostfile at {self._hostfile}.")

        # e.g., worker-0 slots=16
        with self._hostfile.open("r") as file:
            resource_pool = {}
            master_addr = None
            for line in file.readlines():
                line = line.strip()
                if line == "":
                    # skip empty lines
                    continue
                try:
                    hostname, slots = line.split()
                    _, slot_count = slots.split("=")
                    slot_count = int(slot_count)
                    if master_addr is None:
                        master_addr = hostname
                except ValueError as err:
                    logger.error("Hostfile is not formatted correctly, unable to proceed with training/inference.")
                    raise err
                if hostname in resource_pool:
                    logger.error("Hostfile contains duplicate hosts, unable to proceed with training/inference.")
                    raise ValueError(f"Host {hostname} is already defined")
                resource_pool[hostname] = slot_count

        return master_addr
