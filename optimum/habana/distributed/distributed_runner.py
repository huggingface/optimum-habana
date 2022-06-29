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

from optimum.utils import logging


logger = logging.get_logger(__name__)


class DistributedRunner:
    """
    Set up training hardware configurations and run distributed training commands.
    """

    def __init__(
        self,
        command_list=[],
        world_size=1,
        use_mpi=False,
        use_env=False,
        map_by="socket",
        multi_hls=False,
    ):
        self.__commands = command_list
        self.__world_size = world_size
        self.__map_by = map_by
        self.__multi_hls = multi_hls
        self.__use_env = use_env
        self.__interpreter = f"{sys.executable} "

        self.__model_env_vars = {}

        logger.info(
            f"Training is {'not ' if self.__world_size == 1 else ''}distributed, world_size = {self.__world_size}"
        )
        # Distributed training
        if self.__world_size > 1:
            # Multi-node training
            if self.__multi_hls:
                self.create_multi_hls_setup()
            elif use_mpi:
                # Single-node multi-card training with MPI
                self.__model_env_vars["MASTER_ADDR"] = "localhost"
                self.__model_env_vars["MASTER_PORT"] = "12345"
                self.create_single_hls_setup_mpirun()
            else:
                # Single-node multi-card training with torch.distributed
                self.create_single_hls_setup()
        else:
            # Single-card training
            self.create_single_card_setup()

    def setup_config_env(self):
        hccl_over_tcp = os.getenv("HCCL_OVER_TCP")
        hccl_over_ofi = os.getenv("HCCL_OVER_OFI")
        if hccl_over_tcp or hccl_over_ofi:
            if hccl_over_tcp:
                hccl_over_tcp = hccl_over_tcp.lower() in ["1", "true"]
            if hccl_over_ofi:
                hccl_over_ofi = hccl_over_ofi.lower() in ["1", "true"]
            logger.info(f"HCCL_OVER_TCP={os.getenv('HCCL_OVER_TCP')}")
            logger.info(f"HCCL_OVER_OFI={os.getenv('HCCL_OVER_OFI')}")

        logger.info(f"HLS ({self.__world_size})")

    def get_peval(self):
        cmd1 = "lscpu 2>/dev/null | awk '/Socket\(s\)/  { print $2 }'"
        cmd2 = "lscpu 2>/dev/null | awk '/Core\(s\) per socket/  { print $4 }'"
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
            logger.warning(f"Mapping by slot instead of socket")
            self.__map_by = "slot"
        if self.__multi_hls:
            __hls_list = str(os.getenv("MULTI_HLS_IPS", "")).split(",")
            __world_size = self.__world_size
            __per_node_processes = int(__world_size / len(__hls_list))
            peval = (sockets * corespsocket) // __per_node_processes
        else:
            peval = (sockets * corespsocket) // self.__world_size
        return peval, sockets, corespsocket

    def setup_config_env_mpirun(self):
        peval, _, _ = self.get_peval()
        if peval:
            map_cmd = f"--map-by {self.__map_by}:PE={peval}"
        return map_cmd

    def create_single_card_setup(self):
        """
        Single-card setup.
        """

        self.setup_config_env()
        self.__interpreter = f"{sys.executable} "

    def create_single_hls_setup_mpirun(self):
        """
        Single-node multi-cards configuration setup for mpirun.
        """

        self.setup_config_env()
        mpi_cmd = self.setup_config_env_mpirun()
        self.__interpreter = (
            f"mpirun -n {self.__world_size} --bind-to core {mpi_cmd} --rank-by core --report-bindings"
            f" --allow-run-as-root {sys.executable} "
        )

    def create_single_hls_setup(self):
        """
        Single-node multi-cards configuration setup.
        """

        use_env_param = "--use_env" if self.__use_env else ""

        self.setup_config_env()
        self.__interpreter = (
            f"{sys.executable} -um torch.distributed.launch --nproc_per_node={self.__world_size} {use_env_param} "
        )

    def create_multi_hls_setup(self):
        """
        Multi-node configuration setup for mpirun.
        """

        self.setup_config_env()
        envlist = [
            "MAX_WAIT_ATTEMPTS",
            "LOG_LEVEL_ALL",
            "LOG_LEVEL_SYN_API",
            "LD_LIBRARY_PATH",
            "PYTORCH_MODULES_ROOT_PATH",
            "BUILD_ROOT_LATEST",
            "PYTHONPATH",
            "HABANA_LOGS",
            "GC_KERNEL_PATH",
            "HCCL_BOX_SIZE",
            "HCCL_OVER_TCP",
            "HCCL_COMM_ID",
            "HCCL_SOCKET_IFNAME",
            "SOCKET_NTHREADS",
            "NSOCK_PERTHREAD",
            "HCCL_DEFAULT_NIC_COUNT",
            "PT_HCCL_SLICE_SIZE_MB",
            "HCCL_OVER_OFI",
            "MULTI_HLS_IPS",
            "https_proxy",
            "HTTPS_PROXY",
            "http_proxy",
            "HTTP_PROXY",
            "MULTI_STREAMS_ENABLE",
        ]
        assert os.getenv("MULTI_HLS_IPS"), "environment variable MULTI_HLS_IPS is not set"
        __hls_list = str(os.getenv("MULTI_HLS_IPS", "")).split(",")
        __world_size = self.__world_size
        __sshport = os.getenv("DOCKER_SSHD_PORT", 3022)
        __master_port = os.getenv("MASTER_PORT", 12345)
        __master_addr = os.getenv("MASTER_ADDR", __hls_list[0])
        __per_node_processes = int(__world_size / len(__hls_list))
        __pe_val, __sockets, _ = self.get_peval()
        __cores_per_node = __per_node_processes * __pe_val
        __process_per_socket = __per_node_processes // __sockets
        envset_cmds = []
        for __env in envlist:
            __val = os.getenv(__env, None)
            if __val:
                __arg = f'-x {__env}="{__val}"'
                envset_cmds.append(__arg)
        envset_cmd = " ".join(envset_cmds)
        hls_nodes = []
        for hls in __hls_list:
            hls_node = hls.split("-")[0]
            hls_nodes.append(f"{hls_node}:{__cores_per_node}")
        hls_info = ",".join(hls_nodes)
        __master_addr = __hls_list[0]
        network = __master_addr.split(".")
        network[-1] = "0"
        network_id = ".".join(network) + "/16"
        cmd = "mpirun --allow-run-as-root "
        cmd += f" {envset_cmd} "
        cmd += f"--prefix {os.getenv('MPI_ROOT', '/usr/local/openmpi')} "
        cmd += f"--mca btl_tcp_if_include {network_id} "
        cmd += f"-x MASTER_ADDR={__master_addr} "
        cmd += f"-x MASTER_PORT={__master_port} "
        cmd += f'--mca plm_rsh_args "-p {__sshport}" --bind-to core '
        cmd += f"-H {hls_info} -n {__world_size} "
        if __process_per_socket > 0:
            cmd += f"--map-by ppr:{__process_per_socket}:socket:PE={__pe_val} "
        else:
            cmd += f"--map-by ppr:{__per_node_processes}:node:PE={__pe_val} "
        cmd += f"--rank-by core --report-bindings {sys.executable} "
        self.__interpreter = cmd

    def run(self):
        try:
            if self.__model_env_vars:
                print("Running with the following model specific env vars: ")
                for env_name, env_val in [
                    *self.__model_env_vars.items()
                ]:  # iterate key value pairs of self.__model_env_vars
                    print(f"{env_name}={env_val}")
                    if "LD_PRELOAD" in str(env_name) and os.environ.get(str(env_name), None):
                        os.environ[str(env_name)] = str(env_val) + ":" + os.environ.get(str(env_name), None)
                    else:
                        os.environ[str(env_name)] = str(env_val)
            for command in self.__commands:
                command = self.__interpreter + command
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
            if self.__model_env_vars:
                for env_name in [*self.__model_env_vars.keys()]:  # iterate keys of self.__model_env_vars
                    del os.environ[str(env_name)]
        except Exception as exc:
            raise RuntimeError(f"Error in {self.__class__.__name__} run()") from exc
