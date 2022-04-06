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

"""Generate an HCL Config File
Generate hcl config file and set HCL_CONFIG_PATH
Globals:
  MULTI_HLS_IPS
Arguments:
  Path to save the file
  Number of devices per HLS
  HLS type, default HLS1
"""
import json
import os
import socket
import sys
from pathlib import Path

from optimum.habana.distributed.multi_node_utils import (
    get_multi_node_config_nodes,
    is_hccl_api,
    is_horovod_hierarchical,
    is_valid_multi_node_config,
    run_cmd_as_subprocess,
)


def _get_hcl_ranks(devices_per_hls):
    def gen(devices_per_hls):
        for ip in get_multi_node_config_nodes():
            for _ in range(devices_per_hls):
                yield ip

    return list(gen(devices_per_hls))


HCL_PORT_DEFAULT = 53432


def _generate_hcl_config_content(file_path, devices_per_hls, hls_type):
    config = {}

    config["HCL_PORT"] = int(os.environ.get("HCL_PORT", HCL_PORT_DEFAULT))

    if not is_valid_multi_node_config() or is_horovod_hierarchical():
        config["HCL_COUNT"] = devices_per_hls
    else:
        config["HCL_RANKS"] = _get_hcl_ranks(devices_per_hls)
    return json.dumps(config, indent="\t")


def _get_default_hcl_config_path(file_path, num_workers_total):
    file_name = f"hcl_config_{num_workers_total}.json"
    os.makedirs(os.path.abspath(file_path), mode=0o777, exist_ok=True)
    return Path(os.path.abspath(file_path)).joinpath(file_name)


def generate_hcl_config_r(file_path, devices_per_hls, hls_type="HLS1"):
    host_name = socket.gethostname()
    hcl_config_path = os.environ.get("HCL_CONFIG_PATH")

    if hcl_config_path:
        print(f"{host_name}: Path: {str(hcl_config_path)}")
        with open(hcl_config_path, "r") as f:
            print(f"{host_name}: HCL Config: \n{f.read()}")
        return hcl_config_path
    else:
        try:
            print(f"{host_name}: Generating HCL Config...")
            num_workers_total = devices_per_hls
            if is_valid_multi_node_config() and not is_horovod_hierarchical():
                num_nodes = get_multi_node_config_nodes()
                num_workers_total *= len(num_nodes)

            hcl_config_path = _get_default_hcl_config_path(file_path, num_workers_total)

            if os.path.exists(hcl_config_path):
                cmd = f"rm -f {str(hcl_config_path)}"
                run_cmd_as_subprocess(cmd)
            print(f"{host_name}: Path: {str(hcl_config_path)}")

            with open(hcl_config_path, "a") as out_fid:
                config_str = _generate_hcl_config_content(file_path, devices_per_hls, hls_type)
                print(f"{host_name}: HCL Config: \n{config_str}")
                out_fid.write(config_str)

            os.environ["HCL_CONFIG_PATH"] = str(hcl_config_path)
            return hcl_config_path
        except Exception as exc:
            raise Exception(
                f"{host_name}: Error in {__file__} generate_hcl_config_r({file_path}, {devices_per_hls}, {hls_type})"
            ) from exc


def generate_hcl_config_unless_hccl(file_path, devices_per_hls, hls_type="HLS1"):
    host_name = socket.gethostname()
    hcl_config_path = os.environ.get("HCL_CONFIG_PATH")

    if hcl_config_path:
        if is_hccl_api():
            print(
                "\n".join(
                    [
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                        "!                                                                             !",
                        "! WARNING: HCL_CONFIG_PATH environment variable is set, while HCCL API is     !",
                        "!          used internally by HPU collective operations.                      !",
                        "!          HCCL does not require HCL_CONFIG_PATH, but it will honor it.       !",
                        "!          If you do not know what it is for, please unset HCL_CONFIG_PATH.   !",
                        "!                                                                             !",
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                    ]
                )
            )
        print(f"{host_name}: Path: {str(hcl_config_path)}")
        with open(hcl_config_path, "r") as f:
            print(f"{host_name}: HCL Config: \n{f.read()}")
        return hcl_config_path
    else:
        # For HCCL API used by HPU collectives, preparing HCL Config and setting HCL_CONFIG_PATH is not required.
        if not is_hccl_api():
            try:
                print(f"{host_name}: Generating HCL Config...")
                num_workers_total = devices_per_hls
                if is_valid_multi_node_config() and not is_horovod_hierarchical():
                    num_nodes = get_multi_node_config_nodes()
                    num_workers_total *= len(num_nodes)

                hcl_config_path = os.environ.get(
                    "HCL_CONFIG_AUTOGEN_PATH", _get_default_hcl_config_path(file_path, num_workers_total)
                )

                if os.path.exists(hcl_config_path):
                    cmd = f"rm -f {str(hcl_config_path)}"
                    run_cmd_as_subprocess(cmd)
                print(f"{host_name}: Path: {str(hcl_config_path)}")

                with open(hcl_config_path, "a") as out_fid:
                    config_str = _generate_hcl_config_content(file_path, devices_per_hls, hls_type)
                    print(f"{host_name}: HCL Config: \n{config_str}")
                    out_fid.write(config_str)

                os.environ["HCL_CONFIG_PATH"] = str(hcl_config_path)
                return hcl_config_path
            except Exception as exc:
                raise Exception(
                    f"{host_name}: Error in {__file__} generate_hcl_config_unless_hccl({file_path}, {devices_per_hls}, {hls_type})"
                ) from exc
        else:
            return None


if __name__ == "__main__":
    host_name = socket.gethostname()
    print(f"{host_name}: In {sys.argv[0]}")
    print(f'{host_name}: called with arguments: "{sys.argv[1]} {sys.argv[2]} {sys.argv[3]}"')
    file_path = sys.argv[1]
    devices_per_hls = int(sys.argv[2])
    hls_type = sys.argv[3]
    print(f"{host_name}: MULTI_HLS_IPS = {os.environ.get('MULTI_HLS_IPS')}")
    print(f"{host_name}: HCL_CONFIG_PATH = {os.environ.get('HCL_CONFIG_PATH')}")
    generate_hcl_config_unless_hccl(file_path, devices_per_hls, hls_type)
