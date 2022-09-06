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
"""
A simple launcher script for HPU training
::
    >>> python gaudi_spawn.py --world_size=NUM_CARDS_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)
"""


import sys
from argparse import REMAINDER, ArgumentParser

from optimum.habana.distributed import DistributedRunner
from optimum.utils import logging


logger = logging.get_logger(__name__)


def parse_args():
    """
    Helper function parsing the command line options.
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description=(
            "Habana Gaudi distributed training launch helper utility that will spawn up multiple distributed"
            " processes."
        )
    )

    # Optional arguments for the launch helper
    parser.add_argument("--world_size", type=int, default=1, help="Number of HPUs to use (1 or 8)")
    parser.add_argument("--process_per_node", type=int, default=0, metavar="N", help="Number of processes per node")
    parser.add_argument("--use_mpi", action="store_true", help="Use MPI for distributed training")
    parser.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed for distributed training")

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help=(
            "The full path to the single HPU training "
            "program/script to be launched in parallel, "
            "followed by all the arguments for the "
            "training script."
        ),
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.use_mpi and args.use_deepspeed:
        raise ValueError("--use_mpi and --use_deepspeed cannot be both True.")

    # Patch sys.argv
    sys.argv = [args.training_script] + args.training_script_args
    command_list = [" ".join(sys.argv)]

    if args.process_per_node > 0:
        logger.warning("Multi-node is not officially supported yet, run it at your own risk.")
        multi_hls = True
    else:
        multi_hls = False

    distributed_runner = DistributedRunner(
        command_list=command_list,
        world_size=args.world_size,
        use_mpi=args.use_mpi,
        use_deepspeed=args.use_deepspeed,
        multi_hls=multi_hls,
    )

    ret_code = distributed_runner.run()
    sys.exit(ret_code)


if __name__ == "__main__":
    main()
