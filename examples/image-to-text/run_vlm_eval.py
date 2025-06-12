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
# Copyright (C) 2020-2025 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import json
import multiprocessing as mp
import os
from typing import Literal, Optional
from unittest import result

import psutil
import torch
import torch.nn.functional as F
import transformers
from lm_eval import evaluator, utils
from lm_eval.models.huggingface import HFLM, TemplateLM

import copy
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers

# Local imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor, pipeline
from transformers.generation import GenerationConfig
from utils import finalize_quantization, save_model

from optimum.habana.transformers.generation import GaudiGenerationConfig
from optimum.habana.utils import HabanaGenerationTime, get_hpu_memory_stats
import lm_eval


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = utils.eval_logger

# This hack is a workaround to limitations of lm_eval which always allocates
# mp.Pool with max cpu count which explodes on multinode scenarios and for hpu
# create multiprocess with spawn context
OrigPool = mp.Pool


def LimitedSpawnPool(_):
    spawn_context = mp.get_context("spawn")
    physical_cpu_count = psutil.cpu_count(logical=False)
    pool_size = physical_cpu_count
    world_size = int(os.getenv("WORLD_SIZE", 1))
    pool_size //= max(world_size, 1)
    if (pool_size * world_size) != physical_cpu_count:
        pool_size -= 1
    return spawn_context.Pool(pool_size)


mp.Pool = LimitedSpawnPool


def setup_lm_eval_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Evaluation script for HPU"
    )

    parser.add_argument(
        "--output_file", "-o", type=str, help="Output file with end results and runtime parameters", required=True
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Tasks to run",
        default=["mmmu_val"],
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=True,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pre-trained model"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bfloat16 precision"
    )
    parser.add_argument(
        "--world_size", 
        type=int,
        default=1,
        help="Number of processes for distributed training"
    )

    return parser.parse_args()

eval_logger = logging.getLogger(__name__)

def main() -> None:
    args = setup_lm_eval_parser()
    transformers.GenerationConfig = GaudiGenerationConfig

    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "0"))
    args.global_rank =  int(os.getenv("RANK", "0"))

    os.environ["PT_HPUGRAPH_DISABLE_TENSOR_CACHE"] = "1"
    os.environ.setdefault("EXPERIMENTAL_WEIGHT_SHARING", "FALSE")
    if args.world_size > 0:
        os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")
        os.environ.setdefault("DEEPSPEED_USE_HABANA_FRAMEWORKS_DETERMINISTIC_API", "1")

    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    adapt_transformers_to_gaudi()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model_type = config.model_type

    if args.bf16:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    os.environ["PT_HPUGRAPH_DISABLE_TENSOR_CACHE"] = "1"
    os.environ.setdefault("EXPERIMENTAL_WEIGHT_SHARING", "FALSE")

    from run_pipeline import initialize_distributed_model
    if args.world_size > 1:
        os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")
        os.environ.setdefault("DEEPSPEED_USE_HABANA_FRAMEWORKS_DETERMINISTIC_API", "1")
        import deepspeed

        with deepspeed.OnDevice(dtype=model_dtype, device="cpu"):
            model = AutoModelForVision2Seq.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype)
        if model_type == "mllama":
            model = initialize_distributed_model(args, model, logger, model_dtype)
        else:
            model = initialize_distributed_model(args, model, logger, model_dtype)
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_name_or_path,
            torch_dtype=model_dtype,
        )

    model = model.eval().to("hpu")
    lm_eval_model = lm_eval.models.hf_vlms.HFMultimodalLM(pretrained=model)

    with HabanaGenerationTime() as timer:
        results = lm_eval.simple_evaluate(
                lm_eval_model,
                tasks=args.tasks,
                batch_size=1,
                device="hpu"
        )

    torch.hpu.synchronize()
    results["args"] = vars(args)
    results["duration"] = timer.last_duration

    if args.local_rank == 0:
        if torch.hpu.is_available():
            mem = get_hpu_memory_stats()
            for k, v in mem.items():
                print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))

        json_str = json.dumps(results, indent=2, default=utils.handle_non_serializable, ensure_ascii=False)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(json_str)
        if args.show_config:
            print(json_str)

if __name__ == "__main__":
    main()
