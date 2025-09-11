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
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Union

import psutil

# Local imports
from run_generation import setup_parser
from utils import finalize_quantization, initialize_model, save_model


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = logging.getLogger(__name__)

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


def try_parse_json(value: str) -> Union[str, dict, None]:
    """
    From https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.1/lm_eval/__main__.py
    """
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if "{" in value:
            raise argparse.ArgumentTypeError(f"Invalid JSON: {value}. Hint: Use double quotes for JSON strings.")
        return value


def setup_lm_eval_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Evaluation script for HPU"
    )
    parser.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        help="Input length buckets to use with static_shapes",
        default=[16, 32, 64, 128, 189, 284, 384],
    )

    parser.add_argument(
        "--output_file", "-o", type=str, help="Output file with end results and runtime parameters", required=True
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Tasks to run",
        default=["hellaswag", "lambada_openai", "piqa", "winogrande"],
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        metavar="N|0<N<1",
        help="Limit the number of examples per task. If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="If True, prints extra-logs for all tasks",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to be used in the prompt",
    )
    parser.add_argument("--max_graphs", type=int, help="Maximum number of HPU graphs", default=None)
    parser.add_argument(
        "--gen_kwargs",
        type=try_parse_json,
        default=None,
        help=(
            "Either comma delimited string or JSON formatted arguments for model generation on greedy_until tasks,"
            """ e.g. '{"temperature":0.7,"until":["hello"]}' or temperature=0,top_p=0.1."""
        ),
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="If True, uses the fewshot as a multi-turn conversation",
    )
    parser.add_argument(
        "--metadata",
        type=json.loads,
        default=None,
        help="""JSON string metadata to pass to task configs, for example '{"max_seq_lengths":[4096,8192]}'. Will be merged with model_args. Can also be set in task config.""",
    )
    parser.add_argument(
        "--apply_chat_template",
        type=str,
        nargs="?",
        const=True,
        default=False,
        help=(
            "If True, apply chat template to the prompt. "
            "Providing `--apply_chat_template` without an argument will apply the default chat template to the prompt. "
            "To apply a specific template from the available list of templates, provide the template name as an argument. "
            "E.g. `--apply_chat_template template_name`"
        ),
    )
    parser.add_argument(
        "--samples",
        "-E",
        default=None,
        type=str,
        metavar="/path/to/json",
        help='JSON string or path to JSON file containing doc indices of selected examples to test. Format: {"task_name":[indices],...}',
    )
    parser.add_argument(
        "--confirm_run_unsafe_code",
        action="store_true",
        help="Confirm that you understand the risks of running unsafe code for tasks that require it",
    )
    args = setup_parser(parser)
    return args


def main() -> None:
    # Modified based on cli_evaluate function in https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.1/lm_eval/__main__.py#L301
    args = setup_lm_eval_parser()
    model, _, tokenizer, generation_config = initialize_model(args, logger)

    import torch
    from lm_eval import evaluator, utils
    from model_adapter import HabanaModelAdapter

    max_length = None
    metadata = None
    if args.metadata:
        metadata = args.metadata if isinstance(args.metadata, dict) else utils.sample_parse_args_string(args.metadata)
        max_length = args.metadata.get("max_length")

    if args.fewshot_as_multiturn and args.apply_chat_template is False:
        raise ValueError(
            "When `fewshot_as_multiturn` is selected, `apply_chat_template` must be set (either to `True` or to the chosen template name)."
        )
    if args.samples:
        assert args.limit is None, "If --samples is not None, then --limit must be None."
        if (samples := Path(args.samples)).is_file():
            args.samples = json.loads(samples.read_text())
        else:
            args.samples = json.loads(args.samples)

    with torch.no_grad():
        lm = HabanaModelAdapter(tokenizer, model, args, generation_config, max_length=max_length)

    from optimum.habana.utils import HabanaGenerationTime, get_hpu_memory_stats

    with HabanaGenerationTime() as timer:
        with torch.no_grad():
            results = evaluator.simple_evaluate(
                lm,
                tasks=args.tasks,
                limit=args.limit,
                samples=args.samples,
                log_samples=args.log_samples,
                num_fewshot=args.num_fewshot,
                fewshot_as_multiturn=args.fewshot_as_multiturn,
                gen_kwargs=args.gen_kwargs,
                system_instruction=args.system_instruction,
                apply_chat_template=args.apply_chat_template,
                metadata=metadata,
                confirm_run_unsafe_code=args.confirm_run_unsafe_code,
            )
        if args.device == "hpu":
            import habana_frameworks.torch.hpu as torch_hpu

            torch_hpu.synchronize()

    results["args"] = vars(args)
    results["duration"] = timer.last_duration

    if args.local_rank == 0:
        if args.device == "hpu":
            mem = get_hpu_memory_stats()
            for k, v in mem.items():
                print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))

        json_str = json.dumps(results, indent=2, default=utils.handle_non_serializable, ensure_ascii=False)
        from pathlib import Path

        output_dir = Path(args.output_file).parent
        if not output_dir.exists():
            os.makedirs(output_dir)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(json_str)
        if args.show_config:
            print(json_str)

    if args.quant_config:
        finalize_quantization(model)
    if args.save_quantized_model_with_inc:
        save_model(model, tokenizer, args.saved_model_path)
    if args.pt2e_save and args.pt2e_path:
        from quantization_tools.pt2e import pt2e_save

        pt2e_save(model)

    if args.const_serialization_path and os.path.isdir(args.const_serialization_path):
        import shutil

        shutil.rmtree(args.const_serialization_path)


if __name__ == "__main__":
    main()
