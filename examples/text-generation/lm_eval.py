#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import json
import logging
import os
import time

import lm_eval.evaluator
import lm_eval.tasks
import torch
import torch.nn.functional as F
from run_generation import setup_parser
from utils import initialize_model

from optimum.habana.utils import get_hpu_memory_stats


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = logging.getLogger(__name__)


def setup_parser_eval():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Evaluation script for HPU"
    )
    parser.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        help="Input length buckets to use with static_shapes",
        default=[16, 32, 64, 128, 189, 284, 512],
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
    parser.add_argument("--limit_iters", type=int, help="limit examples to run that many iterations", default=None)
    args = setup_parser(parser)

    return args


class HabanaModelAdapter(lm_eval.base.BaseLM):
    def __init__(self, tokenizer, model, args, options):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self._batch_size = args.batch_size
        self.buckets = sorted(args.buckets)
        self.options = options
        self._device = args.device
        if args.warmup:
            self.warm_up()

    def warm_up(self):
        for bucket_size in reversed(self.buckets):
            inps = torch.ones((self._batch_size, bucket_size), dtype=torch.int64)
            self._model_call(inps)
            pass

    @property
    def eot_token_id(self):
        return self.model.config.eos_token_id

    @property
    def max_length(self):
        return self.buckets[-1]

    @property
    def max_gen_toks(self):
        raise NotImplementedError()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        # We need to do padding ourselves, otherwise we'll end up with recompilations
        # Returning 'cpu' to keep tensors on CPU in lm_eval code
        return "cpu"

    def tok_encode(self, string):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError()

    def find_bucket(self, length):
        return [b for b in self.buckets if b >= length][0]

    def _model_call(self, inps):
        seq_length = inps.shape[-1]
        padding_length = 0
        if self.options.static_shapes:
            bucket_length = self.find_bucket(seq_length)
            padding_length = bucket_length - seq_length
            inps = F.pad(inps, (0, padding_length), value=self.model.config.pad_token_id)
        logits = self.model(
            inps.to(self._device),
            use_cache=self.options.reuse_cache,
            attn_softmax_bf16=self.options.attn_softmax_bf16,
        )["logits"].cpu()
        if self.options.static_shapes and padding_length > 0:
            logits = logits[:, :-padding_length, :]
        logits = logits.to(torch.float32)
        return logits


def main():
    args = setup_parser_eval()
    model, tokenizer, generation_config = initialize_model(args, logger)

    lm_tasks = lm_eval.tasks.get_task_dict(args.tasks)
    lm = HabanaModelAdapter(tokenizer, model, args, generation_config)

    eval_start = time.perf_counter()
    results = lm_eval.evaluator.evaluate(lm, lm_tasks, limit=args.limit_iters)
    if args.device == "hpu":
        pass

    eval_end = time.perf_counter()

    results["args"] = vars(args)
    results["duration"] = eval_end - eval_start

    if args.local_rank == 0:
        if args.device == "hpu":
            mem = get_hpu_memory_stats()
            for k, v in mem.items():
                print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))
        json.dump(results, open(args.output_file, "w"), indent=2)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
