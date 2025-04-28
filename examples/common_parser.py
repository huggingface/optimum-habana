from argparse import ArgumentParser


def add_profiling_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--profiling_warmup_steps",
        default=0,
        type=int,
        help="Number of steps to ignore for profiling.",
    )
    parser.add_argument(
        "--profiling_steps",
        default=0,
        type=int,
        help="Number of steps to capture for profiling.",
    )
    parser.add_argument(
        "--profiling_record_shapes",
        action="store_true",
        help="Record shapes when enabling profiling.",
    )
