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
"""Tool to create or update a diff file between transformers and optimum examples."""

import re
import subprocess
import tempfile
from argparse import ArgumentParser
from pathlib import Path

from git import Repo


DIFF_DIRECTORY = Path(__file__).parent.resolve() / "example_diff"


def _ask_yes_or_no_question(message: str) -> str:
    if message[-1] == "?":
        message = message[:-1]
    message = f"{message} (y/n) ? "
    continue_ = True
    while continue_:
        res = input(message)
        if res not in ["y", "n"]:
            print(f"You must answer by either y (yes) or n (no), but {res} was provided.\n")
        else:
            continue_ = False
    return res


def diff(filename1: Path, filename2: Path) -> str:
    if not filename1.exists() or not filename2.exists():
        raise FileNotFoundError(
            f"Cannot compute the diff because at least one of the files does not exist: {filename1} and/or {filename2}."
        )
    cmd_line = ["diff", str(filename1), str(filename2)]
    p = subprocess.Popen(cmd_line, stdout=subprocess.PIPE)
    outs, _ = p.communicate()
    return outs.decode("utf-8")


def _colorize_lines(content):
    lines = content.split("\n")
    color_mapping = {
        "<": "\033[0;31m",  # Red
        ">": "\033[0;32m",  # Green
        "-": "",
        "default": "\033[0;36m",  # Blue
    }
    end_color = "\033[0;0m"
    for i, line in enumerate(lines):
        if not line:
            continue
        start_char = color_mapping.get(line[0], color_mapping["default"])
        lines[i] = "".join([start_char, line, end_color])
    return "\n".join(lines)


def create_diff_content(raw_diff: str, keep_all_diffs: bool = False) -> str:
    matches = list(re.finditer(r"^[^><-]+", raw_diff, flags=re.MULTILINE))
    final_diff = []
    for m1, m2 in zip(matches, matches[1:] + [None]):
        start, end = m1.span()[0], m2.span()[0] if m2 is not None else None
        if end is not None and raw_diff[end - 1] == "\n":
            end = end - 1
        content = raw_diff[start:end]
        if not keep_all_diffs:
            print(_colorize_lines(content))
            keep_diff = _ask_yes_or_no_question("Keep this diff")
            if keep_diff == "n":
                continue
        final_diff.append(content)
    return "\n".join(final_diff)


def auto_diff():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Clone the Transformers GH repo
        Repo.clone_from("https://github.com/huggingface/transformers.git", tmpdirname)

        # Get paths to Optimum and Transformers examples
        path_to_optimum_examples = Path(__file__).resolve().parent / "../examples/"
        optimum_example_dirs = [directory for directory in path_to_optimum_examples.iterdir() if directory.is_dir()]
        path_to_transformers_examples = Path(f"{tmpdirname}/examples/pytorch/")
        transformers_example_dirs = [
            directory for directory in path_to_transformers_examples.iterdir() if directory.is_dir()
        ]

        # Loop over Optimum examples to compare them with their Transformers counterpart
        for directory in optimum_example_dirs:
            # Check if the example is in Transformers
            if directory.name in [folder.name for folder in transformers_example_dirs]:
                path_to_transformers = path_to_transformers_examples / directory.name
                # Loop over all the "run_*.py" scripts in the example folder
                for file in directory.iterdir():
                    if file.is_file() and file.name.startswith("run_"):
                        transformers_file = path_to_transformers / file.name
                        if transformers_file.is_file():
                            final_diff = create_diff_content(
                                diff(
                                    transformers_file,
                                    file,
                                ),
                                keep_all_diffs=True,
                            )
                            diff_filename = DIFF_DIRECTORY / f"{file.stem}.txt"
                            with open(diff_filename, "w") as fp:
                                fp.write(final_diff)


def parse_args():
    parser = ArgumentParser(
        description="Tool to create or update a diff file between transformers and optimum examples."
    )
    parser.add_argument("--transformers", type=Path, help="The path to the transformers example")
    parser.add_argument("--optimum", type=Path, help="The path to the optimum example")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Whether to automatically write diff files or not. If true, all diffs will be accepted.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.auto:
        auto_diff()
    else:
        if args.transformers is None and args.optimum is None:
            raise ValueError("`--transformers` and `--optimum` must be both set if `--auto` is not set.")
        raw_diff = diff(args.transformers, args.optimum)
        print(f"Creating the diff file between {args.transformers} and {args.optimum}:\n")
        final_diff = create_diff_content(raw_diff)
        print(f"Difference between {args.transformers} and {args.optimum}:\n")
        print(_colorize_lines(final_diff))
        print("\n")

        default_filename = DIFF_DIRECTORY / f"{args.transformers.stem}.txt"
        filename = input(f"Would you like to save this file at {default_filename} (y/n/other path)? ")
        if filename == "y":
            filename = default_filename
        if filename != "n":
            filename = Path(filename)
            should_override = True
            if filename.exists():
                should_override = _ask_yes_or_no_question("This file already exists, do you want to overwrite it")
                should_override = should_override == "y"

            if should_override:
                with open(filename, "w") as fp:
                    fp.write(final_diff)

                print(f"Content saved at: {filename}")


if __name__ == "__main__":
    main()
