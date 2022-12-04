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

import shutil
from os import PathLike
from pathlib import Path
from typing import List, Tuple, Union

import pytest
from git import Repo

from .create_diff_file_for_example import DIFF_DIRECTORY, diff


TRANSFORMERS_REPO_URL = "https://github.com/huggingface/transformers.git"
TRANSFORMERS_REPO_PATH = Path("transformers")


def get_examples(
    transformers_example_dir: Union[str, PathLike],
    optimum_example_dir: Union[str, PathLike],
    include_readmes: bool = False,
) -> List[Tuple[str]]:
    """Retrieves the common example filenames between the transformers and the optimum-habana repos."""
    # TODO: validate for include README.
    glob_pattern = "*/run_*.py" if not include_readmes else "*/(run_*|README).(py|md)"

    transformers_files = list(Path(transformers_example_dir).glob(glob_pattern))
    transformer_example_names = {p.name for p in transformers_files}
    optimum_files = list(Path(optimum_example_dir).glob(glob_pattern))
    optimum_example_names = {p.name for p in optimum_files}

    transformer_files = sorted(p for p in transformers_files if p.name in optimum_example_names)
    optimum_files = sorted(p for p in optimum_files if p.name in transformer_example_names)

    return list(zip(transformer_files, optimum_files))


cloned_repo = Repo.clone_from(TRANSFORMERS_REPO_URL, TRANSFORMERS_REPO_PATH)
EXAMPLES = get_examples(TRANSFORMERS_REPO_PATH / "examples" / "pytorch", "examples")


@pytest.mark.parametrize("filename1,filename2", EXAMPLES, ids=lambda filename: str(filename.name))
def test_diff_match(filename1: Path, filename2: Path):
    reference_diff_filename = DIFF_DIRECTORY / f"{filename1.stem}.txt"
    try:
        with open(reference_diff_filename) as fp:
            reference_diff = fp.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find the reference diff file for example {filename1.name}, you can create it manually or with"
            " the command line tool located at: optimum-habana/tests/create_diff_file_for_example.py"
        )

    current_diff = diff(filename1, filename2)
    assert reference_diff == current_diff


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    # A bit hacky: this fixture will be called twice: at the beginning of the session, and at the end.
    # The goal is to cleanup the transformers repository at the end of the test session.
    # To do that, we first do nothing (yield some random value), which is executed at the beginning of the session, and
    # then remove the repo, which is executed at the end of the session.
    yield True
    shutil.rmtree(TRANSFORMERS_REPO_PATH)
