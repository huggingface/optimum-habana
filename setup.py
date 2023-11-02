# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import re

from setuptools import find_namespace_packages, setup


# Ensure we match the version set in optimum/habana/version.py
try:
    filepath = "optimum/habana/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)


INSTALL_REQUIRES = [
    "transformers >= 4.33.0, < 4.35.0",
    "optimum",
    "torch",
    "accelerate >= 0.23.0",
    "diffusers >= 0.18.0",
]

TESTS_REQUIRE = [
    "pytest",
    "psutil",
    "parameterized",
    "GitPython",
    "optuna",
    "sentencepiece",
    "datasets",
    "safetensors",
]

QUALITY_REQUIRES = [
    "black",
    "ruff",
    "hf_doc_builder",
]

EXTRAS_REQUIRE = {
    "tests": TESTS_REQUIRE,
    "quality": QUALITY_REQUIRES,
}

setup(
    name="optimum-habana",
    version=__version__,
    description=(
        "Optimum Habana is the interface between the Hugging Face Transformers and Diffusers libraries and Habana's"
        " Gaudi processor (HPU). It provides a set of tools enabling easy model loading, training and inference on"
        " single- and multi-HPU settings for different downstream tasks."
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, diffusers, mixed-precision training, fine-tuning, gaudi, hpu",
    url="https://huggingface.co/hardware/habana",
    author="HuggingFace Inc. Special Ops Team",
    author_email="hardware@huggingface.co",
    license="Apache",
    packages=find_namespace_packages(include=["optimum*"]),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    zip_safe=False,
)
