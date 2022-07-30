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
    "transformers >= 4.20.0",
    "optimum",
    "datasets",
    "tokenizers",
    "torch",
    "sentencepiece",
    "scipy",
    "pillow",
]

TESTS_REQUIRE = [
    "pytest",
    "psutil",
    "parameterized",
    "GitPython",
    "optuna",
]

EXTRAS_REQUIRE = {
    "tests": TESTS_REQUIRE,
}

setup(
    name="optimum-habana",
    version=__version__,
    description=(
        "Optimum Habana is the interface between the Hugging Face Transformers library and Habana Gaudi Processor"
        " (HPU). It provides a set of tools enabling easy model loading and fine-tuning on single- and multi-HPU"
        " settings for different downstream tasks."
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
    keywords="transformers, mixed-precision training, fine-tuning, gaudi, hpu",
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
