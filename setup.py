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
    "transformers >= 4.26.0",
    "optimum",
    "torch",
    "accelerate",
    "diffusers >= 0.12.0",
]

TESTS_REQUIRE = [
    "pytest",
    "psutil",
    "parameterized",
    "GitPython",
    "optuna",
    "sentencepiece",
    "datasets",
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
