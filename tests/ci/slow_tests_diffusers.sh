#!/bin/bash

python -m pip install --upgrade pip
export RUN_SLOW=true
huggingface-cli login --token $1
make test_installs
CUSTOM_BF16_OPS=1 python -m pytest tests/test_diffusers.py -v -s -k "test_no_throughput_regression_autocast"
make slow_tests_diffusers
