#!/bin/bash

python -m pip install --upgrade pip
export RUN_SLOW=true
make test_installs
CUSTOM_BF16_OPS=1 python -m pytest tests/test_diffusers.py -v -s -k "test_stable_diffusion_default"
make slow_tests_diffusers
