#!/bin/bash

python -m pip install --upgrade pip huggingface_hub
export RUN_SLOW=true
huggingface-cli login --token $1
make slow_tests_8x
