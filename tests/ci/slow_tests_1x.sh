#!/bin/bash

python -m pip install --upgrade --user pip
export RUN_SLOW=true
make slow_tests_1x
