#!/bin/bash

python -m pip install --upgrade pip
export RUN_SLOW=true
make slow_tests_trl_ddpo && make slow_tests_trl_grpo
