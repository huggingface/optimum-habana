#!/bin/bash

pip install --upgrade pip
export RUN_SLOW=true
make slow_tests_1x
