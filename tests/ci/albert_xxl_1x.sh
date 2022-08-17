#!/bin/bash

python -m pip install --upgrade pip
export RUN_SLOW=true
export RUN_ALBERT_XXL_1X=true
python -m pip install .[tests]
python -m pytest tests/test_examples.py -v -s -k "albert-xxlarge-v1_single_card"
