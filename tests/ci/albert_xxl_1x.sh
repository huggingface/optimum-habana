#!/bin/bash

pip install --upgrade pip
export RUN_SLOW=true
export RUN_ALBERT_XXL_1X=true
pip install .[tests]
python -m pytest tests/test_examples.py -v -s -k "albert-xxlarge-v1_single_card"
