#!/bin/bash

pip install --user --upgrade pip
export RUN_SLOW=true
export RUN_ALBERT_XXL_1X=true
pip install --user --force-reinstall .[tests]
python -m pytest tests/test_examples.py -v -s -k "albert-xxlarge-v1_single_card"
