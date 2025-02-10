#!/bin/bash

OPTIMUM_HABANA_PATH=${CI_OPTIMUM_HABANA_PATH:-/root/workspace/optimum-habana}
ACCELERATE_PATH=${CI_SENTENCE_TRANSFORMER_PATH:-/root/workspace/accelerate}

python -m pip install --upgrade pip
python -m pip install $OPTIMUM_HABANA_PATH[tests]
cd $ACCELERATE_PATH
python -m pip install .
RUN_SLOW=1 pytest tests/test_accelerator.py
