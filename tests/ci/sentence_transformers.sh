#!/bin/bash

python -m pip install --upgrade pip
python -m pip install /root/workspace/optimum-habana[tests]
cd /root/workspace/sentence-transformers/tests
python -m pip install ..
python -m pytest test_compute_embeddings.py test_evaluator.py test_multi_process.py test_pretrained_stsb.py test_util.py
cd /root/workspace/optimum-habana/tests
python -m pytest test_sentence_transformers.py
