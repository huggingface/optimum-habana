#!/bin/bash

python -m pip install --upgrade pip
python -m pip install /root/workspace/optimum-habana[tests]
cd /root/workspace/sentence-transformers
python -m pip install .
python -m pytest test_compute_embeddings.py
cd /root/workspace/optimum-habana
python -m pytest test_sentence_transformers.py
