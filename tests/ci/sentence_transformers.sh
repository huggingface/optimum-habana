#!/bin/bash

cd /root/workspace/sentence-transformers
python -m pytest test_compute_embeddings.py
cd /root/workspace/optimum-habana
python -m pytest test_sentence_transformers.py
