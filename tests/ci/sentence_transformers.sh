#!/bin/bash

cd ../sentence-transformers
python -m pytest test_compute_embeddings.py
cd ../optimum-habana
python -m pytest test_sentence_transformers.py
