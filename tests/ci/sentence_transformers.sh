#!/bin/bash

python -m pip install --upgrade pip
python -m pip install /root/workspace/optimum-habana[tests]
cd /root/workspace/sentence-transformers/tests
python -m pip install ..
pytest test_cmnrl.py test_evaluator.py test_multi_process.py test_train_stsb.py test_compute_embeddings.py test_model_card_data.py test_trainer.py test_util.py test_pretrained_stsb.py
cd /root/workspace/optimum-habana/tests
python -m pytest test_sentence_transformers.py
