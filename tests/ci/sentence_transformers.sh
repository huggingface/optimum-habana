#!/bin/bash

OPTIMUM_HABANA_PATH=${CI_OPTIMUM_HABANA_PATH:-/root/workspace/optimum-habana}
SENTENCE_TRANSFORMER_PATH=${CI_SENTENCE_TRANSFORMER_PATH:-/root/workspace/sentence-transformers}

python -m pip install --upgrade pip
python -m pip install $OPTIMUM_HABANA_PATH[tests]
cd $SENTENCE_TRANSFORMER_PATH/tests
python -m pip install ..
pytest test_cmnrl.py test_evaluator.py test_multi_process.py test_train_stsb.py test_compute_embeddings.py test_model_card_data.py test_trainer.py test_util.py test_pretrained_stsb.py
cd $OPTIMUM_HABANA_PATH/tests
python -m pytest test_sentence_transformers.py
