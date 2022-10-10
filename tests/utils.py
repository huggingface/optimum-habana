# coding=utf-8
# Copyright 2022 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Mapping between model families and specific model names with their configuration
MODELS_TO_TEST_MAPPING = {
    "bert": [
        ("bert-base-uncased", "Habana/bert-base-uncased"),
        ("bert-large-uncased-whole-word-masking", "Habana/bert-large-uncased-whole-word-masking"),
    ],
    "roberta": [
        ("roberta-base", "Habana/roberta-base"),
        ("roberta-large", "Habana/roberta-large"),
    ],
    "albert": [
        ("albert-large-v2", "Habana/albert-large-v2"),
        ("albert-xxlarge-v1", "Habana/albert-xxlarge-v1"),
    ],
    "distilbert": [
        ("distilbert-base-uncased", "Habana/distilbert-base-uncased"),
    ],
    "gpt2": [
        ("gpt2", "Habana/gpt2"),
        ("gpt2-xl", "Habana/gpt2"),
    ],
    "t5": [
        ("t5-small", "Habana/t5"),
    ],
    "vit": [
        ("google/vit-base-patch16-224-in21k", "Habana/vit"),
    ],
}

VALID_MODELS_FOR_QUESTION_ANSWERING = [
    "bert",
    "roberta",
    "albert",
    "distilbert",
]

# Only BERT is officially supported for sequence classification
VALID_MODELS_FOR_SEQUENCE_CLASSIFICATION = [
    "bert",
    # "roberta",
    # "albert",
    # "distilbert",
]

VALID_MODELS_FOR_CAUSAL_LANGUAGE_MODELING = ["gpt2"]

VALID_SEQ2SEQ_MODELS = ["t5"]

VALID_MODELS_FOR_IMAGE_CLASSIFICATION = ["vit"]

# Only RoBERTa is tested in CI for MLM
VALID_MODELS_FOR_MASKED_LANGUAGE_MODELING = [
    # "bert",
    "roberta",
    # "albert",
    # "distilbert",
]
