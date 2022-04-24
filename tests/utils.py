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
# TODO: add configuration names once they have been pushed to the hub
MODELS_TO_TEST_MAPPING = {
    "bert": [
        # ("bert-base-uncased", ""),  # removed from CI to save time
        ("bert-large-uncased-whole-word-masking", ""),
    ],
    "roberta": [
        ("roberta-base", ""),
        ("roberta-large", ""),
    ],
    "albert": [
        ("albert-large-v2", ""),
        # ("albert-xxlarge-v1", ""),  # make Github action job exceed the limit of 6 hours
    ],
    "distilbert": [
        ("distilbert-base-uncased", ""),
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
