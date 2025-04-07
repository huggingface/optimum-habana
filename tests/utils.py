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
    "audio-spectrogram-transformer": [
        ("MIT/ast-finetuned-speech-commands-v2", "Habana/wav2vec2"),
    ],
    "bert": [
        # ("bert-base-uncased", "Habana/bert-base-uncased"),
        ("bert-large-uncased-whole-word-masking", "Habana/bert-large-uncased-whole-word-masking"),
    ],
    "roberta": [
        # ("roberta-base", "Habana/roberta-base"),
        ("roberta-large", "Habana/roberta-large"),
    ],
    "albert": [
        ("albert-large-v2", "Habana/albert-large-v2"),
        # ("albert-xxlarge-v1", "Habana/albert-xxlarge-v1"),
    ],
    "distilbert": [
        # ("distilbert-base-uncased", "Habana/distilbert-base-uncased"),
    ],
    "gpt2": [
        # ("gpt2", "Habana/gpt2"),
        ("gpt2-xl", "Habana/gpt2"),
    ],
    "t5": [
        # ("t5-small", "Habana/t5"),
        ("google/flan-t5-xxl", "Habana/t5"),
    ],
    "vit": [
        ("google/vit-base-patch16-224-in21k", "Habana/vit"),
    ],
    "wav2vec2": [
        ("facebook/wav2vec2-base", "Habana/wav2vec2"),
        ("facebook/wav2vec2-large-lv60", "Habana/wav2vec2"),
    ],
    "swin": [("microsoft/swin-base-patch4-window7-224-in22k", "Habana/swin")],
    "clip": [("./clip-roberta", "Habana/clip")],
    "bridgetower": [("BridgeTower/bridgetower-large-itm-mlm-itc", "Habana/clip")],
    "gpt_neox": [("EleutherAI/gpt-neox-20b", "Habana/gpt2")],
    "llama": [("huggyllama/llama-7b", "Habana/llama"), ("meta-llama/Llama-3.1-8B", "Habana/llama")],
    "falcon": [("tiiuae/falcon-40b", "Habana/falcon")],
    "bloom": [("bigscience/bloom-7b1", "Habana/roberta-base")],
    "whisper": [("openai/whisper-small", "Habana/whisper")],
    "llama_guard": [("meta-llama/LlamaGuard-7b", "Habana/llama")],
    "code_llama": [("codellama/CodeLlama-13b-Instruct-hf", "Habana/llama")],
    "protst": [("mila-intel/protst-esm1b-for-sequential-classification", "Habana/gpt2")],
    "qwen2": [("Qwen/Qwen2-7B", "Habana/qwen"), ("Qwen/Qwen2-72B", "Habana/qwen")],
    "idefics2": [("HuggingFaceM4/idefics2-8b", "Habana/gpt2")],
    "mllama": [("meta-llama/Llama-3.2-11B-Vision-Instruct", "Habana/gpt2")],
    "gemma": [("google/gemma-2b-it", "Habana/gpt2")],
    "chatglm": [("THUDM/chatglm3-6b", "Habana/gpt2")],
    "llava": [("llava-hf/llava-1.5-7b-hf", "Habana/gpt2")],
}

MODELS_TO_TEST_FOR_QUESTION_ANSWERING = [
    # "bert",
    "roberta",
    # "albert",
    # "distilbert",
]

# Only BERT has been officially validated for sequence classification
MODELS_TO_TEST_FOR_SEQUENCE_CLASSIFICATION = [
    "bert",
    "llama_guard",
    # "roberta",
    # "albert",
    # "distilbert",
]

MODELS_TO_TEST_FOR_CAUSAL_LANGUAGE_MODELING = ["gpt2", "gpt_neox", "bloom", "code_llama", "gemma", "chatglm"]

MODELS_TO_TEST_FOR_SEQ2SEQ = ["t5"]

MODELS_TO_TEST_FOR_IMAGE_CLASSIFICATION = ["vit", "swin"]

# Only RoBERTa is tested in CI for MLM
MODELS_TO_TEST_FOR_MASKED_LANGUAGE_MODELING = [
    # "bert",
    "roberta",
    # "albert",
    # "distilbert",
]

MODELS_TO_TEST_FOR_AUDIO_CLASSIFICATION = ["wav2vec2", "audio-spectrogram-transformer"]

MODELS_TO_TEST_FOR_SPEECH_RECOGNITION = ["wav2vec2", "whisper"]

MODELS_TO_TEST_FOR_IMAGE_TEXT = ["clip"]

# This will be configured by conftest.py at the start of the pytest session
OH_DEVICE_CONTEXT = "unknown"
