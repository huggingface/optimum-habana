# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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


def adapt_sentence_transformers_to_gaudi():
    """
    Replaces some SentenceTransformer' methods for equivalent methods optimized
    for Gaudi.
    """

    from optimum.habana.sentence_transformers import st_gaudi_encode, st_gaudi_transformer_tokenize, st_gaudi_data_collator_call
    from sentence_transformers import SentenceTransformer
    SentenceTransformer.encode = st_gaudi_encode

    from sentence_transformers.models import Transformer
    Transformer.tokenize = st_gaudi_transformer_tokenize

    from sentence_transformers.data_collator import SentenceTransformerDataCollator
    SentenceTransformerDataCollator.__call__ = st_gaudi_data_collator_call

