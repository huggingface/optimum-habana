#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from transformers.generation import GenerationMixin
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.albert.modeling_albert import AlbertModel
from transformers.models.vit.modeling_vit import ViTSelfAttention
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

from .generation import GaudiGenerationMixin
from .models import (
    gaudi_albert_forward,
    gaudi_get_extended_attention_mask,
    gaudi_invert_attention_mask,
    gaudi_vit_self_attention_forward,
    gaudi_wav2vec2_forward,
)


def adapt_transformers_to_gaudi(use_habana_mixed_precision: bool):
    """
    Replaces some Transformers' methods for equivalent methods optimized
    for Gaudi.

    Args:
        use_habana_mixed_precision (bool): whether HMP is used or not.
    """

    # Optimization tweak for ViT
    ViTSelfAttention.forward = gaudi_vit_self_attention_forward

    # Optimization tweak for Wav2Vec2
    # modeling_wav2vec2._compute_mask_indices = _gaudi_wav2vec2_compute_mask_indices
    # modeling_wav2vec2._sample_negative_indices = _gaudi_wav2vec2_sample_negative_indices
    # Wav2Vec2Model._mask_hidden_states = _gaudi_wav2vec2_mask_hidden_states
    Wav2Vec2Model.forward = gaudi_wav2vec2_forward

    # Generation is modified to run faster in lazy mode
    GenerationMixin.generate = GaudiGenerationMixin.generate
    GenerationMixin.greedy_search = GaudiGenerationMixin.greedy_search
    GenerationMixin.sample = GaudiGenerationMixin.sample
    GenerationMixin.beam_search = GaudiGenerationMixin.beam_search
    GenerationMixin.beam_sample = GaudiGenerationMixin.beam_sample
    GenerationMixin.group_beam_search = GaudiGenerationMixin.group_beam_search
    GenerationMixin.constrained_beam_search = GaudiGenerationMixin.constrained_beam_search

    if use_habana_mixed_precision:
        # When HMP is enabled, replace invert_attention_mask and get_extended_attention_mask
        # so that HMP is disabled for specific parts of the code
        ModuleUtilsMixin.invert_attention_mask = gaudi_invert_attention_mask
        ModuleUtilsMixin.get_extended_attention_mask = gaudi_get_extended_attention_mask
        # AlbertModel.forward does not rely on get_extended_attention_mask so it also needs
        # to be replaced when using HMP
        AlbertModel.forward = gaudi_albert_forward
