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

import transformers.models.bloom.modeling_bloom as modeling_bloom
import transformers.models.gpt2.modeling_gpt2 as modeling_gpt2
import transformers.models.gpt_neox.modeling_gpt_neox as modeling_gpt_neox
import transformers.models.gptj.modeling_gptj as modeling_gptj
import transformers.models.opt.modeling_opt as modeling_opt
import transformers.models.t5.modeling_t5 as modeling_t5
from transformers import pytorch_utils
from transformers.generation import GenerationMixin
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.albert.modeling_albert import AlbertModel
from transformers.models.esm.modeling_esm import EsmOutput, EsmSelfOutput
from transformers.models.esm.modeling_esmfold import (
    EsmFoldAttention,
    EsmFoldingTrunk,
    EsmFoldSelfAttention,
    EsmForProteinFolding,
)
from transformers.models.esm.openfold_utils import rigid_utils
from transformers.models.vit.modeling_vit import ViTSelfAttention
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

from .generation import GaudiGenerationMixin
from .models import (
    GaudiBloomForCausalLM,
    GaudiBloomMLP,
    GaudiGPT2Attention,
    GaudiGPT2LMHeadModel,
    GaudiGPTJForCausalLM,
    GaudiGPTNeoXForCausalLM,
    GaudiOPTForCausalLM,
    GaudiOPTLearnedPositionalEmbedding,
    GaudiT5DenseActDense,
    GaudiT5DenseGatedActDense,
    GaudiT5LayerCrossAttention,
    GaudiT5LayerFF,
    GaudiT5LayerSelfAttention,
    GaudiT5Stack,
    _gaudi_esmfold_attention_wrap_up,
    gaudi_albert_forward,
    gaudi_bloom_attention_forward,
    gaudi_bloom_block_forward,
    gaudi_bloom_convert_to_bloom_cache,
    gaudi_bloom_convert_to_standard_cache,
    gaudi_bloom_model_forward,
    gaudi_conv1d_forward,
    gaudi_esm_for_protein_folding_forward,
    gaudi_esmfold_self_attention_forward,
    gaudi_esmfolding_trunk_forward,
    gaudi_esmoutput_forward,
    gaudi_esmselfoutput_forward,
    gaudi_get_extended_attention_mask,
    gaudi_gpt2_block_forward,
    gaudi_gpt2_forward,
    gaudi_gpt_neox_attention_forward,
    gaudi_gpt_neox_layer_forward,
    gaudi_gpt_neox_model_forward,
    gaudi_gptj_attention_forward,
    gaudi_gptj_block_forward,
    gaudi_gptj_model_forward,
    gaudi_invert_attention_mask,
    gaudi_opt_attention_forward,
    gaudi_opt_decoder_forward,
    gaudi_opt_decoder_layer_forward,
    gaudi_opt_model_forward,
    gaudi_rot_matmul,
    gaudi_rot_vec_mul,
    gaudi_T5Attention_forward,
    gaudi_vit_self_attention_forward,
    gaudi_wav2vec2_forward,
)


def adapt_transformers_to_gaudi():
    """
    Replaces some Transformers' methods for equivalent methods optimized
    for Gaudi.
    """

    # optimize Conv1D
    pytorch_utils.Conv1D.forward = gaudi_conv1d_forward

    # Optimization tweak for ViT
    ViTSelfAttention.forward = gaudi_vit_self_attention_forward

    # Optimization tweak for Wav2Vec2
    # modeling_wav2vec2._compute_mask_indices = _gaudi_wav2vec2_compute_mask_indices
    # modeling_wav2vec2._sample_negative_indices = _gaudi_wav2vec2_sample_negative_indices
    # Wav2Vec2Model._mask_hidden_states = _gaudi_wav2vec2_mask_hidden_states
    Wav2Vec2Model.forward = gaudi_wav2vec2_forward

    # Generation is modified to run faster in lazy mode
    GenerationMixin.generate = GaudiGenerationMixin.generate
    GenerationMixin._update_model_kwargs_for_generation = GaudiGenerationMixin._update_model_kwargs_for_generation
    GenerationMixin._expand_inputs_for_generation = staticmethod(GaudiGenerationMixin._expand_inputs_for_generation)
    GenerationMixin.greedy_search = GaudiGenerationMixin.greedy_search
    GenerationMixin.sample = GaudiGenerationMixin.sample
    GenerationMixin.beam_search = GaudiGenerationMixin.beam_search
    GenerationMixin.beam_sample = GaudiGenerationMixin.beam_sample
    GenerationMixin.group_beam_search = GaudiGenerationMixin.group_beam_search
    GenerationMixin.constrained_beam_search = GaudiGenerationMixin.constrained_beam_search

    # Optimization for BLOOM generation on Gaudi
    modeling_bloom.BloomAttention.forward = gaudi_bloom_attention_forward
    modeling_bloom.BloomBlock.forward = gaudi_bloom_block_forward
    modeling_bloom.BloomModel.forward = gaudi_bloom_model_forward
    modeling_bloom.BloomMLP = GaudiBloomMLP
    modeling_bloom.BloomForCausalLM = GaudiBloomForCausalLM
    modeling_bloom.BloomPreTrainedModel._convert_to_standard_cache = gaudi_bloom_convert_to_standard_cache
    modeling_bloom.BloomPreTrainedModel._convert_to_bloom_cache = gaudi_bloom_convert_to_bloom_cache

    # Replace invert_attention_mask and get_extended_attention_mask
    # so that HMP is disabled for specific parts of the code
    ModuleUtilsMixin.invert_attention_mask = gaudi_invert_attention_mask
    ModuleUtilsMixin.get_extended_attention_mask = gaudi_get_extended_attention_mask
    # AlbertModel.forward does not rely on get_extended_attention_mask so it also needs to be replaced
    AlbertModel.forward = gaudi_albert_forward

    # From Transformers 4.27, the bias in the GPT2Attention layer is a Boolean
    # Since HCCL cannot handle this dtype, we revert it back to uint8 (same behaviour as Transformers <= 4.26)
    modeling_gpt2.GPT2Attention = GaudiGPT2Attention
    modeling_gpt2.GPT2Model.forward = gaudi_gpt2_forward
    modeling_gpt2.GPT2LMHeadModel = GaudiGPT2LMHeadModel
    modeling_gpt2.GPT2Block.forward = gaudi_gpt2_block_forward

    # Optimization for EsmFold on Gaudi
    EsmFoldingTrunk.forward = gaudi_esmfolding_trunk_forward
    EsmForProteinFolding.forward = gaudi_esm_for_protein_folding_forward
    EsmFoldAttention._wrap_up = _gaudi_esmfold_attention_wrap_up
    EsmFoldSelfAttention.forward = gaudi_esmfold_self_attention_forward
    rigid_utils.rot_matmul = gaudi_rot_matmul
    rigid_utils.rot_vec_mul = gaudi_rot_vec_mul
    EsmSelfOutput.forward = gaudi_esmselfoutput_forward
    EsmOutput.forward = gaudi_esmoutput_forward

    # Optimization for OPT generation on Gaudi
    modeling_opt.OPTAttention.forward = gaudi_opt_attention_forward
    modeling_opt.OPTDecoder.forward = gaudi_opt_decoder_forward
    modeling_opt.OPTForCausalLM = GaudiOPTForCausalLM
    modeling_opt.OPTModel.forward = gaudi_opt_model_forward
    modeling_opt.OPTDecoderLayer.forward = gaudi_opt_decoder_layer_forward
    modeling_opt.OPTLearnedPositionalEmbedding = GaudiOPTLearnedPositionalEmbedding

    # Optimization for GPTJ on Gaudi
    modeling_gptj.GPTJAttention.forward = gaudi_gptj_attention_forward
    modeling_gptj.GPTJForCausalLM = GaudiGPTJForCausalLM
    modeling_gptj.GPTJBlock.forward = gaudi_gptj_block_forward
    modeling_gptj.GPTJModel.forward = gaudi_gptj_model_forward

    # Optimization for gpt-neox generation on Gaudi
    modeling_gpt_neox.GPTNeoXForCausalLM = GaudiGPTNeoXForCausalLM
    modeling_gpt_neox.GPTNeoXModel.forward = gaudi_gpt_neox_model_forward
    modeling_gpt_neox.GPTNeoXLayer.forward = gaudi_gpt_neox_layer_forward
    modeling_gpt_neox.GPTNeoXAttention.forward = gaudi_gpt_neox_attention_forward

    # Dropout kernel improvement for Flan-T5
    modeling_t5.T5Stack = GaudiT5Stack
    modeling_t5.T5DenseGatedActDense = GaudiT5DenseGatedActDense
    modeling_t5.T5LayerFF = GaudiT5LayerFF
    modeling_t5.T5LayerSelfAttention = GaudiT5LayerSelfAttention
    modeling_t5.T5LayerCrossAttention = GaudiT5LayerCrossAttention
    modeling_t5.T5DenseActDense = GaudiT5DenseActDense
    modeling_t5.T5Attention.forward = gaudi_T5Attention_forward
