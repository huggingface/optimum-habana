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

import transformers

from .generation import GaudiGenerationConfig, GaudiGenerationMixin
from .models import (
    GaudiBloomForCausalLM,
    GaudiBloomMLP,
    GaudiFalconForCausalLM,
    GaudiFalconModel,
    GaudiGPT2Attention,
    GaudiGPT2LMHeadModel,
    GaudiGPTJAttention,
    GaudiGPTJForCausalLM,
    GaudiGPTNeoXForCausalLM,
    GaudiLlamaForCausalLM,
    GaudiOPTForCausalLM,
    GaudiOPTLearnedPositionalEmbedding,
    _gaudi_esmfold_attention_wrap_up,
    gaudi_albert_forward,
    gaudi_BartAttention_forward,
    gaudi_BartDecoder_forward,
    gaudi_BartDecoderLayer_forward,
    gaudi_BartEncoder_forward,
    gaudi_BartEncoderLayer_forward,
    gaudi_BartForConditionalGeneration_forward,
    gaudi_BartForConditionalGeneration_prepare_inputs_for_generation,
    gaudi_BartLearnedPositionalEmbedding,
    gaudi_BartModel_forward,
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
    gaudi_falcon_attention_forward,
    gaudi_falcon_decoder_layer_forward,
    gaudi_falcon_rotary_embedding_forward,
    gaudi_get_extended_attention_mask,
    gaudi_gpt2_block_forward,
    gaudi_gpt2_forward,
    gaudi_gpt_neox_attention_forward,
    gaudi_gpt_neox_layer_forward,
    gaudi_gpt_neox_model_forward,
    gaudi_gptj_block_forward,
    gaudi_gptj_model_forward,
    gaudi_invert_attention_mask,
    gaudi_llama_attention_forward,
    gaudi_llama_decoder_layer_forward,
    gaudi_llama_model_forward,
    gaudi_llama_rmsnorm_forward,
    gaudi_opt_attention_forward,
    gaudi_opt_decoder_forward,
    gaudi_opt_decoder_layer_forward,
    gaudi_opt_model_forward,
    gaudi_rot_matmul,
    gaudi_rot_vec_mul,
    gaudi_vit_self_attention_forward,
    gaudi_wav2vec2_forward,
)


def adapt_transformers_to_gaudi():
    """
    Replaces some Transformers' methods for equivalent methods optimized
    for Gaudi.
    """

    # optimize Conv1D
    transformers.pytorch_utils.Conv1D.forward = gaudi_conv1d_forward

    # Optimization tweak for ViT
    transformers.models.vit.modeling_vit.ViTSelfAttention.forward = gaudi_vit_self_attention_forward

    # Optimization tweak for Wav2Vec2
    # transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices = _gaudi_wav2vec2_compute_mask_indices
    # transformers.models.wav2vec2.modeling_wav2vec2._sample_negative_indices = _gaudi_wav2vec2_sample_negative_indices
    # transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states = _gaudi_wav2vec2_mask_hidden_states
    transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model.forward = gaudi_wav2vec2_forward

    # Generation is modified to run faster in lazy mode
    transformers.generation.GenerationMixin.generate = GaudiGenerationMixin.generate
    transformers.generation.GenerationMixin._update_model_kwargs_for_generation = (
        GaudiGenerationMixin._update_model_kwargs_for_generation
    )
    transformers.generation.GenerationMixin._expand_inputs_for_generation = staticmethod(
        GaudiGenerationMixin._expand_inputs_for_generation
    )
    transformers.generation.GenerationMixin._prepare_attention_mask_for_generation = (
        GaudiGenerationMixin._prepare_attention_mask_for_generation
    )
    transformers.generation.GenerationMixin._prepare_decoder_input_ids_for_generation = (
        GaudiGenerationMixin._prepare_decoder_input_ids_for_generation
    )
    transformers.generation.GenerationMixin._get_stopping_criteria = GaudiGenerationMixin._get_stopping_criteria
    transformers.generation.GenerationMixin._prepare_past_key_values = GaudiGenerationMixin._prepare_past_key_values
    transformers.generation.GenerationMixin._prepare_decoder_attention_mask = (
        GaudiGenerationMixin._prepare_decoder_attention_mask
    )
    transformers.generation.GenerationMixin._validate_model_kwargs = GaudiGenerationMixin._validate_model_kwargs
    transformers.generation.GenerationMixin.greedy_search = GaudiGenerationMixin.greedy_search
    transformers.generation.GenerationMixin.sample = GaudiGenerationMixin.sample
    transformers.generation.GenerationMixin.beam_search = GaudiGenerationMixin.beam_search
    transformers.generation.GenerationMixin.beam_sample = GaudiGenerationMixin.beam_sample
    transformers.generation.GenerationMixin.group_beam_search = GaudiGenerationMixin.group_beam_search
    transformers.generation.GenerationMixin.constrained_beam_search = GaudiGenerationMixin.constrained_beam_search
    transformers.generation.GenerationConfig = GaudiGenerationConfig
    transformers.modeling_utils.GenerationConfig = GaudiGenerationConfig

    # Optimization for BLOOM generation on Gaudi
    transformers.models.bloom.modeling_bloom.BloomAttention.forward = gaudi_bloom_attention_forward
    transformers.models.bloom.modeling_bloom.BloomBlock.forward = gaudi_bloom_block_forward
    transformers.models.bloom.modeling_bloom.BloomModel.forward = gaudi_bloom_model_forward
    transformers.models.bloom.modeling_bloom.BloomMLP = GaudiBloomMLP
    transformers.models.bloom.modeling_bloom.BloomForCausalLM = GaudiBloomForCausalLM
    transformers.models.bloom.modeling_bloom.BloomPreTrainedModel._convert_to_standard_cache = (
        gaudi_bloom_convert_to_standard_cache
    )
    transformers.models.bloom.modeling_bloom.BloomPreTrainedModel._convert_to_bloom_cache = (
        gaudi_bloom_convert_to_bloom_cache
    )

    # Optimization for BART generation on Gaudi
    transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding = gaudi_BartLearnedPositionalEmbedding
    transformers.models.bart.modeling_bart.BartAttention.forward = gaudi_BartAttention_forward
    transformers.models.bart.modeling_bart.BartEncoderLayer.forward = gaudi_BartEncoderLayer_forward
    transformers.models.bart.modeling_bart.BartDecoderLayer.forward = gaudi_BartDecoderLayer_forward
    transformers.models.bart.modeling_bart.BartEncoder.forward = gaudi_BartEncoder_forward
    transformers.models.bart.modeling_bart.BartDecoder.forward = gaudi_BartDecoder_forward
    transformers.models.bart.modeling_bart.BartModel.forward = gaudi_BartModel_forward
    transformers.models.bart.modeling_bart.BartForConditionalGeneration.forward = (
        gaudi_BartForConditionalGeneration_forward
    )
    transformers.models.bart.modeling_bart.BartForConditionalGeneration.prepare_inputs_for_generation = (
        gaudi_BartForConditionalGeneration_prepare_inputs_for_generation
    )

    # Replace invert_attention_mask and get_extended_attention_mask
    # so that HMP is disabled for specific parts of the code
    transformers.modeling_utils.ModuleUtilsMixin.invert_attention_mask = gaudi_invert_attention_mask
    transformers.modeling_utils.ModuleUtilsMixin.get_extended_attention_mask = gaudi_get_extended_attention_mask
    # AlbertModel.forward does not rely on get_extended_attention_mask so it also needs to be replaced
    transformers.models.albert.modeling_albert.AlbertModel.forward = gaudi_albert_forward

    # From Transformers 4.27, the bias in the GPT2Attention layer is a Boolean
    # Since HCCL cannot handle this dtype, we revert it back to uint8 (same behaviour as Transformers <= 4.26)
    transformers.models.gpt2.modeling_gpt2.GPT2Attention = GaudiGPT2Attention
    transformers.models.gpt2.modeling_gpt2.GPT2Model.forward = gaudi_gpt2_forward
    transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel = GaudiGPT2LMHeadModel
    transformers.models.gpt2.modeling_gpt2.GPT2Block.forward = gaudi_gpt2_block_forward

    # Optimization for EsmFold on Gaudi
    transformers.models.esm.modeling_esmfold.EsmFoldingTrunk.forward = gaudi_esmfolding_trunk_forward
    transformers.models.esm.modeling_esmfold.EsmForProteinFolding.forward = gaudi_esm_for_protein_folding_forward
    transformers.models.esm.modeling_esmfold.EsmFoldAttention._wrap_up = _gaudi_esmfold_attention_wrap_up
    transformers.models.esm.modeling_esmfold.EsmFoldSelfAttention.forward = gaudi_esmfold_self_attention_forward
    transformers.models.esm.openfold_utils.rigid_utils.rot_matmul = gaudi_rot_matmul
    transformers.models.esm.openfold_utils.rigid_utils.rot_vec_mul = gaudi_rot_vec_mul
    transformers.models.esm.modeling_esm.EsmSelfOutput.forward = gaudi_esmselfoutput_forward
    transformers.models.esm.modeling_esm.EsmOutput.forward = gaudi_esmoutput_forward

    # Optimization for OPT generation on Gaudi
    transformers.models.opt.modeling_opt.OPTAttention.forward = gaudi_opt_attention_forward
    transformers.models.opt.modeling_opt.OPTDecoder.forward = gaudi_opt_decoder_forward
    transformers.models.opt.modeling_opt.OPTForCausalLM = GaudiOPTForCausalLM
    transformers.models.opt.modeling_opt.OPTModel.forward = gaudi_opt_model_forward
    transformers.models.opt.modeling_opt.OPTDecoderLayer.forward = gaudi_opt_decoder_layer_forward
    transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding = GaudiOPTLearnedPositionalEmbedding

    # Optimization for GPTJ on Gaudi
    # From Transformers 4.27, the bias in the GPT2Attention layer is a Boolean
    # Since HCCL cannot handle this dtype, we revert it back to uint8 (same behaviour as Transformers <= 4.26)
    transformers.models.gptj.modeling_gptj.GPTJAttention = GaudiGPTJAttention
    transformers.models.gptj.modeling_gptj.GPTJForCausalLM = GaudiGPTJForCausalLM
    transformers.models.gptj.modeling_gptj.GPTJBlock.forward = gaudi_gptj_block_forward
    transformers.models.gptj.modeling_gptj.GPTJModel.forward = gaudi_gptj_model_forward

    # Optimization for gpt-neox generation on Gaudi
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM = GaudiGPTNeoXForCausalLM
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXModel.forward = gaudi_gpt_neox_model_forward
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXLayer.forward = gaudi_gpt_neox_layer_forward
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention.forward = gaudi_gpt_neox_attention_forward

    # Optimization for llama generation on Gaudi
    transformers.models.llama.modeling_llama.LlamaForCausalLM = GaudiLlamaForCausalLM
    transformers.models.llama.modeling_llama.LlamaModel.forward = gaudi_llama_model_forward
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = gaudi_llama_decoder_layer_forward
    transformers.models.llama.modeling_llama.LlamaAttention.forward = gaudi_llama_attention_forward
    transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = gaudi_llama_rmsnorm_forward

    # Optimization for falcon generation on Gaudi
    transformers.models.falcon.modeling_falcon.FalconForCausalLM = GaudiFalconForCausalLM
    transformers.models.falcon.modeling_falcon.FalconModel = GaudiFalconModel
    transformers.models.falcon.modeling_falcon.FalconDecoderLayer.forward = gaudi_falcon_decoder_layer_forward
    transformers.models.falcon.modeling_falcon.FalconAttention.forward = gaudi_falcon_attention_forward
    transformers.models.falcon.modeling_falcon.FalconRotaryEmbedding.forward = gaudi_falcon_rotary_embedding_forward
