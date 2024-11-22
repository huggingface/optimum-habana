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
import os

import accelerate
import transformers
import transformers.utils.fx

from ..accelerate.utils import extract_model_from_parallel
from .generation import (
    GaudiGenerationConfig,
    GaudiGenerationMixin,
    gaudi_EosTokenCriteria_call,
    gaudi_MaxLengthCriteria_call,
    gaudi_MaxTimeCriteria_call,
    gaudi_StoppingCriteriaList_call,
)
from .models import (
    GAUDI_WHISPER_ATTENTION_CLASSES,
    DeciLMConfig,
    DeciLMForCausalLM,
    Gaudi2Idefics2ImageProcessor,
    GaudiBloomForCausalLM,
    GaudiBloomMLP,
    GaudiCLIPAttention,
    GaudiCLIPEncoder,
    GaudiCLIPEncoderLayer,
    GaudiCLIPVisionEmbeddings,
    GaudiCLIPVisionModel,
    GaudiCLIPVisionTransformer,
    GaudiCodeGenAttention,
    GaudiCodeGenForCausalLM,
    GaudiCohereDecoderLayer,
    GaudiCohereForCausalLM,
    GaudiDynamicMoeBlock,
    GaudiFalconAttention,
    GaudiFalconDecoderLayer,
    GaudiFalconForCausalLM,
    GaudiFalconMLP,
    GaudiFalconModel,
    GaudiGemmaAttention,
    GaudiGemmaDecoderLayer,
    GaudiGemmaForCausalLM,
    GaudiGemmaMLP,
    GaudiGemmaModel,
    GaudiGPT2Attention,
    GaudiGPT2Block,
    GaudiGPT2DoubleHeadsModel,
    GaudiGPT2LMHeadModel,
    GaudiGPTBigCodeAttention,
    GaudiGPTBigCodeForCausalLM,
    GaudiGPTJAttention,
    GaudiGPTJBlock,
    GaudiGPTJForCausalLM,
    GaudiGPTJModel,
    GaudiGPTNeoForCausalLM,
    GaudiGPTNeoXAttention,
    GaudiGPTNeoXForCausalLM,
    GaudiGPTNeoXLayer,
    GaudiIdefics2ForConditionalGeneration,
    GaudiIdefics2Model,
    GaudiIdefics2VisionEmbeddings,
    GaudiLlamaAttention,
    GaudiLlamaDecoderLayer,
    GaudiLlamaDynamicNTKScalingRotaryEmbedding,
    GaudiLlamaForCausalLM,
    GaudiLlamaLinearScalingRotaryEmbedding,
    GaudiLlamaMLP,
    GaudiLlamaModel,
    GaudiLlamaRotaryEmbedding,
    GaudiLlavaForConditionalGeneration,
    GaudiLlavaNextForConditionalGeneration,
    GaudiMistralAttention,
    GaudiMistralDecoderLayer,
    GaudiMistralForCausalLM,
    GaudiMistralModel,
    GaudiMixtralAttention,
    GaudiMixtralDecoderLayer,
    GaudiMixtralForCausalLM,
    GaudiMixtralModel,
    GaudiMllamaCrossAttentionDecoderLayer,
    GaudiMllamaForCausalLM,
    GaudiMllamaForConditionalGeneration,
    GaudiMllamaSelfAttentionDecoderLayer,
    GaudiMllamaTextCrossAttention,
    GaudiMllamaTextModel,
    GaudiMllamaTextSelfAttention,
    GaudiMllamaVisionModel,
    GaudiMptAttention,
    GaudiMptBlock,
    GaudiMptForCausalLM,
    GaudiMptModel,
    GaudiOPTForCausalLM,
    GaudiOPTLearnedPositionalEmbedding,
    GaudiPersimmonAttention,
    GaudiPersimmonDecoderLayer,
    GaudiPersimmonForCausalLM,
    GaudiPhiAttention,
    GaudiPhiDecoderLayer,
    GaudiPhiForCausalLM,
    GaudiPhiModel,
    GaudiQwen2Attention,
    GaudiQwen2DecoderLayer,
    GaudiQwen2ForCausalLM,
    GaudiQwen2MLP,
    GaudiQwen2Model,
    GaudiQwen2MoeAttention,
    GaudiQwen2MoeDecoderLayer,
    GaudiQwen2MoeForCausalLM,
    GaudiQwen2MoeMLP,
    GaudiQwen2MoeModel,
    GaudiStableLmAttention,
    GaudiStableLmDecoderLayer,
    GaudiStableLmForCausalLM,
    GaudiStarcoder2Attention,
    GaudiStarcoder2DecoderLayer,
    GaudiStarcoder2ForCausalLM,
    GaudiStarcoder2Model,
    GaudiWhisperDecoder,
    GaudiWhisperDecoderLayer,
    GaudiWhisperForConditionalGeneration,
    GaudiWhisperModel,
    GaudiWhisperSdpaAttention,
    GaudiXGLMForCausalLM,
    LlamaConfig,
    MistralConfig,
    MixtralConfig,
    _gaudi_wav2vec2_compute_mask_indices,
    _gaudi_wav2vec2_mask_hidden_states,
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
    gaudi_BertModel_forward,
    gaudi_BlipForConditionalGeneration_generate,
    gaudi_BlipForQuestionAnswering_generate,
    gaudi_BlipTextAttention_forward,
    gaudi_BlipTextEncoder_forward,
    gaudi_BlipTextLayer_forward,
    gaudi_BlipTextLMHead_forward,
    gaudi_BlipTextLMHead_prepare_inputs_for_generation,
    gaudi_BlipTextModel_forward,
    gaudi_BlipTextSelfAttention_forward,
    gaudi_bloom_attention_forward,
    gaudi_bloom_block_forward,
    gaudi_bloom_convert_to_bloom_cache,
    gaudi_bloom_convert_to_standard_cache,
    gaudi_bloom_model_forward,
    gaudi_check_and_enable_sdpa,
    gaudi_codegen_block_forward,
    gaudi_codegen_model_forward,
    gaudi_cohere_attention_forward,
    gaudi_cohere_model_forward,
    gaudi_conv1d_forward,
    gaudi_DetrConvModel_forward,
    gaudi_esm_for_protein_folding_forward,
    gaudi_esmfolding_trunk_forward,
    gaudi_falcon_linear_forward,
    gaudi_generate_speech,
    gaudi_get_extended_attention_mask,
    gaudi_gpt2_forward,
    gaudi_gpt_bigcode_block_forward,
    gaudi_gpt_bigcode_model_forward,
    gaudi_gpt_neo_attention_forward,
    gaudi_gpt_neo_block_forward,
    gaudi_gpt_neo_model_forward,
    gaudi_gpt_neo_selfattention_forward,
    gaudi_gpt_neox_model_forward,
    gaudi_invert_attention_mask,
    gaudi_llama_rmsnorm_forward,
    gaudi_MambaForCausalLM_prepare_inputs_for_generation,
    gaudi_MambaForCausalLM_update_model_kwargs_for_generation,
    gaudi_mistral_rmsnorm_forward,
    gaudi_mixtral_block_sparse_moe_forward,
    gaudi_mixtral_rmsnorm_forward,
    gaudi_opt_attention_forward,
    gaudi_opt_decoder_forward,
    gaudi_opt_decoder_layer_forward,
    gaudi_opt_model_forward,
    gaudi_owlvitclasspredictionhead_forward,
    gaudi_persimmon_model_forward,
    gaudi_qwen2_rmsnorm_forward,
    gaudi_qwen2moe_block_sparse_moe_forward,
    gaudi_qwen2moe_rmsnorm_forward,
    gaudi_rot_matmul,
    gaudi_rot_vec_mul,
    gaudi_SeamlessM4TAttention_forward,
    gaudi_SeamlessM4TCodeHifiGan_get_output_hifigan_lengths,
    gaudi_SeamlessM4TDecoder_forward,
    gaudi_SeamlessM4TDecoderLayer_forward,
    gaudi_SeamlessM4TForTextToSpeech_forward,
    gaudi_SeamlessM4TForTextToSpeech_generate,
    gaudi_SeamlessM4TForTextToSpeech_prepare_inputs_for_generation,
    gaudi_SeamlessM4TTextToUnitForConditionalGeneration_forward,
    gaudi_SeamlessM4TTextToUnitForConditionalGeneration_prepare_inputs_for_generation,
    gaudi_SeamlessM4TTextToUnitModel_forward,
    gaudi_SpeechT5Attention_forward,
    gaudi_SpeechT5Decoder_forward,
    gaudi_SpeechT5DecoderLayer_forward,
    gaudi_stablelm_model_forward,
    gaudi_t5_layernorm_forward,
    gaudi_T5Attention_forward,
    gaudi_T5Block_forward,
    gaudi_T5ForConditionalGeneration_forward,
    gaudi_T5ForConditionalGeneration_prepare_inputs_for_generation,
    gaudi_T5LayerSelfAttention_forward,
    gaudi_T5Stack_forward,
    gaudi_table_transformer_conv_encoder_forward,
    gaudi_unconstrained_rational_quadratic_spline,
    gaudi_VisionEncoderDecoderModel_prepare_inputs_for_generation,
    gaudi_vit_self_attention_forward,
    gaudi_wav2vec2_encoder_forward,
    gaudi_wav2vec2_forward,
    gaudi_wav2vec2_tdnnlayer_forward,
    gaudi_wav2vec2forctc_forward,
    gaudi_xglm_attention_forward,
    gaudi_xglm_decoder_layer_forward,
    gaudi_xglm_model_forward,
)


def adapt_transformers_to_gaudi():
    """
    Replaces some Transformers' methods for equivalent methods optimized
    for Gaudi.
    """
    accelerate.utils.extract_model_from_parallel = extract_model_from_parallel
    accelerate.utils.other.extract_model_from_parallel = extract_model_from_parallel
    accelerate.accelerator.extract_model_from_parallel = extract_model_from_parallel

    # models that support symbolic tracing should be added to this list
    models_with_tracing_support = []

    # optimize Conv1D
    transformers.pytorch_utils.Conv1D.forward = gaudi_conv1d_forward

    # Optimization tweak for ViT
    transformers.models.vit.modeling_vit.ViTSelfAttention.forward = gaudi_vit_self_attention_forward

    # Optimization tweak for Wav2Vec2
    transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices = _gaudi_wav2vec2_compute_mask_indices
    # transformers.models.wav2vec2.modeling_wav2vec2._sample_negative_indices = _gaudi_wav2vec2_sample_negative_indices
    transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states = (
        _gaudi_wav2vec2_mask_hidden_states
    )
    transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model.forward = gaudi_wav2vec2_forward
    transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Encoder.forward = gaudi_wav2vec2_encoder_forward
    transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.forward = gaudi_wav2vec2forctc_forward
    transformers.models.wav2vec2.modeling_wav2vec2.TDNNLayer.forward = gaudi_wav2vec2_tdnnlayer_forward

    # Generation is modified to run faster in lazy mode
    transformers.generation.GenerationMixin.generate = GaudiGenerationMixin.generate
    transformers.generation.GenerationMixin._update_model_kwargs_for_generation = (
        GaudiGenerationMixin._update_model_kwargs_for_generation
    )
    transformers.generation.GenerationMixin.update_model_kwargs_for_bucketing = (
        GaudiGenerationMixin.update_model_kwargs_for_bucketing
    )
    transformers.generation.GenerationMixin._get_hpu_graphs_kwargs = GaudiGenerationMixin._get_hpu_graphs_kwargs
    transformers.generation.GenerationMixin._pad_past_key_values = GaudiGenerationMixin._pad_past_key_values
    transformers.generation.GenerationMixin._remove_past_key_values = GaudiGenerationMixin._remove_past_key_values
    transformers.generation.GenerationMixin._expand_inputs_for_generation = staticmethod(
        GaudiGenerationMixin._expand_inputs_for_generation
    )
    transformers.generation.GenerationMixin._prepare_attention_mask_for_generation = (
        GaudiGenerationMixin._prepare_attention_mask_for_generation
    )
    transformers.generation.GenerationMixin._prepare_decoder_input_ids_for_generation = (
        GaudiGenerationMixin._prepare_decoder_input_ids_for_generation
    )
    transformers.generation.GenerationMixin._prepare_decoder_attention_mask = (
        GaudiGenerationMixin._prepare_decoder_attention_mask
    )
    transformers.generation.GenerationMixin._prepare_generation_config = (
        GaudiGenerationMixin._prepare_generation_config
    )
    transformers.generation.GenerationMixin._prepare_generated_length = GaudiGenerationMixin._prepare_generated_length
    transformers.generation.GenerationMixin._get_stopping_criteria = GaudiGenerationMixin._get_stopping_criteria
    transformers.generation.GenerationMixin._validate_model_kwargs = GaudiGenerationMixin._validate_model_kwargs
    transformers.generation.GenerationMixin._dola_decoding = GaudiGenerationMixin._dola_decoding
    transformers.generation.GenerationMixin._sample = GaudiGenerationMixin._sample
    transformers.generation.GenerationMixin._beam_search = GaudiGenerationMixin._beam_search
    transformers.generation.GenerationMixin._group_beam_search = GaudiGenerationMixin._group_beam_search
    transformers.generation.GenerationMixin._constrained_beam_search = GaudiGenerationMixin._constrained_beam_search
    transformers.generation.GenerationMixin._contrastive_search = GaudiGenerationMixin._contrastive_search
    transformers.generation.GenerationMixin._assisted_decoding = GaudiGenerationMixin._assisted_decoding
    transformers.generation.GenerationMixin._get_candidate_generator = GaudiGenerationMixin._get_candidate_generator
    transformers.generation.GenerationMixin._prepare_cache_for_generation = (
        GaudiGenerationMixin._prepare_cache_for_generation
    )
    transformers.generation.GenerationConfig = GaudiGenerationConfig
    transformers.generation.configuration_utils.GenerationConfig = GaudiGenerationConfig
    transformers.modeling_utils.GenerationConfig = GaudiGenerationConfig
    transformers.generation.MaxLengthCriteria.__call__ = gaudi_MaxLengthCriteria_call
    transformers.generation.MaxTimeCriteria.__call__ = gaudi_MaxTimeCriteria_call
    transformers.generation.EosTokenCriteria.__call__ = gaudi_EosTokenCriteria_call
    transformers.generation.StoppingCriteriaList.__call__ = gaudi_StoppingCriteriaList_call

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

    # Optimization for BERT on Gaudi
    transformers.models.bert.modeling_bert.BertModel.forward = gaudi_BertModel_forward

    # Optimization for codegen generation on Gaudi
    transformers.models.codegen.modeling_codegen.CodeGenAttention = GaudiCodeGenAttention
    transformers.models.codegen.modeling_codegen.CodeGenForCausalLM = GaudiCodeGenForCausalLM
    transformers.models.codegen.modeling_codegen.CodeGenModel.forward = gaudi_codegen_model_forward
    transformers.models.codegen.modeling_codegen.CodeGenBlock.forward = gaudi_codegen_block_forward

    # Replace invert_attention_mask and get_extended_attention_mask
    # so that Torch Autocast is disabled for specific parts of the code
    transformers.modeling_utils.ModuleUtilsMixin.invert_attention_mask = gaudi_invert_attention_mask
    transformers.modeling_utils.ModuleUtilsMixin.get_extended_attention_mask = gaudi_get_extended_attention_mask

    # Override sdpa check on Gaudi
    transformers.modeling_utils.PreTrainedModel._check_and_enable_sdpa = gaudi_check_and_enable_sdpa

    # AlbertModel.forward does not rely on get_extended_attention_mask so it also needs to be replaced
    transformers.models.albert.modeling_albert.AlbertModel.forward = gaudi_albert_forward

    # Optimization for GPT2 on Gaudi
    transformers.models.gpt2.modeling_gpt2.GPT2Attention = GaudiGPT2Attention
    transformers.models.gpt2.modeling_gpt2.GPT2Model.forward = gaudi_gpt2_forward
    transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel = GaudiGPT2LMHeadModel
    transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModel = GaudiGPT2DoubleHeadsModel
    transformers.models.gpt2.modeling_gpt2.GPT2Block = GaudiGPT2Block
    models_with_tracing_support.extend((GaudiGPT2Attention, GaudiGPT2LMHeadModel))

    # Optimization for EsmFold on Gaudi
    transformers.models.esm.modeling_esmfold.EsmFoldingTrunk.forward = gaudi_esmfolding_trunk_forward
    transformers.models.esm.modeling_esmfold.EsmForProteinFolding.forward = gaudi_esm_for_protein_folding_forward
    transformers.models.esm.openfold_utils.rigid_utils.rot_matmul = gaudi_rot_matmul
    transformers.models.esm.openfold_utils.rigid_utils.rot_vec_mul = gaudi_rot_vec_mul

    # Optimization for OPT generation on Gaudi
    transformers.models.opt.modeling_opt.OPTAttention.forward = gaudi_opt_attention_forward
    transformers.models.opt.modeling_opt.OPTDecoder.forward = gaudi_opt_decoder_forward
    transformers.models.opt.modeling_opt.OPTForCausalLM = GaudiOPTForCausalLM
    transformers.models.opt.modeling_opt.OPTModel.forward = gaudi_opt_model_forward
    transformers.models.opt.modeling_opt.OPTDecoderLayer.forward = gaudi_opt_decoder_layer_forward
    transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding = GaudiOPTLearnedPositionalEmbedding

    # Optimization for GPTJ on Gaudi
    transformers.models.gptj.modeling_gptj.GPTJAttention = GaudiGPTJAttention
    transformers.models.gptj.modeling_gptj.GPTJForCausalLM = GaudiGPTJForCausalLM
    transformers.models.gptj.modeling_gptj.GPTJBlock = GaudiGPTJBlock
    transformers.models.gptj.modeling_gptj.GPTJModel = GaudiGPTJModel

    # Optimization for GPTBigCode on Gaudi
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeAttention = GaudiGPTBigCodeAttention
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeForCausalLM = GaudiGPTBigCodeForCausalLM
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeBlock.forward = gaudi_gpt_bigcode_block_forward
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeModel.forward = gaudi_gpt_bigcode_model_forward
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBIGCODE_ATTENTION_CLASSES.update(
        {"eager": GaudiGPTBigCodeAttention}
    )

    # Optimization for gpt-neo generation on Gaudi
    transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM = GaudiGPTNeoForCausalLM
    transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoModel.forward = gaudi_gpt_neo_model_forward
    transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoBlock.forward = gaudi_gpt_neo_block_forward
    transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoAttention.forward = gaudi_gpt_neo_attention_forward
    transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention.forward = gaudi_gpt_neo_selfattention_forward

    # Optimization for gpt-neox generation on Gaudi
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention = GaudiGPTNeoXAttention
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM = GaudiGPTNeoXForCausalLM
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXLayer = GaudiGPTNeoXLayer
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXModel.forward = gaudi_gpt_neox_model_forward

    # Optimization for llama generation on Gaudi
    transformers.models.llama.modeling_llama.LlamaForCausalLM = GaudiLlamaForCausalLM
    transformers.models.llama.modeling_llama.LlamaModel = GaudiLlamaModel
    transformers.models.llama.modeling_llama.LlamaAttention = GaudiLlamaAttention
    transformers.models.llama.modeling_llama.LlamaMLP = GaudiLlamaMLP
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = GaudiLlamaDecoderLayer
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = GaudiLlamaRotaryEmbedding
    transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding = GaudiLlamaLinearScalingRotaryEmbedding
    transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding = (
        GaudiLlamaDynamicNTKScalingRotaryEmbedding
    )
    transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = gaudi_llama_rmsnorm_forward
    transformers.models.llama.configuration_llama.LlamaConfig = LlamaConfig

    # Optimization for llava on Gaudi
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration = GaudiLlavaForConditionalGeneration
    transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration = (
        GaudiLlavaNextForConditionalGeneration
    )

    # Optimization for idefics2 on Gaudi
    transformers.models.idefics2.modeling_idefics2.Idefics2ForConditionalGeneration = (
        GaudiIdefics2ForConditionalGeneration
    )
    transformers.models.idefics2.modeling_idefics2.Idefics2Model = GaudiIdefics2Model
    transformers.models.idefics2.image_processing_idefics2.Idefics2ImageProcessor = Gaudi2Idefics2ImageProcessor
    transformers.models.idefics2.modeling_idefics2.Idefics2VisionEmbeddings = GaudiIdefics2VisionEmbeddings

    # Optimization for Clip on Gaudi
    transformers.models.clip.modeling_clip.CLIPVisionEmbeddings = GaudiCLIPVisionEmbeddings
    transformers.models.clip.modeling_clip.CLIPAttention = GaudiCLIPAttention
    transformers.models.clip.modeling_clip.CLIPEncoderLayer = GaudiCLIPEncoderLayer
    transformers.models.clip.modeling_clip.CLIPEncoder = GaudiCLIPEncoder
    transformers.models.clip.modeling_clip.CLIPVisionTransformer = GaudiCLIPVisionTransformer
    transformers.models.clip.modeling_clip.CLIPVisionModel = GaudiCLIPVisionModel

    # Optimization for falcon generation on Gaudi
    transformers.models.falcon.modeling_falcon.FalconAttention = GaudiFalconAttention
    transformers.models.falcon.modeling_falcon.FalconForCausalLM = GaudiFalconForCausalLM
    transformers.models.falcon.modeling_falcon.FalconMLP = GaudiFalconMLP
    transformers.models.falcon.modeling_falcon.FalconModel = GaudiFalconModel
    transformers.models.falcon.modeling_falcon.FalconDecoderLayer = GaudiFalconDecoderLayer
    transformers.models.falcon.modeling_falcon.FalconLinear.forward = gaudi_falcon_linear_forward

    # Optimization for t5 on Gaudi
    transformers.models.t5.modeling_t5.T5LayerNorm.forward = gaudi_t5_layernorm_forward
    transformers.models.t5.modeling_t5.T5Stack.forward = gaudi_T5Stack_forward
    transformers.models.t5.modeling_t5.T5LayerSelfAttention.forward = gaudi_T5LayerSelfAttention_forward
    transformers.models.t5.modeling_t5.T5ForConditionalGeneration.forward = gaudi_T5ForConditionalGeneration_forward
    transformers.models.t5.modeling_t5.T5ForConditionalGeneration.prepare_inputs_for_generation = (
        gaudi_T5ForConditionalGeneration_prepare_inputs_for_generation
    )
    transformers.models.t5.modeling_t5.T5Attention.forward = gaudi_T5Attention_forward
    transformers.models.t5.modeling_t5.T5Block.forward = gaudi_T5Block_forward

    # Optimization for table transformer on Gaudi
    transformers.models.table_transformer.modeling_table_transformer.TableTransformerConvEncoder.forward = (
        gaudi_table_transformer_conv_encoder_forward
    )

    # Optimization for mpt on Gaudi
    transformers.models.mpt.modeling_mpt.MptForCausalLM = GaudiMptForCausalLM
    transformers.models.mpt.modeling_mpt.MptModel = GaudiMptModel
    transformers.models.mpt.modeling_mpt.MptAttention = GaudiMptAttention
    transformers.models.mpt.modeling_mpt.MptBlock = GaudiMptBlock

    # Optimization for mistral on Gaudi
    transformers.models.mistral.modeling_mistral.MistralForCausalLM = GaudiMistralForCausalLM
    transformers.models.mistral.modeling_mistral.MistralAttention = GaudiMistralAttention
    transformers.models.mistral.modeling_mistral.MistralDecoderLayer = GaudiMistralDecoderLayer
    transformers.models.mistral.modeling_mistral.MistralModel = GaudiMistralModel
    transformers.models.mistral.modeling_mistral.MistralRMSNorm.forward = gaudi_mistral_rmsnorm_forward
    transformers.models.mistral.configuration_mistral.MistralConfig = MistralConfig

    # Optimization for phi on Gaudi
    transformers.models.phi.modeling_phi.PhiForCausalLM = GaudiPhiForCausalLM
    transformers.models.phi.modeling_phi.PhiAttention = GaudiPhiAttention
    transformers.models.phi.modeling_phi.PhiDecoderLayer = GaudiPhiDecoderLayer
    transformers.models.phi.modeling_phi.PhiModel = GaudiPhiModel

    # Optimization for gemma on Gaudi
    transformers.models.gemma.modeling_gemma.GemmaForCausalLM = GaudiGemmaForCausalLM
    transformers.models.gemma.modeling_gemma.GemmaMLP = GaudiGemmaMLP
    transformers.models.gemma.modeling_gemma.GemmaAttention = GaudiGemmaAttention
    transformers.models.gemma.modeling_gemma.GemmaDecoderLayer = GaudiGemmaDecoderLayer
    transformers.models.gemma.modeling_gemma.GemmaModel = GaudiGemmaModel

    # Optimization for blip Text model on Gaudi
    transformers.models.blip.BlipTextModel.forward = gaudi_BlipTextModel_forward
    transformers.models.blip.modeling_blip_text.BlipTextLMHeadModel.forward = gaudi_BlipTextLMHead_forward
    transformers.models.blip.modeling_blip_text.BlipTextLMHeadModel.prepare_inputs_for_generation = (
        gaudi_BlipTextLMHead_prepare_inputs_for_generation
    )
    transformers.models.blip.modeling_blip_text.BlipTextEncoder.forward = gaudi_BlipTextEncoder_forward
    transformers.models.blip.modeling_blip_text.BlipTextLayer.forward = gaudi_BlipTextLayer_forward
    transformers.models.blip.modeling_blip_text.BlipTextAttention.forward = gaudi_BlipTextAttention_forward
    transformers.models.blip.modeling_blip_text.BlipTextSelfAttention.forward = gaudi_BlipTextSelfAttention_forward
    transformers.models.blip.BlipForQuestionAnswering.generate = gaudi_BlipForQuestionAnswering_generate
    transformers.models.blip.BlipForConditionalGeneration.generate = gaudi_BlipForConditionalGeneration_generate

    # Optimization for mixtral on Gaudi
    transformers.models.mixtral.modeling_mixtral.MixtralAttention = GaudiMixtralAttention
    transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM = GaudiMixtralForCausalLM
    transformers.models.mixtral.modeling_mixtral.MixtralModel = GaudiMixtralModel
    # We need this workaround until moe op in hpu is supporting fp8
    if os.environ.get("QUANT_CONFIG"):
        transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock.forward = (
            gaudi_mixtral_block_sparse_moe_forward
        )
    else:
        transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock = GaudiDynamicMoeBlock
    transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer = GaudiMixtralDecoderLayer
    transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm.forward = gaudi_mixtral_rmsnorm_forward
    transformers.models.mixtral.configuration_mixtral.MixtralConfig = MixtralConfig

    # Optimization for speecht5 on Gaudi
    transformers.models.speecht5.modeling_speecht5.SpeechT5Decoder.forward = gaudi_SpeechT5Decoder_forward
    transformers.models.speecht5.modeling_speecht5.SpeechT5DecoderLayer.forward = gaudi_SpeechT5DecoderLayer_forward
    transformers.models.speecht5.modeling_speecht5.SpeechT5Attention.forward = gaudi_SpeechT5Attention_forward
    transformers.models.speecht5.modeling_speecht5._generate_speech = gaudi_generate_speech

    # Optimization for persimmon on Gaudi
    transformers.models.persimmon.modeling_persimmon.PersimmonAttention = GaudiPersimmonAttention
    transformers.models.persimmon.modeling_persimmon.PersimmonDecoderLayer = GaudiPersimmonDecoderLayer
    transformers.models.persimmon.modeling_persimmon.PersimmonForCausalLM = GaudiPersimmonForCausalLM
    transformers.models.persimmon.modeling_persimmon.PersimmonModel.forward = gaudi_persimmon_model_forward

    # Optimization for seamless m4t on Gaudi
    transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TAttention.forward = (
        gaudi_SeamlessM4TAttention_forward
    )
    transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TDecoderLayer.forward = (
        gaudi_SeamlessM4TDecoderLayer_forward
    )
    transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TDecoder.forward = (
        gaudi_SeamlessM4TDecoder_forward
    )
    transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TTextToUnitModel.forward = (
        gaudi_SeamlessM4TTextToUnitModel_forward
    )
    transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TTextToUnitForConditionalGeneration.forward = (
        gaudi_SeamlessM4TTextToUnitForConditionalGeneration_forward
    )

    transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TTextToUnitForConditionalGeneration.prepare_inputs_for_generation = gaudi_SeamlessM4TTextToUnitForConditionalGeneration_prepare_inputs_for_generation

    transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TCodeHifiGan._get_output_hifigan_lengths = (
        gaudi_SeamlessM4TCodeHifiGan_get_output_hifigan_lengths
    )

    transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.forward = (
        gaudi_SeamlessM4TForTextToSpeech_forward
    )

    transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.generate = (
        gaudi_SeamlessM4TForTextToSpeech_generate
    )

    transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.prepare_inputs_for_generation = (
        gaudi_SeamlessM4TForTextToSpeech_prepare_inputs_for_generation
    )

    transformers.models.vits.modeling_vits._unconstrained_rational_quadratic_spline = (
        gaudi_unconstrained_rational_quadratic_spline
    )

    # Optimization for starcoder2 on Gaudi
    transformers.models.starcoder2.modeling_starcoder2.Starcoder2ForCausalLM = GaudiStarcoder2ForCausalLM
    transformers.models.starcoder2.modeling_starcoder2.Starcoder2Model = GaudiStarcoder2Model
    transformers.models.starcoder2.modeling_starcoder2.Starcoder2Attention = GaudiStarcoder2Attention
    transformers.models.starcoder2.modeling_starcoder2.Starcoder2DecoderLayer = GaudiStarcoder2DecoderLayer

    # Optimization for qwen2 on Gaudi
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM = GaudiQwen2ForCausalLM
    transformers.models.qwen2.modeling_qwen2.Qwen2Model = GaudiQwen2Model
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention = GaudiQwen2Attention
    transformers.models.qwen2.modeling_qwen2.Qwen2MLP = GaudiQwen2MLP
    transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer = GaudiQwen2DecoderLayer
    transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm.forward = gaudi_qwen2_rmsnorm_forward

    # Optimization for qwen2Moe on Gaudi
    transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeMLP = GaudiQwen2MoeMLP
    transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeAttention = GaudiQwen2MoeAttention
    transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeDecoderLayer = GaudiQwen2MoeDecoderLayer
    transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeModel = GaudiQwen2MoeModel
    transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeForCausalLM = GaudiQwen2MoeForCausalLM
    transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeRMSNorm.forward = gaudi_qwen2moe_rmsnorm_forward
    transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock.forward = (
        gaudi_qwen2moe_block_sparse_moe_forward
    )

    # Optimization for stablelm on Gaudi
    transformers.models.stablelm.modeling_stablelm.StableLmAttention = GaudiStableLmAttention
    transformers.models.stablelm.modeling_stablelm.StableLmDecoderLayer = GaudiStableLmDecoderLayer
    transformers.models.stablelm.modeling_stablelm.StableLmForCausalLM = GaudiStableLmForCausalLM
    transformers.models.stablelm.modeling_stablelm.StableLmModel.forward = gaudi_stablelm_model_forward

    transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder.VisionEncoderDecoderModel.prepare_inputs_for_generation = gaudi_VisionEncoderDecoderModel_prepare_inputs_for_generation

    # Optimization for Owl ViT model on Gaudi
    transformers.models.owlvit.modeling_owlvit.OwlViTClassPredictionHead.forward = (
        gaudi_owlvitclasspredictionhead_forward
    )

    # Optimization for DETR model on Gaudi
    transformers.models.detr.modeling_detr.DetrConvModel.forward = gaudi_DetrConvModel_forward

    # Tell transformers which Gaudi models support tracing
    transformers.utils.fx._SUPPORTED_MODELS += tuple(cls.__name__ for cls in models_with_tracing_support)

    # Optimization for mamba on Gaudi
    transformers.models.mamba.modeling_mamba.MambaForCausalLM.prepare_inputs_for_generation = (
        gaudi_MambaForCausalLM_prepare_inputs_for_generation
    )
    transformers.models.mamba.modeling_mamba.MambaForCausalLM._update_model_kwargs_for_generation = (
        gaudi_MambaForCausalLM_update_model_kwargs_for_generation
    )

    # Optimization for Whisper on Gaudi
    transformers.models.whisper.modeling_whisper.WhisperSdpaAttention = GaudiWhisperSdpaAttention
    transformers.models.whisper.modeling_whisper.WhisperDecoderLayer = GaudiWhisperDecoderLayer
    transformers.models.whisper.modeling_whisper.WhisperDecoder = GaudiWhisperDecoder
    transformers.models.whisper.modeling_whisper.WhisperModel = GaudiWhisperModel
    transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration = GaudiWhisperForConditionalGeneration
    transformers.models.whisper.modeling_whisper.WHISPER_ATTENTION_CLASSES = GAUDI_WHISPER_ATTENTION_CLASSES

    # Optimization for mllama on Gaudi
    transformers.models.mllama.modeling_mllama.MllamaSelfAttentionDecoderLayer = GaudiMllamaSelfAttentionDecoderLayer
    transformers.models.mllama.modeling_mllama.MllamaCrossAttentionDecoderLayer = GaudiMllamaCrossAttentionDecoderLayer
    transformers.models.mllama.modeling_mllama.MllamaForCausalLM = GaudiMllamaForCausalLM
    transformers.models.mllama.modeling_mllama.MllamaTextSelfAttention = GaudiMllamaTextSelfAttention
    transformers.models.mllama.modeling_mllama.MllamaTextCrossAttention = GaudiMllamaTextCrossAttention
    transformers.models.mllama.modeling_mllama.MllamaForConditionalGeneration = GaudiMllamaForConditionalGeneration
    transformers.models.mllama.modeling_mllama.MllamaTextModel = GaudiMllamaTextModel
    transformers.models.mllama.modeling_mllama.MllamaVisionModel = GaudiMllamaVisionModel

    transformers.AutoConfig.register("deci", DeciLMConfig)
    transformers.AutoModelForCausalLM.register(DeciLMConfig, DeciLMForCausalLM)

    # Optimization for cohere on Gaudi
    transformers.models.cohere.modeling_cohere.CohereDecoderLayer = GaudiCohereDecoderLayer
    transformers.models.cohere.modeling_cohere.CohereForCausalLM = GaudiCohereForCausalLM
    transformers.models.cohere.modeling_cohere.CohereModel.forward = gaudi_cohere_model_forward
    transformers.models.cohere.modeling_cohere.CohereAttention.forward = gaudi_cohere_attention_forward

    # Optimization for xglm on Gaudi
    transformers.models.xglm.modeling_xglm.XGLMForCausalLM = GaudiXGLMForCausalLM
    transformers.models.xglm.modeling_xglm.XGLMModel.forward = gaudi_xglm_model_forward
    transformers.models.xglm.modeling_xglm.XGLMAttention.forward = gaudi_xglm_attention_forward
    transformers.models.xglm.modeling_xglm.XGLMDecoderLayer.forward = gaudi_xglm_decoder_layer_forward
