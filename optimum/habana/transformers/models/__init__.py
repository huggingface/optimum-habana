from .albert import gaudi_albert_forward
from .bart import (
    gaudi_BartAttention_forward,
    gaudi_BartDecoder_forward,
    gaudi_BartDecoderLayer_forward,
    gaudi_BartEncoder_forward,
    gaudi_BartEncoderLayer_forward,
    gaudi_BartForConditionalGeneration_forward,
    gaudi_BartForConditionalGeneration_prepare_inputs_for_generation,
    gaudi_BartLearnedPositionalEmbedding,
    gaudi_BartModel_forward,
)
from .bert import gaudi_BertModel_forward
from .blip import (
    gaudi_BlipForConditionalGeneration_generate,
    gaudi_BlipForQuestionAnswering_generate,
    gaudi_BlipTextAttention_forward,
    gaudi_BlipTextEncoder_forward,
    gaudi_BlipTextLayer_forward,
    gaudi_BlipTextLMHead_forward,
    gaudi_BlipTextLMHead_prepare_inputs_for_generation,
    gaudi_BlipTextModel_forward,
    gaudi_BlipTextSelfAttention_forward,
)
from .bloom import (
    GaudiBloomForCausalLM,
    GaudiBloomMLP,
    gaudi_bloom_attention_forward,
    gaudi_bloom_block_forward,
    gaudi_bloom_convert_to_bloom_cache,
    gaudi_bloom_convert_to_standard_cache,
    gaudi_bloom_model_forward,
)
from .clip import (
    GaudiCLIPAttention,
    GaudiCLIPEncoder,
    GaudiCLIPEncoderLayer,
    GaudiCLIPVisionEmbeddings,
    GaudiCLIPVisionModel,
    GaudiCLIPVisionTransformer,
)
from .codegen import (
    GaudiCodeGenAttention,
    GaudiCodeGenForCausalLM,
    gaudi_codegen_block_forward,
    gaudi_codegen_model_forward,
)
from .decilm import (
    DeciLMConfig,
    DeciLMForCausalLM,
)
from .detr import gaudi_DetrConvModel_forward
from .esm import (
    gaudi_esm_for_protein_folding_forward,
    gaudi_esmfolding_trunk_forward,
    gaudi_rot_matmul,
    gaudi_rot_vec_mul,
)
from .falcon import (
    GaudiFalconAttention,
    GaudiFalconDecoderLayer,
    GaudiFalconForCausalLM,
    GaudiFalconMLP,
    GaudiFalconModel,
    gaudi_falcon_linear_forward,
)
from .gemma import (
    GaudiGemmaDecoderLayer,
    GaudiGemmaForCausalLM,
    gaudi_gemma_attention_forward,
    gaudi_gemma_model_forward,
)
from .gpt2 import GaudiGPT2Attention, GaudiGPT2Block, GaudiGPT2LMHeadModel, gaudi_gpt2_forward
from .gpt_bigcode import (
    GaudiGPTBigCodeForCausalLM,
    gaudi_gpt_bigcode_attention_forward,
    gaudi_gpt_bigcode_block_forward,
    gaudi_gpt_bigcode_model_forward,
)
from .gpt_neox import (
    GaudiGPTNeoXForCausalLM,
    gaudi_gpt_neox_attention_forward,
    gaudi_gpt_neox_layer_forward,
    gaudi_gpt_neox_model_forward,
    gaudi_gpt_neox_rotary_embedding_set_cos_sin_cache,
)
from .gptj import (
    GaudiGPTJAttention,
    GaudiGPTJBlock,
    GaudiGPTJForCausalLM,
    GaudiGPTJModel,
)
from .llama import (
    GaudiLlamaAttention,
    GaudiLlamaDecoderLayer,
    GaudiLlamaDynamicNTKScalingRotaryEmbedding,
    GaudiLlamaForCausalLM,
    GaudiLlamaLinearScalingRotaryEmbedding,
    GaudiLlamaMLP,
    GaudiLlamaModel,
    GaudiLlamaRotaryEmbedding,
    LlamaConfig,
    gaudi_llama_rmsnorm_forward,
)
from .llava import GaudiLlavaForConditionalGeneration
from .llava_next import GaudiLlavaNextForConditionalGeneration
from .mamba import (
    gaudi_MambaForCausalLM_prepare_inputs_for_generation,
    gaudi_MambaForCausalLM_update_model_kwargs_for_generation,
)
from .mistral import (
    GaudiMistralAttention,
    GaudiMistralDecoderLayer,
    GaudiMistralForCausalLM,
    GaudiMistralModel,
    MistralConfig,
    gaudi_mistral_rmsnorm_forward,
)
from .mixtral import (
    GaudiMixtralAttention,
    GaudiMixtralDecoderLayer,
    GaudiMixtralForCausalLM,
    GaudiMixtralModel,
    MixtralConfig,
    gaudi_mixtral_block_sparse_moe_forward,
    gaudi_mixtral_rmsnorm_forward,
)
from .modeling_all_models import (
    gaudi_check_and_enable_sdpa,
    gaudi_conv1d_forward,
    gaudi_get_extended_attention_mask,
    gaudi_invert_attention_mask,
)
from .mpt import (
    GaudiMptForCausalLM,
    GaudiMptModel,
    gaudi_mpt_attention_forward,
    gaudi_mpt_block_forward,
)
from .opt import (
    GaudiOPTForCausalLM,
    GaudiOPTLearnedPositionalEmbedding,
    gaudi_opt_attention_forward,
    gaudi_opt_decoder_forward,
    gaudi_opt_decoder_layer_forward,
    gaudi_opt_model_forward,
)
from .owlvit import gaudi_owlvitclasspredictionhead_forward
from .persimmon import (
    GaudiPersimmonForCausalLM,
    gaudi_persimmon_attention_forward,
    gaudi_persimmon_decoder_layer_forward,
    gaudi_persimmon_model_forward,
)
from .phi import (
    GaudiPhiAttention,
    GaudiPhiDecoderLayer,
    GaudiPhiForCausalLM,
    GaudiPhiModel,
)
from .qwen2 import (
    GaudiQwen2Attention,
    GaudiQwen2DecoderLayer,
    GaudiQwen2ForCausalLM,
    GaudiQwen2MLP,
    GaudiQwen2Model,
    gaudi_qwen2_rmsnorm_forward,
)
from .seamless_m4t import (
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
)
from .speecht5 import (
    gaudi_generate_speech,
    gaudi_SpeechT5Attention_forward,
    gaudi_SpeechT5Decoder_forward,
    gaudi_SpeechT5DecoderLayer_forward,
)
from .stablelm import (
    GaudiStableLmDecoderLayer,
    GaudiStableLmForCausalLM,
    gaudi_stablelm_attention_forward,
    gaudi_stablelm_model_forward,
)
from .starcoder2 import (
    GaudiStarcoder2DecoderLayer,
    GaudiStarcoder2ForCausalLM,
    gaudi_starcoder2_attention_forward,
    gaudi_starcoder2_model_forward,
)
from .t5 import (
    gaudi_t5_layernorm_forward,
    gaudi_T5Attention_forward,
    gaudi_T5Block_forward,
    gaudi_T5ForConditionalGeneration_forward,
    gaudi_T5ForConditionalGeneration_prepare_inputs_for_generation,
    gaudi_T5LayerSelfAttention_forward,
    gaudi_T5Stack_forward,
)
from .table_transformer import gaudi_table_transformer_conv_encoder_forward
from .vision_encoder_decoder import (
    gaudi_VisionEncoderDecoderModel_prepare_inputs_for_generation,
)
from .vit import gaudi_vit_self_attention_forward
from .vits import gaudi_unconstrained_rational_quadratic_spline
from .wav2vec2 import (
    _gaudi_wav2vec2_compute_mask_indices,
    _gaudi_wav2vec2_mask_hidden_states,
    _gaudi_wav2vec2_sample_negative_indices,
    gaudi_wav2vec2_encoder_forward,
    gaudi_wav2vec2_forward,
    gaudi_wav2vec2_tdnnlayer_forward,
    gaudi_wav2vec2forctc_forward,
)
