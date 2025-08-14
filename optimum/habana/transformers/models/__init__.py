from .albert import gaudi_albert_forward
from .baichuan import (
    BaichuanConfig,
    BaichuanForCausalLM,
    BaichuanTokenizer,
)
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
from .bert import (
    gaudi_Bert_Sdpa_SelfAttention_forward,
    gaudi_BertModel_forward,
)
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
from .chatglm import (
    ChatGLMConfig,
    ChatGLMForConditionalGeneration,
    ChatGLMForSequenceClassification,
    ChatGLMTokenizer,
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
from .cohere import (
    GaudiCohereAttention,
    GaudiCohereDecoderLayer,
    GaudiCohereForCausalLM,
    gaudi_cohere_model_forward,
)
from .decilm import (
    DeciLMConfig,
    DeciLMForCausalLM,
)
from .deepseek_v2 import (
    DeepseekTokenizerFast,
    DeepseekV2Config,
    DeepseekV2ForCausalLM,
)
from .deepseek_v3 import (
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
)
from .detr import (
    gaudi_DetrConvModel_forward,
    gaudi_DetrHungarianMatcher_forward,
    gaudi_DetrLoss_forward,
    gaudi_DetrLoss_loss_boxes,
    gaudi_DetrLoss_loss_cardinality,
    gaudi_DetrLoss_loss_labels,
)
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
from .falcon_mamba import (
    gaudi_FalconMambaForCausalLM_prepare_inputs_for_generation,
    gaudi_FalconMambaModel_forward,
)
from .gemma import (
    GaudiGemmaAttention,
    GaudiGemmaDecoderLayer,
    GaudiGemmaForCausalLM,
    GaudiGemmaMLP,
    GaudiGemmaModel,
)
from .gemma2 import (
    GaudiGemma2Attention,
    GaudiGemma2DecoderLayer,
    GaudiGemma2ForCausalLM,
    GaudiGemma2MLP,
    GaudiGemma2Model,
    GaudiGemma2RotaryEmbedding,
)
from .glm4v import (
    ChatGLM4Tokenizer,
    GLM4VConfig,
    GLM4VForConditionalGeneration,
    GLM4VForSequenceClassification,
)
from .gpt2 import (
    GaudiGPT2Attention,
    GaudiGPT2Block,
    GaudiGPT2DoubleHeadsModel,
    GaudiGPT2LMHeadModel,
    gaudi_gpt2_forward,
)
from .gpt_bigcode import (
    GaudiGPTBigCodeAttention,
    GaudiGPTBigCodeForCausalLM,
    gaudi_gpt_bigcode_block_forward,
    gaudi_gpt_bigcode_model_forward,
)
from .gpt_neo import (
    GaudiGPTNeoForCausalLM,
    gaudi_gpt_neo_attention_forward,
    gaudi_gpt_neo_block_forward,
    gaudi_gpt_neo_model_forward,
    gaudi_gpt_neo_selfattention_forward,
)
from .gpt_neox import (
    GaudiGPTNeoXAttention,
    GaudiGPTNeoXForCausalLM,
    GaudiGPTNeoXLayer,
    gaudi_gpt_neox_model_forward,
)
from .gptj import (
    GaudiGPTJAttention,
    GaudiGPTJBlock,
    GaudiGPTJForCausalLM,
    GaudiGPTJModel,
)
from .idefics2 import (
    Gaudi2Idefics2ImageProcessor,
    GaudiIdefics2ForConditionalGeneration,
    GaudiIdefics2Model,
    GaudiIdefics2VisionEmbeddings,
)
from .llama import (
    GaudiLlamaAttention,
    GaudiLlamaDecoderLayer,
    GaudiLlamaForCausalLM,
    GaudiLlamaMLP,
    GaudiLlamaModel,
    GaudiLlamaRotaryEmbedding,
    LlamaConfig,
    gaudi_llama_rmsnorm_forward,
)
from .llava import GaudiLlavaForConditionalGeneration
from .llava_next import GaudiLlavaNextForConditionalGeneration
from .llava_onevision import GaudiLlavaOnevisionForConditionalGeneration
from .mamba import (
    gaudi_MambaForCausalLM_prepare_inputs_for_generation,
    gaudi_MambaForCausalLM_update_model_kwargs_for_generation,
)
from .minicpm import MiniCPM3Config, MiniCPM3ForCausalLM
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
    GaudiMixtralSparseMoeBlock,
    MixtralConfig,
    gaudi_mixtral_rmsnorm_forward,
)
from .mllama import (
    GaudiMllamaCrossAttentionDecoderLayer,
    GaudiMllamaForCausalLM,
    GaudiMllamaForConditionalGeneration,
    GaudiMllamaSelfAttentionDecoderLayer,
    GaudiMllamaTextCrossAttention,
    GaudiMllamaTextModel,
    GaudiMllamaTextSelfAttention,
    GaudiMllamaVisionEncoder,
    GaudiMllamaVisionEncoderLayer,
    GaudiMllamaVisionModel,
    GaudiMllamaVisionSdpaAttention,
)
from .modeling_all_models import (
    KVCache,
    Matmul,
    apply_customized_rope_module,
    gaudi_check_and_enable_sdpa,
    gaudi_conv1d_forward,
    gaudi_get_extended_attention_mask,
    gaudi_invert_attention_mask,
)
from .mpt import (
    GaudiMptAttention,
    GaudiMptBlock,
    GaudiMptForCausalLM,
    GaudiMptModel,
)
from .opt import (
    GaudiOPTDecoderLayer,
    GaudiOPTForCausalLM,
    GaudiOPTLearnedPositionalEmbedding,
    gaudi_opt_attention_forward,
    gaudi_opt_decoder_forward,
    gaudi_opt_model_forward,
)
from .owlvit import gaudi_owlvitclasspredictionhead_forward
from .paligemma import GaudiPaliGemmaForConditionalGeneration
from .persimmon import (
    GaudiPersimmonAttention,
    GaudiPersimmonDecoderLayer,
    GaudiPersimmonForCausalLM,
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
    GaudiQwen2ForSequenceClassification,
    GaudiQwen2ForTokenClassification,
    GaudiQwen2MLP,
    GaudiQwen2Model,
    gaudi_qwen2_rmsnorm_forward,
)
from .qwen2_moe import (
    GaudiQwen2MoeAttention,
    GaudiQwen2MoeDecoderLayer,
    GaudiQwen2MoeForCausalLM,
    GaudiQwen2MoeMLP,
    GaudiQwen2MoeModel,
    gaudi_qwen2moe_block_sparse_moe_forward,
    gaudi_qwen2moe_rmsnorm_forward,
)
from .qwen2_vl import (
    GaudiQwen2VisionTransformerPretrainedModel,
    GaudiQwen2VLDecoderLayer,
    GaudiQwen2VLForConditionalGeneration,
    GaudiQwen2VLModel,
    GaudiQwen2VLSdpaAttention,
    GaudiQwen2VLVisionBlock,
    GaudiVisionSdpaAttention,
)
from .qwen3 import (
    GaudiQwen3Attention,
    GaudiQwen3DecoderLayer,
    GaudiQwen3ForCausalLM,
    GaudiQwen3ForSequenceClassification,
    GaudiQwen3ForTokenClassification,
    GaudiQwen3MLP,
    GaudiQwen3Model,
    gaudi_qwen3_rmsnorm_forward,
)
from .qwen3_moe import (
    GaudiQwen3MoeAttention,
    GaudiQwen3MoeDecoderLayer,
    GaudiQwen3MoeForCausalLM,
    GaudiQwen3MoeForSequenceClassification,
    GaudiQwen3MoeForTokenClassification,
    GaudiQwen3MoeMLP,
    GaudiQwen3MoeModel,
    GaudiQwen3MoeSparseMoeBlock,
    gaudi_qwen3moe_rmsnorm_forward,
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
from .siglip import (
    GaudiSiglipAttention,
    GaudiSiglipEncoder,
    GaudiSiglipEncoderLayer,
    GaudiSiglipVisionEmbeddings,
    GaudiSiglipVisionModel,
    GaudiSiglipVisionTransformer,
)
from .speecht5 import (
    gaudi_generate_speech,
    gaudi_SpeechT5Attention_forward,
    gaudi_SpeechT5Decoder_forward,
    gaudi_SpeechT5DecoderLayer_forward,
)
from .stablelm import (
    GaudiStableLmAttention,
    GaudiStableLmDecoderLayer,
    GaudiStableLmForCausalLM,
    gaudi_stablelm_model_forward,
)
from .starcoder2 import (
    GaudiStarcoder2Attention,
    GaudiStarcoder2DecoderLayer,
    GaudiStarcoder2ForCausalLM,
    GaudiStarcoder2Model,
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
from .video_llava import GaudiVideoLlavaForConditionalGeneration, GaudiVideoLlavaProcessor
from .vision_encoder_decoder import (
    gaudi_VisionEncoderDecoderModel_prepare_inputs_for_generation,
)
from .vit import gaudi_vit_self_attention_forward
from .vits import gaudi_unconstrained_rational_quadratic_spline
from .wav2vec2 import (
    GaudiWav2Vec2SdpaAttention,
    _gaudi_wav2vec2_compute_mask_indices,
    _gaudi_wav2vec2_mask_hidden_states,
    _gaudi_wav2vec2_sample_negative_indices,
    gaudi_wav2vec2_encoder_forward,
    gaudi_wav2vec2_forward,
    gaudi_wav2vec2_tdnnlayer_forward,
    gaudi_wav2vec2forctc_forward,
)
from .whisper import (
    GAUDI_WHISPER_ATTENTION_CLASSES,
    GaudiWhisperDecoder,
    GaudiWhisperDecoderLayer,
    GaudiWhisperForConditionalGeneration,
    GaudiWhisperModel,
    GaudiWhisperSdpaAttention,
)
from .xglm import (
    GaudiXGLMForCausalLM,
    gaudi_xglm_attention_forward,
    gaudi_xglm_decoder_layer_forward,
    gaudi_xglm_model_forward,
)
from .xlm_roberta import gaudi_XLMRoberta_Sdpa_SelfAttention_forward
