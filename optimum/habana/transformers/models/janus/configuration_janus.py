from transformers.configuration_utils import PretrainedConfig

from ..llama.configuration_llama import LlamaConfig


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    image_size: int = 384
    model_name: str = "siglip_large_patch16_384"
    select_feature: str = "same"
    select_layer: int = -1


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    depth: int = 2
    input_dim: int = 1024
    n_embed: int = 4096
    projector_type: str = "mlp_gelu"


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    image_token_size: int = 16384
    n_embed: int = 8


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    depth: int = 2
    input_dim: int = 8
    n_embed: int = 4096
    projector_type: str = "mlp_gelu"


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    image_token_embed: int = 4096
    image_token_size: int = 16384
    n_embed: int = 4096


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)
