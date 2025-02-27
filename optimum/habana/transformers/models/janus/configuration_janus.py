from transformers.configuration_utils import PretrainedConfig
from typing import List, Optional

from ..llama.configuration_llama import LlamaConfig


class VisionConfig(PretrainedConfig):
    model_type = "vision"

    # For siglip

    # From params
    image_size: int = 384
    model_name: str = "siglip_large_patch16_384"
    select_feature: str = "same"
    select_layer: int = -1

    def __init__(
        self,
        image_size: int = 384,
        patch_size: int = 16,
        width: int = 1024,
        layers: int = 24,
        heads: int = 16,
        mlp_ratio: int = 4,
        global_pool: str = "map",
        use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        params: dict = kwargs.get("params", {})
        self.image_size = image_size # params.get("image_size", 384)
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.global_pool = global_pool
        self.use_checkpoint = use_checkpoint
        self.model_name = params.get("model_name", "siglip_large_patch16_384")
        self.select_feature = params.get("select_feature", "same")
        self.select_layer = params.get("select_layer", -1)


class MlpProjectorConfig(PretrainedConfig):
    depth: int
    input_dim: int
    n_embed: int
    projector_type: str


class AlignerConfig(MlpProjectorConfig):
    model_type = "aligner"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        params: dict = kwargs.get("params", {})
        self.depth: int = params.get("depth", 2)
        self.input_dim: int = params.get("input_dim", 1024)
        self.n_embed: int = params.get("n_embed", 4096)
        self.projector_type: str = params.get("projector_type", "mlp_gelu")


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"

    # From params
    image_token_size: int = 16384
    n_embed: int = 8

    def __init__(
        self,
        codebook_size: int = 16384,
        codebook_embed_dim: int = 8,
        codebook_l2_norm: bool = True,
        codebook_show_usage: bool = True,
        commit_loss_beta: float = 0.25,
        entropy_loss_ratio: float = 0.0,
        z_channels: int = 256,
        dropout_p: float = 0.0,
        encoder_ch_mult: Optional[List[int]] = None,
        decoder_ch_mult: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size
        self.codebook_embed_dim = codebook_embed_dim
        self.codebook_l2_norm = codebook_l2_norm
        self.codebook_show_usage = codebook_show_usage
        self.commit_loss_beta = commit_loss_beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.z_channels = z_channels
        self.dropout_p = dropout_p

        params: dict = kwargs.get("params", {})
        self.encoder_ch_mult = encoder_ch_mult or params.get("encoder_ch_mult", [1, 1, 2, 2, 4])
        self.decoder_ch_mult = decoder_ch_mult or params.get("decoder_ch_mult", [1, 1, 2, 2, 4])

        # FIXME: Not used. Maybe remove?
        self.image_token_size = params.get("image_token_size", 16384)
        self.n_embed = params.get("n_embed", 8)


class GenAlignerConfig(MlpProjectorConfig):
    model_type = "gen_aligner"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        params: dict = kwargs.get("params", {})
        self.depth: int = params.get("depth", 2)
        self.input_dim: int = params.get("input_dim", 8)
        self.n_embed: int = params.get("n_embed", 4096)
        self.projector_type: str = params.get("projector_type", "mlp_gelu")


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"

    # From params
    image_token_embed: int = 4096
    image_token_size: int = 16384
    n_embed: int = 4096

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        params: dict = kwargs.get("params", {})
        self.image_token_embed = params.get("image_token_embed", 4096)
        self.image_token_size = params.get("image_token_size", 16384)
        self.n_embed = params.get("n_embed", 4096)


class JanusMultiModalConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    def __init__(self, **kwargs):
        vision_config = kwargs.pop("vision_config", {})
        aligner_config = kwargs.pop("aligner_config", {})
        gen_vision_config = kwargs.pop("gen_vision_config", {})
        gen_aligner_config = kwargs.pop("gen_aligner_config", {})
        gen_head_config = kwargs.pop("gen_head_config", {})
        language_config = kwargs.pop("language_config", {})

        super().__init__(**kwargs)

        self.vision_config = VisionConfig(**vision_config)
        self.aligner_config = AlignerConfig(**aligner_config)
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)
