from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    pass

import copy
from transformers.models.mpt.configuration_mpt import PretrainedConfig
from transformers.models.mpt.configuration_mpt import MptAttentionConfig


class MptConfig(PretrainedConfig):
    """
    Copied from: https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/mpt/configuration_mpt.py
    Changes:
    - add `rope_scaling` `rope_theta` and `_rope_scaling_validation` (inspired from Llama)
    """

    model_type = "mpt"
    attribute_map = {
        "num_attention_heads": "n_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        expansion_ratio: int = 4,
        max_seq_len: int = 2048,
        vocab_size: int = 50368,
        resid_pdrop: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        emb_pdrop: float = 0.0,
        learned_pos_emb: bool = True,
        attn_config: MptAttentionConfig = None,
        init_device: str = "cpu",
        logit_scale: Optional[Union[float, str]] = None,
        no_bias: bool = True,
        verbose: int = 0,
        embedding_fraction: float = 1.0,
        norm_type: str = "low_precision_layernorm",
        use_cache: bool = False,
        initializer_range=0.02,
        rope_scaling=None,
        rope_theta=10000,
        **kwargs,
    ):
        if attn_config is None:
            self.attn_config = MptAttentionConfig()
        elif isinstance(attn_config, dict):
            self.attn_config = MptAttentionConfig(**attn_config)
        else:
            self.attn_config = attn_config
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.verbose = verbose
        self.embedding_fraction = embedding_fraction
        self.norm_type = norm_type
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self._rope_scaling_validation()

    def _rope_scaling_validation(self):
        """
        Taken from: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/configuration_llama.py#L172
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
