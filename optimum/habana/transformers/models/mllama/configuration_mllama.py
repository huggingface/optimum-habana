from transformers.models.mllama.configuration_mllama import MllamaTextConfig


class MllamaTextConfig(MllamaTextConfig):
    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 4096,
        hidden_act: str = "silu",
        num_hidden_layers: int = 40,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 14_336,
        rope_theta: float = 500_000,
        rope_scaling: Optional[Dict] = None,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 131_072,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        cross_attention_layers: Optional[List[int]] = None,
        dropout: float = 0,
        bos_token_id: int = 128000,
        eos_token_id: int = 128001,
        pad_token_id: Optional[int] = 128004,
        fused_qkv=False,
        parallel_strategy=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            hidden_act,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            rope_theta,
            rope_scaling,
            rms_norm_eps,
            max_position_embeddings,
            initializer_range,
            use_cache,
            tie_word_embeddings,
            cross_attention_layers,
            dropout,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            **kwargs,
        )

        self.fused_qkv = fused_qkv
        self.parallel_strategy = parallel_strategy
