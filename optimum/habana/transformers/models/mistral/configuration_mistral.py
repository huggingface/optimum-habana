from transformers.modeling_rope_utils import rope_config_validation
from transformers.models.mistral.configuration_mistral import MistralConfig


class MistralConfig(MistralConfig):
    """
    Copied from: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/mistral/configuration_mistral.py#L29
    Changes:
    - add `rope_scaling` and `_rope_scaling_validation` (inspired from Llama)
    """

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=4096,
        attention_dropout=0.0,
        rope_scaling=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            pad_token_id,
            bos_token_id,
            eos_token_id,
            tie_word_embeddings,
            rope_theta,
            sliding_window,
            attention_dropout,
            **kwargs,
        )

        self.rope_scaling = rope_scaling

        # Validate the correctness of rotary position embeddings parameters
        rope_config_validation(self)
