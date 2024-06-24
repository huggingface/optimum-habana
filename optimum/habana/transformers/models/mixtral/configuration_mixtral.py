from transformers.models.mixtral.configuration_mixtral import MixtralConfig


class MixtralConfig(MixtralConfig):
    """
    Copied from: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/mixtral/configuration_mixtral.py#L28
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
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=1e6,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
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
            num_experts_per_tok,
            num_local_experts,
            output_router_logits,
            router_aux_loss_coef,
            router_jitter_noise,
            **kwargs,
        )

        self.rope_scaling = rope_scaling
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
