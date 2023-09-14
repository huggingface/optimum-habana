from transformers.generation import GenerationConfig


class GaudiGenerationConfig(GenerationConfig):
    """
    This class extends [`transformers.generation.GenerationConfig`](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py)
    to add HPU-specific arguments for generation.

    Arg:
    trim_logit (`bool`, *optional):
        Calculate logits only for the last token to save memory in the first step.
    static_shapes (`bool`, *optional*):
        Whether to use static shapes for generation or not. It will run faster on HPUs with static shapes
        but not all models support it. If not specified, it will automatically be set to `True` if the given
        model supports it.
    ignore_eos (`bool`, *optional*):
        Whether to ignore finished sequences (faster in lazy mode and with HPU graphs) or not (eager mode).
        If not specified, it will automatically be set to `True` if lazy mode is on.
    attn_softmax_bf16 (`bool`, *optional*):
        Whether to run attention softmax layer in lower precision provided that the model supports it and
        is also running in lower precision.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trim_logits = kwargs.get("trim_logits", None)
        self.static_shapes = kwargs.get("static_shapes", None)
        self.ignore_eos = kwargs.get("ignore_eos", None)
        self.attn_softmax_bf16 = kwargs.get("attn_softmax_bf16", None)
