"""
Adapted from the following sources:
https://huggingface.co/Deci/DeciLM-7B/blob/main/configuration_decilm.py
"""

from transformers.models.llama.configuration_llama import LlamaConfig


class DeciLMConfig(LlamaConfig):
    r"""
    Args:
        num_key_value_heads_per_layer (`List[int]`):
            The number of key-value heads per layer.
    """

    model_type = "deci"

    def __init__(
        self,
        num_key_value_heads_per_layer: list = None,
        **kwargs,
    ):
        self.num_key_value_heads_per_layer = num_key_value_heads_per_layer
        super().__init__(**kwargs)
