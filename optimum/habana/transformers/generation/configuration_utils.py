from transformers.generation import GenerationConfig


class GaudiGenerationConfig(GenerationConfig):
    """
    ignore_eos (`bool`, *optional*):
        Whether to ignore finished sequences (faster in lazy mode and with HPU graphs) or not (eager mode).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.static_shapes = kwargs.get("static_shapes", None)
        self.ignore_eos = kwargs.get("ignore_eos", None)
