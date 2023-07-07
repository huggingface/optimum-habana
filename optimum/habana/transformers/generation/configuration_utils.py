from transformers.generation.configuration_utils import GenerationConfig


class GaudiGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.static_shapes = kwargs.get("static_shapes", False)
        self.ignore_eos = kwargs.get("ignore_eos", False)
