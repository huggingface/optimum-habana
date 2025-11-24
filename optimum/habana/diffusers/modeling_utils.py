import diffusers

from .models.transformers import (
    GaudiQwenTimestepProjEmbeddings,
    GaudiQwenEmbedRope,
    GaudiQwenDoubleStreamAttnProcessor2_0,
)

def adapt_transformers_to_gaudi():
    """
    Replaces some Transformers' methods for equivalent methods optimized
    for Gaudi.
    """

    diffusers.models.transformers.transformer_qwenimage.QwenTimestepProjEmbeddings = GaudiQwenTimestepProjEmbeddings
    diffusers.models.transformers.transformer_qwenimage.QwenEmbedRope = GaudiQwenEmbedRope
    diffusers.models.transformers.transformer_qwenimage.QwenDoubleStreamAttnProcessor2_0 = GaudiQwenDoubleStreamAttnProcessor2_0
