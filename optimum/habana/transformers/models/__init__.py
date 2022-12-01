from .albert import gaudi_albert_forward
from .modeling_all_models import gaudi_get_extended_attention_mask, gaudi_invert_attention_mask
from .vit import gaudi_vit_self_attention_forward
from .wav2vec2 import (
    _gaudi_wav2vec2_compute_mask_indices,
    _gaudi_wav2vec2_mask_hidden_states,
    _gaudi_wav2vec2_sample_negative_indices,
    gaudi_wav2vec2_forward,
)
