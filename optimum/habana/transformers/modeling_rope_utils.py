import math
from typing import Optional, Tuple

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, _compute_default_rope_parameters


def gaudi_compute_llama3_parameters(
    config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None, **rope_kwargs
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len, **rope_kwargs)

    factor = config.rope_scaling["factor"]  # `8` in the original implementation
    low_freq_factor = config.rope_scaling["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = config.rope_scaling["high_freq_factor"]  # `4` in the original implementation
    old_context_len = config.rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama, attention_factor


GAUDI_ROPE_INIT_FUNCTIONS = ROPE_INIT_FUNCTIONS
GAUDI_ROPE_INIT_FUNCTIONS["llama3"] = gaudi_compute_llama3_parameters
