import os
from typing import Optional

import torch
from habana_frameworks.torch.hpex.kernels import FusedSDPA


def gaudi_fused_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    bsz, num_heads, tgt_len, head_dim = query.shape

    if tgt_len == 1:
        # next token
        softmax_mode = True if os.getenv("QUANT_CONFIG", "") else False
        recompute_mode = False
    else:
        # first token
        softmax_mode = "fast" if os.getenv("FLASH_ATTENTION_FAST_SOFTMAX") == "1" else "None"
        recompute_mode = True if os.getenv("FLASH_ATTENTION_RECOMPUTE") == "1" else False

    attn_output = FusedSDPA.apply(
        query,
        key,
        value,
        attention_mask,
        dropout,
        is_causal,
        None,
        softmax_mode,
        recompute_mode,
    )

    if attn_output.size() != (bsz, num_heads, tgt_len, head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, tgt_len, head_dim)}, but is {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)

    return attn_output, None
