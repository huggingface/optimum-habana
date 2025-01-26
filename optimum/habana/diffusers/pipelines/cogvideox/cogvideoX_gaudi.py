from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None

#  FusedScaledDotProductAttention
class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        self._hpu_kernel_fsdpa = fusedSDPA

    def forward(self, query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode):
        return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode)


from diffusers.models.attention import Attention

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/embeddings.py#L697
    """
    cos_, sin_ = freqs_cis  # [S, D]

    cos = cos_[None, None]
    sin = sin_[None, None]
    cos, sin = cos.to(x.device), sin.to(x.device)

    x = torch.ops.hpu.rotary_pos_embedding(x, sin, cos, None, 0, 1)

    return x


class CogVideoXAttnProcessorGaudi:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA) if FusedSDPA else None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = self.fused_scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_casual=False, scale=None, softmax_mode='fast'
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states

import torch.nn.functional as F
from diffusers.models import attention_processor


attention_processor.CogVideoXAttnProcessor2_0 = CogVideoXAttnProcessorGaudi

from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXSafeConv3d
from diffusers.models.autoencoders.vae import DecoderOutput


class CogVideoXCausalConv3dGaudi(nn.Module):
    r"""A 3D causal convolution layer that pads the input tensor to ensure causality in CogVideoX Model.

    Args:
        in_channels (`int`): Number of channels in the input tensor.
        out_channels (`int`): Number of output channels produced by the convolution.
        kernel_size (`int` or `Tuple[int, int, int]`): Kernel size of the convolutional kernel.
        stride (`int`, defaults to `1`): Stride of the convolution.
        dilation (`int`, defaults to `1`): Dilation rate of the convolution.
        pad_mode (`str`, defaults to `"constant"`): Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        self.temporal_dim = 2
        self.time_kernel_size = time_kernel_size

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = CogVideoXSafeConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )


    def fake_context_parallel_forward(
        self, inputs: torch.Tensor, conv_cache: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        kernel_size = self.time_kernel_size
        if kernel_size > 1:
            cached_inputs = [conv_cache] if conv_cache is not None else [inputs[:, :, :1]] * (kernel_size - 1)
            inputs = torch.cat(cached_inputs + [inputs], dim=2)
        return inputs

    def forward(self, inputs: torch.Tensor, conv_cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        inputs = self.fake_context_parallel_forward(inputs, conv_cache)
        #conv_cache = inputs[:, :, -self.time_kernel_size + 1 :].clone()

        padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
        inputs_pad = F.pad(inputs, padding_2d, mode="constant", value=0)

        output = self.conv(inputs_pad)
        if self.time_kernel_size>1:
            if conv_cache is not None and conv_cache.shape == inputs[:, :, -self.time_kernel_size + 1:].shape:
                conv_cache.copy_(inputs[:, :, -self.time_kernel_size + 1:])
            else:
                conv_cache = inputs[:, :, -self.time_kernel_size + 1:].clone()
        return output, conv_cache

from diffusers.models.autoencoders import autoencoder_kl_cogvideox


autoencoder_kl_cogvideox.CogVideoXCausalConv3d = CogVideoXCausalConv3dGaudi

from diffusers.models.autoencoders.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX

def adapt_cogvideo_to_gaudi():
    import diffusers
    diffusers.models.autoencoders.autoencoder_kl_cogvideox.CogVideoXCausalConv3d  = CogVideoXCausalConv3dGaudi
    #diffusers.models.autoencoders.autoencoder_kl_cogvideox.AutoencoderKLCogVideoX = AutoencoderKLCogVideoXGaudi
    diffusers.models.attention_processor.CogVideoXAttnProcessor2_0 = CogVideoXAttnProcessorGaudi
    #diffusers.models.transformers.cogvideox_transformer_3d.CogVideoXTransformer3DModel = CogVideoXTransformer3DModelGaudi


