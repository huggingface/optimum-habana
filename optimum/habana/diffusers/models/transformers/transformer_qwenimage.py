# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adapted from
https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/transformers/transformer_qwenimage.py
and modified RoPE from complex to cos/sin
"""

import functools
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cos: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
    freqs_sin: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2].unsqueeze(1)
    cos = cos.unsqueeze(0)
    sin = freqs_sin[..., 1::2].unsqueeze(1)
    sin = sin.unsqueeze(0)

    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos

    return out.type_as(x)


class GaudiQwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, hidden_states):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))  # (N, D)

        conditioning = timesteps_emb

        return conditioning


class GaudiQwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        freqs_cos = []
        freqs_sin = []
        for i in range(3):
            cos, sin = self.rope_params(pos_index, self.axes_dim[i], self.theta)
            freqs_cos.append(cos)
            freqs_sin.append(sin)
        self.pos_freqs_cos = torch.cat(freqs_cos, dim=1)
        self.pos_freqs_sin = torch.cat(freqs_sin, dim=1)

        freqs_cos = []
        freqs_sin = []
        for i in range(3):
            cos, sin = self.rope_params(neg_index, self.axes_dim[i], self.theta)
            freqs_cos.append(cos)
            freqs_sin.append(sin)
        self.neg_freqs_cos = torch.cat(freqs_cos, dim=1)
        self.neg_freqs_sin = torch.cat(freqs_sin, dim=1)

        self.rope_cache_cos = {}
        self.rope_cache_sin = {}

        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0

        is_mps = index.device.type == "mps"
        is_npu = index.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64

        freqs_dtype = torch.float32
        freqs_cos, freqs_sin = get_1d_rotary_pos_embed(
            dim,
            index,
            theta,
            use_real=True,
            repeat_interleave_real=True,
            freqs_dtype=freqs_dtype,
        )
        return freqs_cos, freqs_sin

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        """
        if self.pos_freqs_cos.device != device:
            self.pos_freqs_cos = self.pos_freqs_cos.to(device)
            self.pos_freqs_sin = self.pos_freqs_sin.to(device)
            self.neg_freqs_cos = self.neg_freqs_cos.to(device)
            self.neg_freqs_sin = self.neg_freqs_sin.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs_cos = []
        vid_freqs_sin = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if not torch.compiler.is_compiling():
                if rope_key not in self.rope_cache_cos:
                    self.rope_cache_cos[rope_key], self.rope_cache_sin[rope_key] = self._compute_video_freqs(
                        frame, height, width, idx
                    )
                video_freq_cos = self.rope_cache_cos[rope_key]
                video_freq_sin = self.rope_cache_sin[rope_key]
            else:
                video_freq_cos, video_freq_sin = self._compute_video_freqs(frame, height, width, idx)

            video_freq_cos, video_freq_sin = video_freq_cos.to(device), video_freq_sin.to(device)
            vid_freqs_cos.append(video_freq_cos)
            vid_freqs_sin.append(video_freq_sin)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs_cos = self.pos_freqs_cos[max_vid_index : max_vid_index + max_len, ...]
        txt_freqs_sin = self.pos_freqs_sin[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs_cos = torch.cat(vid_freqs_cos, dim=0)
        vid_freqs_sin = torch.cat(vid_freqs_sin, dim=0)

        return vid_freqs_cos, vid_freqs_sin, txt_freqs_cos, txt_freqs_sin

    @functools.lru_cache(maxsize=None)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width

        freqs_pos_cos = self.pos_freqs_cos.split(self.axes_dim, dim=1)
        freqs_pos_sin = self.pos_freqs_sin.split(self.axes_dim, dim=1)
        freqs_neg_cos = self.neg_freqs_cos.split(self.axes_dim, dim=1)
        freqs_neg_sin = self.neg_freqs_sin.split(self.axes_dim, dim=1)

        freqs_cos_frame = freqs_pos_cos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        freqs_sin_frame = freqs_pos_sin[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_cos_height = torch.cat(
                [freqs_neg_cos[1][-(height - height // 2) :], freqs_pos_cos[1][: height // 2]], dim=0
            )
            freqs_cos_height = freqs_cos_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_sin_height = torch.cat(
                [freqs_neg_sin[1][-(height - height // 2) :], freqs_pos_sin[1][: height // 2]], dim=0
            )
            freqs_sin_height = freqs_sin_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_cos_width = torch.cat(
                [freqs_neg_cos[2][-(width - width // 2) :], freqs_pos_cos[2][: width // 2]], dim=0
            )
            freqs_cos_width = freqs_cos_width.view(1, 1, width, -1).expand(frame, height, width, -1)
            freqs_sin_width = torch.cat(
                [freqs_neg_sin[2][-(width - width // 2) :], freqs_pos_sin[2][: width // 2]], dim=0
            )
            freqs_sin_width = freqs_sin_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_cos_height = freqs_pos_cos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_sin_height = freqs_pos_sin[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_cos_width = freqs_pos_cos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
            freqs_sin_width = freqs_pos_sin[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs_cos = torch.cat([freqs_cos_frame, freqs_cos_height, freqs_cos_width], dim=-1).reshape(seq_lens, -1)
        freqs_sin = torch.cat([freqs_sin_frame, freqs_sin_height, freqs_sin_width], dim=-1).reshape(seq_lens, -1)
        return freqs_cos.clone().contiguous(), freqs_sin.clone().contiguous()


class GaudiQwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "GaudiQwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("GaudiQwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]  #

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs_cos, img_freqs_sin, txt_freqs_cos, txt_freqs_sin = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, freqs_cos=img_freqs_cos, freqs_sin=img_freqs_sin)
            img_key = apply_rotary_emb_qwen(img_key, freqs_cos=img_freqs_cos, freqs_sin=img_freqs_sin)
            txt_query = apply_rotary_emb_qwen(txt_query, freqs_cos=txt_freqs_cos, freqs_sin=txt_freqs_sin)
            txt_key = apply_rotary_emb_qwen(txt_key, freqs_cos=txt_freqs_cos, freqs_sin=txt_freqs_sin)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output
