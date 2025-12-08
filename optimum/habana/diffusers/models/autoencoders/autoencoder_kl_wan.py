# Copyright 2023 The HuggingFace Team. All rights reserved.
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


import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F


CACHE_T = 2


def WanAvgDown3DForwardGaudi(self, x: torch.Tensor) -> torch.Tensor:
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/autoencoders/autoencoder_kl_wan.py#L37
    workaround a GC issue on G3: use transpose to replace permute to bypass extractDataMovementMultiNodes
    """
    pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
    pad = (0, 0, 0, 0, pad_t, 0)
    x = F.pad(x, pad)
    B, C, T, H, W = x.shape
    x = x.view(
        B,
        C,
        T // self.factor_t,
        self.factor_t,
        H // self.factor_s,
        self.factor_s,
        W // self.factor_s,
        self.factor_s,
    )

    # the original code is permute(0, 1, 3, 5, 7, 2, 4, 6)
    # split it by transpose(6, 7) and permute(0, 1, 3, 5, 6, 2, 4, 7)
    x = x.transpose(6, 7).contiguous()
    htcore.mark_step()
    x = x.permute(0, 1, 3, 5, 6, 2, 4, 7).contiguous()

    x = x.view(
        B,
        C * self.factor,
        T // self.factor_t,
        H // self.factor_s,
        W // self.factor_s,
    )
    x = x.view(
        B,
        self.out_channels,
        self.group_size,
        T // self.factor_t,
        H // self.factor_s,
        W // self.factor_s,
    )
    x = x.mean(dim=2)
    return x


def WanDupUp3DForwardGaudi(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/autoencoders/autoencoder_kl_wan.py#L90
    workaround a GC issue on G3: use transpose to replace permute to bypass extractDataMovementMultiNodes
    """
    x = x.repeat_interleave(self.repeats, dim=1)
    x = x.view(
        x.size(0),
        self.out_channels,
        self.factor_t,
        self.factor_s,
        self.factor_s,
        x.size(2),
        x.size(3),
        x.size(4),
    )

    # the original code is permute(0, 1, 5, 2, 6, 3, 7, 4)
    # split it by transpose(2, 5), permute(0, 1, 2, 5, 6, 3, 4, 7) and transpose(6, 7)
    x = x.transpose(2, 5).contiguous()
    htcore.mark_step()
    x = x.permute(0, 1, 2, 5, 6, 3, 4, 7).contiguous()
    htcore.mark_step()
    x = x.transpose(6, 7).contiguous()

    x = x.view(
        x.size(0),
        self.out_channels,
        x.size(2) * self.factor_t,
        x.size(4) * self.factor_s,
        x.size(6) * self.factor_s,
    )
    if first_chunk:
        x = x[:, :, self.factor_t - 1 :, :, :]
        x = x.contiguous()
    return x


def WanDecoder3dForwardGaudi(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/autoencoders/autoencoder_kl_wan.py#L874
    only add mark_step() for memory optimization.
    """
    ## conv1
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_in(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_in(x)

    ## middle
    x = self.mid_block(x, feat_cache, feat_idx)
    htcore.mark_step()

    ## upsamples
    for up_block in self.up_blocks:
        x = up_block(x, feat_cache, feat_idx, first_chunk=first_chunk)
        htcore.mark_step()

    ## head
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_out(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_out(x)
    return x


def WanEncoder3dForwardGaudi(self, x, feat_cache=None, feat_idx=[0]):
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/autoencoders/autoencoder_kl_wan.py#L586
    only add mark_step() for first iters caused too many time on graph build in lazy mode.
    """
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_in(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_in(x)
    htcore.mark_step()
    ## downsamples
    for layer in self.down_blocks:
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)
        htcore.mark_step()

    ## middle
    x = self.mid_block(x, feat_cache, feat_idx)
    htcore.mark_step()

    ## head
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv_out(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv_out(x)
    htcore.mark_step()
    return x
