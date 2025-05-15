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

from typing import Optional, Union

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DecoderOutput


def tiled_decode_gaudi(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
    r"""
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py#L1374
    Decode a batch of images using a tiled decoder.

    Args:
        z (`torch.Tensor`): Input batch of latent vectors.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

    Returns:
        [`~models.vae.DecoderOutput`] or `tuple`:
            If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
            returned.
    """
    # Rough memory assessment:
    #   - In CogVideoX-2B, there are a total of 24 CausalConv3d layers.
    #   - The biggest intermediate dimensions are: [1, 128, 9, 480, 720].
    #   - Assume fp16 (2 bytes per value).
    # Memory required: 1 * 128 * 9 * 480 * 720 * 24 * 2 / 1024**3 = 17.8 GB
    #
    # Memory assessment when using tiling:
    #   - Assume everything as above but now HxW is 240x360 by tiling in half
    # Memory required: 1 * 128 * 9 * 240 * 360 * 24 * 2 / 1024**3 = 4.5 GB

    print("run gaudi pipelined tiled decode!")
    batch_size, num_channels, num_frames, height, width = z.shape

    overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor_height))
    overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor_width))
    blend_extent_height = int(self.tile_sample_min_height * self.tile_overlap_factor_height)
    blend_extent_width = int(self.tile_sample_min_width * self.tile_overlap_factor_width)
    row_limit_height = self.tile_sample_min_height - blend_extent_height
    row_limit_width = self.tile_sample_min_width - blend_extent_width
    frame_batch_size = self.num_latent_frames_batch_size

    import habana_frameworks.torch.core as htcore

    # Split z into overlapping tiles and decode them separately.
    # The tiles have an overlap to avoid seams between tiles.
    rows = []
    for i in range(0, height, overlap_height):
        row = []
        for j in range(0, width, overlap_width):
            num_batches = max(num_frames // frame_batch_size, 1)
            conv_cache = None
            time = []

            for k in range(num_batches):
                remaining_frames = num_frames % frame_batch_size
                start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                end_frame = frame_batch_size * (k + 1) + remaining_frames
                tile = z[
                    :,
                    :,
                    start_frame:end_frame,
                    i : i + self.tile_latent_min_height,
                    j : j + self.tile_latent_min_width,
                ].clone()
                if self.post_quant_conv is not None:
                    tile = self.post_quant_conv(tile)
                tile, conv_cache = self.decoder(tile, conv_cache=conv_cache)
                time.append(tile.clone())
                htcore.mark_step()

            row.append(torch.cat(time, dim=2))
        rows.append(row)

    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            # blend the above tile and the left tile
            # to the current tile and add the current tile to the result row
            if i > 0:
                tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
            if j > 0:
                tile = self.blend_h(row[j - 1], tile, blend_extent_width)
            result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
        result_rows.append(torch.cat(result_row, dim=4))

    dec = torch.cat(result_rows, dim=3)

    if not return_dict:
        return (dec,)

    return DecoderOutput(sample=dec)


def CogVideoXCausalConv3dforwardGaudi(
    self, inputs: torch.Tensor, conv_cache: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py#L129
    change conv_cache.clone() to conv_cache.copy_().
    """

    # print('run gaudi CogVideoXCausalConv3d forward!')
    inputs = self.fake_context_parallel_forward(inputs, conv_cache)
    # conv_cache = inputs[:, :, -self.time_kernel_size + 1 :].clone()

    if self.pad_mode == "replicate":
        conv_cache = None
    else:
        if self.time_kernel_size > 1:
            if conv_cache is not None and conv_cache.shape == inputs[:, :, -self.time_kernel_size + 1 :].shape:
                conv_cache.copy_(inputs[:, :, -self.time_kernel_size + 1 :])
            else:
                conv_cache = inputs[:, :, -self.time_kernel_size + 1 :].clone()

    output = self.conv(inputs)
    return output, conv_cache
