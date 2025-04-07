# Copyright 2024 The HuggingFace Team. All rights reserved.
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
PyTorch utilities: Utilities related to PyTorch
"""

import torch
from torch.fft import (
    fftn,
    fftshift,
    ifftn,
    ifftshift,
)


def gaudi_fourier_filter(x_in: "torch.Tensor", threshold: int, scale: int) -> "torch.Tensor":
    r"""
    Copied from https://github.com/huggingface/diffusers/blob/v0.29.2/src/diffusers/utils/torch_utils.py#L93
    Changes:
      - Use the cpu for the fft operations, because the HPU cannot support now.
    """
    x = x_in
    B, C, H, W = x.shape

    # FFT
    # Moving to CPU as torch.fft operations are not supported on HPU
    x = x.to(device="cpu", dtype=torch.float32)
    x_freq = fftn(x, dim=(-2, -1))
    x_freq = fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)
    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = ifftshift(x_freq, dim=(-2, -1))
    x_filtered = ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered.to(device=x_in.device, dtype=x_in.dtype)
