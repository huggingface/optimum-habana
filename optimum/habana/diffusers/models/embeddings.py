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

from typing import Tuple, Union

import torch


class RotaryPosEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freqs_cis):
        cos_, sin_ = freqs_cis  # [S, D]

        cos = cos_[None, None]
        sin = sin_[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        ctx.save_for_backward(cos, sin)
        x = torch.ops.hpu.rotary_pos_embedding(x, sin, cos, None, 0, 1)
        return x

    @staticmethod
    def backward(ctx, x_grad_in):
        (cos, sin) = ctx.saved_tensors
        x_embed_grad =torch.ops.hpu.rotary_pos_embedding_backward(x_grad_in, sin, cos, None, 0, 1)
        return x_embed_grad, None
