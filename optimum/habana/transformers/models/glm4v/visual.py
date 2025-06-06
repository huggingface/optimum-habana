# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
# Copyright (C) 2025 Habana Labs, Ltd. an Intel Company
###############################################################################

"""
Adapted from the following source:
https://huggingface.co/THUDM/glm-4v-9b/blob/main/visual.py
"""

import math
from argparse import Namespace

import habana_frameworks.torch.hpu as ht
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm
from transformers.activations import ACT2FN


try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Cannot import Fused SDPA from Habana Torch")
    FusedSDPA = None


def standard_attention(query_layer, key_layer, value_layer, scaling_attention_score=True):
    if scaling_attention_score:
        query_layer = query_layer / math.sqrt(query_layer.shape[-1])
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    attention_probs = F.softmax(attention_scores, dim=-1)

    context_layer = torch.matmul(attention_probs, value_layer)
    return context_layer


def attention_fn_default(query_layer, key_layer, value_layer, scaling_attention_score=True):
    if scaling_attention_score and FusedSDPA is not None:
        with ht.sdp_kernel(enable_recompute=True):
            # softmax_mode = "None"
            attn_output = FusedSDPA.apply(query_layer, key_layer, value_layer, None, 0.0, False, None, "None")
        return attn_output
    else:
        return standard_attention(query_layer, key_layer, value_layer, scaling_attention_score=scaling_attention_score)


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.cls_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: tensor(B, C, H, W) -> tensor(B, L, D)
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.position_embedding.weight.unsqueeze(0)
        return x


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = head_dim**-0.5
        self.query_key_value = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 3, B, H, L, D
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = attention_fn_default(q, k, v)
        output = self.dense(out.transpose(1, 2).reshape(B, L, -1))
        output = self.output_dropout(output)
        return output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Attention(config)
        self.mlp = MLP(config)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        attention_input = hidden_states
        attention_output = self.input_layernorm(self.attention(attention_input))
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states

        # https://github.com/THUDM/GLM-4/issues/350
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input)).to(mlp_input.device)
        output = mlp_input + mlp_output
        return output


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class GLU(nn.Module):
    def __init__(self, config, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, config.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.dense_4h_to_h = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x


class EVA2CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = PatchEmbedding(vision_config)
        self.transformer = Transformer(vision_config)
        self.linear_proj = GLU(config, in_features=config.hidden_size)
        self.conv = nn.Conv2d(
            in_channels=vision_config.hidden_size, out_channels=config.hidden_size, kernel_size=2, stride=2
        )
        self.boi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.scaling_factor = vision_config.scaling_factor

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(images)
        x = self.transformer(x)
        x = x[:, 1:]

        b, s, h = x.shape
        grid_size = int(s**0.5)
        x = x.view(b, grid_size, grid_size, h).permute(0, 3, 1, 2)
        x = self.conv(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.linear_proj(x)

        # https://github.com/THUDM/GLM-4/issues/350
        boi = self.boi.expand(x.shape[0], -1, -1).to(x.device)
        eoi = self.eoi.expand(x.shape[0], -1, -1).to(x.device)
        x = torch.cat((boi, x, eoi), dim=1)
        x = x / self.scaling_factor
        return x
