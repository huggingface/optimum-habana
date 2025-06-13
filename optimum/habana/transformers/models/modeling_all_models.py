# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import warnings
from typing import Tuple

import torch
from transformers.modeling_utils import ModuleUtilsMixin, PretrainedConfig
from transformers.utils.import_utils import is_torch_sdpa_available


try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE
except ImportError:
    print("Not using HPU fused kernel for apply_rotary_pos_emb")
    FusedRoPE = None


class Matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.matmul(*args, **kwargs)


class KVCache(torch.nn.Module):
    def __init__(self):
        super(KVCache, self).__init__()
        self.cache = None
        self.inp_seq_len = -1

    def allocate(self, inp_seq_len, dtype, device, shape):
        if self.cache is None or self.cache.shape != shape:
            self.inp_seq_len = inp_seq_len
            self.cache = torch.zeros(shape, dtype=dtype, device=device)
        else:
            assert self.inp_seq_len == inp_seq_len, (
                f"inp_seq_len must be the same. self.inp_seq_len:{self.inp_seq_len} inp_seq_len:{inp_seq_len}"
            )
            self.cache.fill_(0)

    @staticmethod
    def update(prev, cur, dim, idx, inp_seq_len):
        orig_cur = cur
        if prev.shape == cur.shape:
            prev.copy_(cur)
            return orig_cur
        if cur.shape[2] > 1 and cur.shape[2] <= prev.shape[2]:
            # Initialize
            prev[:, :, :inp_seq_len, :].copy_(cur)
            return orig_cur
        assert cur.shape[2] == 1, f"Cannot update kv-cache. Unsupported shapes. prev:{prev.shape} cur:{cur.shape}"
        if idx is not None:
            prev.index_copy_(dim, idx - 1, cur)
            return prev
        else:
            return torch.cat((prev, cur), dim=dim)

    def get_shape(self):
        if self.cache is None:
            return None
        return self.cache.shape

    def forward(self, cur, dim, idx):
        return self.update(self.cache, cur, dim, idx, self.inp_seq_len)


def apply_customized_rope_module(q, k, cos, sin, position_ids, training=True):
    if training:
        rope_q = FusedRoPE.apply(q, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0), position_ids)
        rope_k = FusedRoPE.apply(k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0), position_ids)

    else:
        if q.dtype == torch.bfloat16:
            rope_q = FusedRoPE.apply(
                q,
                cos.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                sin.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                position_ids,
            )
        else:
            rope_q = FusedRoPE.apply(q, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0), position_ids)
        if k.dtype == torch.bfloat16:
            rope_k = FusedRoPE.apply(
                k,
                cos.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                sin.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                position_ids,
            )
        else:
            rope_k = FusedRoPE.apply(k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0), position_ids)
    return rope_q, rope_k


def gaudi_invert_attention_mask(self, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Same as https://github.com/huggingface/transformers/blob/a9eee2ffecc874df7dd635b2c6abb246fdb318cc/src/transformers/modeling_utils.py#L640
    except that mixed precision is disabled for computing:
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min
    """
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    # torch.finfo must take the dtype of encoder_extended_attention_mask
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # bf16 compatibility
    encoder_extended_attention_mask = 1.0 - encoder_extended_attention_mask
    #  Fixes issue where the model is not in bf16 and mul is casting it to values out of range resulting in nan
    with torch.autocast(enabled=False, device_type="hpu"):
        encoder_extended_attention_mask = (
            encoder_extended_attention_mask * torch.finfo(encoder_extended_attention_mask.dtype).min
        )

    return encoder_extended_attention_mask


def gaudi_get_extended_attention_mask(
    self, attention_mask: torch.Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None
) -> torch.Tensor:
    """
    Same as https://github.com/huggingface/transformers/blob/a9eee2ffecc874df7dd635b2c6abb246fdb318cc/src/transformers/modeling_utils.py#L692
    except that mixed precision is disabled for computing:
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    """
    if dtype is None:
        dtype = self.dtype

    if not (attention_mask.dim() == 2 and self.config.is_decoder):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder:
            extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                input_shape, attention_mask, device
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    # torch.finfo must take the dtype of encoder_extended_attention_mask
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # bf16 compatibility
    extended_attention_mask = 1.0 - extended_attention_mask
    with torch.autocast(enabled=False, device_type="hpu"):
        extended_attention_mask = extended_attention_mask * torch.finfo(extended_attention_mask.dtype).min

    return extended_attention_mask


def gaudi_conv1d_forward(self, x):
    """
    Same as https://github.com/huggingface/transformers/blob/3335724376319a0c453049d0cd883504f530ff52/src/transformers/pytorch_utils.py#L100
    but moves reshape before view for tpc auto fusion.
    """
    size_out = x.size()[:-1] + (self.nf,)
    x = torch.mm(x.view(-1, x.size(-1)), self.weight)
    x = x.view(size_out)
    x = x + self.bias
    return x


# Adapted from transformers.modeling_utils.PreTrainedModel._check_and_enable_sdpa
@classmethod
def gaudi_check_and_enable_sdpa(cls, config, hard_check_only: bool = False) -> PretrainedConfig:
    # This model doesn't support SDPA in Gaudi yet, fallback to original code.
    MODELS_ATTN_IMPLEMENTATION_EAGER = ["albert", "bart", "gpt_bigcode", "mistral", "mixtral", "roberta"]

    if config.model_type in MODELS_ATTN_IMPLEMENTATION_EAGER:
        config._attn_implementation = "eager"
        return config

    # Otherwise, fallback to original implementation
    # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/modeling_utils.py#L1542
    if hard_check_only:
        if not cls._supports_sdpa:
            raise ValueError(
                f"{cls.__name__} does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet."
                " Please request the support for this architecture: https://github.com/huggingface/transformers/issues/28005. If you believe"
                ' this error is a bug, please open an issue in Transformers GitHub repository and load your model with the argument `attn_implementation="eager"` meanwhile. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="eager")`'
            )
        if not is_torch_sdpa_available():
            raise ImportError("PyTorch SDPA requirements in Transformers are not met. Please install torch>=2.1.1.")

    if not is_torch_sdpa_available() or not cls._supports_sdpa:
        return config

    _is_bettertransformer = getattr(cls, "use_bettertransformer", False)
    if _is_bettertransformer:
        return config

    if not hard_check_only:
        config._attn_implementation = "sdpa"

    return config


# Splitting DeepSpeed LinearAllReduce to three parts to avoid redundant memory consumption
class ScopedLinearAllReduce(torch.nn.Module):
    def __init__(self, mod, *args, **kwargs):
        self.__dict__.update(mod.__dict__)

    def forward(self, input):
        # pre_all_reduce

        output = torch.matmul(input, self.weight.transpose(-1, -2))
        return output

    def all_reduce(self, input):
        if self.mp_group is not None:
            from deepspeed import comm as dist

            dist.inference_all_reduce(input, group=self.mp_group)

    def post_all_reduce(self, input):
        output = input + self.bias if (self.bias is not None) else input
        return output
