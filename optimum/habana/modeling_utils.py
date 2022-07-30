#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import copy
from typing import Tuple

import torch

from habana_frameworks.torch.hpex import hmp
from transformers import PreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin


PRETRAINED_TO_GAUDI_REGISTRY = {}


def register(transformers_cls):
    """
    Class wrapper to register models that require accelerated generation.
    """

    def wrapper(cls):
        if transformers_cls is None:
            raise ValueError(f"None cannot be registered, please provide a valid Transformers model class.")
        PRETRAINED_TO_GAUDI_REGISTRY[transformers_cls] = cls
        return cls

    return wrapper


def to_gaudi_for_accelerated_generation(model: PreTrainedModel) -> PreTrainedModel:
    """
    Convert a model so that its generate method is executed efficiently
    in lazy mode.

    Args:
        model (PreTrainedModel): model to convert.

    Raises:
        KeyError: when the class of the given model is not registered.

    Returns:
        PreTrainedModel: converted model.
    """

    model_cls = model.__class__
    gaudi_cls = PRETRAINED_TO_GAUDI_REGISTRY.get(model_cls, None)
    if gaudi_cls is not None:
        return gaudi_cls.from_transformers(model)
    else:
        raise KeyError(f"{model_cls.__name__} Gaudi version not found in registry: {PRETRAINED_TO_GAUDI_REGISTRY}")


class GaudiMixin:
    @classmethod
    def from_transformers(cls, model: PreTrainedModel) -> PreTrainedModel:
        """
        Convert a model so that its generate method is executed efficiently
        in lazy mode.

        Args:
            model (PreTrainedModel): model to convert.

        Returns:
            PreTrainedModel: converted model.
        """

        config = copy.deepcopy(model.config)
        gaudi_model = cls(config)
        gaudi_model.load_state_dict(model.state_dict())

        return gaudi_model


def gaudi_invert_attention_mask(self, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Invert an attention mask (e.g., switches 0. and 1.).
    Args:
        encoder_attention_mask (`torch.Tensor`): An attention mask.
    Returns:
        `torch.Tensor`: The inverted attention mask.
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
    # HMP is disabled because (1.0 - encoder_extended_attention_mask) may be converted into bf16
    # while torch.finfo(self.dtype).min is the min value of fp32, which leads to NaNs
    with hmp.disable_casts():
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

    return encoder_extended_attention_mask


def gaudi_get_extended_attention_mask(
    self, attention_mask: torch.Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None
) -> torch.Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.
    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
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
    # HMP is disabled because (1.0 - encoder_extended_attention_mask) may be converted into bf16
    # while torch.finfo(self.dtype).min is the min value of fp32, which leads to NaNs
    with hmp.disable_casts():
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min

    return extended_attention_mask
