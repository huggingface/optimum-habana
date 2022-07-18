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

import torch

from transformers import PreTrainedModel


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
