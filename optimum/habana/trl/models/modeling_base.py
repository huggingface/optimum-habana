# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import os

import torch
from trl import PreTrainedModelWrapper

from optimum.habana.utils import to_device_dtype


def adapt_PreTrainedModelWrapper_to_gaudi():
    PreTrainedModelWrapper._get_current_device = gaudi_get_current_device
    PreTrainedModelWrapper.save_pretrained = gaudi_save_pretrained


def gaudi_get_current_device():
    """
    Copied from PreTrainedModelWrapper._get_current_device: https://github.com/huggingface/trl/blob/v0.7.6/trl/models/modeling_base.py#L392
    - add hpu device
    """
    if hasattr(torch, "hpu") and torch.hpu.is_available():
        return "hpu"
    else:
        return "cpu"


def gaudi_save_pretrained(self, *args, **kwargs):
    """
    Copied from PreTrainedModelWrapper.save_pretrained: https://github.com/huggingface/trl/blob/v0.7.6/trl/models/modeling_base.py#L528
    - to cpu if model dict is in hpu
    """
    state_dict = kwargs.get("state_dict")
    if state_dict is None:
        state_dict = self.state_dict()
        kwargs["state_dict"] = state_dict

    if self.__class__._get_current_device() == "hpu":
        state_dict = to_device_dtype(state_dict, target_device=torch.device("cpu"))

    # if it is a peft model only save the `v_head` state_dict and
    # pop the `state_dict` from the kwargs to avoid slient bugs with `peft`
    if self.is_peft_model:
        save_path = args[0]
        save_path = os.path.join(save_path, "pytorch_model.bin")
        torch.save(state_dict, save_path)
        _ = kwargs.pop("state_dict", None)

    if self.__class__._get_current_device() == "hpu":
        state_dict = self.pretrained_model.state_dict()
        state_dict = to_device_dtype(state_dict, target_device=torch.device("cpu"))
        kwargs["state_dict"] = state_dict

    return self.pretrained_model.save_pretrained(*args, **kwargs)
