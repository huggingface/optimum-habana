# coding=utf-8
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

###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modeling_exaone

from .modeling_gaudi_exaone import (
    GaudiExaoneForCausalLM,
    GaudiExaoneSdpaAttention,
    gaudi_exaone_model_forward,
    gaudi_exaone_block_forward,
    gaudi_exaone_rmsnorm_forward,
)

def adapt_exaone_to_gaudi(logger):
    """
    Replaces some Exaone' methods for equivalent methods optimized
    for Gaudi.
    """
    logger.info("`optimum_habana` is set, Optimization for exaone generation on Gaudi")
    # Optimization for exaone generation on Gaudi
    modeling_exaone.ExaoneForCausalLM.forward = GaudiExaoneForCausalLM.forward
    modeling_exaone.ExaoneForCausalLM.prepare_inputs_for_generation = GaudiExaoneForCausalLM.prepare_inputs_for_generation
    modeling_exaone.ExaoneSdpaAttention = GaudiExaoneSdpaAttention
    modeling_exaone.ExaoneBlock.forward = gaudi_exaone_block_forward
    modeling_exaone.ExaoneModel.forward = gaudi_exaone_model_forward
    modeling_exaone.ExaoneRMSNorm.forward = gaudi_exaone_rmsnorm_forward

