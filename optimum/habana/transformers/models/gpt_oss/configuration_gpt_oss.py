# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""openai model configuration"""

#from transformers.configuration_utils import PretrainedConfig#, layer_type_validation not in 4.51.0
#from transformers.modeling_rope_utils import rope_config_validation
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

class GptOssConfig(GptOssConfig):
    def __init__(
        self,
        num_hidden_layers: int = 36,
        num_local_experts: int = 128,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        intermediate_size: int = 2880,
        head_dim: int = 64,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        sliding_window: int = 128,
        rope_theta: float = 150000.0,
        tie_word_embeddings=False,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        max_position_embeddings=131072,
        rms_norm_eps: float = 1e-5,
        rope_scaling={"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False},
        attention_dropout: float = 0.0,
        num_experts_per_tok=4,
        router_aux_loss_coef: float = 0.9,
        output_router_logits=False,
        use_cache=True,
        layer_types=None,
        **kwargs,
    ):
        super().__init__(
            num_hidden_layers,
            num_local_experts,
            vocab_size,
            hidden_size,
            intermediate_size,
            head_dim,
            num_attention_heads,
            num_key_value_heads,
            sliding_window,
            rope_theta,
            tie_word_embeddings,
            hidden_act,
            initializer_range,
            max_position_embeddings,
            rms_norm_eps,
            rope_scaling,
            attention_dropout,
            num_experts_per_tok,
            router_aux_loss_coef,
            output_router_logits,
            use_cache,
            layer_types,   
            **kwargs,
        )
