
from typing import Any, Optional, Dict

import torch
from torch import nn


from transformers.utils import (
    logging,
)

from transformers.models.mamba.modeling_mamba import (
    MambaCache,
)
from transformers.utils import ModelOutput
logger = logging.get_logger(__name__)
import habana_frameworks.torch.core as htcore 


# from https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/mamba/modeling_mamba.py#L240
#  changes: index_select replace and mark_step
# fmt: off
def gaudi_MambaMixer_slow_forward(self, input_states, cache_params: Optional[MambaCache]=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediade_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []

        for i in range(seq_len):

            index = torch.tensor(i, device=discrete_A.device)
            #ssm_state = torch.select(discrete_A, 2, index) * ssm_state + torch.select(discrete_A, 2, index)
            #scan_output = torch.matmul(ssm_state.to(dtype),  torch.select(C, 1, index).unsqueeze(-1))
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]

            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)                                # [batch, seq_len, intermediade_size]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2))             # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on


def gaudi_MambaForCausalLM_prepare_inputs_for_generation(
        self, input_ids, cache_params: Optional[MambaCache] = None, inputs_embeds=None, attention_mask=None, **kwargs
    ):
        token_idx = kwargs.get("token_idx", None)
        token_idx_cpu = kwargs.get("token_idx_cpu", None)
        if cache_params is not None:
            if token_idx is None:
                input_ids = input_ids[:, -1].unsqueeze(-1)
            else:
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)
        else:
            if token_idx is not None:
                input_ids = torch.index_select(input_ids, 1, torch.arange(token_idx_cpu, device=input_ids.device))  
        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs["cache_params"] = cache_params
        return model_inputs

def gaudi_MambaForCausalLM_update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        token_idx = model_kwargs.get("token_idx", None)
        if token_idx is not None:
            token_idx.add_(1)
            if "token_idx_cpu" in model_kwargs:
                model_kwargs["token_idx_cpu"] += 1
        return model_kwargs
