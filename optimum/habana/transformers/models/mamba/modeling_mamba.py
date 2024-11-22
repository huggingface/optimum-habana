from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from transformers.cache_utils import MambaCache
from transformers.configuration_utils import PretrainedConfig
from transformers.models.mamba.configuration_mamba import MambaConfig
from transformers.activations import ACT2FN
from transformers.utils import (
    ModelOutput,
    logging,
)

from pathlib import Path
import os
base_dir = "/workspace/custom_op_pscan_all" 

custom_op_lib_path = str(
    next(
        Path(
            next(Path(os.path.join(base_dir, "build")).glob("lib.linux-x86_64-*"))
        ).glob("hpu_custom_pscan_all.cpython-*-x86_64-linux-gnu.so")
    )
)    
torch.ops.load_library(custom_op_lib_path)

logger = logging.get_logger(__name__)
is_fast_path_available = False

use_pscan_kernel = True
def Run_Mamba_Forward_Gaudi(in_state, in_x, in_dt, in_A, in_B, in_C, in_D, in_z):

    in_state_h = in_state.unsqueeze(1).transpose(2,3)
    in_x_h = in_x.transpose(1,2).unsqueeze(2)
    in_dt_h = in_dt.unsqueeze(2)
    in_A_h = in_A.unsqueeze(0).unsqueeze(1).transpose(2,3)  
    in_B_h = in_B.unsqueeze(3)
    in_C_h = in_C.unsqueeze(3)
    in_D_h = in_D.unsqueeze(0).unsqueeze(1).unsqueeze(2)
    in_z_h = in_z.transpose(1,2).unsqueeze(2)
    

    if in_state.dtype == torch.float:
        state_out_h = torch.ops.custom_op.custom_pscan(in_state_h, in_x_h, in_dt_h, in_A_h, in_B_h)
        output_h = torch.ops.custom_op.custom_pscan_update(state_out_h, in_x_h, in_C_h, in_D_h, in_z_h)

    else:
        in_A_h = in_A_h.to(torch.bfloat16)
        state_out_h = torch.ops.custom_op.custom_pscan_bf16(in_state_h, in_x_h, in_dt_h, in_A_h, in_B_h)
        output_h = torch.ops.custom_op.custom_pscan_update_bf16(state_out_h, in_x_h, in_C_h, in_D_h, in_z_h)


    output_hpu = output_h.squeeze(2).transpose(1,2)
    state_hpu = state_out_h.transpose(2,3)
    state_out = torch.select(state_hpu,1,output_hpu.shape[2]-1)

    return output_hpu, state_out


def gaudi_MambaCache_update_conv_state(
    self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
) -> torch.Tensor:
    conv_state = self.conv_states[layer_idx]
    cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

    conv_state = conv_state.roll(shifts=-1, dims=-1)
    #conv_state[:, :, cache_position] = new_conv_state.to(conv_state.device)
    for c, i in enumerate(cache_position):
        conv_state[:, :, i] = new_conv_state[:,:,c].to(conv_state.device)

    self.conv_states[layer_idx].zero_()
    self.conv_states[layer_idx] += conv_state
    return self.conv_states[layer_idx]


def gaudi_MambaForCausalLM_update_model_kwargs_for_generation(
    self, outputs: ModelOutput, model_kwargs: Dict[str, Any], num_new_tokens: int = 1, **kwargs
) -> Dict[str, Any]:
    model_kwargs["cache_params"] = outputs.get("cache_params", None)
    token_idx = model_kwargs.get("token_idx", None)
    if (
        model_kwargs.get("use_cache", True)
        and "cache_position" in model_kwargs
        and model_kwargs["cache_position"] is not None
    ):
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens

    if "attention_mask" in model_kwargs:
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
        )

    if token_idx is not None:
        token_idx.add_(1)
        if "token_idx_cpu" in model_kwargs:
            model_kwargs["token_idx_cpu"] += 1

    return model_kwargs


def gaudi_MambaForCausalLM_prepare_inputs_for_generation(
    self,
    input_ids,
    inputs_embeds=None,
    use_cache=None,
    cache_params: Optional[MambaCache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    **kwargs,
):
    token_idx = kwargs.get("token_idx", None)
    token_idx_cpu = kwargs.get("token_idx_cpu", None)
    if use_cache:
        if token_idx is None:
            # `cache_position` should have been initialized in `generate`
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )
            if cache_position[0] > 0:
                input_ids = input_ids[:, -1].unsqueeze(-1)

                if attention_mask is not None:
                    attention_mask = None

            else:
                # we initialize the `cache_position` to full size of `conv_states` at prefill stage
                # considering padding will be applied when input length is shorter, and truncation
                # will be applied when it is longer, so it will be equivalent to always have it match
                # the length of `cache_params.conv_states`, which is `config.conv_kernel`
                cache_position = torch.arange(0, self.config.conv_kernel, device=input_ids.device)
        else:
            idx = token_idx + kwargs.get("inputs_embeds_offset", 0) - 1
            input_ids = torch.index_select(input_ids, 1, idx)
            if attention_mask is not None:
                attention_mask = None
    else:
        if token_idx is not None:
            input_ids = torch.index_select(input_ids, 1, torch.arange(token_idx_cpu, device=input_ids.device))
    if inputs_embeds is not None and cache_params is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids.contiguous()}

    model_inputs.update(
        {
            "cache_params": cache_params,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

class gaudi_MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: MambaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.use_mambapy = config.use_mambapy

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

        if not is_fast_path_available:
            if self.use_mambapy:
                if is_mambapy_available():
                    logger.warning_once(
                        "The fast path is not available because one of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                        " is None. Falling back to the mamba.py backend. To install follow https://github.com/state-spaces/mamba/#installation and"
                        " https://github.com/Dao-AILab/causal-conv1d"
                    )
                else:
                    raise ImportError(
                        "use_mambapy is set to True but the mambapy package is not installed. To install it follow https://github.com/alxndrTL/mamba.py."
                    )
            else:
                logger.warning_once(
                    "The fast path is not available because one of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                    " is None. Falling back to the sequential implementation of Mamba, as use_mambapy is set to False. To install follow https://github.com/state-spaces/mamba/#installation and"
                    " https://github.com/Dao-AILab/causal-conv1d. For the mamba.py backend, follow https://github.com/alxndrTL/mamba.py."
                )

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[MambaCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        if self.training and cache_params is None:  # Doesn't support outputting the states -> used for training
            contextualized_states = mamba_inner_fn(
                projected_states,
                self.conv1d.weight,
                self.conv1d.bias if self.use_conv_bias else None,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias.float() if self.use_bias else None,
                -torch.exp(self.A_log.float()),
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )

        else:
            hidden_states, gate = projected_states.chunk(2, dim=1)

            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(1)

            # 2. Convolution sequence transformation
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            if cache_params is not None and cache_position[0] > 0:
                hidden_states = causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    cache_params.conv_states[self.layer_idx],
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                )
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(
                        hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                    )
                    cache_params.update_conv_state(self.layer_idx, conv_states, cache_position)
                hidden_states = causal_conv1d_fn(
                    hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
                )

            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(1)

            # 3. State Space Model sequence transformation
            # 3.a. input varying initialization of time_step, B and C
            ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
            time_step, B, C = torch.split(
                ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
            )
            discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)

            A = -torch.exp(self.A_log.float())
            # 3.c perform the recurrence y ← SSM(A, B, C)(x)
            time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
            if cache_params is not None and cache_position[0] > 0:
                scan_outputs = selective_state_update(
                    cache_params.ssm_states[self.layer_idx],
                    hidden_states[..., 0],
                    discrete_time_step[..., 0],
                    A,
                    B[:, 0],
                    C[:, 0],
                    self.D,
                    gate[..., 0],
                    time_proj_bias,
                    dt_softplus=True,
                ).unsqueeze(-1)
            else:
                scan_outputs, ssm_state = selective_scan_fn(
                    hidden_states,
                    discrete_time_step,
                    A,
                    B.transpose(1, 2),
                    C.transpose(1, 2),
                    self.D.float(),
                    gate,
                    time_proj_bias,
                    delta_softplus=True,
                    return_last_state=True,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.update_ssm_state(self.layer_idx, ssm_state)

            # 4. Final linear projection
            contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states

    # fmt: off
    def slow_forward(self, input_states, cache_params: Optional[MambaCache]=None, cache_position:Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor] = None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            ssm_state = ssm_state.to(hidden_states.device)
            # use `cache_position.shape[0]` to check whether we are in prefill
            # stage, it's equivalent to check `cache_position[0] == 0`, which
            # breaks dynamo fullgraph constraints
            if cache_position.shape[0] == self.conv_kernel_size:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )

                cache_params.update_conv_state(self.layer_idx, conv_state, cache_position)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
            else:
                conv_state = cache_params.update_conv_state(self.layer_idx, hidden_states, cache_position)
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
        
        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
        if use_pscan_kernel:
            scan_output, ssm_state = Run_Mamba_Forward_Gaudi(
                    ssm_state, 
                    hidden_states, 
                    discrete_time_step,
                    A, 
                    B, 
                    C, 
                    self.D,
                    gate
                    )
        else:
            discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]
            discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
            discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediate_size, seq_len, ssm_state_size]
            deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

            # 3.c perform the recurrence y ← SSM(A, B, C)(x)
            if self.use_mambapy and self.training and cache_params is None:
                hs = pscan(discrete_A.transpose(1, 2), deltaB_u.transpose(1, 2)) # [batch, seq_len, intermediate_size, ssm_state_size]

                scan_output = (hs @ C.unsqueeze(-1)).squeeze(3).transpose(1, 2) # [batch, intermediate_size, seq_len]
                scan_output = scan_output + hidden_states * self.D[None, :, None]
                scan_output = scan_output * self.act(gate)
            else:
                scan_outputs = []
                for i in range(seq_len):
                    ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
                    scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
                    scan_outputs.append(scan_output[:, :, 0])
                scan_output = torch.stack(scan_outputs, dim=-1)                                # [batch, seq_len, intermediade_size]
                scan_output = scan_output + (hidden_states * self.D[None, :, None])
                scan_output = (scan_output * self.act(gate))

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2))  # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(
        self,
        hidden_states,
        cache_params: Optional[MambaCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        if is_fast_path_available and "cuda" in self.x_proj.weight.device.type and not torch._dynamo.is_compiling():
            return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)
        return self.slow_forward(hidden_states, cache_params, cache_position, attention_mask)
