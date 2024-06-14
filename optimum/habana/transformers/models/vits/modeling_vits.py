import numpy as np
import torch
from torch import nn
from transformers.models.vits.modeling_vits import _rational_quadratic_spline
from transformers.utils import logging


logger = logging.get_logger(__name__)


def gaudi_unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    reverse=False,
    tail_bound=5.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    """
    Copied from _unconstrained_rational_quadratic_spline: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/vits/modeling_vits.py#L126
    The only differences are:
    - WA to fix hpu graph accuracy issue
    """
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    log_abs_det = torch.zeros_like(inputs)
    constant = np.log(np.exp(1 - min_derivative) - 1)

    unnormalized_derivatives = nn.functional.pad(unnormalized_derivatives, pad=(1, 1))
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    log_abs_det[outside_interval_mask] = 0.0

    outputs_i, log_abs_det_i = _rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        reverse=reverse,
        tail_bound=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    outputs = outputs_i * inside_interval_mask + outputs * outside_interval_mask
    log_abs_det = log_abs_det_i * inside_interval_mask + log_abs_det * outside_interval_mask
    return outputs, log_abs_det


def gaudi_VitsResidualCouplingLayer_forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
    """
    Copied from VitsResidualCouplingLayer:forward: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/vits/modeling_vits.py
    The only differences are:
    - WA to fix torch.flip issue after conv1d
    """
    first_half, second_half = torch.split(inputs, [self.half_channels] * 2, dim=1)
    hidden_states = self.conv_pre(first_half) * padding_mask
    hidden_states = self.wavenet(hidden_states, padding_mask, global_conditioning)
    mean = self.conv_post(hidden_states) * padding_mask
    log_stddev = torch.zeros_like(mean)

    if not reverse:
        second_half = mean.cpu() + second_half * torch.exp(log_stddev) * padding_mask
        outputs = torch.cat([first_half, second_half], dim=1)
        log_determinant = torch.sum(log_stddev, [1, 2])
        return outputs, log_determinant
    else:
        second_half = (second_half - mean.cpu()) * torch.exp(-log_stddev) * padding_mask
        outputs = torch.cat([first_half, second_half], dim=1)
        return outputs, None
