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

import os
import random
from typing import Optional, Tuple, Union

import torch
from habana_frameworks.torch.hpu import get_device_name
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    Wav2Vec2BaseModelOutput,
)
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import _HIDDEN_STATES_START_POSITION, Wav2Vec2Attention, logger


try:
    from habana_frameworks.torch.hpex.kernels import CTCLoss

    custom_ctc_loss_fwd = CTCLoss.apply
except ImportError:
    print("Could not import Custom CTCLoss kernel. This Kernel is available only for SynapseAI >= 1.15.0")
    custom_ctc_loss_fwd = None

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None


def _gaudi_wav2vec2_compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> torch.Tensor:
    """
    Copied from Transformers: https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L135
    The only differences are (1) that the processing is performed with PyTorch on HPUs (Numpy is used in Transformers), (2) epsilon is generated on HPU instead of CPU, (3) check
    to ensure indices are not larger than sequence length is re-written to avoid host sync.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = torch.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.detach().sum(-1).tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device="hpu")
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = torch.tensor(
            random.sample(range(input_length - (mask_length - 1)), num_masked_span), dtype=torch.int32
        )
        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = torch.cat(
            [
                spec_aug_mask_idx,
                torch.ones(max_num_masked_span - num_masked_span, dtype=torch.int32) * dummy_mask_idx,
            ]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx.to(dtype=torch.long))

    spec_aug_mask_idxs = torch.vstack(spec_aug_mask_idxs).to("hpu")
    # expand masked indices to masked spans
    spec_aug_mask_idxs = torch.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = torch.arange(mask_length, device="hpu")[None, None, :]
    offsets = torch.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if get_device_name() == "GAUDI" or custom_ctc_loss_fwd is None:
        if spec_aug_mask_idxs.max() > sequence_length - 1:
            spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1
    else:
        mask = (spec_aug_mask_idxs > sequence_length - 1) * (spec_aug_mask_idxs.max() > sequence_length - 1)
        inverse_mask = torch.bitwise_not(mask)
        spec_aug_mask_idxs = spec_aug_mask_idxs * inverse_mask + (sequence_length - 1) * mask

    # scatter indices to mask
    spec_aug_mask.scatter_(-1, spec_aug_mask_idxs, 1)

    return spec_aug_mask


def _gaudi_wav2vec2_sample_negative_indices(
    features_shape: Tuple, num_negatives: int, mask_time_indices: Optional[torch.Tensor] = None
):
    """
    Copied from Transformers: https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L254
    The only difference is that the processing is performed with PyTorch on HPUs (Numpy is used in Transformers).
    """
    batch_size, sequence_length = features_shape

    # generate indices of the positive vectors themselves, repeat them `num_negatives` times
    sequence_length_range = torch.arange(sequence_length, device="hpu")

    # get `num_negatives` random vector indices from the same utterance
    sampled_negative_indices = torch.zeros(
        shape=(batch_size, sequence_length, num_negatives), dtype=torch.int32, device="hpu"
    )

    mask_time_indices = (
        mask_time_indices.bool()
        if mask_time_indices is not None
        else torch.ones(features_shape, dtype=torch.bool, device="hpu")
    )

    for batch_idx in range(batch_size):
        high = mask_time_indices[batch_idx].sum() - 1
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

        feature_indices = torch.broadcast_to(torch.arange(high + 1, device="hpu")[:, None], (high + 1, num_negatives))
        sampled_indices = torch.randint(0, high, size=(high + 1, num_negatives), dtype=torch.int16, device="hpu")
        # avoid sampling the same positive vector, but keep the distribution uniform
        sampled_indices[sampled_indices >= feature_indices] += 1

        # remap to actual indices
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]

        # correct for batch size
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices


def gaudi_wav2vec2_encoder_forward(
    self,
    hidden_states: torch.tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    return_dict: bool = True,
):
    """
    Copied from Transformers: https://github.com/huggingface/transformers/blob/7790943c91411f4234d11dfbf4c2f21ce7caf088/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L755
    The only difference is that torch.rand device is set to 'hpu' (required to capture operation as part of HPU graph)
    """
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None

    if attention_mask is not None:
        # make sure padded tokens output 0
        expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
        hidden_states[~expand_attention_mask] = 0

        # extend attention_mask
        attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
        attention_mask = attention_mask.expand(
            attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
        )

    position_embeddings = self.pos_conv_embed(hidden_states)
    hidden_states = hidden_states + position_embeddings
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

    for layer in self.layers:
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = torch.rand([])

        skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
        if not skip_the_layer or synced_gpus:
            # under fsdp or deepspeed zero3 all gpus must run in sync
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                )
            hidden_states = layer_outputs[0]

        if skip_the_layer:
            layer_outputs = (None, None)

        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def gaudi_wav2vec2_forward(
    self,
    input_values: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    mask_time_indices: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
    """
    Copied from Transformers: https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1282
    The only difference is that a clone of `hidden_states` is given to _mask_hidden_states to avoid an error.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    extract_features = self.feature_extractor(input_values)
    extract_features = extract_features.transpose(1, 2)

    if attention_mask is not None:
        # compute reduced attention_mask corresponding to feature vectors
        attention_mask = self._get_feature_vector_attention_mask(
            extract_features.shape[1], attention_mask, add_adapter=False
        )

    hidden_states, extract_features = self.feature_projection(extract_features)
    hidden_states = self._mask_hidden_states(
        hidden_states.clone(), mask_time_indices=mask_time_indices, attention_mask=attention_mask
    )

    encoder_outputs = self.encoder(
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = encoder_outputs[0]

    if self.adapter is not None:
        hidden_states = self.adapter(hidden_states)

    if not return_dict:
        return (hidden_states, extract_features) + encoder_outputs[1:]

    return Wav2Vec2BaseModelOutput(
        last_hidden_state=hidden_states,
        extract_features=extract_features,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def _gaudi_wav2vec2_mask_hidden_states(
    self,
    hidden_states: torch.FloatTensor,
    mask_time_indices: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
):
    """
    Copied from Transformers: https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1227
    Differences are that (1) `mask_time_indices` is not moved to the current device and converted into boolean because this is already done in _compute_mask_indices.
    (2) index_put operation on hidden_states is replaced by combination of simpler ops (more suitable for HPU graphs)
    """

    # `config.apply_spec_augment` can set masking to False
    if not getattr(self.config, "apply_spec_augment", True):
        return hidden_states

    # generate indices & apply SpecAugment along time axis
    batch_size, sequence_length, hidden_size = hidden_states.size()

    if mask_time_indices is not None:
        # apply SpecAugment along time axis with given mask_time_indices
        hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
    elif self.config.mask_time_prob > 0 and self.training:
        mask_time_indices = _gaudi_wav2vec2_compute_mask_indices(
            (batch_size, sequence_length),
            mask_prob=self.config.mask_time_prob,
            mask_length=self.config.mask_time_length,
            attention_mask=attention_mask,
            min_masks=self.config.mask_time_min_masks,
        )
        # replacement of index_put with combination of simpler ops. Assumption made about sizes of hidden_states (3d),
        # mask_time_indices (2d), self.masked_spec_embed (1d), for any other combination better to go back to original code using index_put.
        # hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        inverse_mask_time_indices = torch.bitwise_not(mask_time_indices)
        hidden_states = hidden_states * inverse_mask_time_indices.unsqueeze(2) + self.masked_spec_embed.to(
            hidden_states.dtype
        ).expand(hidden_states.size()) * mask_time_indices.unsqueeze(2)

    if self.config.mask_feature_prob > 0 and self.training:
        # generate indices & apply SpecAugment along feature axis
        mask_feature_indices = _gaudi_wav2vec2_compute_mask_indices(
            (batch_size, hidden_size),
            mask_prob=self.config.mask_feature_prob,
            mask_length=self.config.mask_feature_length,
            min_masks=self.config.mask_feature_min_masks,
        )
        mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
        mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
        hidden_states[mask_feature_indices] = 0

    return hidden_states


def gaudi_wav2vec2forctc_forward(
    self,
    input_values: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    labels: Optional[torch.Tensor] = None,
) -> Union[Tuple, CausalLMOutput]:
    """
    copied from Transformers https://github.com/huggingface/transformers/blob/e770f0316d2a9b787c9d1440f204fcb65e176682/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1950
    only differences are (1) attention_mask tensor generation using ones_like is done on HPU, (2) masked_select is not applied on labels to compute flattened_targets to avoid
    changing flattened_targets tensor shapes across training iterations.
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if labels is not None and labels.max() >= self.config.vocab_size:
        raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

    outputs = self.wav2vec2(
        input_values,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = outputs[0]
    hidden_states = self.dropout(hidden_states)
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        # retrieve loss input_lengths from attention_mask
        attention_mask = (
            attention_mask
            if attention_mask is not None
            else torch.ones_like(input_values, dtype=torch.long, device="hpu")
        )
        input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        # assuming that padded tokens are filled with -100
        # when not being attended to
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        # ctc_loss doesn't support fp16
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        if get_device_name() == "GAUDI" or custom_ctc_loss_fwd is None:
            flattened_targets = labels.masked_select(labels_mask)
            loss = torch.nn.functional.ctc_loss(
                log_probs,
                flattened_targets,
                input_lengths,
                target_lengths,
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )
        else:
            flattened_targets = labels
            loss = custom_ctc_loss_fwd(
                log_probs,
                flattened_targets,
                input_lengths,
                target_lengths,
                self.config.pad_token_id,
                self.config.ctc_loss_reduction,
                self.config.ctc_zero_infinity,
            )

    if not return_dict:
        output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
        return ((loss,) + output) if loss is not None else output
    return CausalLMOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


def gaudi_wav2vec2_tdnnlayer_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Copied from Transformers: https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2290
    v4.38.2 implementation caused accuracy issue to run pytest Wav2Vec2RobustModelTest.
    """
    hidden_states = hidden_states.unsqueeze(1)
    hidden_states = torch.nn.functional.unfold(
        hidden_states,
        (self.kernel_size, self.in_conv_dim),
        stride=(1, self.in_conv_dim),
        dilation=(self.dilation, 1),
    )
    hidden_states = hidden_states.transpose(1, 2)
    hidden_states = self.kernel(hidden_states)

    hidden_states = self.activation(hidden_states)
    return hidden_states


class GaudiWav2Vec2SdpaAttention(Wav2Vec2Attention):
    # Copied from transformers.models.bart.modeling_bart.BartSdpaAttention.forward with Bart->Wav2Vec2
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Wav2Vec2Config] = None,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            is_decoder,
            bias,
            is_causal,
            config,
        )
        self.use_flash_attention = True if os.getenv("USE_FLASH_ATTENTION") == "1" else False
        self.flash_attention_fast_softmax = True if os.getenv("FLASH_ATTENTION_FAST_SOFTMAX") == "1" else False
        self.flash_attention_recompute = True if os.getenv("FLASH_ATTENTION_RECOMPUTE") == "1" else False

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Copied from Wav2Vec2SdpaAttention.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py
        The only difference is If the `USE_FLASH_ATTENTION` switch is enabled, then use the HPU's fused SDPA; otherwise, use PyTorch's native SDPA
        """
        """Input shape: Batch x Time x Channel"""
        if output_attentions or layer_head_mask is not None:
            # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Wav2Vec2Model is using Wav2Vec2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True` or `layer_head_mask` not None. Falling back to the manual attention"
                ' implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        query_states = self._shape(query_states, tgt_len, bsz)

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case tgt_len == 1.
        is_causal = True if self.is_causal and attention_mask is None and tgt_len > 1 else False

        if self.use_flash_attention and FusedSDPA:
            if tgt_len == 1:
                # next token
                softmax_mode = True if os.getenv("QUANT_CONFIG", "") else False
                recompute_mode = False
            else:
                # first token
                softmax_mode = "fast" if self.flash_attention_fast_softmax else "None"
                recompute_mode = self.flash_attention_recompute

            attn_output = FusedSDPA.apply(
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.dropout if self.training else 0.0,
                is_causal,
                None,  # scale ; Non -> default scale
                softmax_mode,  # "None"  -> Non fast softmax
                recompute_mode,  # -> recompute_mode = false
            )
        else:
            # NOTE: SDPA with memory-efficient backend is currently (torch==2.1.2) bugged when using non-contiguous inputs and a custom attn_mask,
            # but we are fine here as `_shape` do call `.contiguous()`. Reference: https://github.com/pytorch/pytorch/issues/112577
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value
