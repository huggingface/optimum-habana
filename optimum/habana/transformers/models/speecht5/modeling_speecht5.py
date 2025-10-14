from typing import Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithSpeechPrenet, SpeechT5PreTrainedModel
from transformers.utils import logging

from ...modeling_attn_mask_utils import (
    _gaudi_prepare_4d_causal_attention_mask,
)


logger = logging.get_logger(__name__)


def gaudi_SpeechT5Attention_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional["Cache"] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    position_bias: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    cache_position: Optional[torch.Tensor] = None,
    token_idx: Optional[torch.Tensor] = None,
    layer_idx: Optional[int] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Copied from SpeechT5Attention.forward (transformers 4.55.4)
    The only differences are:
    - add new arg `token_idx`
    """

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    bsz, tgt_len, _ = hidden_states.size()

    # get query projection
    query_states = self.q_proj(hidden_states) * self.scaling

    # retrieve cache entry for this layer
    if past_key_value is not None:
        if is_cross_attention:
            curr_past = past_key_value.cross_attention_cache
        else:
            curr_past = past_key_value.self_attention_cache
    else:
        curr_past = None

    # compute key/value
    current_states = key_value_states if is_cross_attention else hidden_states
    if curr_past is not None and curr_past.is_updated.get(layer_idx, False) and is_cross_attention:
        key_states = curr_past.layers[layer_idx].keys
        value_states = curr_past.layers[layer_idx].values
    else:
        key_states = self.k_proj(current_states)
        value_states = self.v_proj(current_states)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            # update Cache (new HF 4.55+ mechanism)
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past.update(
                key_states, value_states, layer_idx, {"cache_position": cache_position}
            )
            if is_cross_attention:
                past_key_value.is_updated[layer_idx] = True

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
    query_states = query_states.reshape(*proj_shape)
    key_states = key_states.reshape(*proj_shape)
    value_states = value_states.reshape(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, "
            f"but is {attn_weights.size()}"
        )

    # relative attention bias
    if position_bias is not None:
        reshape_q = query_states.contiguous().view(bsz * self.num_heads, -1, self.head_dim).transpose(0, 1)
        rel_pos_bias = torch.matmul(reshape_q, position_bias.transpose(-2, -1))
        rel_pos_bias = rel_pos_bias.transpose(0, 1).view(
            bsz * self.num_heads, position_bias.size(0), position_bias.size(1)
        )
        attn_weights += rel_pos_bias

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        if layer_head_mask.size() != (self.num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            )
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if output_attentions:
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, "
            f"but is {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_reshaped


def gaudi_SpeechT5DecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional["Cache"] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = True,
    cache_position: Optional[torch.Tensor] = None,
    token_idx: Optional[torch.Tensor] = None,
):
    """
    Copied from SpeechT5DecoderLayer.forward (transformers 4.55.4)
    The only differences are:
    - add token_idx argument in self-attention
    """
    residual = hidden_states

    # Self-Attention (HF 4.55.4 style)
    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states,
        past_key_value=past_key_value,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
        cache_position=cache_position,
        token_idx=token_idx,  # Gaudi extension
    )

    hidden_states = self.dropout(hidden_states)
    hidden_states = residual + hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)

    cross_attn_weights = None
    if encoder_hidden_states is not None:
        residual = hidden_states
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

    hidden_states = hidden_states + self.feed_forward(hidden_states)
    hidden_states = self.final_layer_norm(hidden_states)

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights, cross_attn_weights)

    return outputs


def gaudi_SpeechT5Decoder_forward(
    self,
    hidden_states: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional["Cache"] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.Tensor] = None,
    token_idx: Optional[torch.Tensor] = None,
) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
    """
    Copied from SpeechT5Decoder.forward (transformers 4.55.4)
    The only differences are:
    - add token_idx args for Gaudi
    - use _gaudi_prepare_4d_causal_attention_mask
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    input_shape = hidden_states.size()[:-1]
    past_seen_tokens = past_key_values.get_usable_length(cache_position) if past_key_values is not None else 0

    attention_mask = _gaudi_prepare_4d_causal_attention_mask(
        attention_mask, input_shape, hidden_states, past_seen_tokens
    )

    if encoder_hidden_states is not None and encoder_attention_mask is not None:
        encoder_attention_mask = _prepare_4d_attention_mask(
            encoder_attention_mask, hidden_states.dtype, tgt_len=input_shape[-1]
        )

    synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        skip_the_layer = False
        if self.training:
            dropout_probability = torch.rand([])
            skip_the_layer = dropout_probability < self.layerdrop
        if skip_the_layer and not synced_gpus:
            continue

        hidden_states, self_attn_weights, cross_attn_weights = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            token_idx=token_idx,  # Gaudi extension
        )

        if output_attentions:
            all_self_attentions = all_self_attentions + (self_attn_weights,)
            if encoder_hidden_states is not None:
                all_cross_attentions = all_cross_attentions + (cross_attn_weights,)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )


def gaudi_generate_speech(
    model: SpeechT5PreTrainedModel,
    input_values: torch.FloatTensor,
    speaker_embeddings: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    threshold: float = 0.5,
    minlenratio: float = 0.0,
    maxlenratio: float = 20.0,
    vocoder: Optional[nn.Module] = None,
    output_cross_attentions: bool = False,
    return_output_lengths: bool = False,
):
    """
    Copied from _generate_speech (transformers 4.55.4)
    Differences:
    - wrapped with HPU graphs
    - static-shape kv-cache (Cache API)
    - disable dropout in prenet
    """
    if speaker_embeddings is None:
        raise ValueError(
            "`speaker_embeddings` must be provided (e.g. from https://huggingface.co/datasets/regisss/cmu-arctic-xvectors)."
        )

    from habana_frameworks.torch.hpu import wrap_in_hpu_graph

    if not hasattr(model.speecht5.encoder, "clear_cache"):
        model.speecht5.encoder = wrap_in_hpu_graph(model.speecht5.encoder)
    if not hasattr(model.speecht5.decoder.wrapped_decoder, "clear_cache"):
        model.speecht5.decoder.wrapped_decoder = wrap_in_hpu_graph(model.speecht5.decoder.wrapped_decoder)
    if not hasattr(model.speecht5.decoder.prenet, "clear_cache"):
        model.speecht5.decoder.prenet = wrap_in_hpu_graph(model.speecht5.decoder.prenet)

    encoder_attention_mask = (
        1 - (input_values == model.config.pad_token_id).int() if attention_mask is None else attention_mask
    )

    bsz = input_values.size(0)
    encoder_out = model.speecht5.encoder(
        input_values=input_values,
        attention_mask=encoder_attention_mask,
        return_dict=True,
    )
    encoder_hidden_states = encoder_out.last_hidden_state

    # Downsample attention mask if prenet used
    if isinstance(model.speecht5.encoder, SpeechT5EncoderWithSpeechPrenet):
        encoder_attention_mask = model.speecht5.encoder.prenet._get_feature_vector_attention_mask(
            encoder_hidden_states.shape[1], encoder_attention_mask
        )

    maxlen = int(encoder_hidden_states.size(1) * maxlenratio / model.config.reduction_factor)
    minlen = int(encoder_hidden_states.size(1) * minlenratio / model.config.reduction_factor)

    output_sequence = encoder_hidden_states.new_zeros(bsz, 1, model.config.num_mel_bins)
    output_sequence = torch.nn.functional.pad(output_sequence, (0, 0, 0, maxlen - 1), value=model.config.pad_token_id)
    spectrogram, cross_attentions, result_spectrogram = [], [], {}
    token_idx = torch.tensor(1, device=output_sequence.device)
    attention_mask = torch.zeros((bsz, maxlen), dtype=torch.long, device=output_sequence.device)

    past_key_values = EncoderDecoderCache(model.speecht5.decoder.config)

    idx = 0
    while True:
        idx += 1
        attention_mask.index_fill_(1, token_idx - 1, 1)

        decoder_hidden_states = model.speecht5.decoder.prenet(output_sequence, speaker_embeddings)
        decoder_inputs = (
            decoder_hidden_states
            if past_key_values.get_seq_length() == 0
            else torch.index_select(decoder_hidden_states, 1, token_idx - 1)
        )

        decoder_out = model.speecht5.decoder.wrapped_decoder(
            hidden_states=decoder_inputs,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=output_cross_attentions,
            return_dict=True,
            token_idx=token_idx,
        )

        if output_cross_attentions:
            cross_attentions.append(torch.cat(decoder_out.cross_attentions, dim=0))

        last_output = decoder_out.last_hidden_state[:, 0:1, :].squeeze(1)
        spectrum = model.speech_decoder_postnet.feat_out(last_output)
        spectrum = spectrum.view(bsz, model.config.reduction_factor, model.config.num_mel_bins)
        spectrogram.append(spectrum)

        output_sequence.index_copy_(1, token_idx, spectrum[:, -1, :].view(bsz, 1, model.config.num_mel_bins))
        prob = torch.sigmoid(model.speech_decoder_postnet.prob_out(last_output))
        token_idx.add_(1)

        if idx < minlen:
            continue
        meet_indexes = (
            torch.where(torch.sum(prob, dim=-1) >= threshold)[0].tolist() if idx < maxlen else range(len(prob))
        )
        meet_indexes = [i for i in meet_indexes if i not in result_spectrogram]
        if meet_indexes:
            spectrograms = torch.stack(spectrogram).transpose(0, 1).flatten(1, 2)
            spectrograms = model.speech_decoder_postnet.postnet(spectrograms)
            for mi in meet_indexes:
                result_spectrogram[mi] = spectrograms[mi]
        if len(result_spectrogram) >= bsz:
            break

    spectrograms = [result_spectrogram[i] for i in range(len(result_spectrogram))]
    if not return_output_lengths:
        spectrogram = spectrograms[0] if bsz == 1 else torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        outputs = vocoder(spectrogram) if vocoder is not None else spectrogram
        if output_cross_attentions:
            cross_attentions = torch.cat(cross_attentions, dim=2)
            if bsz > 1:
                cross_attentions = cross_attentions.view(
                    bsz, int(cross_attentions.size(0) / bsz), *cross_attentions.size()[-3:]
                )
            outputs = (outputs, cross_attentions)
    else:
        lengths = [s.size(0) for s in spectrograms]
        spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        if vocoder is None:
            outputs = (spectrograms, lengths)
        else:
            waveforms = vocoder(spectrograms)
            waveform_lengths = [int(waveforms.size(1) / max(lengths)) * i for i in lengths]
            outputs = (waveforms, waveform_lengths)
        if output_cross_attentions:
            cross_attentions = torch.cat(cross_attentions, dim=2).view(
                bsz, int(cross_attentions.size(0) / bsz), *cross_attentions.size()[-3:]
            )
            outputs = (*outputs, cross_attentions)
    return outputs
