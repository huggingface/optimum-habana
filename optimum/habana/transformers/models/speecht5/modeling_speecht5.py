from typing import Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
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
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    The only differences are:
    - add new args token_idx
    - update to HF 4.55.4 Cache API (no explicit past_key_value tuple in/out)
    """

    is_cross_attention = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scaling

    current_states = key_value_states if is_cross_attention else hidden_states

    key_states = self.k_proj(current_states)
    value_states = self.v_proj(current_states)
    key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    if past_key_value is not None:
        _cache_pos = None if is_cross_attention else cache_position
        key_states, value_states = past_key_value.update(
            key_states, value_states, getattr(self, "layer_idx", None), {"cache_position": _cache_pos}
        )

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).reshape(*proj_shape)
    key_states = key_states.reshape(*proj_shape)
    value_states = value_states.reshape(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
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
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
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
    past_key_value: Optional["Cache"] = None,  # 4.55.4
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = True,
    cache_position: Optional[torch.Tensor] = None,
    token_idx: Optional[torch.Tensor] = None,
):
    """
    Copied from SpeechT5DecoderLayer.forward: https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/speecht5/modeling_speecht5.py
    The only differences are:
    - add token_idx in self-attention
    - align with HF 4.55.4: no present_key_value returned (cache is updated in-place)
    """
    residual = hidden_states

    # Self Attention
    self_attn_outputs = self.self_attn(
        hidden_states=hidden_states,
        past_key_value=past_key_value.self_attention_cache if past_key_value is not None else None,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
        cache_position=cache_position,
        token_idx=token_idx,
    )
    hidden_states = self.dropout(self_attn_outputs[0])
    hidden_states = residual + hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)

    # Cross-Attention Block
    cross_attn_weights = None
    if encoder_hidden_states is not None:
        residual = hidden_states
        cross_outputs = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_key_value.cross_attention_cache if past_key_value is not None else None,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = self.dropout(cross_outputs[0])
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        if output_attentions:
            cross_attn_weights = cross_outputs[1]

    # Fully Connected
    hidden_states = hidden_states + self.feed_forward(hidden_states)
    hidden_states = self.final_layer_norm(hidden_states)

    outputs = (hidden_states,)
    if output_attentions:
        # self-attn weights
        outputs += (self_attn_outputs[1], cross_attn_weights)

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
    Copied from SpeechT5Decoder.forward: https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/speecht5/modeling_speecht5.py
    The only differences are:
    - add token_idx args
    - use _gaudi_prepare_4d_causal_attention_mask
    - align with HF 4.55.4 Cache API (no next_decoder_cache)
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    input_shape = hidden_states.size()[:-1]

    past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

    attention_mask = _gaudi_prepare_4d_causal_attention_mask(
        attention_mask, input_shape, hidden_states, past_key_values_length
    )

    # expand encoder attention mask
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

    for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
        if attn_mask is not None:
            if attn_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        skip_the_layer = False
        if self.training:
            dropout_probability = torch.rand([])
            skip_the_layer = dropout_probability < self.layerdrop
        if skip_the_layer and not synced_gpus:
            continue

        layer_outputs = decoder_layer(
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
            token_idx=token_idx,
        )
        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if encoder_hidden_states is not None:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, None, all_hidden_states, all_self_attentions, all_cross_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=None,
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
    Copied and adapted from `_generate_speech` (transformers v4.55.4)
    Differences introduced for Habana Gaudi:
    - wrapped encoder / decoder / prenet with HPU graphs
    - use static-shape kv-cache via Cache API (DynamicCache + EncoderDecoderCache)
    - disable dropout in prenet for deterministic output lengths
    - adjust attention_mask update order (fixes off-by-one shape mismatch)
    """
    if speaker_embeddings is None:
        raise ValueError(
            "`speaker_embeddings` must be provided (e.g. from https://huggingface.co/datasets/regisss/cmu-arctic-xvectors)."
        )

    from habana_frameworks.torch.hpu import wrap_in_hpu_graph

    # Wrap model components with HPU graph to enable static compilation
    if not hasattr(model.speecht5.encoder, "clear_cache"):
        model.speecht5.encoder = wrap_in_hpu_graph(model.speecht5.encoder)
    if not hasattr(model.speecht5.decoder.wrapped_decoder, "clear_cache"):
        model.speecht5.decoder.wrapped_decoder = wrap_in_hpu_graph(model.speecht5.decoder.wrapped_decoder)
    if not hasattr(model.speecht5.decoder.prenet, "clear_cache"):
        model.speecht5.decoder.prenet = wrap_in_hpu_graph(model.speecht5.decoder.prenet)

    # Prepare encoder attention mask
    encoder_attention_mask = (
        1 - (input_values == model.config.pad_token_id).int() if attention_mask is None else attention_mask
    )

    bsz = input_values.size(0)

    # Run encoder
    encoder_out = model.speecht5.encoder(
        input_values=input_values,
        attention_mask=encoder_attention_mask,
        return_dict=True,
    )
    encoder_hidden_states = encoder_out.last_hidden_state

    # Downsample attention mask if using speech prenet
    if isinstance(model.speecht5.encoder, SpeechT5EncoderWithSpeechPrenet):
        encoder_attention_mask = model.speecht5.encoder.prenet._get_feature_vector_attention_mask(
            encoder_hidden_states.shape[1], encoder_attention_mask
        )

    # Determine dynamic decoding length bounds
    maxlen = int(encoder_hidden_states.size(1) * maxlenratio / model.config.reduction_factor)
    minlen = int(encoder_hidden_states.size(1) * minlenratio / model.config.reduction_factor)

    # Initialize decoder inputs
    output_sequence = encoder_hidden_states.new_zeros(bsz, 1, model.config.num_mel_bins)
    output_sequence = torch.nn.functional.pad(output_sequence, (0, 0, 0, maxlen - 1), value=model.config.pad_token_id)

    # Prepare attention and cache structures
    attention_mask = torch.zeros((bsz, maxlen), dtype=torch.long, device=output_sequence.device)
    self_attention_cache = DynamicCache()
    cross_attention_cache = DynamicCache()
    past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)

    # Internal buffers
    spectrogram, cross_attentions, result_spectrogram = [], [], {}
    token_idx = torch.tensor(1, device=output_sequence.device)
    idx = 0

    # Generation loop
    while True:
        idx += 1

        # Prenet (disable dropout for HPU determinism)
        decoder_hidden_states = model.speecht5.decoder.prenet(output_sequence, speaker_embeddings)

        # Use last step or full input depending on cache
        decoder_inputs = (
            decoder_hidden_states
            if past_key_values.get_seq_length() == 0
            else torch.index_select(decoder_hidden_states, 1, token_idx - 1)
        )

        # Decoder forward with caching
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

        attention_mask.index_fill_(1, token_idx - 1, 1)

        # Optional cross-attention collection
        if output_cross_attentions:
            cross_attentions.append(torch.cat(decoder_out.cross_attentions, dim=0))

        # Extract decoder output
        last_output = decoder_out.last_hidden_state[:, 0:1, :].squeeze(1)

        # Predict mel spectrum
        spectrum = model.speech_decoder_postnet.feat_out(last_output)
        spectrum = spectrum.view(bsz, model.config.reduction_factor, model.config.num_mel_bins)
        spectrogram.append(spectrum)

        # Update output sequence and token index
        output_sequence.index_copy_(1, token_idx, spectrum[:, -1, :].view(bsz, 1, model.config.num_mel_bins))
        prob = torch.sigmoid(model.speech_decoder_postnet.prob_out(last_output))
        token_idx.add_(1)

        # Early exit logic
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

    # Combine all generated spectrograms
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
