# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Llava model."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast, LlavaForConditionalGeneration
from transformers.utils import logging


logger = logging.get_logger(__name__)


def _pad_inputs(
    input_ids, attention_mask, image_token_index, num_patches, pad_token_id, vision_feature_select_strategy=None
):
    """
    pad inputs for static shape
    """

    if vision_feature_select_strategy == "default":
        num_patches = num_patches
    elif vision_feature_select_strategy == "full":
        num_patches = num_patches + 1
    else:
        raise ValueError(f"Unexpected select feature strategy: {vision_feature_select_strategy}")
    image_offset = 0
    new_input_ids = []
    new_attention_mask = []
    tokens_pos = []
    for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask):
        image_token_indices = torch.where(cur_input_ids == image_token_index)[0].tolist() + [cur_input_ids.shape[0]]

        cur_input_ids_extend = []
        cur_attention_mask_extend = []
        cur_token_pos = []
        start = 0
        for i, image_token_indice in enumerate(image_token_indices):
            token_pos = len(cur_input_ids_extend)
            cur_input_ids_extend.extend(cur_input_ids[start:image_token_indice].cpu().tolist())
            cur_attention_mask_extend.extend(cur_attention_mask[start:image_token_indice].cpu().tolist())
            cur_token_pos.extend(list(range(token_pos, len(cur_input_ids_extend))))
            if i != len(image_token_indices) - 1:
                cur_input_ids_extend.extend([image_token_index] * num_patches)
                cur_attention_mask_extend.extend([1] * num_patches)
                cur_token_pos.append(image_token_indice)
            start = image_token_indice + 1

        new_input_ids.append(cur_input_ids_extend)
        new_attention_mask.append(cur_attention_mask_extend)
        tokens_pos.append(cur_token_pos)

    max_len = max(len(x) for x in new_input_ids)
    image_offset += max_len - input_ids.shape[1]

    # padding
    new_input_ids_padded = []
    new_attention_mask_padded = []
    tokens_pos_padded = []

    # left padding for no image in example, so we don't need change token_idx
    for cur_new_ids, cur_attention_mask, cur_token_pos in zip(new_input_ids, new_attention_mask, tokens_pos):
        pad_len = max_len - len(cur_new_ids)
        new_input_ids_padded.append([pad_token_id] * pad_len + cur_new_ids)
        new_attention_mask_padded.append([0] * pad_len + cur_attention_mask)
        tokens_pos_padded.append([x + pad_len for x in cur_token_pos])

    input_ids = torch.tensor(new_input_ids_padded).to(input_ids.device)
    attention_mask = torch.tensor(new_attention_mask_padded).to(input_ids.device)
    tokens_pos = torch.tensor(tokens_pos_padded).to(input_ids.device)

    return input_ids, attention_mask, image_offset, tokens_pos


def _merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, image_token_index):
    """
    Merge text and images
    """
    batch_size, sequence_length, embed_dim = inputs_embeds.shape

    batch_indices, image_indices = torch.where(input_ids == image_token_index)

    inputs_embeds[batch_indices, image_indices] = image_features.contiguous().reshape(-1, embed_dim)

    return inputs_embeds.contiguous()


class GaudiLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: Optional[torch.Tensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        image_offset: Optional[int] = None,
        tokens_pos: Optional[torch.LongTensor] = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        **lm_kwargs,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        """
        Inherits from LlavaForConditionalGeneration: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/llava/modeling_llava.py#L362
        The only differences are:
        - add new args token_idx
        - add new args image_offset
        - add new args tokens_pos
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        # 2. Merge text and images
        if pixel_values is not None and input_ids.shape[1] != 1:
            image_outputs = self.vision_tower(
                pixel_values,
                output_hidden_states=True,
                use_flash_attention=use_flash_attention,
                flash_attention_recompute=flash_attention_recompute,
            )
            # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

            image_features = self.multi_modal_projector(selected_image_feature)
            inputs_embeds = _merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, self.config.image_token_index
            )

        if token_idx is not None:
            outputs = self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                token_idx=token_idx + image_offset,
                use_flash_attention=use_flash_attention,
                flash_attention_recompute=flash_attention_recompute,
                **lm_kwargs,
            )

            if input_ids.shape[1] != 1 and pixel_values is not None and tokens_pos is not None:
                batch_size, seq_len = tokens_pos.shape
                batch_indices = torch.arange(batch_size).repeat_interleave(seq_len)
                logits = outputs[0][batch_indices, tokens_pos.reshape(-1), :].reshape(batch_size, seq_len, -1)
            else:
                logits = outputs[0]

            loss = None

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return LlavaCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                image_hidden_states=image_features if pixel_values is not None else None,
            )

        else:
            outputs = self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                use_flash_attention=use_flash_attention,
                flash_attention_recompute=flash_attention_recompute,
                **lm_kwargs,
            )

            logits = outputs[0]

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                if attention_mask is not None:
                    shift_attention_mask = attention_mask[..., 1:]
                    shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                    shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
                else:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
                )

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return LlavaCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                image_hidden_states=image_features if pixel_values is not None else None,
            )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        """
        Inherits from LlavaForConditionalGeneration: https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llava/modeling_llava.py
        The only differences are:
        - add new args token_idx
        - add new args image_offset
        - add new args tokens_pos
        - from step2 when enable KV cache, slice next_input_ids from input_ids base on the token_idx
        - from step2 when enable KV cache, slice next_position_ids from position_ids base on the token_id
        """
        token_idx = kwargs.get("token_idx", None)
        image_offset = 0
        tokens_pos = None
        legacy_processing = (
            (input_ids == self.config.image_token_index).sum(1).max() < self.config.image_seq_length
        ) or ((input_ids.shape[-1] == 1 if token_idx is None else token_idx == 1) and pixel_values is not None)
        if token_idx is not None and pixel_values is not None and legacy_processing:
            input_ids, attention_mask, image_offset, tokens_pos = _pad_inputs(
                input_ids,
                attention_mask,
                self.config.image_token_index,
                self.vision_tower.vision_model.embeddings.num_patches,
                self.pad_token_id,
                vision_feature_select_strategy=self.config.vision_feature_select_strategy,
            )

        past_length = 0
        if past_key_values is not None:
            if token_idx is None:
                if isinstance(past_key_values, Cache):
                    cache_length = past_key_values.get_seq_length()
                    past_length = past_key_values.seen_tokens
                else:
                    cache_length = past_length = past_key_values[0][0].shape[2]
                # Keep only the unprocessed tokens:
                # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
                # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
                # input)
                if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                    input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
                # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
                # input_ids based on the past_length.
                elif past_length < input_ids.shape[1]:
                    input_ids = input_ids[:, past_length:]
                    # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
                elif self.config.image_token_index in input_ids:
                    input_ids = input_ids[:, input_ids.shape[1] - 1 :]
                    # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
                    # older attention values, as their corresponding values are not part of the input.
                    if cache_length < past_length and attention_mask is not None:
                        attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]
            else:
                # past_length += token_idx
                input_ids = torch.index_select(input_ids, 1, token_idx + image_offset - 1)

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                if token_idx is not None:
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                else:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        use_flash_attention = kwargs.get("use_flash_attention", False)
        flash_attention_recompute = kwargs.get("flash_attention_recompute", False)

        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "token_idx": token_idx,
                "image_offset": image_offset,
                "tokens_pos": tokens_pos,
                "use_flash_attention": use_flash_attention,
                "flash_attention_recompute": flash_attention_recompute,
            }
        )

        return model_inputs
