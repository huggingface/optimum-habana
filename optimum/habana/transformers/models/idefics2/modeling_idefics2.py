# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Idefics2 model."""

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.idefics2.modeling_idefics2 import (
    Idefics2BaseModelOutputWithPast,
    Idefics2CausalLMOutputWithPast,
    Idefics2ForConditionalGeneration,
    Idefics2Model,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)


class GaudiIdefics2Model(Idefics2Model):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Idefics2BaseModelOutputWithPast]:
        """
        Inherits from Idefics2Model::forward https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/idefics2/modeling_idefics2.py#L1303
        The only differences are:
        - add new args token_idx
        - ignoring new Cache path for HPU
        - unfold is not supported in HPU, fallback to cpu
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_seen_tokens = 0
        return_legacy_cache = True
        use_new_cache = False  # Ignoring new Cache path for HPU
        if use_cache and use_new_cache:
            if not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
                return_legacy_cache = True
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()
        else:
            if past_key_values is not None:
                past_seen_tokens = past_key_values[0][0].shape[2]
        if inputs_embeds is not None and input_ids is None and past_seen_tokens == 0:
            raise ValueError("When first calling the model, if input_embeds are passed, input_ids should not be None.")

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.to(dtype=self.dtype)  # fp16 compatibility
            pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

            # Remove padding images - padding images are full 0.
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
            pixel_values = pixel_values[real_images_inds].contiguous()

            # Handle the vision attention mask
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(
                    size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                    dtype=torch.bool,
                    device=pixel_values.device,
                )
            else:
                # Remove padding images from the mask/pP p
                pixel_attention_mask = pixel_attention_mask.view(
                    batch_size * num_images, *pixel_attention_mask.shape[2:]
                )
                pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

            patch_size = self.config.vision_config.patch_size
            patches_subgrid = pixel_attention_mask.cpu().unfold(dimension=1, size=patch_size, step=patch_size)
            patches_subgrid = patches_subgrid.cpu().unfold(dimension=2, size=patch_size, step=patch_size)
            patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

            # Get sequence from the vision encoder
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
            ).last_hidden_state

            # Modality projection & resampling
            image_hidden_states = self.connector(
                image_hidden_states, attention_mask=patch_attention_mask.view(pixel_values.size(0), -1)
            )

        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if past_seen_tokens == 0 and inputs_embeds is not None and image_hidden_states is not None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_idx=token_idx,
        )

        if return_legacy_cache and use_cache:
            outputs.past_key_values = (
                outputs.past_key_values.to_legacy_cache()
                if isinstance(outputs.past_key_values, Cache)
                else outputs.past_key_values
            )

        if not return_dict:
            return tuple(v for v in [*outputs, image_hidden_states] if v is not None)

        return Idefics2BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


class GaudiIdefics2ForConditionalGeneration(Idefics2ForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Idefics2CausalLMOutputWithPast]:
        """
        Inherits from Idefics2ForConditionalGeneration::forward https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/idefics2/modeling_idefics2.py#L1505
        The only differences are:
        - add new args token_idx
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_idx=token_idx,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Idefics2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Inherits from Idefics2ForConditionalGeneration::forward https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/idefics2/modeling_idefics2.py#L1622
        The only differences are:
        - add new args token_idx
        - add None "Cache" past_key_values support
        """
        past_length = 0
        token_idx = kwargs.get("token_idx", None)
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            if isinstance(past_key_values, Cache):
                past_length = past_key_values.get_seq_length()
                max_cache_length = past_key_values.get_max_length()
            else:
                past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if token_idx is not None:
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)
            elif attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and past_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                if token_idx is not None:
                    position_ids = torch.index_select(position_ids, 1, token_idx - 1)
                else:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_length == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        image_hidden_states = kwargs.get("image_hidden_states", None)
        if image_hidden_states is not None:
            pixel_values = None
            pixel_attention_mask = None
        else:
            pixel_values = kwargs.get("pixel_values", None)
            pixel_attention_mask = kwargs.get("pixel_attention_mask", None)
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_attention_mask,
                "image_hidden_states": image_hidden_states,
                "token_idx": token_idx,
            }
        )
        return model_inputs
