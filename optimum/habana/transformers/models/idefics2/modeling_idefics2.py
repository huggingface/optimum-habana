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
    Idefics2VisionEmbeddings,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)


class GaudiIdefics2VisionEmbeddings(Idefics2VisionEmbeddings):
    def forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Inherits from Idefics2VisionEmbeddings::forward https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/idefics2/modeling_idefics2.py#L159
        The only differences are:
        - add int() in nb_patches_h. nb_patches_w to avoid overflow in torch.arange. sometimes return shape is nb_patches_h/nb_patch_w + 1
        - delete to("cpu") of p_attn_mask
        """

        batch_size, _, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full(
            size=(batch_size, max_nb_patches_h * max_nb_patches_w),
            fill_value=0,
            device=self.position_embedding.weight.device,
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = int(p_attn_mask[:, 0].sum())
            nb_patches_w = int(p_attn_mask[0].sum())

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids.to(self.position_embedding.weight.device)
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class GaudiIdefics2Model(Idefics2Model):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Idefics2BaseModelOutputWithPast]:
        """
        Inherits from Idefics2Model::forward https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/idefics2/modeling_idefics2.py#L1303
        The only differences are:
        - ignoring new Cache path for HPU
        - unfold is not supported in HPU, replace with conv2d
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
            conv_kernel = torch.ones(
                [1, 1, patch_size, patch_size], dtype=pixel_values.dtype, device=pixel_values.device
            )
            patches_subgrid = torch.nn.functional.conv2d(
                pixel_attention_mask.unsqueeze(1).to(conv_kernel.dtype), conv_kernel, stride=patch_size
            ).squeeze(1)
            patch_attention_mask = torch.eq(patches_subgrid, (patch_size * patch_size))

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
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=return_dict,
        )

        if return_legacy_cache and use_cache:
            if return_dict:
                outputs.past_key_values = (
                    outputs.past_key_values.to_legacy_cache()
                    if isinstance(outputs.past_key_values, Cache)
                    else outputs.past_key_values
                )
            else:
                outputs[1] = outputs[1].to_legacy_cache() if isinstance(outputs[1], Cache) else outputs[1]

        if not return_dict:
            return tuple(v for v in [*outputs, image_hidden_states] if v is not None)

        return Idefics2BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )

    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.Tensor],
        image_hidden_states: Optional[torch.Tensor],
    ):
        """
        Inherits from Idefics2Model::inputs_merger https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/idefics2/modeling_idefics2.py#L1268
        The only differences are:
        - replace `==` with torch.where to fix the issue in hpu graph
        """
        num_images, _, vision_hidden_size = image_hidden_states.shape
        special_image_token_mask = torch.where(input_ids == self.image_token_id)
        new_inputs_embeds = inputs_embeds.clone()
        reshaped_image_hidden_states = image_hidden_states.view(-1, vision_hidden_size)
        new_inputs_embeds[special_image_token_mask] = reshaped_image_hidden_states.to(new_inputs_embeds.device)
        return new_inputs_embeds


class GaudiIdefics2ForConditionalGeneration(Idefics2ForConditionalGeneration):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        token_idx: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Idefics2CausalLMOutputWithPast]:
        """
        Inherits from Idefics2ForConditionalGeneration::forward https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/idefics2/modeling_idefics2.py#L1505
        The only differences are:
        - add new args token_idx
        """
        if token_idx is not None:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
                raise ValueError(
                    "When first calling the model, if input_embeds are passed, input_ids should not be None."
                )

            if inputs_embeds is None:
                inputs_embeds = self.model.text_model.get_input_embeddings()(input_ids)

            # START VISUAL INPUTS INTEGRATION
            if pixel_values is not None and image_hidden_states is not None:
                raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
            elif image_hidden_states is not None:
                image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

            if past_seen_tokens == 0 and inputs_embeds is not None and image_hidden_states is not None:
                # When we generate, we don't want to replace the potential image_token_id that we generated by images
                # that simply don't exist
                inputs_embeds = self.model.inputs_merger(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    image_hidden_states=image_hidden_states,
                )

            outputs = self.model.text_model(
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
                if return_dict:
                    outputs.past_key_values = (
                        outputs.past_key_values.to_legacy_cache()
                        if isinstance(outputs.past_key_values, Cache)
                        else outputs.past_key_values
                    )
                else:
                    outputs[1] = outputs[1].to_legacy_cache() if isinstance(outputs[1], Cache) else outputs[1]

            hidden_states = outputs[0]
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                # Shift so that tokens < n predict n
                if attention_mask is not None:
                    shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
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
                image_hidden_states=image_hidden_states,
            )
        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                image_hidden_states=image_hidden_states,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                return_dict=return_dict,
            )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Inherits from Idefics2ForConditionalGeneration::forward https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/idefics2/modeling_idefics2.py#L1622
        The only differences are:
        - add new args token_idx
        - add None "Cache" past_key_values support
        - move vision_model to prepare_input_for_generation
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

        if token_idx is not None and pixel_values is not None:
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
            conv_kernel = torch.ones(
                [1, 1, patch_size, patch_size], dtype=pixel_values.dtype, device=pixel_values.device
            )
            patches_subgrid = torch.nn.functional.conv2d(
                pixel_attention_mask.unsqueeze(1).to(conv_kernel.dtype), conv_kernel, stride=patch_size
            ).squeeze(1)
            patch_attention_mask = torch.eq(patches_subgrid, (patch_size * patch_size))

            # Get sequence from the vision encoder
            image_hidden_states = self.model.vision_model(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
            ).last_hidden_state

            # Modality projection & resampling
            image_hidden_states = self.model.connector(
                image_hidden_states, attention_mask=patch_attention_mask.view(pixel_values.size(0), -1)
            )
            pixel_values = None
            pixel_attention_mask = None

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
