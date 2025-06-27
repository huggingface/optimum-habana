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
"""PyTorch VideoLlava model."""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.video_llava.modeling_video_llava import (
    VideoLlavaCausalLMOutputWithPast,
    VideoLlavaConfig,
    VideoLlavaForConditionalGeneration,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)


class GaudiVideoLlavaForConditionalGeneration(VideoLlavaForConditionalGeneration):
    def __init__(self, config: VideoLlavaConfig):
        super().__init__(config)
        self.feature_offset = 0

    def _merge_input_ids_with_visual_features(
        self, visual_features, inputs_embeds, input_ids, attention_mask, labels, token_idx, num_frames=1
    ):
        r"""
        Copied from VideoLlavaForConditionalGeneration._merge_input_ids_with_visual_features: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/video_llava/modeling_video_llava.py
        The only differences are:
        - add new args token_idx
        - add self.feature_offset param
        """
        num_images, num_image_patches, embed_dim = visual_features.shape
        batch_size, sequence_length = input_ids.shape
        last_token_idx = token_idx + self.feature_offset
        left_padding = not torch.sum(input_ids[:, last_token_idx - 1] == torch.tensor(self.pad_token_id))
        special_vision_token = self.config.video_token_index if num_frames > 1 else self.config.image_token_index

        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == special_vision_token
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_seq_len = (num_special_image_tokens.max() * (num_image_patches * num_frames - 1)) + sequence_length
        self.feature_offset = self.feature_offset + max_seq_len - sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != special_vision_token)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = (
            torch.cumsum((special_image_token_mask * (num_image_patches * num_frames - 1) + 1), dim=-1) - 1
        )
        nb_image_pad = max_seq_len - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        # expand input ids so that the second "merge" with videos does not fail
        final_embedding = torch.zeros(
            batch_size, max_seq_len, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_seq_len, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_seq_len), self.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_image_indices]
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_seq_len), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]
        else:
            final_labels = None

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.full((batch_size, max_seq_len), True, dtype=torch.bool, device=inputs_embeds.device)
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != visual_features.shape[:-1].numel():
            visual_type = "videos" if num_frames == 8 else "images"
            num_images //= num_frames
            raise ValueError(
                f"The input provided to the model are wrong. The number of {visual_type} tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of {visual_type} given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = visual_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

    def _get_vision_features(
        self,
        pixel_values_images: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        if pixel_values_images is None and pixel_values_videos is None:
            raise ValueError("You have to specify `pixel_values_images` or `pixel_values_videos`")

        # videos do not need to select features and it's always "full" (as it is done in the orig implementation)
        if pixel_values_videos is not None:
            batch_size_vid, num_frames, channels, height, width = pixel_values_videos.shape

            pixel_values = pixel_values_videos.reshape(batch_size_vid * num_frames, channels, height, width)
            video_outputs = self.video_tower(pixel_values, output_hidden_states=True)
            video_outputs = video_outputs.hidden_states[vision_feature_layer].squeeze(1)
        else:
            video_outputs = None
            num_frames = 0

        if pixel_values_images is not None:
            image_outputs = self.image_tower(pixel_values_images, output_hidden_states=True)
            image_outputs = image_outputs.hidden_states[vision_feature_layer].squeeze(1)

            if vision_feature_select_strategy == "default":
                image_outputs = image_outputs[:, 1:]
            elif vision_feature_select_strategy == "full":
                image_outputs = image_outputs
            else:
                raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")
        else:
            image_outputs = None

        return image_outputs, video_outputs, num_frames

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values_images: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
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
        token_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, VideoLlavaCausalLMOutputWithPast]:
        r"""
        Copied from VideoLlavaForConditionalGeneration.forward: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/video_llava/modeling_video_llava.py
        The only differences are:
        - add new args token_idx
        - add new args attn_softmax_bf16
        - add new args reuse_cache
        - add new args use_flash_attention
        - add new args flash_attention_recompute
        - add new args flash_attention_causal_mask
        - add new args flash_attention_fast_softmax
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            logits_to_keep=0,
            token_idx=token_idx,
            **kwargs,
        )

        logits = outputs[0]
        if logits.shape[1] > 1:
            logits = logits[:, self.feature_offset :, :]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
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

        return VideoLlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=kwargs.get("image_features", None) if pixel_values_images is not None else None,
            video_hidden_states=kwargs.get("video_features", None) if pixel_values_videos is not None else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values_images=None,
        pixel_values_videos=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        token_idx = kwargs.get("token_idx", None)
        if token_idx is None:
            return super().prepare_inputs_for_generation(
                input_ids=input_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values_images=pixel_values_images,
                pixel_values_videos=pixel_values_videos,
                attention_mask=attention_mask,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        # Else, we need to update token_idx when merging features from videos/images with input embeddings
        labels = kwargs.get("labels", None)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if (pixel_values_images is not None or pixel_values_videos is not None) and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        legacy_processing = False
        inputs_not_expanded = False
        if input_ids is not None:
            img_token_not_enough = (input_ids == self.config.image_token_index).sum(
                1
            ).max() < self.config.image_seq_length
            video_token_not_enough = (input_ids == self.config.video_token_index).sum(
                1
            ).max() < self.config.video_seq_length
            # if the number of image/video tokens is more than image embeddings seq length, then prob we expanded it in processing
            # not very reliable, but we don't expect one to actually pass 500+ images for one prompt
            inputs_not_expanded = (img_token_not_enough and pixel_values_images is not None) or (
                video_token_not_enough and pixel_values_videos is not None
            )
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        position_ids = model_inputs["position_ids"]
        cache_position = model_inputs["cache_position"]
        attention_mask = model_inputs["attention_mask"]
        inputs_embeds = model_inputs.get("inputs_embeds", None)
        input_ids = model_inputs.get("input_ids", None)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            pixels_present = input_ids.shape[-1] == 1 and (
                pixel_values_images is not None or pixel_values_videos is not None
            )
            legacy_processing = inputs_not_expanded or pixels_present

        vision_feature_layer = kwargs.get("vision_feature_layer", None)
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = kwargs.get("vision_feature_select_strategy", None)
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        if pixel_values_images is not None or pixel_values_videos is not None:
            image_outputs, video_outputs, num_frames = self._get_vision_features(
                pixel_values_images=pixel_values_images,
                pixel_values_videos=pixel_values_videos,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

            image_features = video_features = None
            if image_outputs is not None:
                image_features = self.multi_modal_projector(image_outputs)
            if video_outputs is not None:
                video_features = self.multi_modal_projector(video_outputs)

            if legacy_processing:
                logger.warning_once(
                    "Expanding inputs for image tokens in Video-LLaVa should be done in processing. "
                    "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                    "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                    "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
                )
                if input_ids.shape[1] != 1:
                    self.feature_offset = 0
                    for features, frames in ((image_features, 1), (video_features, num_frames)):
                        if features is not None:
                            (
                                inputs_embeds,
                                attention_mask,
                                labels,
                                position_ids,
                                input_ids,
                            ) = self._merge_input_ids_with_visual_features(
                                features,
                                inputs_embeds,
                                input_ids,
                                attention_mask,
                                labels,
                                token_idx,
                                num_frames=frames,
                            )
                    cache_position = torch.arange(attention_mask.shape[1], device=attention_mask.device)
                else:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                    target_length = input_ids.shape[1]
                    past_length = first_layer_past_key_value.shape[-1]

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], past_length),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    # Filter out only the tokens that can be un-attended, this can happen
                    # if one uses Llava + Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]

                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0
                    new_token_idx = token_idx + self.feature_offset
                    extended_attention_mask[:, new_token_idx - 1 + target_length :] = 0
                    attention_mask = extended_attention_mask.clone()
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                    cache_position = new_token_idx

            # TODO: @raushan retain only the new behavior after v4.47
            else:
                if image_outputs is not None:
                    special_image_mask = (
                        (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    )
                    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

                if video_outputs is not None:
                    special_image_mask = (
                        (input_ids == self.config.video_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    )
                    video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, video_features)

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "attention_mask": attention_mask,
                "token_idx": token_idx + self.feature_offset,
                "inputs_embeds": inputs_embeds,
            }
        )
        if legacy_processing or (cache_position is not None and cache_position[0]) == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values_images"] = pixel_values_images
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["image_features"] = image_features
            model_inputs["video_features"] = video_features
        return model_inputs
