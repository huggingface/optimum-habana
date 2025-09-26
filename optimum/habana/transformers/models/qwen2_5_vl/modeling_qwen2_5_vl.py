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
"""PyTorch Gaudi Qwen2.5-VL model."""

from math import ceil
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, StaticCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLAttention,
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLConfig,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VLVisionConfig,
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
)
from transformers.utils import is_torchdynamo_compiling, logging


try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE

    has_fused_rope = True
except ImportError:
    has_fused_rope = False
    print("Not using HPU fused kernel for apply_rotary_pos_emb")

try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm as FusedRMSNorm

    has_fused_rms_norm = True
except ImportError:
    has_fused_rms_norm = False
    print("Not using HPU fused kernel for RMSNorm")

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None

import habana_frameworks.torch.core as htcore


logger = logging.get_logger(__name__)

VISION_BUCKETS = [1600, 3136, 4096, 6400, 7744, 9216, 12544]


def apply_customized_rope(query, key, cos, sin, vision=False, mrope_section=None):
    if query.device.type == "hpu" and has_fused_rope:
        if mrope_section is not None:
            # multimodal
            mrope_section = mrope_section * 2
            cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
            sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)
        if key.dtype == torch.bfloat16:
            return FusedRoPE.apply(
                query, cos.unsqueeze(1).to(torch.bfloat16), sin.unsqueeze(1).to(torch.bfloat16), None
            ), FusedRoPE.apply(key, cos.unsqueeze(1).to(torch.bfloat16), sin.unsqueeze(1).to(torch.bfloat16), None)
        else:
            return FusedRoPE.apply(
                query,
                cos.unsqueeze(1),
                sin.unsqueeze(1),
                None,
            ), FusedRoPE.apply(key, cos.unsqueeze(1), sin.unsqueeze(1), None)
    elif vision:
        return apply_rotary_pos_emb_vision(query, key, cos, sin)
    else:
        return apply_multimodal_rotary_pos_emb(query, key, cos, sin, mrope_section)


class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        self._hpu_kernel_fsdpa = fusedSDPA

    def forward(self, query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode):
        return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode)


class GaudiQwen2_5_VLVisionAttention(Qwen2_5_VLVisionAttention):
    def __init__(self, config: Qwen2_5_VLVisionConfig):
        super().__init__(config)
        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA) if FusedSDPA else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_customized_rope(query_states, key_states, cos, sin, vision=True)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        if cu_seqlens is not None:
            # performs window attention
            # we assume image is 112 aligned in both h/w dims
            # in other words, x % 64 = 0
            # that simplifies the slicing of window attention
            # in patches of 64
            outputs = []
            cu_seqlens = list(range(0, hidden_states.shape[0] + 1, 64))

            if FusedSDPA is not None and use_flash_attention:
                for i in range(1, len(cu_seqlens)):
                    # For large image, we add mark step here
                    # for every 100th step to make compile time shorter
                    if i % 100 == 0:
                        htcore.mark_step()
                    start_idx = cu_seqlens[i - 1]
                    end_idx = cu_seqlens[i]
                    q_i = query_states[:, :, start_idx:end_idx]
                    k_i = key_states[:, :, start_idx:end_idx]
                    v_i = value_states[:, :, start_idx:end_idx]
                    a_i = attn_mask[:, :, :, start_idx:end_idx]

                    output_i = self.fused_scaled_dot_product_attention(q_i, k_i, v_i, a_i, 0.0, False, None, "None")
                    output_i = output_i.transpose(1, 2)
                    outputs.append(output_i)
                attn_output = torch.cat(outputs, dim=1)
            else:
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states, attn_mask, dropout_p=0.0
                )
                attn_output = attn_output.transpose(1, 2)
        else:
            fullatt_block_attn_mask = attn_mask

            (batch_size, _, seq_len_N_t, _) = query_states.shape
            (batch_size, _, seq_len_N_s, _) = key_states.shape
            mask_shape = (batch_size, 1, seq_len_N_t, seq_len_N_s)
            attn_mask = fullatt_block_attn_mask.reshape(batch_size, 1, seq_len_N_t, seq_len_N_s, -1)[:, :, :, :, 0]
            assert attn_mask.shape == mask_shape

            if query_states.shape[2] <= 65536:
                if FusedSDPA is not None and use_flash_attention:  # need to investigate this crosspoint
                    attn_output = FusedSDPA.apply(
                        query_states, key_states, value_states, attn_mask, 0.0, False, None, "None"
                    )
                else:
                    attn_output = F.scaled_dot_product_attention(
                        query_states, key_states, value_states, attn_mask, dropout_p=0.0
                    )
            else:
                raise ValueError("To be implemented for long sequence")

            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.squeeze(0)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


# from: https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L292
class GaudiQwen2_5_VLVisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__(config, attn_implementation)
        self.attn = GaudiQwen2_5_VLVisionAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Copied from https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L300
        - add new args use_flash_attention
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            attn_mask=attn_mask,
            use_flash_attention=use_flash_attention,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class GaudiQwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):
    """
    Here we add new methods and overwrite some of the methods of
    Qwen2_5_VisionTransformerPretrainedModel to make the model more friendly
    to static shapes. Specifically, we split the forward  method into:
      - pre_attn (dynamic)
      - forward (static shape)
      - post_attn (dynamic)
    and we should call get_image_embeds instead of forward, allowing
    the forward method to run with HPU_Graphs, whereas the
    pre_attn and post_attn methods are allowed to be dynamic.
    """

    def pad_multimodal_data_wh(self, pixel_values, image_grid_thw, constant_value=0):
        self.vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size
        grid_t, grid_h, grid_w = image_grid_thw[0]
        # x has already been flatted by spatial_merge_unit
        flatten_grid_h = grid_h // self.spatial_merge_size
        flatten_grid_w = grid_w * self.spatial_merge_size
        # pad grid_w and grid_h
        padded_grid_h = (
            ceil(grid_h / self.spatial_merge_size / self.vit_merger_window_size)
            * self.spatial_merge_size
            * self.vit_merger_window_size
        )
        padded_grid_w = (
            ceil(grid_w / self.spatial_merge_size / self.vit_merger_window_size)
            * self.spatial_merge_size
            * self.vit_merger_window_size
        )
        flatten_padded_grid_h = padded_grid_h // self.spatial_merge_size
        flatten_padded_grid_w = padded_grid_w * self.spatial_merge_size
        pad_h = flatten_padded_grid_h - flatten_grid_h
        pad_w = flatten_padded_grid_w - flatten_grid_w
        # pad pixel_values
        _, embed_size = pixel_values.size()
        pixel_values = pixel_values.reshape(grid_t, flatten_grid_h, flatten_grid_w, -1)
        pixel_values = F.pad(pixel_values, (0, 0, 0, pad_w, 0, pad_h, 0, 0), "constant", constant_value)
        pixel_values = pixel_values.reshape(-1, embed_size)
        new_grid_thw = torch.Tensor([[grid_t, padded_grid_h, padded_grid_w]]).to(torch.int32).to(image_grid_thw.device)

        return pixel_values, new_grid_thw

    def pad_multimodal_data(self, pixel_values, image_grid_thw, constant_value=0):
        pixel_values, image_grid_thw = self.pad_multimodal_data_wh(pixel_values, image_grid_thw, constant_value)

        if pixel_values.shape[0] >= VISION_BUCKETS[-1]:
            desired_number_of_pixels = pixel_values.shape[0]
        else:
            for b in VISION_BUCKETS:
                if pixel_values.shape[0] <= b:
                    desired_number_of_pixels = b
                    break
        padding_len = desired_number_of_pixels - pixel_values.shape[0]
        if padding_len <= 0:
            return pixel_values, image_grid_thw
        logger_msg = (
            "[Multimodal] Padding current number pixel "
            + str(pixel_values.shape[0])
            + " to "
            + str(desired_number_of_pixels)
        )
        logger.debug(logger_msg)
        # needs to make sure padding_len is even
        assert padding_len % 64 == 0, "padding needs to be multiple of 64"
        pixel_values = F.pad(pixel_values, (0, 0, 0, padding_len), "constant", constant_value)
        image_grid_thw = torch.cat(
            [image_grid_thw, torch.tensor([[1, 8, padding_len // 8]], device=image_grid_thw.device)]
        )
        assert image_grid_thw.prod(-1).sum() == desired_number_of_pixels
        return pixel_values, image_grid_thw

    def pre_attn(self, x: torch.Tensor, grid_thw: torch.Tensor):
        seq_len = x.shape[0]

        rot_pos_emb, cu_window_seqlens, window_index, attention_mask = self.prepare_for_attn_cpu(seq_len, grid_thw)
        hidden_states, rot_pos_emb, attention_mask = self.pre_attn_hpu(
            x, rot_pos_emb, attention_mask, window_index, grid_thw
        )
        return hidden_states, rot_pos_emb, None, cu_window_seqlens, window_index, attention_mask.bool(), grid_thw

    def prepare_for_attn_cpu(self, seq_len, grid_thw: torch.Tensor):
        grid_thw_cpu = grid_thw.to("cpu")

        rotary_pos_emb = self.rot_pos_emb(grid_thw_cpu)

        # pad input tensors
        attention_mask = torch.ones(seq_len, 1)
        rotary_pos_emb, _ = self.pad_multimodal_data(rotary_pos_emb, grid_thw_cpu, -100)

        attention_mask, padded_grid_thw_cpu = self.pad_multimodal_data(attention_mask, grid_thw_cpu, 0)

        # windows attention
        window_index, cu_window_seqlens = self.get_window_index(padded_grid_thw_cpu)

        cu_window_seqlens = torch.tensor(cu_window_seqlens, device=self.device, dtype=torch.int32)

        rotary_pos_emb = rotary_pos_emb.to(device=self.device, dtype=self.dtype)
        attention_mask = attention_mask.bool().to(device=self.device)

        return rotary_pos_emb, cu_window_seqlens, window_index, attention_mask

    def pre_attn_hpu(self, x: torch.Tensor, rotary_pos_emb, attention_mask, window_index, grid_thw):
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        hidden_states, _ = self.pad_multimodal_data(hidden_states, grid_thw, 0)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        attention_mask = attention_mask.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        attention_mask = attention_mask[window_index, :]
        attention_mask = attention_mask.reshape(1, 1, 1, seq_len)

        return hidden_states, rotary_pos_emb, attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        fullattn_mask: Optional[torch.Tensor],
        windowattn_mask: Optional[torch.Tensor],
        rotary_pos_emb: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
        use_flash_attention: Optional[bool] = False,
    ) -> torch.Tensor:
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for layer_num, blk in enumerate(self.blocks):
            cu_seqlens = None if layer_num in self.fullatt_block_indexes else cu_window_seqlens
            attn_mask = fullattn_mask if layer_num in self.fullatt_block_indexes else windowattn_mask

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__,
                    hidden_states,
                    cu_seqlens,
                    None,
                    position_embeddings,
                    attn_mask=attn_mask,
                    use_flash_attention=use_flash_attention,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    attn_mask=attn_mask,
                    use_flash_attention=use_flash_attention,
                )

        return hidden_states

    def post_attn(
        self,
        hidden_states: torch.Tensor,
        window_index: torch.Tensor,
        grid_thw: torch.Tensor,
        grid_thw_padded: torch.Tensor,
    ):
        # adapter
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)

        hidden_states = hidden_states[reverse_indices, :]

        grid_t, gird_h, grid_w = grid_thw[0]
        llm_grid_h = gird_h // self.spatial_merge_size
        llm_grid_w = grid_w // self.spatial_merge_size
        grid_t, padded_grid_h, padded_grid_w = grid_thw_padded
        padded_llm_grid_h = padded_grid_h // self.spatial_merge_size
        padded_llm_grid_w = padded_grid_w // self.spatial_merge_size
        hidden_states = hidden_states[: grid_t * padded_llm_grid_h * padded_llm_grid_w]
        hidden_states = hidden_states.reshape(grid_t, padded_llm_grid_h, padded_llm_grid_w, -1)
        orig_hidden_states = hidden_states[:, :llm_grid_h, :llm_grid_w, :]
        orig_hidden_states = orig_hidden_states.reshape(grid_t * llm_grid_h * llm_grid_w, -1)

        return orig_hidden_states

    def get_image_embeds(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        use_flash_attention: Optional[bool] = False,
    ) -> torch.Tensor:
        offset = 0
        results = []
        # process each image one by one
        for img_idx in range(grid_thw.shape[0]):
            img_shape = grid_thw[img_idx, :].unsqueeze(0).clone()
            # For video, we process frames separately
            grid_t = grid_thw[img_idx, 0]
            img_shape[0, 0] = 1
            curr_img_size = img_shape.prod()
            for _ in torch.arange(0, grid_t):
                pixel_values_curr_img = pixel_values[offset : offset + curr_img_size, :]

                offset += curr_img_size

                (
                    pixel_values_curr_img_padded,
                    rot_pos_emb,
                    _,
                    cu_window_seqlens,
                    window_index,
                    attention_mask,
                    img_shape_padded,
                ) = self.pre_attn(pixel_values_curr_img, img_shape)

                fullatt_block_attn_mask = attention_mask[0, 0, :, :] * attention_mask[0, 0, 0, :].unsqueeze(1)

                htcore.mark_step()
                hidden_states = self.forward(
                    pixel_values_curr_img_padded,
                    rotary_pos_emb=rot_pos_emb,
                    fullattn_mask=fullatt_block_attn_mask,
                    windowattn_mask=attention_mask,
                    cu_window_seqlens=cu_window_seqlens,
                    use_flash_attention=use_flash_attention,
                )
                htcore.mark_step()

                image_embeds = self.post_attn(hidden_states, window_index, img_shape, img_shape_padded[0])
                # slice image_embeds to remove the padded parts
                pad_index = img_shape_padded[0].prod() // self.spatial_merge_unit
                results += [image_embeds[:pad_index, :]]

        results_cat = torch.concat(results)
        image_embeds = results_cat
        return image_embeds


class GaudiQwen2_5_VLAttention(Qwen2_5_VLAttention):
    """
    Qwen2_5 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def __init__(self, config: Qwen2_5_VLTextConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx=layer_idx)
        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA) if FusedSDPA else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        use_flash_attention: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Copied from https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L656
        The only differences are:
        - add new args use_flash_attention
        - add FusedSDPA
        """

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings

        query_states, key_states = apply_customized_rope(
            query_states, key_states, cos, sin, mrope_section=self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False
        if FusedSDPA is not None and use_flash_attention:
            attn_output = self.fused_scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                causal_mask,
                self.attention_dropout if self.training else 0.0,
                is_causal,
                None,  # scale
                "None",  #'fast'
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


# from: https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L709
class GaudiQwen2_5_VLDecoderLayer(Qwen2_5_VLDecoderLayer):
    def __init__(self, config: Qwen2_5_VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GaudiQwen2_5_VLAttention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        use_flash_attention: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Copied from https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L726
        The only differences are:
        - add new kwargs use_flash_attention
        """
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            use_flash_attention=use_flash_attention,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# from: https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L793
class GaudiQwen2_5_VLTextModel(Qwen2_5_VLTextModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_flash_attention: Optional[bool] = False,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # NOTE: we need to pass text position ids for packing. Qwen2-VL uses 3D positions
        # where each dim indicates visual spatial positions for temporal/height/width grids.
        # There are two scenarios when FA2-like packed masking might be activated.
        # 1. User specifically passed packed `position_ids` and no attention mask.
        #    In this case we expect the useer to create correct position ids for all 3 grids
        #    and prepend text-only position ids to it. The final tensor will be [4, bs, seq-len]
        # 2. User runs forward with no attention mask and no position ids. In this case, position ids
        #    are prepared by the model (`get_rope_index`) as `[4, bs, seq-len]` tensor. Text-only positions are
        #    prepended by us when creating positions so that the mask is constructed correctly. NOTE: failing to pass
        #    text-only positions will cause incorrect mask construction, do not change `prepare_input_for_generation`
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=text_position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_flash_attention=use_flash_attention,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# from: https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L950
class GaudiQwen2_5_VLModel(Qwen2_5_VLModel):
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
        use_flash_attention: Optional[bool] = False,
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual.get_image_embeds(
            pixel_values_videos, grid_thw=video_grid_thw, use_flash_attention=use_flash_attention
        )
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return video_embeds

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        use_flash_attention: Optional[bool] = False,
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual.get_image_embeds(
            pixel_values, grid_thw=image_grid_thw, use_flash_attention=use_flash_attention
        )
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
        *kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Copied from Qwen2_5_VLModel https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1234
        The only differences are:
        - add new arg use_flash_attention
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(
                pixel_values, image_grid_thw, use_flash_attention=use_flash_attention
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw, use_flash_attention=use_flash_attention
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids += delta.to(position_ids.device)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            use_flash_attention=use_flash_attention,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

        return output if return_dict else output.to_tuple()


# https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1377
class GaudiQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    _supports_static_cache = True

    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self._supports_cache_class = True

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        token_idx: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        """
        Adapted from https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1422
        The only differences are:
        - add new arg token_idx
        - add new arg use_flash_attention
        - add Gaudi Example
        """
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor
        >>> from optimum.habana.transformers.models import GaudiQwen2_5_VLForConditionalGeneration
        >>> from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
        >>> from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        >>> adapt_transformers_to_gaudi()
        >>> model = GaudiQwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> model = model.to("hpu")
        >>> wrap_in_hpu_graph(model)
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], return_tensors="pt")
        >>> inputs = inputs.to("hpu")
        >>> generate_kwargs = {
                "lazy_mode": True,
                "hpu_graphs": True,
                "use_cache": True,
                "cache_implementation": "static",
                "use_flash_attention": True
            }
        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=30, **generate_kwargs)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene in what appears to be a Chinatown area. The focal point is a red stop sign on the left side of the..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            use_flash_attention=use_flash_attention,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        """
        Copied from https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1869
        The only differences are:
        - handle new args token_idx
        - handle new args use_flash_attention
        """

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        token_idx = kwargs.get("token_idx", None)
        use_flash_attention = kwargs.get("use_flash_attention", False)

        if token_idx is not None:
            if isinstance(past_key_values, StaticCache):
                if cache_position.shape[0] > 1:
                    input_ids = input_ids[:, :token_idx]
                    attention_mask = attention_mask[:, :token_idx]
                    cache_position = cache_position[:token_idx]
                else:
                    # over-write with token idx
                    cache_position[0] = token_idx - 1
        model_inputs.update(
            {
                "token_idx": token_idx,
                "use_flash_attention": use_flash_attention,
            }
        )

        # Qwen2-5-VL position_ids are prepared with rope_deltas
        if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            if cache_position[0] == 0 or self.model.rope_deltas is None:
                vision_positions, rope_deltas = self.model.get_rope_index(
                    model_inputs.get("input_ids", None),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            elif "position_ids" in model_inputs:
                position_ids = model_inputs["position_ids"][1:, :, :]
                delta = self.model.rope_deltas
                delta = delta.repeat_interleave(position_ids.shape[1] // delta.shape[0], dim=0)
                vision_positions = position_ids + delta.expand_as(position_ids)
                vision_positions = vision_positions.expand(3, vision_positions.shape[1], -1)

            # Concatenate "text + vision" positions into [4, bs, seq-len]
            if "position_ids" not in model_inputs:
                text_positions = torch.arange(input_ids, device=input_ids.device)[None, None, :]
                model_inputs["position_ids"] = torch.cat([text_positions, vision_positions], dim=0)

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs
