from typing import Optional, Union

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPTextConfig, CLIPVisionConfig
from transformers.models.clip.modeling_clip import (
    CLIPMLP,
    CLIPAttention,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPVisionEmbeddings,
    CLIPVisionModel,
    CLIPVisionTransformer,
)

from ..modeling_all_models import Matmul


try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None


class GaudiCLIPVisionEmbeddings(CLIPVisionEmbeddings):
    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        if not interpolate_pos_encoding and (height != self.image_size or width != self.image_size):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size}*{self.image_size})."
            )
        target_dtype = self.patch_embedding.weight.dtype
        # if HQT quantization enabled, remove the explicit cast to float8 to avoid HQT casting error
        if "float8" in str(target_dtype) and pixel_values.device.type == "hpu":
            target_dtype = torch.bfloat16
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        self._hpu_kernel_fsdpa = fusedSDPA

    def forward(self, query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode):
        return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode)


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim=None, invAttnHead=None):
        return torch.nn.functional.softmax(x, dim, dtype=torch.float32)


class GaudiCLIPAttention(CLIPAttention):
    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        super().__init__(config=config)
        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA) if FusedSDPA else None
        self.bmm1 = Matmul()
        self.bmm2 = Matmul()
        self.softmax = Softmax()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Copied from CLIPAttention.forward: https://github.com/huggingface/transformers/blob/ab0f050b42d903f34d6eb97f3f8c0c07f0517ad2/src/transformers/models/clip/modeling_clip.py
        The only differences are:
        - add new args use_flash_attention to enable FusedSDPA
        - add new args flash_attention_recompute
        - add new args flash_attention_fast_softmax
        """
        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        # CLIP text model uses both `causal_attention_mask` and `attention_mask`
        # in case FA2 kernel is called, `is_causal` should be inferred from `causal_attention_mask`
        if self.config._attn_implementation == "flash_attention_2":
            self.is_causal = causal_attention_mask is not None
        else:
            if attention_mask is not None and causal_attention_mask is not None:
                attention_mask = attention_mask + causal_attention_mask
            elif causal_attention_mask is not None:
                attention_mask = causal_attention_mask

        if FusedSDPA and use_flash_attention:
            import habana_frameworks.torch.hpu as ht

            softmax_mode = "fast" if flash_attention_fast_softmax else "None"
            with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                attn_output = self.fused_scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    attention_mask,
                    self.dropout,
                    False,
                    1,
                    softmax_mode,
                )
        else:
            attn_weights = self.bmm1(queries, keys.transpose(-1, -2)) * self.scale
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = self.softmax(attn_weights, dim=-1).to(queries.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

            attn_output = self.bmm2(attn_weights, values)

        # attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class GaudiCLIPEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        super(CLIPEncoderLayer, self).__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = GaudiCLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
    ) -> tuple[torch.FloatTensor]:
        """
        Copied from CLIPEncoderLayer.forward: https://github.com/huggingface/transformers/blob/ab0f050b42d903f34d6eb97f3f8c0c07f0517ad2/src/transformers/models/clip/modeling_clip.py
        The only differences are:
        - add new args use_flash_attention
        - add new args flash_attention_recompute
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class GaudiCLIPEncoder(CLIPEncoder):
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
    ) -> BaseModelOutput:
        """
        Copied from CLIPEncoder.forward: https://github.com/huggingface/transformers/blob/ab0f050b42d903f34d6eb97f3f8c0c07f0517ad2/src/transformers/models/clip/modeling_clip.py
        The only differences are:
        - add new args use_flash_attention
        - add new args flash_attention_recompute
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
                use_flash_attention=use_flash_attention,
                flash_attention_recompute=flash_attention_recompute,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class GaudiCLIPVisionTransformer(CLIPVisionTransformer):
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
    ) -> BaseModelOutputWithPooling:
        """
        Copied from CLIPVisionTransformer.forward: https://github.com/huggingface/transformers/blob/ab0f050b42d903f34d6eb97f3f8c0c07f0517ad2/src/transformers/models/clip/modeling_clip.py
        The only differences are:
        - add new args use_flash_attention
        - add new args flash_attention_recompute
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class GaudiCLIPVisionModel(CLIPVisionModel):
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        """
        Copied from CLIPVisionModel.forward: https://github.com/huggingface/transformers/blob/ab0f050b42d903f34d6eb97f3f8c0c07f0517ad2/src/transformers/models/clip/modeling_clip.py
        The only differences are:
        - add new args use_flash_attention
        - add new args flash_attention_recompute
        """

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
        )
