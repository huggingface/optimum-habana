from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.utils import USE_PEFT_BACKEND


try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None

#  FusedScaledDotProductAttention
class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        self._hpu_kernel_fsdpa = fusedSDPA

    def forward(self, query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode):
        return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode)



def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/embeddings.py#L697
    """
    cos_, sin_ = freqs_cis  # [S, D]

    cos = cos_[None, None]
    sin = sin_[None, None]
    cos, sin = cos.to(x.device), sin.to(x.device)

    x = torch.ops.hpu.rotary_pos_embedding(x, sin, cos, None, 0, 1)

    return x

class CogVideoXAttnProcessorGaudi:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA) if FusedSDPA else None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        print(f'run gaudi transformer attention_processor with fused SDPA!')
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = self.fused_scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_casual=False, scale=None, softmax_mode='fast'
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states

def cogvideoXTransformerForwardGaudi(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: Union[int, float, torch.LongTensor],
    timestep_cond: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
):
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    batch_size, num_frames, channels, height, width = hidden_states.shape

    # 1. Time embedding
    timesteps = timestep
    t_emb = self.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)

    # 2. Patch embedding
    hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
    hidden_states = self.embedding_dropout(hidden_states)

    text_seq_length = encoder_hidden_states.shape[1]
    encoder_hidden_states = hidden_states[:, :text_seq_length]
    hidden_states = hidden_states[:, text_seq_length:]

    print(f'baymax debug run gaudi transformer forward!')
    # 3. Transformer blocks
    for i, block in enumerate(self.transformer_blocks):
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                emb,
                image_rotary_emb,
                **ckpt_kwargs,
            )
        else:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
            )
        htcore.mark_step()

    if not self.config.use_rotary_positional_embeddings:
        # CogVideoX-2B
        hidden_states = self.norm_final(hidden_states)
    else:
        # CogVideoX-5B
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states[:, text_seq_length:]

    # 4. Final block
    hidden_states = self.norm_out(hidden_states, temb=emb)
    hidden_states = self.proj_out(hidden_states)

    # 5. Unpatchify
    # Note: we use `-1` instead of `channels`:
    #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
    #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
    p = self.config.patch_size
    output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
    output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)

import habana_frameworks.torch.core as htcore
def tiled_decode_gaudi(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
    r"""
    Decode a batch of images using a tiled decoder.

    Args:
        z (`torch.Tensor`): Input batch of latent vectors.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

    Returns:
        [`~models.vae.DecoderOutput`] or `tuple`:
            If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
            returned.
    """
    # Rough memory assessment:
    #   - In CogVideoX-2B, there are a total of 24 CausalConv3d layers.
    #   - The biggest intermediate dimensions are: [1, 128, 9, 480, 720].
    #   - Assume fp16 (2 bytes per value).
    # Memory required: 1 * 128 * 9 * 480 * 720 * 24 * 2 / 1024**3 = 17.8 GB
    #
    # Memory assessment when using tiling:
    #   - Assume everything as above but now HxW is 240x360 by tiling in half
    # Memory required: 1 * 128 * 9 * 240 * 360 * 24 * 2 / 1024**3 = 4.5 GB

    print('run gaudi pipelined tiled decode!')
    batch_size, num_channels, num_frames, height, width = z.shape

    overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor_height))
    overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor_width))
    blend_extent_height = int(self.tile_sample_min_height * self.tile_overlap_factor_height)
    blend_extent_width = int(self.tile_sample_min_width * self.tile_overlap_factor_width)
    row_limit_height = self.tile_sample_min_height - blend_extent_height
    row_limit_width = self.tile_sample_min_width - blend_extent_width
    frame_batch_size = self.num_latent_frames_batch_size

    # Split z into overlapping tiles and decode them separately.
    # The tiles have an overlap to avoid seams between tiles.
    rows = []
    for i in range(0, height, overlap_height):
        row = []
        for j in range(0, width, overlap_width):
            num_batches = max(num_frames // frame_batch_size, 1)
            conv_cache = None
            time = []

            for k in range(num_batches):
                remaining_frames = num_frames % frame_batch_size
                start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                end_frame = frame_batch_size * (k + 1) + remaining_frames
                tile = z[
                    :,
                    :,
                    start_frame:end_frame,
                    i : i + self.tile_latent_min_height,
                    j : j + self.tile_latent_min_width,
                ].clone()
                if self.post_quant_conv is not None:
                    tile = self.post_quant_conv(tile)
                tile, conv_cache = self.decoder(tile, conv_cache=conv_cache)
                time.append(tile.clone())
                htcore.mark_step()

            row.append(torch.cat(time, dim=2))
        rows.append(row)

    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            # blend the above tile and the left tile
            # to the current tile and add the current tile to the result row
            if i > 0:
                tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
            if j > 0:
                tile = self.blend_h(row[j - 1], tile, blend_extent_width)
            result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
        result_rows.append(torch.cat(result_row, dim=4))

    dec = torch.cat(result_rows, dim=3)

    if not return_dict:
        return (dec,)

    return DecoderOutput(sample=dec)


def CogVideoXCausalConv3dforwardGaudi(self, inputs: torch.Tensor, conv_cache: Optional[torch.Tensor] = None) -> torch.Tensor:
    #print('run gaudi CogVideoXCausalConv3d forward!')
    inputs = self.fake_context_parallel_forward(inputs, conv_cache)
    #conv_cache = inputs[:, :, -self.time_kernel_size + 1 :].clone()

    padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
    inputs_pad = F.pad(inputs, padding_2d, mode="constant", value=0)

    output = self.conv(inputs_pad)
    if self.time_kernel_size>1:
        if conv_cache is not None and conv_cache.shape == inputs[:, :, -self.time_kernel_size + 1:].shape:
            conv_cache.copy_(inputs[:, :, -self.time_kernel_size + 1:])
        else:
            conv_cache = inputs[:, :, -self.time_kernel_size + 1:].clone()
    return output, conv_cache

