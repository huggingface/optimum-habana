import torch
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings


class GaudiCLIPVisionEmbeddings(CLIPVisionEmbeddings):
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        # if HQT quantization enabled, remove the explicit cast to float8 to avoid HQT casting error
        if "float8" in str(target_dtype) and pixel_values.device.type == "hpu":
            target_dtype = torch.bfloat16
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
