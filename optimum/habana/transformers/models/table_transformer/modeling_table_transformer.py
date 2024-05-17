import torch
import torch.nn as nn


def gaudi_table_transformer_conv_encoder_forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
    """
    Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/table_transformer/modeling_table_transformer.py#L319
    Only differences:
    - Changed indexing after interpolate from `[0]` to `[:, 0]`

    """
    # send pixel_values through the model to get list of feature maps
    features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps

    out = []
    for feature_map in features:
        # downsample pixel_mask to match shape of corresponding feature_map
        mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[:, 0]
        out.append((feature_map, mask))
    return out
