from typing import Optional, Tuple

import torch


def gaudi_owlvitclasspredictionhead_forward(
    self,
    image_embeds: torch.FloatTensor,
    query_embeds: Optional[torch.FloatTensor],
    query_mask: Optional[torch.Tensor],
) -> Tuple[torch.FloatTensor]:
    """
    Copied from modeling_owlvit: https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/owlvit/modeling_owlvit.py#L1233
    The only modification is:
        - Replace the torch.where by torch.clamp
    """

    image_class_embeds = self.dense0(image_embeds)
    if query_embeds is None:
        device = image_class_embeds.device
        batch_size, num_patches = image_class_embeds.shape[:2]
        pred_logits = torch.zeros((batch_size, num_patches, self.query_dim)).to(device)
        return (pred_logits, image_class_embeds)

    # Normalize image and text features
    image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
    query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

    # Get class predictions
    pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)

    # Apply a learnable shift and scale to logits
    logit_shift = self.logit_shift(image_embeds)
    logit_scale = self.logit_scale(image_embeds)
    logit_scale = self.elu(logit_scale) + 1
    pred_logits = (pred_logits + logit_shift) * logit_scale

    if query_mask is not None:
        if query_mask.ndim > 1:
            query_mask = torch.unsqueeze(query_mask, dim=-2)

        pred_logits = pred_logits.to(torch.float64)
        pred_logits = torch.clamp(pred_logits, min=-1e6)
        pred_logits = pred_logits.to(torch.float32)

    return (pred_logits, image_class_embeds)
