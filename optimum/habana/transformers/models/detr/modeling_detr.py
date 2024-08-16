def gaudi_DetrConvModel_forward(self, pixel_values, pixel_mask):
    """
    Copied from modeling_detr: https://github.com/huggingface/transformers/blob/main/src/transformers/models/detr/modeling_detr.py#L398
    The modications are:
        - Use CPU to calculate the position_embeddings and transfer back to HPU
    """

    # send pixel_values and pixel_mask through backbone to get list of (feature_map, pixel_mask) tuples
    out = self.conv_encoder(pixel_values, pixel_mask)
    pos = []
    self.position_embedding = self.position_embedding.to("cpu")

    for feature_map, mask in out:
        # position encoding
        feature_map = feature_map.to("cpu")
        mask = mask.to("cpu")
        pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype).to("hpu"))

    return out, pos
