

import torch
from torch import nn
import transformers
from transformers.models.detr.modeling_detr import generalized_box_iou, center_to_corners_format
from scipy.optimize import linear_sum_assignment


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


@torch.no_grad()
def gaudi_DetrHungarianMatcher_forward(self, outputs, targets):
    """
    Copied from https://github.com/huggingface/transformers/tree/v4.40.2 
    https://github.com/huggingface/transformers/blob/4fdf58afb72b0754da30037fc800b6044e7d9c99/src/transformers/models/detr/modeling_detr.py#L2287
    The modications are:
        - Convert cost_matrix on HPU to float32 before moving it to CPU
    """
    batch_size, num_queries = outputs["logits"].shape[:2]

    # We flatten to compute the cost matrices in a batch
    out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
    out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

    # Also concat the target labels and boxes
    target_ids = torch.cat([v["class_labels"] for v in targets])
    target_bbox = torch.cat([v["boxes"] for v in targets])

    # Compute the classification cost. Contrary to the loss, we don't use the NLL,
    # but approximate it in 1 - proba[target class].
    # The 1 is a constant that doesn't change the matching, it can be ommitted.
    class_cost = -out_prob[:, target_ids]

    # Compute the L1 cost between boxes
    bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

    # Compute the giou cost between boxes
    giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

    # Final cost matrix
    cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
    cost_matrix = cost_matrix.view(batch_size, num_queries, -1).to(torch.float32).cpu()

    sizes = [len(v["boxes"]) for v in targets]
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def gaudi_DetrLoss_loss_labels(self, outputs, targets, indices, num_boxes):
    """
    Copied from https://github.com/huggingface/transformers/tree/v4.40.2 
    https://github.com/huggingface/transformers/blob/4fdf58afb72b0754da30037fc800b6044e7d9c99/src/transformers/models/detr/modeling_detr.py#L2074
    The modications are:
        - Move cross entropy computation to CPU
    """
    if "logits" not in outputs:
        raise KeyError("No logits were found in the outputs")
    source_logits = outputs["logits"]

    idx = self._get_source_permutation_idx(indices)
    target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = torch.full(
        source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
    )
    target_classes[idx] = target_classes_o

    if source_logits.device == torch.device("cpu") :
        loss_ce = nn.functional.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
    else :
        source_logits_cpu = source_logits.to('cpu').float()
        target_classes_cpu = target_classes.to('cpu')
        empty_weight_cpu = self.empty_weight.to('cpu').float()
        loss_ce_cpu = nn.functional.cross_entropy(source_logits_cpu.transpose(1, 2), target_classes_cpu, empty_weight_cpu)
        loss_ce = loss_ce_cpu.to('hpu')
    losses = {"loss_ce": loss_ce}

    return losses
