import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from transformers.loss.loss_deformable_detr import center_to_corners_format, generalized_box_iou
from transformers.utils import is_accelerate_available


if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import reduce


def gaudi_DetrConvModel_forward(self, pixel_values, pixel_mask):
    """
    Copied from modeling_detr: https://github.com/huggingface/transformers/blob/main/src/transformers/models/detr/modeling_detr.py#L398
    The modifications are:
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


def gaudi_DetrLoss_get_targets_without_no_objects(self, targets):
    target_copy = targets.copy()
    tcopy_iter = iter(target_copy)
    for v in targets:
        entries = []
        for x in v["class_labels"].to("cpu").numpy():
            if x != self.num_classes:
                entries.append(x)
        y = next(tcopy_iter)
        y["class_labels"] = torch.as_tensor(entries, dtype=torch.int64)
        y["boxes"] = v["boxes"].to("cpu")[0 : len(y["class_labels"])]
    return target_copy


@torch.no_grad()
def gaudi_DetrHungarianMatcher_forward(self, outputs, targets):
    """
    Copied from https://github.com/huggingface/transformers/tree/v4.40.2
    https://github.com/huggingface/transformers/blob/4fdf58afb72b0754da30037fc800b6044e7d9c99/src/transformers/models/detr/modeling_detr.py#L2287
    The modifications are:
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
    # The 1 is a constant that doesn't change the matching, it can be omitted.
    class_cost = -out_prob[:, target_ids]

    # HPU Eager mode requires tensors to be on the same device
    out_bbox = out_bbox.to(target_bbox.device)
    class_cost = class_cost.to(target_bbox.device)

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
    The modifications are:
        - Move cross entropy computation to CPU
    """
    if "logits" not in outputs:
        raise KeyError("No logits were found in the outputs")
    source_logits = outputs["logits"]

    idx = self._get_source_permutation_idx(indices)
    target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = torch.full(
        source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=target_classes_o.device
    )
    target_classes[idx] = target_classes_o

    if source_logits.device == torch.device("cpu"):
        loss_ce = nn.functional.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
    else:
        source_logits_cpu = source_logits.to("cpu").float()
        target_classes_cpu = target_classes.to("cpu")
        empty_weight_cpu = self.empty_weight.to("cpu").float()
        loss_ce_cpu = nn.functional.cross_entropy(
            source_logits_cpu.transpose(1, 2), target_classes_cpu, empty_weight_cpu
        )
        loss_ce = loss_ce_cpu.to("hpu")
    losses = {"loss_ce": loss_ce}

    return losses


def gaudi_DetrLoss_loss_boxes(self, outputs, targets, indices, num_boxes):
    """
    Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.
    Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
    are expected in format (center_x, center_y, w, h), normalized by the image size.
    """
    if "pred_boxes" not in outputs:
        raise KeyError("No predicted boxes found in outputs")
    idx = self._get_source_permutation_idx(indices)
    source_boxes = outputs["pred_boxes"][idx]
    target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

    # HPU eager mode requires both source and target tensors to be on same device
    source_boxes = source_boxes.to(target_boxes.device)
    loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

    losses = {}
    losses["loss_bbox"] = loss_bbox.sum() / num_boxes

    loss_giou = 1 - torch.diag(
        generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
    )
    losses["loss_giou"] = loss_giou.sum() / num_boxes
    return losses


@torch.no_grad()
def gaudi_DetrLoss_loss_cardinality(self, outputs, targets, indices, num_boxes):
    """
    Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.
    This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
    """
    logits = outputs["logits"]
    target_lengths = torch.as_tensor([len(v) for v in targets], device="cpu")
    # Count the number of predictions that are NOT "no-object" (which is the last class)
    card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
    card_err = nn.functional.l1_loss(card_pred.to("cpu").float(), target_lengths.float())
    losses = {"cardinality_error": card_err}
    return losses


def gaudi_DetrLoss_forward(self, outputs, targets):
    """
    This performs the loss computation.
    Args:
            outputs (`dict`, *optional*):
            Dictionary of tensors, see the output specification of the model for the format.
            targets (`List[dict]`, *optional*):
            List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
            losses applied, see each loss' doc.
    """
    outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

    # Retrieve the matching between the outputs of the last layer and the targets
    device = outputs["logits"].device
    target_copy = self.gaudi_DetrLoss_get_targets_without_no_objects(targets)
    indices = self.matcher(outputs_without_aux, target_copy)

    # Compute the average number of target boxes across all nodes, for normalization purposes
    num_boxes = sum(len(t["class_labels"]) for t in target_copy)
    num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
    world_size = 1
    if is_accelerate_available():
        if PartialState._shared_state != {}:
            num_boxes = reduce(num_boxes)
            world_size = PartialState().num_processes
    num_boxes = torch.clamp(num_boxes / world_size, min=1).item()
    # Compute all the requested losses
    losses = {}
    for loss in self.losses:
        losses.update(self.get_loss(loss, outputs, target_copy, indices, num_boxes))

    # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
    if "auxiliary_outputs" in outputs:
        for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
            indices = self.matcher(auxiliary_outputs, target_copy)
            for loss in self.losses:
                if loss == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                l_dict = self.get_loss(loss, auxiliary_outputs, target_copy, indices, num_boxes)
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)

    for k in losses.keys():
        losses[k] = losses[k].to(device)
    return losses
