#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 Intel Corporation and The HuggingFace Team. All rights reserved.
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

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers.loss.loss_rt_detr import center_to_corners_format, generalized_box_iou


@torch.no_grad()
def gaudi_RTDetrHungarianMatcher_forward(self, outputs, targets):
    """Performs the matching

    Params:
        outputs: This is a dict that contains at least these entries:
             "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
             "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

        targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
             "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                       objects in the target) containing the class labels
             "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

    Returns:
        A list of size batch_size, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
        For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)

    Copied from RTDetrHungarianMatcher.forward: https://github.com/huggingface/transformers/blob/v4.49-release/src/transformers/loss/loss_rt_detr.py#L73
        The only differences are:
        - move output tensors to acceleration device
    """
    batch_size, num_queries = outputs["logits"].shape[:2]

    # We flatten to compute the cost matrices in a batch
    out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
    # Also concat the target labels and boxes
    target_ids = torch.cat([v["class_labels"] for v in targets])
    target_bbox = torch.cat([v["boxes"] for v in targets])
    # Compute the classification cost. Contrary to the loss, we don't use the NLL,
    # but approximate it in 1 - proba[target class].
    # The 1 is a constant that doesn't change the matching, it can be ommitted.
    if self.use_focal_loss:
        out_prob = F.sigmoid(outputs["logits"].flatten(0, 1))
        out_prob = out_prob[:, target_ids]
        neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
        class_cost = pos_cost_class - neg_cost_class
    else:
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        class_cost = -out_prob[:, target_ids]

    # Compute the L1 cost between boxes
    bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)
    # Compute the giou cost betwen boxes
    giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))
    # Compute the final cost matrix
    cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
    cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

    sizes = [len(v["boxes"]) for v in targets]
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]

    return [
        (
            torch.as_tensor(i, dtype=torch.int64, device=target_ids.device),
            torch.as_tensor(j, dtype=torch.int64, device=target_ids.device),
        )
        for i, j in indices
    ]
