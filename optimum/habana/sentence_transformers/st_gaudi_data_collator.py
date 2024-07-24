from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import torch
import math


def st_gaudi_data_collator_call(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """ data collator for sentence transformer """

    columns = list(features[0].keys())

    # We should always be able to return a loss, label or not:
    batch = {"return_loss": True}

    if "dataset_name" in columns:
        columns.remove("dataset_name")
        batch["dataset_name"] = features[0]["dataset_name"]

    # Extract the label column if it exists
    for label_column in self.valid_label_columns:
        if label_column in columns:
            batch["label"] = torch.tensor([row[label_column] for row in features])
            columns.remove(label_column)
            break

    # Extract the feature columns
    cnt = 0
    power2_len=[0, 0]
    for column in columns:
        tokenized = self.tokenize_fn([row[column] for row in features]) 
        for key, value in tokenized.items():
            curr_tokenize_len = value.shape
            if curr_tokenize_len[1] > 4096:
                power2_len[cnt%2] = math.ceil(curr_tokenize_len[1] / 128) * 128
                additional_pad_len = math.ceil(curr_tokenize_len[1] / 128) * 128 - curr_tokenize_len[1]
            else:
                power2_len[cnt%2] = 2 ** math.ceil(math.log2(curr_tokenize_len[1]))
                additional_pad_len = 2 ** math.ceil(math.log2(curr_tokenize_len[1])) - curr_tokenize_len[1]

            if(cnt%2==1) and (power2_len[0]==power2_len[1]):
                additional_pad_len = additional_pad_len + 1

            batch[f"{column}_{key}"] = torch.cat(
                    (
                        value,
                        torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                    ),
                    -1,
                )
        cnt=cnt+1
    return batch
