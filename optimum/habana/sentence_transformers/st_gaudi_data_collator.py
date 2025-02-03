import math
from typing import Any, Dict, List

import torch


def st_gaudi_data_collator_call(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collator for a SentenceTransformers model."""

    column_names = list(features[0].keys())

    # We should always be able to return a loss, label or not:
    batch = {"return_loss": True}

    if "dataset_name" in column_names:
        column_names.remove("dataset_name")
        batch["dataset_name"] = features[0]["dataset_name"]

    if tuple(column_names) not in self._warned_columns:
        self.maybe_warn_about_column_order(column_names)

    # Extract the label column if it exists
    for label_column in self.valid_label_columns:
        if label_column in column_names:
            batch["label"] = torch.tensor([row[label_column] for row in features])
            column_names.remove(label_column)
            break

    # Extract the feature columns
    cnt = 0
    cnt1 = 0
    power2_len = [0, 0]
    for column_name in column_names:
        # If the prompt length has been set, we should add it to the batch
        if column_name.endswith("_prompt_length") and column_name[: -len("_prompt_length")] in column_names:
            batch[column_name] = torch.tensor([row[column_name] for row in features], dtype=torch.int)
            continue

        tokenized = self.tokenize_fn([row[column_name] for row in features])
        for key, value in tokenized.items():
            curr_tokenize_len = value.shape
            if curr_tokenize_len[1] > 4096:
                power2_len[cnt1] = math.ceil(curr_tokenize_len[1] / 128) * 128
            else:
                power2_len[cnt1] = 2 ** math.ceil(math.log2(curr_tokenize_len[1]))
            additional_pad_len = power2_len[cnt1] - curr_tokenize_len[1]
            if (cnt1 == 1) and (power2_len[0] == power2_len[1]):
                additional_pad_len += 1

            batch[f"{column_name}_{key}"] = torch.cat(
                (
                    value,
                    torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                ),
                -1,
            )
        cnt += 1
        cnt1 = cnt & 1
    return batch
