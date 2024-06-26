from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import torch
import math


def st_gaudi_data_collator_call(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    #print(f"--------------- SSS.2 Gaudi DataCollator ---------- \n\n ")
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

    #print(f"-----D.111 ----- DataCollator:  call () ------------tokenize_fn={self.tokenize_fn}\n\n")
    # Extract the feature columns
    cnt = 0
    get_pad_flag = False
    for column in columns:

        tokenized = self.tokenize_fn([row[column] for row in features]) 
                #padding="max_length", return_tensors="pt", max_length=100,        )
        for key, value in tokenized.items():
            #print(f"-----D.222 ----- DataCollator:  call () ------------tokenized = {tokenized}\n\n")
            #print(f"-----D.222 ----- DataCollator:  call () ------------key, value = {key, value}\n\n")
            
            #batch[f"{column}_{key}"] = value
            if(get_pad_flag==False):
                curr_tokenize_len = value.shape
                if curr_tokenize_len[1] > 4096:
                    additional_pad_len = math.ceil(curr_tokenize_len[1] / 128) * 128 - curr_tokenize_len[1]
                else:
                    additional_pad_len = 2 ** math.ceil(math.log2(curr_tokenize_len[1])) - curr_tokenize_len[1]


            """
            if cnt%2 ==0:
                additional_pad_len = additional_pad_len - curr_tokenize_len[1]
            else: 
                additional_pad_len = additional_pad_len - curr_tokenize_len[1] + 1
            """

            batch[f"{column}_{key}"] = torch.cat(
                    (
                        value,
                        torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                    ),
                    -1,
                )
            get_pad_flag = True

        cnt=cnt+1

        if(cnt%2 ==1):
            additional_pad_len = additional_pad_len + 1

    return batch
    """
                curr_tokenize_len = features["input_ids"].shape
                additional_pad_len = 2 ** math.ceil(math.log2(curr_tokenize_len[1])) - curr_tokenize_len[1]
                features["input_ids"] = torch.cat(
                    (
                        features["input_ids"],
                        torch.ones((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                    ),
                    -1,
                )
                features["attention_mask"] = torch.cat(
                    (
                        features["attention_mask"],
                        torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                    ),
                    -1,
                )
                if "token_type_ids" in features:
                    features["token_type_ids"] = torch.cat(
                        (
                            features["token_type_ids"],
                            torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                        ),
                        -1,
                    )
    """

