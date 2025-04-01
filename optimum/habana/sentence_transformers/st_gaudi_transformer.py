import json
import os
from typing import Dict, List, Tuple, Union

import torch

from ..utils import to_device_dtype


def st_gaudi_transformer_save(self, output_path: str, safe_serialization: bool = True) -> None:
    state_dict = self.auto_model.state_dict()
    state_dict = to_device_dtype(state_dict, target_device=torch.device("cpu"))
    self.auto_model.save_pretrained(output_path, state_dict=state_dict, safe_serialization=safe_serialization)
    self.tokenizer.save_pretrained(output_path)

    with open(os.path.join(output_path, "sentence_bert_config.json"), "w") as fOut:
        json.dump(self.get_config_dict(), fOut, indent=2)


def st_gaudi_transformer_tokenize(
    self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]], padding: Union[str, bool] = True
):
    """Tokenizes a text and maps tokens to token-ids"""

    output = {}
    if isinstance(texts[0], str):
        to_tokenize = [texts]
    elif isinstance(texts[0], dict):
        to_tokenize = []
        output["text_keys"] = []
        for lookup in texts:
            text_key, text = next(iter(lookup.items()))
            to_tokenize.append(text)
            output["text_keys"].append(text_key)
        to_tokenize = [to_tokenize]
    else:
        batch1, batch2 = [], []
        for text_tuple in texts:
            batch1.append(text_tuple[0])
            batch2.append(text_tuple[1])
        to_tokenize = [batch1, batch2]

    # strip
    to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

    # Lowercase
    if self.do_lower_case:
        to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

    output.update(
        self.tokenizer(
            *to_tokenize,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.max_seq_length,
        )
    )
    return output
