import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, MT5Config, T5Config


def st_gaudi_transformer_tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]], padding: Union[str, bool] = True):
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

