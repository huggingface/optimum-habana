#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import functools
import logging

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


# Will error if the minimal version of Optimum Habana is not installed. Remove at your own risks.
check_optimum_habana_min_version("1.10.0")


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def tokenize_protein(example, protein_tokenizer=None, padding=None):
    protein_seqs = example["prot_seq"]

    protein_inputs = protein_tokenizer(
        protein_seqs, padding="max_length", truncation=True, add_special_tokens=True, max_length=1024
    )
    example["protein_input_ids"] = protein_inputs.input_ids
    example["protein_attention_mask"] = protein_inputs.attention_mask

    return example


def label_embedding(labels, text_tokenizer, text_model, device):
    # embed label descriptions
    label_feature = []
    with torch.inference_mode():
        for label in labels:
            label_input_ids = text_tokenizer.encode(
                label, max_length=128, truncation=True, add_special_tokens=False, padding="max_length"
            )
            label_input_ids = [text_tokenizer.cls_token_id] + label_input_ids
            label_input_ids = torch.tensor(label_input_ids, dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = label_input_ids != text_tokenizer.pad_token_id
            attention_mask = attention_mask.to(device)
            text_outputs = text_model(label_input_ids, attention_mask=attention_mask)

            label_feature.append(text_outputs["text_feature"].clone())
    label_feature = torch.cat(label_feature, dim=0)
    label_feature = label_feature / label_feature.norm(dim=-1, keepdim=True)

    return label_feature


def zero_shot_eval(logger, device, test_dataset, target_field, protein_model, logit_scale, label_feature):
    # get prediction and target
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    preds, targets = [], []
    with torch.inference_mode():
        for data in tqdm(test_dataloader):
            target = data[target_field]
            targets.append(target)

            protein_input_ids = torch.tensor(data["protein_input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.tensor(data["protein_attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
            protein_outputs = protein_model(protein_input_ids, attention_mask=attention_mask)
            protein_feature = protein_outputs["protein_feature"]
            protein_feature = protein_feature / protein_feature.norm(dim=-1, keepdim=True)
            pred = logit_scale * protein_feature @ label_feature.t()
            preds.append(pred)
    preds = torch.cat(preds, dim=0)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    accuracy = (preds.argmax(dim=-1) == targets).float().mean().item()
    logger.warning("Zero-shot accuracy: %.6f" % accuracy)


if __name__ == "__main__":
    adapt_transformers_to_gaudi()

    device = torch.device("hpu")
    protst_model = AutoModel.from_pretrained(
        "mila-intel/ProtST-esm1b", trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    protein_model = protst_model.protein_model
    text_model = protst_model.text_model
    logit_scale = protst_model.logit_scale

    from habana_frameworks.torch.hpu import wrap_in_hpu_graph

    protein_model = wrap_in_hpu_graph(protein_model)
    text_model = wrap_in_hpu_graph(text_model)
    logit_scale.requires_grad = False
    logit_scale = logit_scale.to(device)
    logit_scale = logit_scale.exp()

    protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
    text_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

    raw_datasets = load_dataset("mila-intel/ProtST-SubcellularLocalization", split="test")
    func_tokenize_protein = functools.partial(tokenize_protein, protein_tokenizer=protein_tokenizer, padding=False)
    test_dataset = raw_datasets.map(
        func_tokenize_protein,
        batched=False,
        remove_columns=["prot_seq"],
        desc="Running tokenize_proteins on dataset",
    )

    labels = load_dataset("mila-intel/subloc_template")["train"]["name"]

    text_tokenizer.encode(labels[0], max_length=128, truncation=True, add_special_tokens=False)
    label_feature = label_embedding(labels, text_tokenizer, text_model, device)
    zero_shot_eval(logger, device, test_dataset, "localization", protein_model, logit_scale, label_feature)
