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
# This script is based on https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb

import time

import habana_frameworks.torch.core as htcore
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


# Will error if the minimal version of Optimum Habana is not installed. Remove at your own risks.
check_optimum_habana_min_version("1.7.2")


def convert_outputs_to_pdb(outputs):
    """
    Converts the model outputs to a PDB file.

    This code comes from the original ESMFold repo, and uses some functions from openfold that have been ported to Transformers.
    """
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


adapt_transformers_to_gaudi()

steps = 4
device = torch.device("hpu")

# This is the sequence for human GNAT1.
# Feel free to substitute your own peptides of interest
# Depending on memory constraints you may wish to use shorter sequences.
test_protein = "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF"  # len = 350

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=False)
model = model.to(device)

# Uncomment this line if you're folding longer (over 600 or so) sequences
model.trunk.set_chunk_size(64)

with torch.no_grad():
    tk = tokenizer([test_protein], return_tensors="pt", add_special_tokens=False)
    tokenized_input = tk["input_ids"]
    print(f"ESMFOLD: input shape = {tokenized_input.shape}")
    tokenized_input = tokenized_input.to(device)

    for batch in range(steps):
        print(f"ESMFOLD: step {batch} start ...")
        start = time.time()
        output = model(tokenized_input)
        htcore.mark_step()
        print(f"ESMFOLD: step {batch} duration: {time.time() - start:.03f} seconds")

pdb = convert_outputs_to_pdb(output)
pdb_file = "save-hpu.pdb"
with open(pdb_file, "w") as fout:
    fout.write(pdb[0])
    print(f"pdb file saved in {pdb_file}")
