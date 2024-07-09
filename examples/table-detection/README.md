<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Table Detection

This folder contains an example for using the [Table Transformer](https://huggingface.co/microsoft/table-transformer-detection) model fine tuned for table detection on the Gaudi platform.

## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

## Single HPU Inference

```bash
python run_example.py \
    --model_name_or_path microsoft/table-transformer-detection \
    --dataset_name nielsr/example-pdf \
    --filename example_pdf.png \
    --use_hpu_graphs \
    --bf16
```

## Models Validated

- [microsoft/table-transformer-detection](https://huggingface.co/microsoft/table-transformer-detection)
