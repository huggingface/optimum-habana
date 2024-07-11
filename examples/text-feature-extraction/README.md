<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

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

# Feature Extraction Examples

This directory contains a script that showcases how to use text embedding models as feature extractors for text embeddings on HPUs.

## Single-HPU inference

```bash
python run_feature_extraction.py \
    --model_name_or_path Supabase/gte-small \
    --source_sentence "What is a deep learning architecture for feature extraction?" \
    --input_texts "There are many different variants of apples created every year." \
        "BERT is a common machine learning architecture for text-based applications." \
        "Alexander Hamilton is one of the founding fathers of the United States." \
    --use_hpu_graphs \
    --bf16
```

Models that have been validated:

- [Supabase/gte-small](https://huggingface.co/Supabase/gte-small)
- [thenlper/gte-small](https://huggingface.co/thenlper/gte-small)
- [thenlper/gte-base](https://huggingface.co/thenlper/gte-base)
- [thenlper/gte-large](https://huggingface.co/thenlper/gte-large)
