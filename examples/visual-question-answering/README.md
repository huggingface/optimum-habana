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

# Visual Question Answering Examples

```bash
pip install -r requirements.txt
```

## Single-HPU inference

The `run_pipeline.py` script showcases how to use the Transformers pipeline API to run visual question answering task on HPUs.

```bash
python3 run_pipeline.py \
    --model_name_or_path Salesforce/blip-vqa-capfilt-large \
    --image_path "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg" \
    --question "how many dogs are in the picture?" \
    --use_hpu_graphs \
    --bf16
```
## OpenCLIP inference

The `run_openclip_vqa.py` can be used to run zero shot image classification with [OpenCLIP Huggingface Models](https://huggingface.co/docs/hub/en/open_clip#using-openclip-at-hugging-face).
The requirements for `run_openclip_vqa.py` can be installed with `openclip_requirements.txt` as follows:

```bash
pip install --no-deps -r openclip_requirements.txt
```

By default, the script runs the sample outlined in [BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 notebook](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/main/biomed_clip_example.ipynb). One can also can also run other OpenCLIP models by specifying model, classifier labels and image URL(s) like so:

```bash
python run_openclip_vqa.py \
    --model_name_or_path laion/CLIP-ViT-g-14-laion2B-s12B-b42K \
    --labels "a dog" "a cat" \
    --image_path "http://images.cocodataset.org/val2017/000000039769.jpg" \
    --use_hpu_graphs \
    --bf16
```
