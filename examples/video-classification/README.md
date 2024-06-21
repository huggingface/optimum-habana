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

# Video Classification

This directory contains an example script to showcase usage of classifying video data.

## Requirements

First, install the requirements:
```bash
pip install -r requirements.txt
```

## Single-HPU inference

### Single video example

```bash
python3 run_example.py \
    --model_name_or_path MCG-NJU/videomae-base-finetuned-kinetics \
    --video_paths "https://ak.picdn.net/shutterstock/videos/21179416/preview/stock-footage-aerial-shot-winter-forest.mp4" \
    --use_hpu_graphs \
    --bf16
```

Outputs:
```
Predicted class for stock-footage-aerial-shot-winter-forest.mp4 is sled dog racing and took 1.243e+00 seconds
```

### Multi-video example

```bash
python3 run_example.py \
    --model_name_or_path MCG-NJU/videomae-base-finetuned-kinetics \
    --use_hpu_graphs \
    --bf16 \
    --warm_up_epochs 3 \
    --video_paths "https://ak.picdn.net/shutterstock/videos/5629184/preview/stock-footage-senior-couple-looking-through-binoculars-on-sailboat-together-shot-on-red-epic-for-high-quality-k.mp4" \
    "https://ak.picdn.net/shutterstock/videos/21179416/preview/stock-footage-aerial-shot-winter-forest.mp4" \
    "https://ak.picdn.net/shutterstock/videos/1063125190/preview/stock-footage-a-beautiful-cookie-with-oranges-lies-on-a-green-tablecloth.mp4" \
    "https://ak.picdn.net/shutterstock/videos/1039695998/preview/stock-footage-japanese-highrise-office-skyscrapers-tokyo-square.mp4" \
    "https://ak.picdn.net/shutterstock/videos/9607838/preview/stock-footage-zrenjanin-serbia-march-fans-watching-live-concert-bokeh-blur-urban-background-x.mp4"
```

Outputs: 
```
Predicted class for stock-footage-senior-couple-looking-through-binoculars-on-sailboat-together-shot-on-red-epic-for-high-quality-k.mp4 is sailing and took 3.372e-01 seconds
Predicted class for stock-footage-aerial-shot-winter-forest.mp4 is sled dog racing and took 3.360e-01 seconds
Predicted class for stock-footage-a-beautiful-cookie-with-oranges-lies-on-a-green-tablecloth.mp4 is cooking sausages and took 3.349e-01 seconds
Predicted class for stock-footage-japanese-highrise-office-skyscrapers-tokyo-square.mp4 is marching and took 3.362e-01 seconds
Predicted class for stock-footage-zrenjanin-serbia-march-fans-watching-live-concert-bokeh-blur-urban-background-x.mp4 is slacklining and took 3.358e-01 seconds
```

Models that have been validated:
- [MCG-NJU/videomae-base-finetuned-kinetics](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)
