<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

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

# ESMFold Example

ESMFold ([paper link](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2)) is a recently released protein folding model from FAIR. Unlike other protein folding models, it does not require external databases or search tools to predict structures, and is up to 60X faster as a result.

The port to the Hugging Face Transformers library is even easier to use, as we've removed the dependency on tools like openfold - once you run `pip install transformers`, you're ready to use this model!

Note that all the code that follows will be running the model locally, rather than calling an external API. This means that no rate limiting applies here - you can predict as many structures as your computer can handle.

## Single-HPU inference

Here we show how to predict the folding of a single chain on HPU:

```bash
python run_esmfold.py
```
The predicted protein structure will be stored in save-hpu.pdb file. We can use some tools like py3Dmol to visualize it.