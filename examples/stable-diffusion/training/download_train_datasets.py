#!/usr/bin/env python
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


# Download Cat-Toy example dataset
local_dir = "./cat"
snapshot_download(
    repo_id="diffusers/cat_toy_example",
    local_dir=local_dir,
    repo_type="dataset",
    ignore_patterns=".gitattributes",
)
cache_dir = Path(local_dir, ".cache")
if cache_dir.is_dir():
    shutil.rmtree(cache_dir)

# Download Dog example dataset
local_dir = "./dog"
snapshot_download(
    repo_id="diffusers/dog-example",
    local_dir=local_dir,
    repo_type="dataset",
    ignore_patterns=".gitattributes",
)
cache_dir = Path(local_dir, ".cache")
if cache_dir.is_dir():
    shutil.rmtree(cache_dir)

# Download ControlNet example images
local_dir = "./cnet"
file_path1 = hf_hub_download(
    repo_id="huggingface/documentation-images",
    subfolder="diffusers/controlnet_training",
    filename="conditioning_image_1.png",
    repo_type="dataset",
    local_dir=local_dir,
)
file_path2 = hf_hub_download(
    repo_id="huggingface/documentation-images",
    subfolder="diffusers/controlnet_training",
    filename="conditioning_image_2.png",
    repo_type="dataset",
    local_dir=local_dir,
)
shutil.copy2(file_path1, local_dir)
shutil.copy2(file_path2, local_dir)
cache_dir = Path(local_dir, ".cache")
if cache_dir.is_dir():
    shutil.rmtree(cache_dir)
sub_dir = Path(local_dir, "diffusers")
if sub_dir.is_dir():
    shutil.rmtree(sub_dir)
