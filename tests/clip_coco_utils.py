import os

# Calculate CLIP score
from functools import partial
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torchmetrics.functional.multimodal import clip_score
from transformers import AutoImageProcessor, AutoTokenizer, VisionTextDualEncoderModel, VisionTextDualEncoderProcessor


COCO_URLS = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/zips/test2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "http://images.cocodataset.org/annotations/image_info_test2017.zip",
]


def download_files(list_of_urls, path=None):
    if path is None:
        path = os.getcwd()

    for url in list_of_urls:
        print(f"Downloading {url}")
        filename = url.split("/")[-1]
        urlretrieve(url, Path(path, filename))
        print(f"{url} downloaded.")


def create_clip_roberta_model():
    print("Generating a CLIP-RoBERTa model...")

    model = VisionTextDualEncoderModel.from_vision_text_pretrained("openai/clip-vit-base-patch32", "roberta-base")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

    # save the model and processor
    model.save_pretrained("clip-roberta")
    processor.save_pretrained("clip-roberta")

    print("Model generated.")


clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)
