import os
from pathlib import Path
from urllib.request import urlretrieve

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
