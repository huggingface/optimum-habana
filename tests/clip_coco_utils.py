import os
from pathlib import Path
from urllib.request import urlretrieve

from transformers import AutoImageProcessor, AutoTokenizer, VisionTextDualEncoderModel, VisionTextDualEncoderProcessor


def download_coco(path=None):
    urls = [
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
        "http://images.cocodataset.org/zips/test2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "http://images.cocodataset.org/annotations/image_info_test2017.zip",
    ]

    if path is None:
        path = os.getcwd()

    print("Downloading COCO...")

    for url in urls:
        filename = url.split("/")[-1]
        urlretrieve(url, Path(path, filename))

    print("COCO downloaded.")


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
