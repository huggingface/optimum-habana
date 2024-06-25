#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import time
from typing import List, Union

import habana_frameworks.torch.core as htcore
import torch
from datasets import DatasetDict, load_dataset
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, LayoutLMv3FeatureExtractor, LiltForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput


logger = logging.getLogger(__name__)

def set_device(device_type: str) -> torch.device:
    if device_type == "hpu":
        device = torch.device("hpu")
    else:
        device = torch.device("cpu")
    return device

def unnormalize_box(bbox: List[int], width: int, height: int) -> List[float]:
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def draw_boxes(image: Image.Image, boxes: List[List[float]], predictions: List[str]) -> Image.Image:
    label2color = {
        "B-HEADER": "blue",
        "B-QUESTION": "red",
        "B-ANSWER": "green",
        "I-HEADER": "blue",
        "I-QUESTION": "red",
        "I-ANSWER": "green",
    }
    width, height = image.size
    normalized_boxes = [unnormalize_box(box, width, height) for box in boxes]

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(predictions, normalized_boxes):
        if prediction == "O":
            continue
        draw.rectangle(box, outline="black")
        draw.rectangle(box, outline=label2color[prediction])
        draw.text((box[0] + 10, box[1] - 10), text=prediction, fill=label2color[prediction], font=font)
    return image

def run_inference(
    images: List[Image.Image],
    model: LiltForTokenClassification,
    feature_extractor: LayoutLMv3FeatureExtractor,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int,
    precision: str,
    use_hpu_graphs: bool,
    output_image: bool = True,
    warm_up_steps: int = 5
) -> List[Union[Image.Image, List[str]]]:
    results = []
    total_inference_time = 0

    # Warm-up phase
    for _ in range(warm_up_steps):
        image = images[0]
        feature_extraction = feature_extractor(images=image, return_tensors="pt")
        words = feature_extraction["words"][0]
        boxes = feature_extraction["boxes"][0]
        encoding = tokenizer(text=words, boxes=boxes, return_tensors="pt", padding="max_length", truncation=True)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        model.to(device)
        if precision == "fp16":
            model.half()
        if use_hpu_graphs and device.type == "hpu":
            htcore.mark_step()
        outputs = model(**encoding)
        if use_hpu_graphs and device.type == "hpu":
            htcore.mark_step()

    # Main inference loop with performance measurement
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_feature_extraction = feature_extractor(images=batch_images, return_tensors="pt")
        batch_words = batch_feature_extraction["words"]
        batch_boxes = batch_feature_extraction["boxes"]

        start_time = time.time()

        feature_extraction_start = time.time()
        feature_extraction = feature_extractor(images=batch_images, return_tensors="pt")
        feature_extraction_end = time.time()

        tokenization_start = time.time()
        encodings = [
            tokenizer(text=words, boxes=boxes, return_tensors="pt", padding="max_length", truncation=True)
            for words, boxes in zip(batch_words, batch_boxes)
        ]
        tokenization_end = time.time()

        for encoding in encodings:
            encoding = {k: v.to(device) for k, v in encoding.items()}
            model.to(device)
            if precision == "fp16":
                model.half()
            if use_hpu_graphs and device.type == "hpu":
                htcore.mark_step()

            inference_start = time.time()
            outputs: TokenClassifierOutput = model(**encoding)

            if use_hpu_graphs and device.type == "hpu":
                htcore.mark_step()

            inference_end = time.time()

            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            labels = [model.config.id2label[prediction] for prediction in predictions]

            unique_boxes = []
            unique_labels = []
            for box, label in zip(encoding["bbox"][0], labels):
                if box.tolist() not in unique_boxes:
                    unique_boxes.append(box.tolist())
                    unique_labels.append(label)

            if output_image:
                results.append(draw_boxes(batch_images[0], unique_boxes, unique_labels))
            else:
                results.append(unique_labels)

            end_time = time.time()
            total_inference_time += end_time - start_time

            logger.info(f"Inference time for this image: {end_time - start_time:.2f} seconds")
            logger.info(f"  Feature extraction time: {feature_extraction_end - feature_extraction_start:.2f} seconds")
            logger.info(f"  Tokenization time: {tokenization_end - tokenization_start:.2f} seconds")
            logger.info(f"  Model inference time: {inference_end - inference_start:.2f} seconds")

    avg_inference_time = total_inference_time / len(images)
    logger.info(f"Average inference time per image: {avg_inference_time:.2f} seconds")

    return results

def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on images using a specified model")
    parser.add_argument('--device_type', type=str, default='hpu', help='Device type: cpu, or hpu')
    parser.add_argument('--dataset_id', type=str, default='nielsr/funsd-layoutlmv3', help='Dataset ID for loading dataset')
    parser.add_argument('--model_path', type=str, default='./results/', help='Path to the pretrained model')
    parser.add_argument('--num_images', type=int, default=20, help='Number of test images to run inference on')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--precision', type=str, default='bf16', help='Precision for inference: bf16 or fp16')
    parser.add_argument('--use_hpu_graphs', action='store_true', help='Use HPU graphs for inference')
    args = parser.parse_args()

    device = set_device(args.device_type)

    dataset: DatasetDict = load_dataset(args.dataset_id)

    logger.info(f"Train dataset size: {len(dataset['train'])}")
    logger.info(f"Test dataset size: {len(dataset['test'])}")

    model = LiltForTokenClassification.from_pretrained(args.model_path)
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    test_images: List[Image.Image] = [dataset["test"][i]["image"].convert("RGB") for i in range(args.num_images)]
    result_images = run_inference(test_images, model, feature_extractor, tokenizer, device, args.batch_size, args.precision, args.use_hpu_graphs)

    for idx, result_image in enumerate(result_images):
        if isinstance(result_image, Image.Image):
            result_image.save(f"result_image_{idx}.png")


if __name__ == "__main__":
    main()
