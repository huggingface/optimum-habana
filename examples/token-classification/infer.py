from datasets import load_dataset
from transformers import LiltForTokenClassification, LayoutLMv3FeatureExtractor, AutoTokenizer
from PIL import Image, ImageDraw, ImageFont
import torch
from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
import habana_frameworks.torch.core as htcore
import time

def set_device(device_type):
    if device_type == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_type == "hpu":
        device = torch.device("hpu")
    else:
        device = torch.device("cpu")
    return device

# Load the FUNSD dataset
dataset_id = "nielsr/funsd-layoutlmv3"
dataset = load_dataset(dataset_id)

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

# Load the model from the specified checkpoint directory
model = LiltForTokenClassification.from_pretrained("./results/checkpoint-2400")
# Load feature extractor and tokenizer
feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=True)
tokenizer = AutoTokenizer.from_pretrained("./results")

def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

label2color = {
    "B-HEADER": "blue",
    "B-QUESTION": "red",
    "B-ANSWER": "green",
    "I-HEADER": "blue",
    "I-QUESTION": "red",
    "I-ANSWER": "green",
}

def draw_boxes(image, boxes, predictions):
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

def run_inference(images, model, feature_extractor, tokenizer, device, output_image=True, warm_up_steps=5):
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
        if device.type == "hpu":
            htcore.mark_step()
        outputs = model(**encoding)
        if device.type == "hpu":
            htcore.mark_step()

    # Main inference loop with performance measurement
    for image in images:
        start_time = time.time()
        
        # Use feature extractor to handle OCR and get bounding boxes
        feature_extraction_start = time.time()
        feature_extraction = feature_extractor(images=image, return_tensors="pt")
        feature_extraction_end = time.time()
        
        words = feature_extraction["words"][0]
        boxes = feature_extraction["boxes"][0]
        
        # Tokenize the words
        tokenization_start = time.time()
        encoding = tokenizer(text=words, boxes=boxes, return_tensors="pt", padding="max_length", truncation=True)
        tokenization_end = time.time()
        
        # Move tensors and model to the specified device
        encoding = {k: v.to(device) for k, v in encoding.items()}
        model.to(device)
        
        if device.type == "hpu":
            # Enable lazy mode for Habana
            htcore.mark_step()
        
        # Model inference
        inference_start = time.time()
        outputs = model(**encoding)
        
        if device.type == "hpu":
            # Enable lazy mode for Habana
            htcore.mark_step()
        
        inference_end = time.time()
        
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        labels = [model.config.id2label[prediction] for prediction in predictions]

        # Ensure only unique bounding boxes and labels are used
        unique_boxes = []
        unique_labels = []
        for box, label in zip(encoding["bbox"][0], labels):
            if box.tolist() not in unique_boxes:
                unique_boxes.append(box.tolist())
                unique_labels.append(label)

        if output_image:
            results.append(draw_boxes(image, unique_boxes, unique_labels))
        else:
            results.append(unique_labels)
        
        end_time = time.time()
        total_inference_time += end_time - start_time
        
        print(f"Inference time for this image: {end_time - start_time:.2f} seconds")
        print(f"  Feature extraction time: {feature_extraction_end - feature_extraction_start:.2f} seconds")
        print(f"  Tokenization time: {tokenization_end - tokenization_start:.2f} seconds")
        print(f"  Model inference time: {inference_end - inference_start:.2f} seconds")
    
    avg_inference_time = total_inference_time / len(images)
    print(f"Average inference time per image: {avg_inference_time:.2f} seconds")
    
    return results

# Set the device type (options: "cuda", "cpu", "hpu")
device_type = "hpu"  # Change this to "cpu" or "cuda" as needed
device = set_device(device_type)

# Test the inference function
test_images = [dataset["test"][i]["image"].convert("RGB") for i in range(20)]
result_images = run_inference(test_images, model, feature_extractor, tokenizer, device)

# Save the resulting images
for idx, result_image in enumerate(result_images):
    result_image.save(f"result_image_{idx}.png")
