from optimum.habana import GaudiTrainer, GaudiTrainingArguments
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from transformers import LayoutLMv3FeatureExtractor, AutoTokenizer, LayoutLMv3Processor, LiltForTokenClassification
import evaluate
import numpy as np
from functools import partial
from datasets import Features, Sequence, ClassLabel, Value, Array2D

# Load and prepare FUNSD dataset
dataset_id = "nielsr/funsd-layoutlmv3"
dataset = load_dataset(dataset_id)

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

# Display an image
image = dataset['train'][34]['image']
image = image.convert("RGB")
image.resize((350, 450))

# Define labels and mappings
labels = dataset['train'].features['ner_tags'].feature.names
print(f"Available labels: {labels}")

id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}

# Feature extractor and tokenizer setup
model_id = "SCUT-DLVCLab/lilt-roberta-en-base"
feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
processor = LayoutLMv3Processor(feature_extractor, tokenizer)

# Define dataset features
features = Features(
    {
        "input_ids": Sequence(feature=Value(dtype="int64")),
        "attention_mask": Sequence(feature=Value(dtype="int64")),
        "bbox": Array2D(dtype="int64", shape=(512, 4)),
        "labels": Sequence(ClassLabel(names=labels)),
    }
)

# Process function
def process(sample, processor=None):
    encoding = processor(
        sample["image"].convert("RGB"),
        sample["tokens"],
        boxes=sample["bboxes"],
        word_labels=sample["ner_tags"],
        padding="max_length",
        truncation=True,
    )
    del encoding["pixel_values"]
    return encoding

# Apply processing
proc_dataset = dataset.map(
    partial(process, processor=processor),
    remove_columns=["image", "tokens", "ner_tags", "id", "bboxes"],
    features=features,
).with_format("torch")

print(proc_dataset["train"].features.keys())

# Fine-tune and evaluate LiLT
model = LiltForTokenClassification.from_pretrained(
    model_id, num_labels=len(labels), label2id=label2id, id2label=id2label, local_files_only=True
)

# Evaluation metric setup
metric = evaluate.load("seqeval")
ner_labels = list(model.config.id2label.values())

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    all_predictions = []
    all_labels = []
    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            all_predictions.append(ner_labels[predicted_idx])
            all_labels.append(ner_labels[label_idx])
    return metric.compute(predictions=[all_predictions], references=[all_labels])

# Training arguments
training_args = GaudiTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    bf16=True,  # Enable bf16 mixed precision training
    learning_rate=5e-5,
    max_steps=2500,
    logging_dir="./results/logs",
    logging_strategy="steps",
    logging_steps=200,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="overall_f1",
    report_to="tensorboard",
    use_habana=True,
    gaudi_config_name="./gaudi_config.json",
    gradient_checkpointing=True,  # Optimize memory usage
    use_hpu_graphs=True,  # Enable HPU Graphs for better performance
    use_lazy_mode=True,  # Enable Lazy Mode
    dataloader_num_workers=4,  # Increase number of DataLoader workers
    non_blocking_data_copy=True,  # Enable Non-Blocking Data Copy
)

# Trainer setup
trainer = GaudiTrainer(
    model=model,
    args=training_args,
    train_dataset=proc_dataset["train"],
    eval_dataset=proc_dataset["test"],
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the processor
processor.save_pretrained("./results")
