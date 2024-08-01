#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import numpy as np
from custom_lilt import CustomLiltSelfAttention
from datasets import Array2D, ClassLabel, DatasetDict, Features, Sequence, Value, load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    LayoutLMv3FeatureExtractor,
    LayoutLMv3Processor,
    LiltForTokenClassification,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging as hf_logging
from transformers.utils import send_example_telemetry

from optimum.habana import GaudiTrainer, GaudiTrainingArguments


class CustomLiltForTokenClassification(LiltForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        # Override the attention mechanism with the custom implementation for all layers
        for layer in self.lilt.encoder.layer:
            layer.attention.self = CustomLiltSelfAttention(config)


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this value if set."
        },
    )


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GaudiTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("token-classification", model_args, data_args)

    hf_logging.set_verbosity_info()
    logger.setLevel(training_args.get_process_log_level())
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()

    logger.info(f"Training Args: {training_args}")
    logger.info(f"Data Args: {data_args}")
    logger.info(f"Model Args: {model_args}")
    last_checkpoint: Optional[str] = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    dataset: DatasetDict = load_dataset(data_args.dataset_name)

    logger.info(f"Train dataset size: {len(dataset['train'])}")
    logger.info(f"Test dataset size: {len(dataset['test'])}")

    labels: List[str] = dataset["train"].features["ner_tags"].feature.names
    logger.info(f"Available labels: {labels}")

    id2label: Dict[int, str] = dict(enumerate(labels))
    label2id: Dict[str, int] = {k: v for v, k in enumerate(labels)}

    model_id: str = model_args.model_name_or_path
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)

    features = Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "attention_mask": Sequence(feature=Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
            "labels": Sequence(ClassLabel(names=labels)),
        }
    )

    def process(sample: Dict[str, Any], processor: LayoutLMv3Processor = None) -> Dict[str, Any]:
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

    proc_dataset = dataset.map(
        partial(process, processor=processor),
        remove_columns=["image", "tokens", "ner_tags", "id", "bboxes"],
        features=features,
    ).with_format("torch")

    logger.info(proc_dataset["train"].features.keys())

    model = CustomLiltForTokenClassification.from_pretrained(
        model_id, num_labels=len(labels), label2id=label2id, id2label=id2label
    )

    metric = evaluate.load("seqeval")
    ner_labels = list(model.config.id2label.values())

    def compute_metrics(p: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        all_predictions: List[str] = []
        all_labels: List[str] = []
        for prediction, label in zip(predictions, labels):
            for predicted_idx, label_idx in zip(prediction, label):
                if label_idx == -100:
                    continue
                all_predictions.append(ner_labels[predicted_idx])
                all_labels.append(ner_labels[label_idx])

        # Use zero_division parameter to handle divisions by zero
        metrics = metric.compute(predictions=[all_predictions], references=[all_labels], zero_division=0)

        # Flatten the nested dictionary
        flat_metrics: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_metrics[f"{key}_{sub_key}"] = sub_value
            else:
                flat_metrics[key] = value

        return flat_metrics

    trainer = GaudiTrainer(
        model=model,
        args=training_args,
        train_dataset=proc_dataset["train"],
        eval_dataset=proc_dataset["test"],
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        logger.info("*** Training ***")
        checkpoint: Optional[str] = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(proc_dataset["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(proc_dataset["train"]))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(proc_dataset["test"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(proc_dataset["test"]))
        trainer.log_metrics("eval", metrics)

        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
