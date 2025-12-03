# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test for Vision-Language Model (VLM) LoRA Fine-tuning
"""

import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from PIL import Image
from transformers.testing_utils import slow


def _create_dummy_vlm_dataset(dataset_dir: Path, num_samples: int = 8):
    """
    Create a small dummy vision-language dataset locally using HuggingFace datasets.
    Saves using save_to_disk() which creates arrow files.
    Args:
        dataset_dir: Directory to save the dataset
        num_samples: Number of samples to generate
    Returns:
        Path to the saved dataset
    """
    # Create dummy images (simple colored squares) as PIL Images
    images = []
    questions = []
    answers = []

    for i in range(num_samples):
        # Create a simple 224x224 colored image
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)
        img_array = np.full((224, 224, 3), color, dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)

        # Create simple Q&A pairs
        questions.append(f"What color is shown in this image? Sample {i}")
        answers.append(f"This is a colored image sample {i}")

    # Create dataset with PIL images
    train_dataset = Dataset.from_dict(
        {
            "image": images,
            "query": questions,
            "label": answers,
        }
    )

    # Create DatasetDict with train split
    dataset_dict = DatasetDict({"train": train_dataset})

    # Save to disk in arrow format
    dataset_dict.save_to_disk(str(dataset_dir))

    return dataset_dir


def _test_vlm_lora_training(
    model_name: str,
    num_samples: int = 8,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    num_epochs: int = 1,
):
    """
    Test VLM LoRA fine-tuning pipeline with local dummy dataset.

    Uses load_from_disk() compatible dataset to avoid network dependencies.

    Args:
        model_name: HuggingFace model name
        num_samples: Number of training samples (default: 8 for quick test)
        batch_size: Training batch size (default: 1)
        gradient_accumulation_steps: Gradient accumulation steps (default: 1)
        num_epochs: Number of training epochs (default: 1)
    """
    from datasets import load_from_disk

    path_to_example_dir = Path(__file__).resolve().parent.parent / "examples"
    env_variables = os.environ.copy()

    # Set Eager mode
    env_variables["PT_HPU_LAZY_MODE"] = "0"

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create dummy dataset locally
        dataset_dir = tmp_path / "dummy_dataset"
        dataset_dir.mkdir(exist_ok=True)
        dataset_path = _create_dummy_vlm_dataset(dataset_dir, num_samples)

        # Verify dataset can be loaded
        test_dataset = load_from_disk(str(dataset_path))
        assert "train" in test_dataset, "Train split not found in dataset"
        assert len(test_dataset["train"]) == num_samples, (
            f"Expected {num_samples} samples, got {len(test_dataset['train'])}"
        )
        print(f"SUCCESS: Created local dummy dataset with {num_samples} samples at {dataset_path}")

        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)

        # Create wrapper script that uses load_from_disk
        wrapper_script = tmp_path / "run_test_wrapper.py"
        wrapper_content = f'''
import sys
sys.path.insert(0, "{path_to_example_dir / "vision-language-modeling"}")

# Patch load_dataset to use load_from_disk for our test dataset
from datasets import load_from_disk
import run_lora_vlm

original_load_dataset = run_lora_vlm.load_dataset

def patched_load_dataset(dataset_name, **kwargs):
    if dataset_name == "DUMMY_DATASET_PATH":
        return load_from_disk("{dataset_path}")
    return original_load_dataset(dataset_name, **kwargs)

run_lora_vlm.load_dataset = patched_load_dataset

# Modify sys.argv for the script
sys.argv = [
    "run_lora_vlm.py",
    "--model_name_or_path", "{model_name}",
    "--output_dir", "{output_dir}",
    "--dataset_name", "DUMMY_DATASET_PATH",
    "--per_device_train_batch_size", "{batch_size}",
    "--gradient_accumulation_steps", "{gradient_accumulation_steps}",
    "--num_train_epochs", "{num_epochs}",
    "--do_train",
    "--bf16",
    "--use_habana",
    "--gaudi_config_name", "Habana/llama",
    "--learning_rate", "0.0002",
    "--logging_steps", "1",
    "--save_steps", "1000",
    "--gradient_checkpointing",
    "--lora_rank", "8",
]

# Run the main function
run_lora_vlm.main()
'''
        wrapper_script.write_text(wrapper_content)

        command = ["python3", str(wrapper_script)]
        print("\n\nRunning VLM LoRA training test with local dummy dataset\n")

        proc = subprocess.run(command, env=env_variables)

        # Ensure the run finished without any issue
        assert proc.returncode == 0, f"Training failed with return code {proc.returncode}"

        # Check if training artifacts exist
        # The script creates: checkpoint-X/, final_model/, training_*.log
        checkpoint_dir = output_dir / "checkpoint-8"
        final_model_dir = output_dir / "final_model"
        training_logs = list(output_dir.glob("training_*.log"))

        assert checkpoint_dir.exists() or final_model_dir.exists(), (
            f"Training artifacts not found. Files in output_dir: {list(output_dir.iterdir())}"
        )

        # Verify training completed successfully
        if checkpoint_dir.exists():
            trainer_state = checkpoint_dir / "trainer_state.json"
            if trainer_state.exists():
                with open(trainer_state) as fp:
                    state = json.load(fp)
                print(f"\nSUCCESS: Training completed successfully. Final step: {state.get('global_step', 'N/A')}")
            else:
                print(f"\nSUCCESS: Training completed successfully. Checkpoint saved at {checkpoint_dir}")

        if final_model_dir.exists():
            print(f"SUCCESS: Final model saved at {final_model_dir}")

        if training_logs:
            print(f"SUCCESS: Training log: {training_logs[0].name}")


@slow
@pytest.mark.parametrize(
    "model_name",
    [
        "llava-hf/llava-v1.6-mistral-7b-hf",
    ],
)
def test_llava_lora_vlm_training(model_name: str):
    """
    Test LLaVA LoRA VLM training with minimal samples.
    This is a smoke test to ensure the training pipeline works.

    Run with:
        RUN_SLOW=true GAUDI2_CI=1 pytest tests/test_vlm_lora.py -v -s -k test_llava_lora_vlm
    """
    _test_vlm_lora_training(
        model_name=model_name,
        num_samples=8,
        batch_size=1,
        gradient_accumulation_steps=1,
        num_epochs=1,
    )
