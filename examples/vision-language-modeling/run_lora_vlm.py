# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

###############################################################################
# Copyright (C) 2020-2025 Habana Labs, Ltd. an Intel Company
###############################################################################

###############################################################################
# This script extends the LoRA fine-tuning approach from run_lora_clm.py
# (optimum-habana/examples/language-modeling) for vision-language models,
# adding multimodal dataset handling, vision tower management, and support
# for LLaVA, Qwen2.5-VL, and Gemma-3 model families.
###############################################################################

import os
import sys

# Early DDP configuration for models with frozen vision towers
# This must be set BEFORE any torch.distributed initialization
if "--model_name_or_path" in sys.argv:
    model_arg_idx = sys.argv.index("--model_name_or_path")
    if model_arg_idx + 1 < len(sys.argv):
        model_name = sys.argv[model_arg_idx + 1]
        # Enable find_unused_parameters for LLaVA models to handle frozen vision tower
        if "llava" in model_name.lower():
            os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "INFO")
            os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch
from datetime import datetime

# HuggingFace & Training
from transformers import (
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

# LoRA/PEFT
from peft import get_peft_model, LoraConfig, TaskType

# Dataset & Data Loading
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from torch.utils.data import DataLoader, Dataset

# Habana-specific
try:
    from optimum.habana import GaudiTrainingArguments, GaudiTrainer
    from optimum.habana.utils import set_seed
    HABANA_AVAILABLE = True
except ImportError:
    HABANA_AVAILABLE = False
    print("[WARNING] optimum-habana not available, falling back to standard PyTorch")

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL REGISTRY & DETECTION LOGIC
# ============================================================================

MODEL_REGISTRY = {
    # LLaVA Family (flexible pattern matching)
    "llava": {
        "aliases": ["llava-1.6", "llava-v1.6", "llava-hf/llava"],
        "model_class": "vision2seq",
        "processor_type": "llava",
        "lora_target_modules": ["q_proj", "v_proj"],  # Only target language model components
        "lora_modules_to_save": [],  # Don't save vision tower parameters
        "supports_bf16": True,
        "vision_feature_select_strategy": "patch",
        "requires_image_processor": True,
        "exclude_vision_tower": True,  # Flag to exclude vision tower from LoRA
    },
    # Qwen2.5-VL
    "qwen2.5-vl": {
        "aliases": ["qwen2.5-vl", "qwen2_5-vl", "qwen2-5-vl"],
        "model_class": "vision2seq",
        "processor_type": "qwen",
        "lora_target_modules": ["q_proj", "v_proj", "k_proj"],
        "supports_bf16": True,
        "vision_feature_select_strategy": "full",
        "requires_image_processor": True,
    },
    # Gemma-3
    "gemma-3": {
        "aliases": ["gemma-3", "gemma3"],
        "model_class": "causal_lm",  # text-only for now; VL version if available
        "processor_type": "gemma",
        "lora_target_modules": ["q_proj", "v_proj"],
        "supports_bf16": True,
        "vision_feature_select_strategy": None,
        "requires_image_processor": False,
    },
}


def detect_model_family(model_name_or_path: str) -> Optional[Dict[str, Any]]:
    """
    Detect model family with flexible pattern matching.
    Handles both HF model IDs (e.g., llava-hf/llava-v1.6-mistral-7b-hf)
    and short names (e.g., llava-1.6).
    """
    model_name_lower = model_name_or_path.lower()
    
    for family_name, config in MODEL_REGISTRY.items():
        for alias in config["aliases"]:
            if alias.lower() in model_name_lower:
                logger.info(f" Detected model family: {family_name} (matched alias: {alias})")
                return {**config, "family": family_name, "family_name": family_name}
    
    logger.error(
        f"ERROR: Unsupported model: {model_name_or_path}\n"
        f"  Supported families: {', '.join(MODEL_REGISTRY.keys())}\n"
        f"  Tips:\n"
        f"    - LLaVA-1.6: llava-hf/llava-v1.6-mistral-7b-hf\n"
        f"    - Qwen2.5-VL: Qwen/Qwen2.5-VL-7B-Instruct\n"
        f"    - Gemma-3: google/gemma-3-12b-it"
    )
    return None


# ============================================================================
# DATASET HANDLING
# ============================================================================

DATASET_CONFIGS = {
    # ChartQA - Validated and working
    "ChartQA": {
        "dataset_name": "HuggingFaceM4/ChartQA",
        "image_column": "image",
        "question_column": "query",  # ChartQA uses 'query' not 'question'
        "answer_column": "label",    # ChartQA uses 'label' not 'answer'
        "supports_train": True,
        "status": "validated",
    },
}


class MultimodalVQADataset(Dataset):
    """
    Flexible dataset loader for VQA/VL tasks.
    Handles ChartQA, DocVQA, TextVQA, InfographicVQA, and custom formats.
    """
    
    def __init__(
        self,
        dataset_name: str,
        processor,
        split: str = "train",
        dataset_config: Optional[Dict] = None,
        max_samples: Optional[int] = None,
        raw_dataset = None,
    ):
        self.processor = processor
        self.dataset_config = dataset_config or {}
        
        # Load dataset (use raw_dataset if provided, otherwise load from hub)
        if raw_dataset is not None:
            logger.info(f"Using pre-loaded dataset: {dataset_name} ({split})")
            self.dataset = raw_dataset
        else:
            logger.info(f"Loading dataset: {dataset_name} ({split})")

            def _load_with_fallback(primary_name: str, fallback_names: List[str]):
                candidates = [primary_name] + [name for name in fallback_names if name]
                last_error = None
                for candidate in candidates:
                    try:
                        if candidate != primary_name:
                            logger.warning(
                                "Dataset '%s' unavailable, trying fallback '%s'",
                                primary_name,
                                candidate,
                            )
                        # Removed trust_remote_code=True as it's deprecated for standard datasets
                        return load_dataset(candidate, split=split)
                    except DatasetNotFoundError as dnfe:
                        last_error = dnfe
                        continue
                    except Exception as generic_error:
                        last_error = generic_error
                        logger.error(
                            "Unexpected error loading dataset '%s': %s",
                            candidate,
                            generic_error,
                        )
                        break
                raise last_error if last_error else RuntimeError(
                    f"Failed to load dataset '{primary_name}' and fallbacks"
                )

            fallback_names = self.dataset_config.get("alternative_names", [])
            try:
                self.dataset = _load_with_fallback(dataset_name, fallback_names)
            except Exception as e:
                logger.error(
                    "Failed to load dataset %s (attempted fallbacks: %s): %s",
                    dataset_name,
                    fallback_names,
                    e,
                )
                raise
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            logger.info(f"Limited dataset to {max_samples} samples")
        
        # Parse dataset schema
        self.image_col = self.dataset_config.get("image_column", "image")
        self.question_col = self.dataset_config.get("question_column", "question")
        self.answer_col = self.dataset_config.get("answer_column", "answer")
        
        logger.info(
            f"Dataset schema: image={self.image_col}, "
            f"question={self.question_col}, answer={self.answer_col}"
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def normalize_sample(self, sample):
        """
        Normalize different dataset schemas to a common format: {image, question, answer}
        Based on the working run_lora_vlm.py approach
        """
        img = sample.get("image")
        if img is None:
            raise ValueError("Missing 'image' column.")

        # Handle different schema patterns like the working script
        if "query" in sample and "label" in sample:
            q = (sample["query"] or "").strip()
            lab = sample["label"]
            a = lab[0] if isinstance(lab, list) and lab else (lab or "")
        elif "question" in sample and "answer" in sample:
            q = (sample["question"] or "").strip()
            a = (sample["answer"] or "").strip()
        else:
            # Fallback to configured column names
            q = (sample.get(self.question_col, "") or "").strip()
            answer_val = sample.get(self.answer_col, "")
            a = answer_val[0] if isinstance(answer_val, list) and answer_val else (answer_val or "")

        return {
            "image": img.convert("RGB") if hasattr(img, "convert") else img,
            "question": q,
            "answer": str(a).strip()
        }

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Return normalized sample - let collate_fn handle the processing
        normalized = self.normalize_sample(sample)
        
        # Skip empty answers
        if not normalized["answer"]:
            logger.warning(f"Empty answer at idx {idx}, using placeholder")
            normalized["answer"] = "No answer provided"
        
        return normalized


# ============================================================================
# LORA CONFIGURATION
# ============================================================================

def get_lora_config(model_config: Dict[str, Any], lora_rank: int = 16) -> LoraConfig:
    """
    Create LoRA config tailored to model family.
    """
    target_modules = model_config.get("lora_target_modules", ["q_proj", "v_proj"])

    family = model_config.get("family") or model_config.get("family_name")

    # For LLaVA models, capture the usual linear layer names; we'll filter to the
    # language-model side after the model is instantiated where we can inspect
    # full module paths.
    if family == "llava":
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        logger.info(
            "LLaVA model detected - will retarget LoRA to language-model linear layers after model load"
        )
    else:
        # Use "all-linear" for broader coverage if specific modules aren't working
        if isinstance(target_modules, list) and len(target_modules) <= 2:
            target_modules = "all-linear"
    
    logger.info(f"LoRA Config: rank={lora_rank}, modules={target_modules}")
    
    # Create base config
    config_args = {
        "task_type": TaskType.CAUSAL_LM,
        "r": lora_rank,
        "lora_alpha": lora_rank,
        "lora_dropout": 0.05,
        "target_modules": target_modules,
        "bias": "none",
    }
    
    # For LLaVA, ensure we're excluding vision tower completely
    if family == "llava":
        logger.info(f"LLaVA LoRA target pattern: {target_modules}")
        # Ensure no vision tower modules are included
    
    return LoraConfig(**config_args)


def configure_llava_lora_targets(model, lora_config: LoraConfig) -> LoraConfig:
    """Restrict LLaVA LoRA targets to language-model linear layers."""

    if not isinstance(lora_config.target_modules, (list, tuple)):
        return lora_config

    attn_suffixes = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )
    mlp_suffixes = (
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    projector_suffixes = (
        "multi_modal_projector.linear_1",
        "multi_modal_projector.linear_2",
    )

    selected_modules: List[str] = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        if any(name.endswith(suffix) for suffix in projector_suffixes):
            selected_modules.append(name)
            continue

        if "language_model.layers" not in name:
            continue

        if any(name.endswith(suffix) for suffix in attn_suffixes + mlp_suffixes):
            selected_modules.append(name)

    if selected_modules:
        selected_modules = sorted(set(selected_modules))
        lora_config.target_modules = selected_modules
        logger.info(
            "Configured %d LLaVA LoRA target modules (language model + projector only)",
            len(selected_modules),
        )
    else:
        logger.warning(
            "No language-model LoRA targets detected for LLaVA; falling back to default target modules"
        )

    return lora_config


# ============================================================================
# GAUDI-SPECIFIC TRAINING ARGUMENTS
# ============================================================================

@dataclass
class GaudiVLMTrainingArguments(GaudiTrainingArguments if HABANA_AVAILABLE else TrainingArguments):
    """
    Extended training arguments with VLM-specific defaults.
    """
    # VLM-specific
    use_vision_features: bool = field(default=True)
    freeze_vision_encoder: bool = field(default=True)
    vision_feature_select_strategy: Optional[str] = field(default="patch")
    
    # Batch & gradient
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=2)
    
    # Optimization
    learning_rate: float = field(default=1e-4)
    warmup_ratio: float = field(default=0.03)
    weight_decay: float = field(default=0.01)
    
    # Precision & Performance
    bf16: bool = field(default=True)
    # gradient_checkpointing: inherits from TrainingArguments (default behavior matches base run_lora_vlm.py)
    use_flash_attention_2: bool = field(default=False)  # Check Gaudi support
    
    # Gaudi-specific
    gaudi_config_name: Optional[str] = field(default=None)
    use_habana: bool = field(default=False)
    use_lazy_mode: bool = field(default=False)  # False = eager (dynamic shapes)
    
    # Logging & Checkpointing
    logging_steps: int = field(default=10)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=100)
    max_steps: int = field(default=-1)  # -1 means derive from dataset size


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

class MultimodalVLMTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Detect model
        self.model_config = detect_model_family(args.model_name_or_path)
        if not self.model_config:
            sys.exit(1)
        
        # Setup logging
        self._setup_logging()
        
        logger.info("="*80)
        logger.info("MULTI-MODAL VLM TRAINING WITH LORA")
        logger.info("="*80)
        logger.info(f"Model: {args.model_name_or_path}")
        logger.info(f"Dataset: {args.dataset_name}")
        logger.info(f"Output Dir: {args.output_dir}")
        logger.info(f"Gaudi Enabled: {HABANA_AVAILABLE and args.use_habana}")
    
    def _setup_logging(self):
        """Setup file logging"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        log_file = os.path.join(
            self.args.output_dir,
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    
    def load_model_and_processor(self):
        """Load model and processor with proper error handling"""
        logger.info(f"Loading model: {self.args.model_name_or_path}")
        
        try:
            # Load processor (trust_remote_code removed for standard models)
            self.processor = AutoProcessor.from_pretrained(
                self.args.model_name_or_path,
            )
            logger.info(" Processor loaded")
            
            # Load model (trust_remote_code removed for standard models)
            model_cls = AutoModelForVision2Seq if self.model_config["model_class"] == "vision2seq" else AutoModelForCausalLM
            self.model = model_cls.from_pretrained(
                self.args.model_name_or_path,
                torch_dtype=torch.bfloat16 if self.args.bf16 else torch.float32,
                device_map="auto" if not HABANA_AVAILABLE else None,
                low_cpu_mem_usage=True,
            )
            
            # Disable cache for training (required for gradient checkpointing)
            self.model.config.use_cache = False
            logger.info(" Model loaded")
            
            # Apply LoRA
            lora_config = get_lora_config(self.model_config, self.args.lora_rank)
            
            # Enable input gradients before applying LoRA (required for gradient checkpointing)
            if hasattr(self.model, 'enable_input_require_grads'):
                self.model.enable_input_require_grads()

            if self.model_config.get("family") == "llava":
                lora_config = configure_llava_lora_targets(self.model, lora_config)
            
            self.model = get_peft_model(self.model, lora_config)
            
            # For LLaVA models, freeze unused vision tower layers to prevent DDP issues
            if self.model_config.get("family") == "llava":
                self._freeze_unused_vision_layers()
            
            # Cast to bfloat16 if needed (after LoRA application)
            if self.args.bf16:
                self.model = self.model.to(torch.bfloat16)
            
            # Apply torch.compile if in TC mode (matches base run_lora_vlm.py behavior)
            is_lazy = os.environ.get('PT_HPU_LAZY_MODE', '1') == '1'
            is_tc = os.environ.get('PT_HPU_TC_MODE', '0') == '1'
            if not is_lazy and is_tc:
                logger.info(" Torch Compile mode enabled (PT_HPU_TC_MODE=1)")
                self.model = torch.compile(self.model, backend="hpu_backend")
                
            logger.info(f" LoRA applied (trainable params: {self.get_trainable_params()})")
            
            # Print trainable parameters for debugging
            if hasattr(self.model, 'print_trainable_parameters'):
                self.model.print_trainable_parameters()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            sys.exit(1)
    
    def _freeze_unused_vision_layers(self):
        """
        Freeze unused vision tower layers for LLaVA models to prevent DDP unused parameter errors.
        
        The issue is that base layer parameters in vision tower (like out_proj, layer_norm, mlp)
        are causing DDP unused parameter errors even when frozen. We need to ensure ALL vision
        tower base parameters are properly excluded from gradient computation.
        """
        if not self.model_config or self.model_config.get("family") != "llava":
            return
            
        frozen_count = 0
        
        for name, param in self.model.named_parameters():
            # Freeze ALL vision tower base parameters (non-LoRA parameters)
            # Keep only LoRA parameters trainable
            if param.requires_grad and "vision_tower" in name:
                if "lora_A" not in name and "lora_B" not in name:
                    param.requires_grad = False
                    frozen_count += 1
                    # Log some examples for debugging
                    if frozen_count <= 5:
                        logger.info(f"Frozen vision tower base parameter: {name}")
        
        if frozen_count > 0:
            logger.info(f" Frozen {frozen_count} vision tower base parameters to fix DDP unused parameter error")
        else:
            logger.info("No additional vision tower parameters needed freezing (already frozen by default)")
    
    def get_trainable_params(self):
        """Count trainable parameters"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return f"{trainable:,} / {total:,}"
    
    def setup_gaudi_config(self):
        """Setup Gaudi configuration"""
        if not (HABANA_AVAILABLE and self.args.use_habana):
            return None
            
        if self.args.gaudi_config_name:
            return self.args.gaudi_config_name
            
        # Create default gaudi config
        default_config = {
            "autocast_bf16_ops": None,
            "autocast_fp32_ops": None,
            "optimum_version": "1.20.0.dev0",
            "transformers_version": "4.55.4", 
            "use_dynamic_shapes": not self.args.use_lazy_mode,  # Eager mode uses dynamic shapes
            "use_fused_adam": True,
            "use_fused_clip_norm": True,
            "use_torch_autocast": True
        }
        
        # Save to output directory
        os.makedirs(self.args.output_dir, exist_ok=True)
        config_path = os.path.join(self.args.output_dir, "gaudi_config.json")
        
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
            
        logger.info(f" Created default Gaudi config: {config_path}")
        return config_path
    
    def prepare_dataset(self):
        """Load and prepare dataset"""
        logger.info(f"Preparing dataset: {self.args.dataset_name}")
        
        dataset_config = DATASET_CONFIGS.get(
            self.args.dataset_name,
            {
                "image_column": "image",
                "question_column": "question",
                "answer_column": "answer",
            }
        )
        
        # Load raw dataset first to handle AUTO-SPLIT logic like original run_lora_vlm.py
        raw_datasets = load_dataset(
            self.args.dataset_name,
            cache_dir=getattr(self.args, 'cache_dir', None),
            token=getattr(self.args, 'token', None),
        )
        
        # AUTO-SPLIT: Create validation split if doing eval but no validation exists
        # This matches the logic from the original run_lora_vlm.py
        if hasattr(self.args, 'do_eval') and self.args.do_eval and "validation" not in raw_datasets:
            logger.warning(
                f"Dataset '{self.args.dataset_name}' has no 'validation' split. "
                f"Creating 80/20 train/validation split from training data (seed=42)."
            )
            split_datasets = raw_datasets["train"].train_test_split(test_size=0.2, seed=42)
            raw_datasets["train"] = split_datasets["train"]
            raw_datasets["validation"] = split_datasets["test"]
            logger.info(f"Train split: {len(raw_datasets['train'])} samples")
            logger.info(f"Validation split: {len(raw_datasets['validation'])} samples")
        
        # Create training dataset wrapper
        self.train_dataset = MultimodalVQADataset(
            dataset_name=self.args.dataset_name,
            processor=self.processor,
            split="train",
            dataset_config=dataset_config,
            max_samples=self.args.max_train_samples,
            raw_dataset=raw_datasets["train"],  # Pass the raw dataset directly
        )
        
        logger.info(f" Loaded {len(self.train_dataset)} training samples")
        
        # Prepare evaluation dataset if needed
        self.eval_dataset = None
        if hasattr(self.args, 'do_eval') and self.args.do_eval:
            if "validation" not in raw_datasets:
                logger.error("Evaluation requires a validation split in the dataset")
                self.eval_dataset = None
            else:
                self.eval_dataset = MultimodalVQADataset(
                    dataset_name=self.args.dataset_name,
                    processor=self.processor,
                    split="validation",
                    dataset_config=dataset_config,
                    max_samples=getattr(self.args, 'max_eval_samples', None),
                    raw_dataset=raw_datasets["validation"],  # Pass the raw dataset directly
                )
                logger.info(f" Loaded {len(self.eval_dataset)} evaluation samples from validation split")
    
    def train(self):
        """Execute training"""
        logger.info("Starting training...")
        
        self.load_model_and_processor()
        self.prepare_dataset()
        
        # Setup Gaudi config
        gaudi_config_name = self.setup_gaudi_config()
        
        # Check if LLaVA model for DDP unused parameters fix
        is_llava_model = self.model_config and self.model_config.get("family") == "llava"
        
        # Training arguments
        training_args = GaudiVLMTrainingArguments(
            output_dir=self.args.output_dir,
            do_train=getattr(self.args, 'do_train', True),  # Default to True if not specified
            do_eval=getattr(self.args, 'do_eval', False),   # Default to False
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            bf16=self.args.bf16,
            use_habana=HABANA_AVAILABLE and self.args.use_habana,
            gaudi_config_name=gaudi_config_name,
            use_lazy_mode=self.args.use_lazy_mode,
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            eval_strategy="steps" if (getattr(self.args, 'do_eval', False) and self.eval_dataset is not None) else "no",
            eval_steps=self.args.save_steps if (getattr(self.args, 'do_eval', False) and self.eval_dataset is not None) else None,
            save_strategy="steps",
            remove_unused_columns=False,
            gradient_checkpointing=self.args.gradient_checkpointing,  # Configurable gradient checkpointing
            dataloader_pin_memory=False,  # Helps with Gaudi compatibility
            dataloader_drop_last=True,  # Ensure consistent batch sizes across ranks
            deepspeed=self.args.deepspeed if hasattr(self.args, 'deepspeed') else None,  # DeepSpeed config
            ddp_find_unused_parameters=is_llava_model,  # Enable for LLaVA models to fix DDP unused parameter error
            ddp_bucket_cap_mb=25,  # Reduce DDP bucket size for better memory management
        )
        
        # Log DDP configuration for LLaVA models
        if is_llava_model:
            logger.info(" LLaVA model detected - enabling ddp_find_unused_parameters for multi-GPU compatibility")
            # Also set environment variables as backup
            os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "INFO")
            os.environ.setdefault("TORCH_DISTRIBUTED_FIND_UNUSED_PARAMETERS", "1")
            logger.info(
                " DDP parameters: find_unused=%s, bucket_cap=%sMB",
                training_args.ddp_find_unused_parameters,
                training_args.ddp_bucket_cap_mb,
            )
            
            # Debug: Log trainable parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f" Model parameters: {trainable_params:,} trainable / {total_params:,} total")
            
            # Debug: Check parameter requires_grad status
            param_count = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param_count += 1
                    if param_count <= 10:  # Log first 10 trainable parameters
                        logger.info(f" Trainable param {param_count}: {name} (shape: {param.shape})")
                elif param_count <= 300 and param_count >= 270:  # Log parameters around the error indices
                    logger.info(f" Non-trainable param {param_count}: {name} (shape: {param.shape})")
                param_count += 1
        
        # Log evaluation strategy
        if getattr(self.args, 'do_eval', False):
            if self.eval_dataset is not None:
                logger.info(f" Evaluation enabled with {len(self.eval_dataset)} samples (strategy: steps)")
            else:
                logger.warning(" Evaluation requested but no eval_dataset available - disabling evaluation")
        
        # Derive max_steps from sample count if user didn't set it explicitly
        if training_args.max_steps == -1:
            eff_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
            steps_per_epoch = max(1, len(self.train_dataset) // eff_bs)
            training_args.max_steps = int(steps_per_epoch * max(1.0, training_args.num_train_epochs))
            logger.info(
                f"Derived max_steps={training_args.max_steps} from N={len(self.train_dataset)} samples, "
                f"bs={training_args.per_device_train_batch_size}, ga={training_args.gradient_accumulation_steps}, "
                f"epochs={training_args.num_train_epochs}"
            )

        # Initialize trainer
        TrainerClass = GaudiTrainer if (HABANA_AVAILABLE and self.args.use_habana) else Trainer
        
        trainer = TrainerClass(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset if training_args.do_train else None,
            eval_dataset=self.eval_dataset if training_args.do_eval else None,
            data_collator=self._collate_fn,
        )
        
        # Additional DDP configuration for LLaVA models
        if is_llava_model and hasattr(trainer, 'model') and hasattr(trainer.model, 'module'):
            logger.info(" Applying DDP configuration to wrapped model")
            # The model is already wrapped in DDP, we need to ensure parameters are properly configured
        
        # Training
        if training_args.do_train:
            logger.info("Starting training...")
            train_result = trainer.train()
            logger.info(" Training completed")
            
            # Save training metrics
            metrics = train_result.metrics
            max_train_samples = (
                getattr(self.args, 'max_train_samples', None) 
                if getattr(self.args, 'max_train_samples', None) is not None 
                else len(self.train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))
            
            logger.info(f"Training metrics: {metrics}")
            
            # Save model
            self.model.save_pretrained(os.path.join(self.args.output_dir, "final_model"))
            logger.info(f" Model saved to {self.args.output_dir}")
        
        # Evaluation
        if training_args.do_eval and self.eval_dataset:
            logger.info("Starting evaluation...")
            eval_metrics = trainer.evaluate()
            
            max_eval_samples = (
                getattr(self.args, 'max_eval_samples', None)
                if getattr(self.args, 'max_eval_samples', None) is not None
                else len(self.eval_dataset)
            )
            eval_metrics["eval_samples"] = min(max_eval_samples, len(self.eval_dataset))
            
            # Calculate perplexity
            try:
                import math
                perplexity = math.exp(eval_metrics["eval_loss"])
            except (OverflowError, KeyError):
                perplexity = float("inf")
            eval_metrics["perplexity"] = perplexity
            
            logger.info(f"Evaluation metrics: {eval_metrics}")
            logger.info(" Evaluation completed")
        
        # Save processor
        if training_args.do_train or training_args.do_eval:
            self.processor.save_pretrained(self.args.output_dir)
            logger.info(" Processor saved")
    
    def _collate_fn(self, batch):
        """
        Custom collate function based on the working run_lora_vlm.py approach.
        Handles vision-language model requirements properly.
        """
        # Extract data from batch
        full_texts, full_images = [], []
        
        for ex in batch:
            # Get normalized data (should have 'image', 'question', 'answer')
            if 'image' in ex and 'question' in ex and 'answer' in ex:
                # This is already processed data from dataset
                image = ex['image']
                question = ex['question'].strip()
                answer = ex.get('answer', '').strip()
            else:
                # This is raw data from the dataset, normalize it first
                dataset_sample = ex
                normalized = self.normalize_sample_dict(dataset_sample)
                image = normalized['image'] 
                question = normalized['question']
                answer = normalized['answer']
            
            # Build conversation format expected by the model
            full_msgs = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            if answer:
                full_msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
            
            # Apply chat template and process
            full_text = self.processor.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            ).strip()
            
            full_texts.append(full_text)
            full_images.append([image])  # Wrap in list for batch processing
        
        # Process the batch
        try:
            enc = self.processor(
                text=full_texts, 
                images=full_images, 
                return_tensors="pt", 
                padding=True
            )
        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            # Fallback: try processing without images
            enc = self.processor(
                text=full_texts,
                return_tensors="pt", 
                padding=True
            )
        
        # Create labels (copy input_ids, will be masked later by trainer)
        if "input_ids" in enc:
            enc["labels"] = enc["input_ids"].clone()
        
        return enc
    
    def normalize_sample_dict(self, sample):
        """Helper to normalize a sample dict outside of dataset context"""
        img = sample.get("image")
        if img is None:
            raise ValueError("Missing 'image' column.")

        # Handle different schema patterns
        if "query" in sample and "label" in sample:
            q = (sample["query"] or "").strip()
            lab = sample["label"]
            a = lab[0] if isinstance(lab, list) and lab else (lab or "")
        elif "question" in sample and "answer" in sample:
            q = (sample["question"] or "").strip()
            a = (sample["answer"] or "").strip()
        else:
            q = (sample.get("query", sample.get("question", "")) or "").strip()
            answer_val = sample.get("label", sample.get("answer", ""))
            a = answer_val[0] if isinstance(answer_val, list) and answer_val else (answer_val or "")

        return {
            "image": img.convert("RGB") if hasattr(img, "convert") else img,
            "question": q,
            "answer": str(a).strip()
        }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal VLM LoRA Training on Gaudi")
    
    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="HF model ID or path")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    
    # Dataset
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceM4/ChartQA",
                        help="Dataset name (ChartQA, DocVQA, TextVQA, etc.)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training samples for debugging")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Limit evaluation samples for debugging")
    
    # Training
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true", 
                        help="Whether to run evaluation")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    
    # Hardware
    parser.add_argument("--use_habana", action="store_true",
                        help="Use Habana Gaudi")
    parser.add_argument("--gaudi_config_name", type=str, default=None,
                        help="Gaudi config name or path (auto-generated if not provided)")
    parser.add_argument("--use_lazy_mode", action="store_true",
                        help="Use Habana lazy mode (dynamic shapes)")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing (HF TrainingArguments default behavior)")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed config file path for ZeRO optimization")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./output_vlm",
                        help="Output directory")
    
    # DeepSpeed / Distributed Training
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set by DeepSpeed/torch.distributed.launch)")
    
    args = parser.parse_args()
    
    # Run training
    trainer = MultimodalVLMTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()

