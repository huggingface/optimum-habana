# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import dataclasses
import inspect
import warnings
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from trl import SFTTrainer
from trl.extras.dataset_formatting import get_formatting_func_from_dataset
from trl.import_utils import is_peft_available
from trl.trainer.utils import (
    DataCollatorForCompletionOnlyLM,
    RichProgressCallback,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from ... import GaudiConfig, GaudiTrainer
from .sft_config import GaudiSFTConfig


class BucketedDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def _get_bucketed_len(self, examples):
        max_sentence_len = max([len(k["input_ids"]) for k in examples])
        if max_sentence_len > self.buckets[-1]:
            self.buckets = np.append(self.buckets, max_sentence_len)
            curr_bucket = max_sentence_len
        else:
            curr_bucket = self.buckets[np.argmin(np.where(max_sentence_len <= self.buckets))]
        return curr_bucket

    # copied from https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/data/data_collator.py#L758
    # change is pad_to_multiple_of=self.pad_to_multiple_of -> pad_to_multiple_of=bucketed_len
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            bucketed_len = self._get_bucketed_len(examples)
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                examples,
                return_tensors="pt",
                pad_to_multiple_of=bucketed_len,  # self.pad_to_multiple_of
            )
        else:
            assert False, "This path has not been implemented/tested yet"
            # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/data/data_collator.py#L765

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


class GaudiSFTTrainer(SFTTrainer, GaudiTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        args: Optional[GaudiSFTConfig] = None,
        gaudi_config: GaudiConfig = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        dataset_text_field: Optional[str] = None,
        packing: Optional[bool] = False,
        formatting_func: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        infinite: Optional[bool] = None,
        num_of_sequences: Optional[int] = 1024,
        chars_per_token: Optional[float] = 3.6,
        dataset_num_proc: Optional[int] = None,
        dataset_batch_size: int = 1000,
        neftune_noise_alpha: Optional[float] = None,
        model_init_kwargs: Optional[Dict] = None,
        dataset_kwargs: Optional[Dict] = None,
        eval_packing: Optional[bool] = None,
        num_buckets: Optional[int] = -1,
    ):
        """
        Copied from SFTTrainer.__init__: https://github.com/huggingface/trl/blob/v0.9.6/trl/trainer/sft_trainer.py#L116
        The only differences are:
        - add new args gaudi_config
        - use GaudiTrainer instead of Trainer
        - cast peft model to bf16.
        - num_buckets: Number of buckets. > 0 means apply bucketing, <= 0  means no bucketing
        """
        if num_buckets > 0:
            assert (
                data_collator is None
            ), "For bucketing (num_buckets > 0), we only support data_collator=None (later it becomes DataCollatorForLanguageModeling)"
        if args is None:
            output_dir = "tmp_trainer"
            warnings.warn(f"No `SFTConfig` passed, using `output_dir={output_dir}`.")
            args = GaudiSFTConfig(output_dir=output_dir)
        elif args is not None and args.__class__.__name__ == "TrainingArguments":
            args_as_dict = args.to_dict()
            # Manually copy token values as TrainingArguments.to_dict() redacts them
            args_as_dict.update({k: getattr(args, k) for k in args_as_dict.keys() if k.endswith("_token")})
            args = GaudiSFTConfig(**args_as_dict)

        if model_init_kwargs is not None:
            warnings.warn(
                "You passed `model_init_kwargs` to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.model_init_kwargs = model_init_kwargs
        if getattr(args, "model_init_kwargs", None) is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_init_kwargs to the SFTConfig, but your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs

            torch_dtype = model_init_kwargs["torch_dtype"]
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)

                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the SFTConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )

            model_init_kwargs["torch_dtype"] = torch_dtype

        if infinite is not None:
            warnings.warn(
                "The `infinite` argument is deprecated and will be removed in a future version of TRL. Use `TrainingArguments.max_steps` or `TrainingArguments.num_train_epochs` instead to control training length."
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the SFTTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if packing:
            warnings.warn(
                "You passed a `packing` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.packing = packing
        if eval_packing is not None:
            warnings.warn(
                "You passed a `eval_packing` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.eval_packing = eval_packing

        if args.packing and data_collator is not None and isinstance(data_collator, DataCollatorForCompletionOnlyLM):
            raise ValueError(
                "You passed a `DataCollatorForCompletionOnlyLM` to the SFTTrainer. This is not compatible with the `packing` argument."
            )

        if is_peft_available() and peft_config is not None:
            if not isinstance(peft_config, PeftConfig):
                raise ValueError(
                    "If you want to use the PeftModel, you need to pass a PeftConfig object to the SFTTrainer."
                    f" and you passed a {type(peft_config)}."
                )

            if not isinstance(model, PeftModel):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )
                gradient_checkpointing_kwargs = getattr(args, "gradient_checkpointing_kwargs", None) or {}
                is_sharded_qlora = False
                # Below is to support QLoRA + FSDP / DS-Zero3 - one should never call
                # peft_module_casting_to_bf16 or prepare_model_for_kbit_training when doing
                # QLoRA + FSDP / DS-Zero3
                if getattr(model, "is_loaded_in_4bit", False):
                    for _, param in model.named_parameters():
                        if param.__class__.__name__ == "Params4bit":
                            is_sharded_qlora = param.data.device.type == "cpu"
                            break
                if getattr(model, "is_loaded_in_8bit", False) or (
                    getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora
                ):
                    prepare_model_kwargs = {
                        "use_gradient_checkpointing": getattr(args, "gradient_checkpointing", False)
                    }

                    if _support_gc_kwargs:
                        prepare_model_kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

                    if args is not None:
                        args = dataclasses.replace(args, gradient_checkpointing=False)
                elif getattr(args, "gradient_checkpointing", False) and (
                    "use_reentrant" not in gradient_checkpointing_kwargs
                    or gradient_checkpointing_kwargs["use_reentrant"]
                ):
                    # For backward compatibility with older versions of transformers
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    else:

                        def make_inputs_require_grad(module, input, output):
                            output.requires_grad_(True)

                        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

                if (
                    "autocast_adapter_dtype" in list(inspect.signature(get_peft_model).parameters)
                    and getattr(model, "is_loaded_in_4bit", False)
                    and is_sharded_qlora
                ):
                    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
                else:
                    model = get_peft_model(model, peft_config)
                if (
                    args is not None
                    and args.bf16
                    and getattr(model, "is_loaded_in_4bit", False)
                    and not is_sharded_qlora
                ):
                    model = model.to(torch.bfloat16)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

        if max_seq_length is not None:
            warnings.warn(
                "You passed a `max_seq_length` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.max_seq_length = max_seq_length

        if args.max_seq_length is None:
            # to overcome some issues with broken tokenizers
            max_seq_length = min(tokenizer.model_max_length, 1024)

            warnings.warn(
                f"You didn't pass a `max_seq_length` argument to the SFTTrainer, this will default to {max_seq_length}"
            )

        if dataset_num_proc is not None:
            warnings.warn(
                "You passed a `dataset_num_proc` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.dataset_num_proc = dataset_num_proc
        self.dataset_num_proc = args.dataset_num_proc

        if dataset_batch_size != args.dataset_batch_size:
            warnings.warn(
                "You passed a `dataset_batch_size` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.dataset_batch_size = dataset_batch_size
        self.dataset_batch_size = args.dataset_batch_size

        self._trainer_supports_neftune = hasattr(args, "neftune_noise_alpha")
        if neftune_noise_alpha is not None and self._trainer_supports_neftune:
            args.neftune_noise_alpha = neftune_noise_alpha
            warnings.warn(
                "You passed a `neftune_noise_alpha` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            # self.neftune_noise_alpha is done at Trainer level
        elif not self._trainer_supports_neftune:
            self.neftune_noise_alpha = neftune_noise_alpha

        if dataset_text_field is not None:
            warnings.warn(
                "You passed a `dataset_text_field` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.dataset_text_field = dataset_text_field

        if formatting_func is None and args.dataset_text_field is None:
            # check if dataset has ChatML format or instruction format and is supported
            # if not stays #None
            formatting_func = get_formatting_func_from_dataset(train_dataset, tokenizer)
            # if a template is detected, we don't need to add special tokens again
            if formatting_func is not None:
                if dataset_kwargs is None:
                    dataset_kwargs = {"add_special_tokens": False}
                else:
                    dataset_kwargs["add_special_tokens"] = False

        if not args.packing:
            # If we aren't skipping data preparation, then a dataset_text_field
            # or formatting_func must be provided.
            if (
                args.dataset_text_field is None
                and formatting_func is None
                and dataset_kwargs is not None
                and "skip_prepare_dataset" in dataset_kwargs
                and dataset_kwargs["skip_prepare_dataset"]
            ):
                raise ValueError(
                    "You passed `packing=False` to the SFTTrainer/SFTConfig, but you didn't pass a `dataset_text_field` or `formatting_func` argument."
                )

            if data_collator is None:
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        if num_of_sequences != args.num_of_sequences:
            warnings.warn(
                "You passed a `num_of_sequences` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.num_of_sequences = num_of_sequences

        if chars_per_token != args.chars_per_token:
            warnings.warn(
                "You passed a `chars_per_token` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.chars_per_token = chars_per_token

        # Pre-process the datasets only once per node. The remaining processes will use the cache.
        with PartialState().local_main_process_first():
            if dataset_kwargs is not None:
                warnings.warn(
                    "You passed a `dataset_kwargs` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
                )
                args.dataset_kwargs = dataset_kwargs
            if args.dataset_kwargs is None:
                args.dataset_kwargs = {}
            if train_dataset is not None:
                train_dataset = self._prepare_dataset(
                    train_dataset,
                    tokenizer,
                    args.packing,
                    args.dataset_text_field,
                    args.max_seq_length,
                    formatting_func,
                    args.num_of_sequences,
                    args.chars_per_token,
                    remove_unused_columns=args.remove_unused_columns if args is not None else True,
                    **args.dataset_kwargs,
                )
            if eval_dataset is not None:
                _multiple = isinstance(eval_dataset, dict)
                _eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}

                eval_packing = args.packing if args.eval_packing is None else args.eval_packing

                for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
                    _eval_datasets[_eval_dataset_name] = self._prepare_dataset(
                        _eval_dataset,
                        tokenizer,
                        eval_packing,
                        args.dataset_text_field,
                        args.max_seq_length,
                        formatting_func,
                        args.num_of_sequences,
                        args.chars_per_token,
                        remove_unused_columns=args.remove_unused_columns if args is not None else True,
                        **args.dataset_kwargs,
                    )
                if not _multiple:
                    eval_dataset = _eval_datasets["singleton"]

        if tokenizer.padding_side is not None and tokenizer.padding_side != "right":
            warnings.warn(
                "You passed a tokenizer with `padding_side` not equal to `right` to the SFTTrainer. This might lead to some unexpected behaviour due to "
                "overflow issues when training a model in half-precision. You might consider adding `tokenizer.padding_side = 'right'` to your code."
            )

        GaudiTrainer.__init__(
            self,
            model=model,
            args=args,
            gaudi_config=gaudi_config,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if self.args.max_steps > 0 and args.packing:
            warnings.warn(
                "You passed `packing=True` to the SFTTrainer/SFTConfig, and you are training your model with `max_steps` strategy. The dataset will be iterated until the `max_steps` are reached."
            )
            self.train_dataset.infinite = True
        elif self.args.max_steps == -1 and args.packing:
            self.train_dataset.infinite = False

        if num_buckets > 0:
            train_dataloader = self.get_train_dataloader()
            batched_sentence_lengths = [batch["input_ids"].shape[1] for batch in train_dataloader]
            buckets = self._get_buckets(batched_sentence_lengths, num_buckets=num_buckets)
            self.data_collator = BucketedDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            self.data_collator.buckets = buckets

        if any(isinstance(callback, RichProgressCallback) for callback in self.callback_handler.callbacks):
            for callback in self.callback_handler.callbacks:
                # Remove the PrinterCallback to avoid duplicated prints in case we passed a `RichProgressCallback`
                if callback.__class__.__name__ == "PrinterCallback":
                    self.callback_handler.pop_callback(callback)

    def _get_buckets(self, sentence_lengths, num_buckets):
        return np.unique(
            np.percentile(
                sentence_lengths,
                np.linspace(0, 100, num_buckets + 1),
                interpolation="lower",
            )[1:]
        )
