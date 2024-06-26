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
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, BatchSampler
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, has_length, seed_worker
from transformers.utils import is_datasets_available
from trl import SFTTrainer
from trl.extras.dataset_formatting import get_formatting_func_from_dataset
from trl.import_utils import is_peft_available
from trl.trainer.utils import (
    DataCollatorForCompletionOnlyLM,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from ... import GaudiConfig, GaudiTrainer, GaudiTrainingArguments

class GaudiSFTTrainer(SFTTrainer, GaudiTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        args: GaudiTrainingArguments = None,
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
    ):
        """
        Copied from SFTTrainer.__init__: https://github.com/huggingface/trl/blob/v0.7.6/trl/trainer/sft_trainer.py#L120
        The only differences are:
        - add new args gaudi_config
        - use GaudiTrainer instead of Trainer
        - cast peft model to bf16.
        """
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the SFTTrainer. But your model is already instantiated.")

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

        if packing and data_collator is not None and isinstance(data_collator, DataCollatorForCompletionOnlyLM):
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
                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                    preprare_model_kwargs = {
                        "use_gradient_checkpointing": getattr(args, "gradient_checkpointing", False)
                    }

                    if _support_gc_kwargs:
                        preprare_model_kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)

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

                model = get_peft_model(model, peft_config)
                if args.bf16:
                    model = model.to(torch.bfloat16)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

        if max_seq_length is None:
            # to overcome some issues with broken tokenizers
            max_seq_length = min(tokenizer.model_max_length, 1024)

            warnings.warn(
                f"You didn't pass a `max_seq_length` argument to the SFTTrainer, this will default to {max_seq_length}"
            )

        self.dataset_num_proc = dataset_num_proc
        self.dataset_batch_size = dataset_batch_size

        self._trainer_supports_neftune = hasattr(args, "neftune_noise_alpha")

        if neftune_noise_alpha is not None and self._trainer_supports_neftune:
            args.neftune_noise_alpha = neftune_noise_alpha
            warnings.warn(
                "You passed a `neftune_noise_alpha` argument to the SFTTrainer, the value you passed will override the one in the `TrainingArguments`."
            )
            # self.neftune_noise_alpha is done at Trainer level
        elif not self._trainer_supports_neftune:
            self.neftune_noise_alpha = neftune_noise_alpha

        if formatting_func is None and dataset_text_field is None:
            # check if dataset has ChatML format or instruction format and is supported
            # if not stays #None
            formatting_func = get_formatting_func_from_dataset(train_dataset, tokenizer)

        if not packing:
            if dataset_text_field is None and formatting_func is None:
                raise ValueError(
                    "You passed `packing=False` to the SFTTrainer, but you didn't pass a `dataset_text_field` or `formatting_func` argument."
                )

            if data_collator is None:
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        if dataset_kwargs is None:
            dataset_kwargs = {}
        if train_dataset is not None:
            train_dataset = self._prepare_dataset(
                train_dataset,
                tokenizer,
                packing,
                dataset_text_field,
                max_seq_length,
                formatting_func,
                num_of_sequences,
                chars_per_token,
                **dataset_kwargs,
            )

        if eval_dataset is not None:
            _multiple = isinstance(eval_dataset, dict)
            _eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}
            for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
                _eval_datasets[_eval_dataset_name] = self._prepare_dataset(
                    _eval_dataset,
                    tokenizer,
                    packing,
                    dataset_text_field,
                    max_seq_length,
                    formatting_func,
                    num_of_sequences,
                    chars_per_token,
                    **dataset_kwargs,
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

        if self.args.max_steps > 0 and packing:
            warnings.warn(
                "You passed `packing=True` to the SFTTrainer, and you are training your model with `max_steps` strategy. The dataset will be iterated until the `max_steps` are reached."
            )
            self.train_dataset.infinite = True
        elif self.args.max_steps == -1 and packing:
            self.train_dataset.infinite = False

    def _get_lengths(self, dataset) -> List[int]:
        """
        Returns list containing length of each example in dataset
        """
        return [len(example) for example in dataset['input_ids']]

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training Dataloader.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            #"batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["batch_sampler"] =  BucketizeBatchSampler(self._get_lengths(train_dataset), num_buckets=100, batch_size=self._train_batch_size)#self._get_train_sampler()
            #dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return DataLoader(train_dataset, **dataloader_params)

class BucketizeBatchSampler(BatchSampler):
    """Buketized BatchSampler for sequential data with different lengths to reduce number of paddings.
    Args:
        lengths (List[int]): The lengths of the samples in the dataset.
        num_buckets (int): The number of buckets to split the data samples.
        min_len (int, optional): The minimum sample lengths to keep.
            (Default: 0)
        max_len (int or None, optional): The maximum sample lengths to keep. Inferred if not provided.
            (Default ``None``)
        max_token_count (int or None, optional): The max number of tokens in one mini-batch.
            (Default: ``None``)
        batch_size (int or None, optional): The number of samples in one mini-batch.
            (Default: ``None``)
        shuffle (bool, optional): Whether to shuffle buckets for non-monotonic length sampling.
            (Default: True)
        drop_last (bool, optional): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
            (Default: False)
        align_buckets (str): Align sequences to bucket boundaries by padding
            (Default: 'none')
    Note:
        ``max_token_count`` and ``batch_size`` are mutually exclusive. Only one argument of the two
        should have value.
    Note:
        ``drop_last`` is only valid when ``batch_size`` argument is given.
    Note:
        if ``shuffle`` is True, it will only shuffle the data once. Please set ``reload_dataloaders_every_n_epochs=1``
        in pytorch_lightning Trainer to enable shuffling every epoch.
    """
    def __init__(
        self,
        lengths: List[int],
        num_buckets: int,
        min_len: int = 0,
        max_len: Optional[int] = None,
        max_token_count: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        align_buckets: str = 'none'
    ) -> None:
        if max_len is None:
            max_len = max(lengths)
        if not (0 <= min_len <= max_len):
            raise AssertionError("``min_len`` should be non-negative and smaller than ``max_len``")
        if max_token_count is not None and batch_size is not None:
            raise AssertionError("The ``max_token_count`` and ``batch_size`` can't be both set.")
        if max_token_count is None and batch_size is None:
            raise AssertionError("One of ``max_token_count`` or ``batch_size`` must be set.")
        if max_token_count is not None:
            assert (
                max_len <= max_token_count
            ), "The  ``max_token_count`` must be greater than or equal to the maximum value of ``lengths``."
        # Filter out samples which are outside the bounds of [min_len, max_len]
        filtered_length_idx = [(length, i) for i, length in enumerate(lengths) if min_len <= length <= max_len]
        if len(filtered_length_idx) == 0:
            raise AssertionError("``lengths`` cannot be empty after filtering.")
        sorted_filtered_length_idx = sorted(filtered_length_idx, key=lambda x: x[0])
        self.lengths = [e[0] for e in sorted_filtered_length_idx]
        self.indices = [e[1] for e in sorted_filtered_length_idx]
        self.max_token_count = max_token_count
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.align_buckets = align_buckets
        self.buckets, self.boundaries = self._get_buckets(self.lengths, num_buckets, min_len, max_len)
        self._update_iter_list()

    def gen_bucket_boundary_fn(self):
        def helper(x):
            idx = np.where(self.boundaries > x)[0][0]
            return int(np.ceil(self.boundaries[idx].item()))
        return helper

    def _get_buckets(self, lengths: List[int], num_buckets: int, min_len: int, max_len: int) -> Dict[int, Tensor]:
        """Generate buckets based on the dataset.
        Args:
            lengths (List[int]): The lengths of the samples in the dataset.
            num_buckets (int): The number of buckets.
            min_len (int): The lower bound of the evenly spaced length intervals to determine bucket width.
            max_len (int): The upper bound of the evenly spaced length intervals to determine bucket width.
        Returns:
            (dict[int, Tensor]): A dictionary in which the key is the bucket index, the value is
                the Tensor of corresponding sample indices.
        """
        buckets = {}
        boundaries = torch.linspace(min_len - 1, max_len + 1, num_buckets + 1)
        bucket_ids = torch.bucketize(torch.tensor(lengths), boundaries)
        for i in range(bucket_ids.size(0)):
            bucket_id = int(bucket_ids[i])
            if bucket_id in buckets:
                buckets[bucket_id].append(i)
            else:
                buckets[bucket_id] = [i]
        for k in buckets:
            buckets[k] = torch.as_tensor(buckets[k], dtype=torch.int)
        buckets = {k: v for k, v in sorted(buckets.items())}
        return buckets, boundaries

    def _update_iter_list(self) -> None:
        if self.shuffle:
            for k in self.buckets:
                self.buckets[k] = self.buckets[k][torch.randperm(self.buckets[k].size(0))]
        self.iter_list = []
        total_len = 0
        batch = []
        max_batch_size = self.max_token_count if self.max_token_count else self.batch_size
        align_to_bucket_boundary = self.gen_bucket_boundary_fn()
        for k in self.buckets:
            for i in range(self.buckets[k].size(0)):
                index = int(self.buckets[k][i])
                sample_length = self.lengths[index] if self.max_token_count else 1
                aligned_len = align_to_bucket_boundary(sample_length) if self.align_buckets=='bucket1' else sample_length
                if total_len + aligned_len <= max_batch_size:
                    batch.append(self.indices[index])
                    total_len += aligned_len
                else:
                    self.iter_list.append(batch)
                    batch = [self.indices[index]]
                    total_len = aligned_len
        if len(batch) > 0 and (self.max_token_count or not self.drop_last):
            self.iter_list.append(batch)

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.iter_list)

    def __len__(self):
        if self.batch_size or (self.max_token_count and not self.shuffle):
            return len(self.iter_list)

class CollateFnDolly:
    """The collate class for HuBERT pre-training and fine-tuning.
    Args:
        feature_type (str): The type of features for KMeans clustering.
            Options: [``mfcc``, ``hubert``].
        pad (bool): If ``True``, the waveforms and labels will be padded to the
            max length in the mini-batch. If ``pad`` is False, the waveforms
            and labels will be cropped to the minimum length in the mini-batch.
            (Default: False)
        rand_crop (bool): if ``True``, the starting index of the waveform
            and label is random if the length is longer than the minimum
            length in the mini-batch.
    """
    def __init__(
        self,
        feature_type: str,
        pad: bool = False,
        rand_crop: bool = True,
        aligner = None
    ) -> None:
        self.feature_type = feature_type
        self.pad = pad
        self.rand_crop = rand_crop
        self.aligner = aligner

    def __call__(self, batch: List[Tuple[Tensor, Tensor, int]]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            batch (List[Tuple(Tensor, Tensor, int)]):
                The list of tuples that contains the waveforms, labels, and audio lengths.
        Returns:
            (Tuple(Tensor, Tensor, Tensor)):
                The Tensor of waveforms with dimensions `(batch, time)`.
                The Tensor of labels with dimensions `(batch, seq)`.
                The Tensor of audio lengths with dimension `(batch,)`.
        """
        if self.pad:
            num_frames = max([sample[0].shape[1] for sample in batch])
        else:
            num_frames = min([sample[0].shape[1] for sample in batch])
        if self.aligner is not None:
            bkt_boundary = self.aligner(num_frames)
            assert bkt_boundary >= num_frames
        else:
            bkt_boundary = None
        waveforms, labels, lengths = [], [], []
        for sample in batch:
            waveform, label, length = sample
            # The MFCC feature is 10ms per frame, while the HuBERT's transformer output
            # is 20ms per frame. Downsample the KMeans label if it's generated by MFCC features.
            if self.feature_type == "mfcc":
                label = label[::2]
            waveform, label, length = _crop_audio_label(waveform, label, length, num_frames, self.rand_crop)
            waveforms.append(waveform)
            lengths.append(length)
            labels.append(label)
        # make sure the shapes are the same if not apply zero-padding
        if not self.pad:
            assert all(
                [waveform.shape[0] == waveforms[0].shape[0] for waveform in waveforms]
            ), "The dimensions of the waveforms should be identical in the same batch."
            assert all(
                [label.shape[0] == labels[0].shape[0] for label in labels]
            ), "The dimensions of the labels should be identical in the same batch."
        if bkt_boundary is not None:
            waveforms.append(torch.zeros([bkt_boundary]))
        waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        if bkt_boundary is not None:
            waveforms = waveforms[:-1,:]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        if bkt_boundary is not None:
            diff = label_len(waveforms.shape[1]) - labels.shape[1]
            labels = torch.nn.functional.pad(labels,(0,diff,0,0))
        lengths = torch.tensor(lengths)
        return waveforms, labels, lengths

#    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
#        """
#        Returns the eval Habana Media Dataloader.
#        """
#        if eval_dataset is None and self.eval_dataset is None:
#            raise ValueError("Trainer: evaluation requires an eval_dataset.")
#        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
#        data_collator = self.data_collator
#
#        if is_datasets_available() and isinstance(eval_dataset, Dataset):
#            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
#        else:
#            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")
#
#        dataloader_params = {
#            "batch_size": self.args.eval_batch_size,
#            "collate_fn": data_collator,
#            "num_workers": self.args.dataloader_num_workers,
#            "pin_memory": self.args.dataloader_pin_memory,
#        }
#
#        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
#            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
#            dataloader_params["drop_last"] = True
#
#        return DataLoader(eval_dataset, **dataloader_params)
#
#    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
#        """
#        Returns the test Habana Media Dataloader.
#        """
#        import pdb; pdb.set_trace()
#        data_collator = self.data_collator
#
#        if is_datasets_available() and isinstance(test_dataset, Dataset):
#            test_dataset = self._remove_unused_columns(test_dataset, description="test")
#        else:
#            data_collator = self._get_collator_with_removed_columns(data_collator, description="test")
#
#        dataloader_params = {
#            "batch_size": self.args.eval_batch_size,
#            "collate_fn": data_collator,
#            "num_workers": self.args.dataloader_num_workers,
#            "pin_memory": self.args.dataloader_pin_memory,
#        }
#
#        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
#            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset)
#            dataloader_params["drop_last"] = True
#
#        # We use the same batch_size as for eval.
#        return DataLoader(test_dataset, **dataloader_params)
