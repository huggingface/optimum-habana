from typing import Callable, List, Optional, Union

import torch
from accelerate.data_loader import (
    _PYTORCH_DATALOADER_KWARGS,
    BatchSamplerShard,
    DataLoaderDispatcher,
    DataLoaderShard,
    IterableDatasetShard,
    SeedableRandomSampler,
)
from accelerate.utils import RNGType
from torch.utils.data import BatchSampler, DataLoader, IterableDataset

from .state import GaudiAcceleratorState


def gaudi_prepare_data_loader(
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    num_processes: Optional[int] = None,
    process_index: Optional[int] = None,
    split_batches: bool = False,
    put_on_device: bool = False,
    rng_types: Optional[List[Union[str, RNGType]]] = None,
    dispatch_batches: Optional[bool] = None,
    even_batches: bool = True,
    slice_fn_for_dispatch: Optional[Callable] = None,
) -> DataLoader:
    """
    Wraps a PyTorch `DataLoader` to generate batches for one of the processes only.

    Depending on the value of the `drop_last` attribute of the `dataloader` passed, it will either stop the iteration
    at the first batch that would be too small / not present on all processes or loop with indices from the beginning.

    Args:
        dataloader (`torch.utils.data.dataloader.DataLoader`):
            The data loader to split across several devices.
        device (`torch.device`):
            The target device for the returned `DataLoader`.
        num_processes (`int`, *optional*):
            The number of processes running concurrently. Will default to the value given by
            [`GaudiAcceleratorState`].
        process_index (`int`, *optional*):
            The index of the current process. Will default to the value given by [`GaudiAcceleratorState`].
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether the resulting `DataLoader` should split the batches of the original data loader across devices or
            yield full batches (in which case it will yield batches starting at the `process_index`-th and advancing of
            `num_processes` batches at each iteration).

            Another way to see this is that the observed batch size will be the same as the initial `dataloader` if
            this option is set to `True`, the batch size of the initial `dataloader` multiplied by `num_processes`
            otherwise.

            Setting this option to `True` requires that the batch size of the `dataloader` is a round multiple of
            `batch_size`.
        put_on_device (`bool`, *optional*, defaults to `False`):
            Whether or not to put the batches on `device` (only works if the batches are nested list, tuples or
            dictionaries of tensors).
        rng_types (list of `str` or [`~utils.RNGType`]):
            The list of random number generators to synchronize at the beginning of each iteration. Should be one or
            several of:

            - `"torch"`: the base torch random number generator
            - `"cuda"`: the CUDA random number generator (GPU only)
            - `"xla"`: the XLA random number generator (TPU only)
            - `"generator"`: the `torch.Generator` of the sampler (or batch sampler if there is no sampler in your
              dataloader) or of the iterable dataset (if it exists) if the underlying dataset is of that type.

        dispatch_batches (`bool`, *optional*):
            If set to `True`, the datalaoder prepared is only iterated through on the main process and then the batches
            are split and broadcast to each process. Will default to `True` when the underlying dataset is an
            `IterableDataset`, `False` otherwise.
        even_batches (`bool`, *optional*, defaults to `True`):
            If set to `True`, in cases where the total batch size across all processes does not exactly divide the
            dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among
            all workers.
        slice_fn_for_dispatch (`Callable`, *optional*`):
            If passed, this function will be used to slice tensors across `num_processes`. Will default to
            [`~utils.slice_tensors`]. This argument is used only when `dispatch_batches` is set to `True` and will be
            ignored otherwise.

    Returns:
        `torch.utils.data.dataloader.DataLoader`: A new data loader that will yield the portion of the batches

    <Tip warning={true}>

    `BatchSampler`s with varying batch sizes are not enabled by default. To enable this behaviour, set `even_batches`
    equal to `False`

    </Tip>
    """
    if dispatch_batches is None:
        if not put_on_device:
            dispatch_batches = False
        else:
            dispatch_batches = isinstance(dataloader.dataset, IterableDataset)

    if dispatch_batches and not put_on_device:
        raise ValueError("Using `dispatch_batches=True` requires `put_on_device=True`.")
    # Grab defaults from GaudiAcceleratorState
    state = GaudiAcceleratorState()
    if num_processes is None:
        num_processes = state.num_processes
    if process_index is None:
        process_index = state.process_index

    # Sanity check
    if split_batches and dataloader.batch_size > 1 and dataloader.batch_size % num_processes != 0:
        raise ValueError(
            f"To use a `DataLoader` in `split_batches` mode, the batch size ({dataloader.batch_size}) "
            f"needs to be a round multiple of the number of processes ({num_processes})."
        )

    new_dataset = dataloader.dataset
    # Iterable dataset doesn't like batch_sampler, but data_loader creates a default one for it
    new_batch_sampler = dataloader.batch_sampler if not isinstance(new_dataset, IterableDataset) else None
    sampler_is_batch_sampler = False
    synchronized_generator = None
    sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler)
    if sampler_is_batch_sampler:
        sampler = dataloader.sampler.sampler
    else:
        sampler = dataloader.batch_sampler.sampler
    # Commenting the block below as it makes the accuracy decrease quite a lot for a few models and tasks
    # e.g. audio classification with Wav2Vec2 or Seq2SeqQA with T5
    # if isinstance(sampler, RandomSampler) and num_processes > 1:
    #     # When iterating through the dataloader during distributed processes
    #     # we want to ensure that on each process we are iterating through the same
    #     # samples in the same order if a seed is set. This requires a tweak
    #     # to the `torch.utils.data.RandomSampler` class (if used).
    #     sampler = SeedableRandomSampler(
    #         data_source=sampler.data_source,
    #         replacement=sampler.replacement,
    #         num_samples=sampler._num_samples,
    #         generator=getattr(sampler, "generator", torch.Generator()),
    #     )

    # No change if no multiprocess
    if num_processes != 1 and not dispatch_batches:
        if isinstance(new_dataset, IterableDataset):
            if getattr(dataloader.dataset, "generator", None) is not None:
                synchronized_generator = dataloader.dataset.generator
            new_dataset = IterableDatasetShard(
                new_dataset,
                batch_size=dataloader.batch_size,
                drop_last=dataloader.drop_last,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
            )
        else:
            # New batch sampler for the current process.
            sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler)
            if sampler_is_batch_sampler:
                sampler = dataloader.sampler.sampler
            else:
                sampler = dataloader.batch_sampler.sampler
            if hasattr(sampler, "generator"):
                if sampler.generator is None:
                    sampler.generator = torch.Generator()
                synchronized_generator = sampler.generator

            batch_sampler = dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler
            new_batch_sampler = BatchSamplerShard(
                batch_sampler,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
                even_batches=even_batches,
            )

    # We ignore all of those since they are all dealt with by our new_batch_sampler
    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
    ]

    if rng_types is not None and synchronized_generator is None and "generator" in rng_types:
        rng_types.remove("generator")

    kwargs = {
        k: getattr(dataloader, k, _PYTORCH_DATALOADER_KWARGS[k])
        for k in _PYTORCH_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }

    # Need to provide batch_size as batch_sampler is None for Iterable dataset
    if new_batch_sampler is None:
        kwargs["drop_last"] = dataloader.drop_last
        kwargs["batch_size"] = (
            dataloader.batch_size // num_processes if split_batches and not dispatch_batches else dataloader.batch_size
        )
    if isinstance(sampler, SeedableRandomSampler):
        if sampler_is_batch_sampler:
            dataloader.sampler.sampler = sampler
        else:
            dataloader.batch_sampler.sampler = sampler
    if dispatch_batches:
        kwargs.pop("generator")
        dataloader = DataLoaderDispatcher(
            new_dataset,
            split_batches=split_batches,
            batch_sampler=new_batch_sampler,
            _drop_last=dataloader.drop_last,
            slice_fn=slice_fn_for_dispatch,
            **kwargs,
        )
    elif sampler_is_batch_sampler:
        dataloader = DataLoaderShard(
            new_dataset,
            device=device if put_on_device else None,
            sampler=new_batch_sampler,
            batch_size=dataloader.batch_size,
            rng_types=rng_types,
            _drop_last=dataloader.drop_last,
            synchronized_generator=synchronized_generator,
            **kwargs,
        )
    else:
        dataloader = DataLoaderShard(
            new_dataset,
            device=device if put_on_device else None,
            batch_sampler=new_batch_sampler,
            rng_types=rng_types,
            synchronized_generator=synchronized_generator,
            _drop_last=dataloader.drop_last,
            **kwargs,
        )

    return dataloader
