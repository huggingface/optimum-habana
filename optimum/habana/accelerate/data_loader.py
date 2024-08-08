from typing import Callable, List, Optional, Union

import torch
from accelerate.data_loader import (
    _PYTORCH_DATALOADER_KWARGS,
    BatchSamplerShard,
    DataLoaderDispatcher,
    DataLoaderShard,
    IterableDatasetShard,
    SeedableRandomSampler,
    get_sampler,
)
from accelerate.state import GradientState
from accelerate.utils import (
    RNGType,
    concatenate,
    find_batch_size,
    get_data_structure,
    is_torch_version,
    send_to_device,
    slice_tensors,
)
from torch.utils.data import BatchSampler, DataLoader, IterableDataset

from .state import GaudiAcceleratorState
from .utils.operations import (
    broadcast,
    broadcast_object_list,
    initialize_tensors,
)


class GaudiDataLoaderDispatcher(DataLoaderDispatcher, DataLoader):
    """
    Subclass of a PyTorch `DataLoader` that will iterate and preprocess on process 0 only, then dispatch on each
    process their part of the batch.

    Args:
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether the resulting `DataLoader` should split the batches of the original data loader across devices or
            yield full batches (in which case it will yield batches starting at the `process_index`-th and advancing of
            `num_processes` batches at each iteration). Another way to see this is that the observed batch size will be
            the same as the initial `dataloader` if this option is set to `True`, the batch size of the initial
            `dataloader` multiplied by `num_processes` otherwise. Setting this option to `True` requires that the batch
            size of the `dataloader` is a round multiple of `batch_size`.
        skip_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning of an iteration.

    **Available attributes:**

        - **total_batch_size** (`int`) -- Total batch size of the dataloader across all processes.
            Equal to the original batch size when `split_batches=True`; otherwise the original batch size * the total
            number of processes

        - **total_dataset_length** (`int`) -- Total length of the inner dataset across all processes.
    """

    def __init__(
        self,
        dataset,
        split_batches: bool = False,
        skip_batches=0,
        _drop_last: bool = False,
        _non_blocking: bool = False,
        slice_fn=None,
        **kwargs,
    ):
        shuffle = False
        if is_torch_version(">=", "1.11.0"):
            from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

            # We need to save the shuffling state of the DataPipe
            if isinstance(dataset, ShufflerIterDataPipe):
                shuffle = dataset._shuffle_enabled
        DataLoader.__init__(self, dataset, **kwargs)
        self.split_batches = split_batches
        if shuffle:
            torch.utils.data.graph_settings.apply_shuffle_settings(dataset, shuffle=shuffle)

        self.gradient_state = GradientState()
        self.state = GaudiAcceleratorState()
        self._drop_last = _drop_last
        self._non_blocking = _non_blocking
        self.skip_batches = skip_batches

        self.slice_fn = slice_tensors if slice_fn is None else slice_fn
        self.iteration = 0

    def _fetch_batches(self, iterator):
        batches, batch = None, None
        # On process 0, we gather the batch to dispatch.
        if self.state.process_index == 0:
            try:
                if self.split_batches:
                    # One batch of the main iterator is dispatched and split.
                    batch = next(iterator)
                else:
                    # num_processes batches of the main iterator are concatenated then dispatched and split.
                    # We add the batches one by one so we have the remainder available when drop_last=False.
                    batches = []
                    for _ in range(self.state.num_processes):
                        batches.append(next(iterator))
                    try:
                        batch = concatenate(batches, dim=0)
                    except RuntimeError as e:
                        raise RuntimeError(
                            "You can't use batches of different size with `dispatch_batches=True` or when using an `IterableDataset`."
                            "either pass `dispatch_batches=False` and have each process fetch its own batch "
                            " or pass `split_batches=True`. By doing so, the main process will fetch a full batch and "
                            "slice it into `num_processes` batches for each process."
                        ) from e
                # In both cases, we need to get the structure of the batch that we will broadcast on other
                # processes to initialize the tensors with the right shape.
                # data_structure, stop_iteration
                batch_info = [get_data_structure(batch), False]
            except StopIteration:
                batch_info = [None, True]
        else:
            batch_info = [None, self._stop_iteration]
        # This is inplace, so after this instruction, every process has the same `batch_info` as process 0.
        broadcast_object_list(batch_info)
        self._stop_iteration = batch_info[1]
        if self._stop_iteration:
            # If drop_last is False and split_batches is False, we may have a remainder to take care of.
            if not self.split_batches and not self._drop_last:
                if self.state.process_index == 0 and len(batches) > 0:
                    batch = concatenate(batches, dim=0)
                    batch_info = [get_data_structure(batch), False]
                else:
                    batch_info = [None, True]
                broadcast_object_list(batch_info)
        return batch, batch_info

    def __iter__(self):
        self.begin()
        self.set_epoch(self.iteration)
        main_iterator = None
        if is_torch_version(">=", "2.0.1"):
            # NOTE PyTorch DataLoader adds forward compatibilities for DataPipes, which broadcasts
            # shared seed to all dist processes. Thus, we need to create iterator for all dist processes.
            # But, we only iterate through the DataLoader on process 0.
            main_iterator = DataLoader.__iter__(self)
        elif self.state.process_index == 0:
            main_iterator = DataLoader.__iter__(self)
        stop_iteration = False
        self._stop_iteration = False
        first_batch = None
        next_batch, next_batch_info = self._fetch_batches(main_iterator)
        batch_index = 0
        while not stop_iteration:
            batch, batch_info = next_batch, next_batch_info

            if self.state.process_index != 0:
                # Initialize tensors on other processes than process 0.
                batch = initialize_tensors(batch_info[0])
            batch = send_to_device(batch, self.state.device, non_blocking=self._non_blocking)
            # Broadcast the batch before splitting it.
            batch = broadcast(batch, from_process=0)

            if not self._drop_last and first_batch is None:
                # We keep at least num processes elements of the first batch to be able to complete the last batch
                first_batch = self.slice_fn(
                    batch,
                    slice(0, self.state.num_processes),
                    process_index=self.state.process_index,
                    num_processes=self.state.num_processes,
                )

            if batch is None:
                raise ValueError(
                    f"Batch does not contain any data (`{batch}`). At the end of all iterable data available before expected stop iteration."
                )

            observed_batch_size = find_batch_size(batch)
            batch_size = observed_batch_size // self.state.num_processes

            stop_iteration = self._stop_iteration
            if not stop_iteration:
                # We may still be at the end of the dataloader without knowing it yet: if there is nothing left in
                # the dataloader since the number of batches is a round multiple of the number of processes.
                next_batch, next_batch_info = self._fetch_batches(main_iterator)
                # next_batch_info[0] is None when there are no more batches, otherwise we still need to process them.
                if self._stop_iteration and next_batch_info[0] is None:
                    stop_iteration = True

            if not self._drop_last and stop_iteration and observed_batch_size % self.state.num_processes != 0:
                # If the last batch is not complete, let's add the first batch to it.
                batch = concatenate([batch, first_batch], dim=0)
                # Batch size computation above is wrong, it's off by 1 so we fix it.
                batch_size += 1

            data_slice = slice(self.state.process_index * batch_size, (self.state.process_index + 1) * batch_size)
            batch = self.slice_fn(
                batch,
                data_slice,
                process_index=self.state.process_index,
                num_processes=self.state.num_processes,
            )

            if stop_iteration:
                self.end_of_dataloader = True
                self.remainder = observed_batch_size
            if batch_index >= self.skip_batches:
                yield batch
            batch_index += 1
        self.iteration += 1
        self.end()


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
    use_seedable_sampler: bool = False,
    non_blocking: bool = False,
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
        use_seedable_sampler (`bool`, *optional*, defaults to `False`):
            Whether to use the [`~data_loader.SeedableRandomSampler`] instead of a `RandomSampler` for better
            reproducability. Comes at a cost of potentially different performances due to different shuffling
            algorithms but ensures results will be the *exact* same. Should be paired with `set_seed()` at every
            `self.set_epoch`
        non_blocking (`bool`, *optional*, defaults to `False`):
            If set to `True`, dataloader will utilize non-blocking host-to-device transfers. If the dataloader has
            `pin_memory` set to `True`, this will help to increase overlap between data transfer and computations.

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
    if split_batches:
        if dataloader.batch_size is not None:
            batch_size_for_check = dataloader.batch_size
        else:
            # For custom batch_sampler
            if hasattr(dataloader.batch_sampler, "batch_size"):
                batch_size_for_check = dataloader.batch_sampler.batch_size
            else:
                raise ValueError(
                    "In order to use `split_batches==True` you must have a `batch_size` attribute either in the passed "
                    "`dataloader` or `dataloader.batch_sampler` objects, and it has to return a natural number. "
                    "Your `dataloader.batch_size` is None and `dataloader.batch_sampler` "
                    f"(`{type(dataloader.batch_sampler)}`) does not have the `batch_size` attribute set."
                )

        if batch_size_for_check > 1 and batch_size_for_check % num_processes != 0:
            raise ValueError(
                f"To use a `DataLoader` in `split_batches` mode, the batch size ({dataloader.batch_size}) "
                f"needs to be a round multiple of the number of processes ({num_processes})."
            )

    new_dataset = dataloader.dataset
    # Iterable dataset doesn't like batch_sampler, but data_loader creates a default one for it
    new_batch_sampler = dataloader.batch_sampler if not isinstance(new_dataset, IterableDataset) else None
    sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler)
    synchronized_generator = None

    sampler = get_sampler(dataloader)
    # Commenting the block below as it makes the accuracy decrease quite a lot for a few models and tasks
    # e.g. audio classification with Wav2Vec2 or Seq2SeqQA with T5
    # if isinstance(sampler, RandomSampler) and use_seedable_sampler:
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
            # The block below was removed in Accelerate but it makes the accuracy decrease quite a lot
            # for a few models and tasks e.g. audio classification with Wav2Vec2 or Seq2SeqQA with T5
            # Keeping it for now
            # New batch sampler for the current process.
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
    if dispatch_batches:
        kwargs.pop("generator")
        dataloader = GaudiDataLoaderDispatcher(
            new_dataset,
            split_batches=split_batches,
            batch_sampler=new_batch_sampler,
            _drop_last=dataloader.drop_last,
            _non_blocking=non_blocking,
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
            _non_blocking=non_blocking,
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
            _non_blocking=non_blocking,
            **kwargs,
        )

    if isinstance(sampler, SeedableRandomSampler) and use_seedable_sampler:
        dataloader.set_sampler(sampler)

    return dataloader
